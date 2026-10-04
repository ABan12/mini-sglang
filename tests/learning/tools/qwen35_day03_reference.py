#!/usr/bin/env python3
"""Independent HF whole-model replay, using fixed Day 1 token IDs and fresh caches."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_qwen35
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

INPUT_IDS = [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388,
             13, 58737, 279, 5918, 303, 799, 11316, 13]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_text_weights(model, checkpoint, dtype, device):
    """Read raw text checkpoint tensors; keep the Day 2 parameter dtype policy."""
    values, restored = {}, []
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as weights:
            for name in weights.keys():
                if not name.startswith("model.language_model."):
                    continue
                value = weights.get_tensor(name)
                target = name.replace("model.language_model.", "model.", 1)
                keep_fp32 = name.endswith((".linear_attn.A_log", ".linear_attn.norm.weight"))
                if keep_fp32:
                    if value.dtype != torch.float32:
                        raise ValueError(f"Expected original FP32 parameter: {name}")
                    restored.append(name)
                values[target] = value.to(device=device, dtype=torch.float32 if keep_fp32 else dtype)
    values["lm_head.weight"] = values["model.embed_tokens.weight"]
    model.load_state_dict(values, strict=True, assign=True)
    return sorted(restored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True).text_config
    config._attn_implementation = "eager"
    # This reference uses the same explicit torch GDN fallback as Day 2.
    hf_qwen35.is_fast_path_available = False
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    restored = load_text_weights(model, args.model, dtype, args.device)
    # RoPE frequencies are nonpersistent buffers, so they are not in the checkpoint.
    model.model.rotary_emb = hf_qwen35.Qwen3_5TextRotaryEmbedding(config).to(args.device)
    model.eval()
    tensors = {}

    def store(name, value):
        tensors[name] = value.detach().cpu().contiguous().clone()

    with torch.inference_mode():
        for request, ids in (("A", INPUT_IDS), ("B", INPUT_IDS[:9])):
            modes = {"full": [len(ids)], "step": [1] * len(ids),
                     "chunked": [5, len(ids) - 5],
                     "reordered": [5, 5, 7] if request == "A" else [3, 4, 2]}
            tokens = torch.tensor([ids], device=args.device)
            for mode, sizes in modes.items():
                cache = DynamicCache(config=config)
                logits, start = [], 0
                for count in sizes:
                    out = model(input_ids=tokens[:, start:start + count],
                                past_key_values=cache, use_cache=True, logits_to_keep=0,
                                position_ids=torch.arange(start, start + count,
                                                          device=args.device)[None, :])
                    logits.append(out.logits)
                    cache = out.past_key_values
                    start += count
                prefix = f"{request}.{mode}"
                store(prefix + ".logits", torch.cat(logits, dim=1))
                for layer, kind in enumerate(config.layer_types):
                    cached = cache.layers[layer]
                    state = f"{prefix}.layer_{layer:02d}"
                    if kind == "linear_attention":
                        store(state + ".conv", cached.conv_states[0])
                        store(state + ".recurrent", cached.recurrent_states[0])
                    else:
                        store(state + ".key", cached.keys)
                        store(state + ".value", cached.values)
    output = args.output / f"model-reference-{args.dtype}.safetensors"
    save_file(tensors, output)
    manifest = {
        "status": "complete", "scope": "HF eager pure-text whole-model fixed-token replay",
        "input_ids": INPUT_IDS, "short_input_ids": INPUT_IDS[:9],
        "config_sha256": sha256(args.model / "config.json"),
        "reference_file": output.name, "reference_sha256": sha256(output),
        "weight_policy": "checkpoint GDN A_log and gated norm remain FP32",
        "restored_fp32_parameters": restored,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(args.device), "dtype": args.dtype,
                        "tf32_matmul": False, "tf32_cudnn": False},
        "source": {"path": inspect.getfile(Qwen3_5ForCausalLM),
                   "sha256": sha256(inspect.getfile(Qwen3_5ForCausalLM))},
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                         "finite": bool(torch.isfinite(value).all())}
                    for key, value in sorted(tensors.items())},
    }
    (args.output / f"manifest-{args.dtype}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "reference": str(output),
                      "tensors": len(tensors), "fp32_parameters": len(restored)}, indent=2))


if __name__ == "__main__":
    main()
