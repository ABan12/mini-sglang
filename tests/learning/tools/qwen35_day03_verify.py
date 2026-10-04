#!/usr/bin/env python3
"""Run a real fresh-process Engine and verify hybrid state, all logits and request isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file, save_file

INPUT_IDS = [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388,
             13, 58737, 279, 5918, 303, 799, 11316, 13]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference_file = args.reference / "model-reference-bfloat16.safetensors"
    refs = load_file(reference_file)  # CPU only: Engine requires CUDA to be uninitialized.
    manifest = json.loads((args.reference / "manifest-bfloat16.json").read_text())
    if manifest["input_ids"] != INPUT_IDS:
        raise ValueError("Reference token IDs differ from the fixed Day 1 input")
    if manifest["config_sha256"] != hashlib.sha256(Path(args.model, "config.json").read_bytes()).hexdigest():
        raise ValueError("Reference model configuration differs")
    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.config import EngineConfig
    from minisgl.engine.engine import Engine
    from minisgl.engine.sample import BatchSamplingArgs

    engine = Engine(EngineConfig(
        model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=2, max_seq_len_override=64, num_page_override=128,
        page_size=1, use_pynccl=False, cuda_graph_bs=[], cuda_graph_max_bs=0,
    ))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"status": "running", "scope": "BF16 TP=1 eager real Engine, fixed 17/9-token requests",
              "capture_path": "all-position shared-head logits from actual Engine text-model hidden; actual sampler-input logits separately checked",
              "input_ids": INPUT_IDS, "short_input_ids": INPUT_IDS[:9],
              "tolerances": {"atol": args.atol, "rtol": args.rtol},
              "environment": {"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "torch": torch.__version__,
                              "transformers": transformers.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(engine.device)},
              "comparisons": {}, "checks": {}, "state_metadata": {}}
    outputs, capture, sample_logits = {}, [], []

    def compare(name, candidate, reference, *, exact=False):
        a, b = candidate.detach().cpu(), reference.detach().cpu()
        shape_ok, dtype_ok = a.shape == b.shape, a.dtype == b.dtype
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        result = {"shape": list(a.shape), "reference_shape": list(b.shape),
                  "dtype": str(a.dtype), "reference_dtype": str(b.dtype),
                  "shape_matches": shape_ok, "dtype_matches": dtype_ok, "finite": finite}
        if shape_ok:
            delta = (a.float() - b.float()).abs()
            result.update(max_abs=delta.max().item(), mean_abs=delta.mean().item(),
                          rms=delta.square().mean().sqrt().item(), exact=bool(torch.equal(a, b)))
            result["passed"] = finite and dtype_ok and (
                torch.equal(a, b) if exact else torch.allclose(a.float(), b.float(), atol=args.atol, rtol=args.rtol))
        else:
            result["passed"] = False
        report["comparisons"][name] = result

    def store(name, value):
        outputs[name] = value.detach().cpu().contiguous().clone()

    text_forward = engine.model.model.forward

    def capture_text(*positional, **kwargs):
        result = text_forward(*positional, **kwargs)
        hidden, _ = result
        capture.append(F.linear(hidden, engine.model.model.embed_tokens.weight).detach().cpu())
        return result

    sampler_forward = engine.sampler.sample

    def capture_sampler(logits, sampling_args):
        sample_logits.append(logits.detach().cpu().clone())
        return sampler_forward(logits, sampling_args)

    engine.model.model.forward = capture_text
    engine.sampler.sample = capture_sampler
    # Noncontiguous locations ensure logical sequence order is read through page_table.
    permutation = torch.randperm(128, generator=torch.Generator().manual_seed(35))
    physical = [permutation[:32].to(engine.device), permutation[32:64].to(engine.device)]
    report["physical_locations"] = [row.cpu().tolist() for row in physical]
    next_uid = 0

    def new_request(table, ids, count):
        nonlocal next_uid
        next_uid += 1
        return Req(torch.tensor(ids[:count], dtype=torch.int32), table_idx=table, cached_len=0,
                   output_len=64, uid=next_uid, sampling_params=SamplingParams(), cache_handle=None)

    def run_batch(work):
        capture.clear()
        sample_logits.clear()
        reqs, ids, positions, locations = [], [], [], []
        for req, sequence, count in work:
            start, end = req.cached_len, req.cached_len + count
            req.input_ids = torch.tensor(sequence[:end], dtype=torch.int32)
            req.device_len = end
            reqs.append(req)
            ids.append(req.input_ids[start:end])
            positions.append(torch.arange(start, end, dtype=torch.int32))
            loc = physical[req.table_idx][start:end]
            locations.append(loc)
            engine.page_table[req.table_idx, start:end] = loc.to(torch.int32)
        phase = "decode" if all(count == 1 and req.cached_len > 0 for req, _, count in work) else "prefill"
        batch = Batch(reqs=reqs, phase=phase)
        batch.padded_reqs = reqs
        batch.input_ids = torch.cat(ids).to(engine.device)
        batch.positions = torch.cat(positions).to(engine.device)
        batch.out_loc = torch.cat(locations).to(torch.int32)
        sampled = engine.forward_batch(batch, BatchSamplingArgs(temperatures=None))
        if len(capture) != len(reqs) or len(sample_logits) != 1:
            raise AssertionError("Engine capture did not match request count")
        expected = sample_logits[0].argmax(-1).to(torch.int32)
        actual = sampled.next_tokens_gpu.cpu()
        if not torch.equal(actual, expected):
            raise AssertionError("Sampler token differs from actual Engine logits argmax")
        return [tensor.clone() for tensor in capture], sample_logits[0].clone()

    def state_tensors(req):
        locations = engine.page_table[req.table_idx, :req.cached_len]
        states = engine.kv_cache.read_states(req.hybrid_state, locations)
        values = {}
        for index, state in enumerate(states):
            names = ("conv", "recurrent") if hasattr(state, "conv") else ("key", "value")
            for name in names:
                values[f"layer_{index:02d}.{name}"] = getattr(state, name).detach().cpu().clone()
        return values

    def finish(prefix, req, parts, last_parts, sizes, reference_prefix=None):
        reference_prefix = reference_prefix or prefix
        logits = torch.cat(parts, dim=1)
        last = torch.stack(last_parts, dim=1)
        store(prefix + ".logits", logits)
        store(prefix + ".engine_last_logits", last)
        compare(prefix + ".logits", logits, refs[reference_prefix + ".logits"])
        ends = torch.tensor(sizes).cumsum(0) - 1
        compare(prefix + ".engine_last_logits", last, refs[reference_prefix + ".logits"][:, ends])
        values = state_tensors(req)
        for suffix, value in values.items():
            name = prefix + "." + suffix
            store(name, value)
            compare(name, value, refs[reference_prefix + "." + suffix])
            report["state_metadata"][name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        return values

    try:
        with torch.inference_mode():
            for mode, sizes in (("full", [17]), ("step", [1] * 17), ("chunked", [5, 12])):
                req = new_request(0, INPUT_IDS, sizes[0])
                parts, last = [], []
                for count in sizes:
                    current, actual_last = run_batch([(req, INPUT_IDS, count)])
                    parts.append(current[0])
                    last.append(actual_last[0:1])
                finish("A." + mode, req, parts, last, sizes)
                engine.kv_cache.release(req.hybrid_state)

            a = new_request(0, INPUT_IDS, 5)
            b = new_request(1, INPUT_IDS[:9], 3)
            parts, last = {"A": [], "B": []}, {"A": [], "B": []}
            schedule = [[("A", a, 5), ("B", b, 3)],
                        [("B", b, 4), ("A", a, 5)],
                        [("A", a, 7), ("B", b, 2)]]
            for step in schedule:
                work = [(req, INPUT_IDS if name == "A" else INPUT_IDS[:9], count)
                        for name, req, count in step]
                current, actual_last = run_batch(work)
                for index, (name, _, _) in enumerate(step):
                    parts[name].append(current[index])
                    last[name].append(actual_last[index:index + 1])
            finish("A.reordered", a, parts["A"], last["A"], [5, 5, 7])
            before_b = finish("B.reordered", b, parts["B"], last["B"], [3, 4, 2])
            old_slot = a.hybrid_state.slot
            engine.kv_cache.release(a.hybrid_state)
            c = new_request(0, INPUT_IDS, 17)
            c.hybrid_state = engine.kv_cache.allocate(c.uid)
            fresh = engine.kv_cache.read_states(c.hybrid_state, physical[0][:0])
            report["checks"]["released_slot_reused"] = c.hybrid_state.slot == old_slot
            report["checks"]["fresh_request_state_layout"] = (
                len(fresh) == 24 and all(state is None for state in fresh))
            slot = c.hybrid_state.slot
            fresh_conv = engine.kv_cache._conv_buffer[:, slot]
            fresh_recurrent = engine.kv_cache._recurrent_buffer[:, slot]
            report["checks"]["new_request_gdn_zero"] = bool(
                torch.count_nonzero(fresh_conv) == 0 and torch.count_nonzero(fresh_recurrent) == 0)
            current, actual_last = run_batch([(c, INPUT_IDS, 17)])
            finish("C.reallocated", c, current, [actual_last[0:1]], [17], "A.full")
            after_b = state_tensors(b)
            for suffix, value in after_b.items():
                compare("B.unchanged_after_C." + suffix, value, before_b[suffix], exact=True)
            report["checks"]["reordered_request_lengths"] = a.cached_len == 17 and b.cached_len == 9
            report["checks"]["sampler_matches_actual_logits"] = True
            report["checks"]["cuda_graph_disabled"] = engine.graph_runner is None
            engine.kv_cache.release(c.hybrid_state)
            engine.kv_cache.release(b.hybrid_state)
    finally:
        engine.model.model.forward = text_forward
        engine.sampler.sample = sampler_forward
        engine.shutdown()
    report["failed"] = [key for key, value in report["comparisons"].items() if not value["passed"]]
    report["failed_checks"] = [key for key, value in report["checks"].items() if not value]
    report["status"] = "passed" if not report["failed"] and not report["failed_checks"] else "failed"
    save_file(outputs, args.output / "engine-outputs.safetensors")
    (args.output / "engine-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "comparisons": len(report["comparisons"]),
                      "failed": report["failed"], "failed_checks": report["failed_checks"]}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
