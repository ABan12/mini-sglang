import importlib
import json
from pathlib import Path

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3_5ForConditionalGeneration": (".qwen3_5", "Qwen35ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
}



def is_model_registered( architecture: str) -> bool:
    return architecture in _MODEL_REGISTRY

def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)

_UNIMPLEMENTED = {
    "DeepseekV3ForCausalLM": "MLA attention/cache and routed/shared MoE are not implemented",
}


def require_supported_model(model_path: str) -> str:
    folder = Path(model_path)
    if folder.is_dir():
        config_path = folder / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        config_path = Path(hf_hub_download(repo_id=model_path, filename="config.json"))

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("config.json must contain an object")
    architectures = raw.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not isinstance(architectures[0], str)
    ):
        raise ValueError("config.json: architectures[0] is required")

    architecture = architectures[0]
    if architecture in _UNIMPLEMENTED:
        raise NotImplementedError(
            f"{architecture}: {_UNIMPLEMENTED[architecture]} in mini-SGLang"
        )
    if not is_model_registered(architecture):
        raise ValueError(f"Model architecture {architecture} not supported")
    return architecture

__all__ = ["get_model_class", "is_model_registered", "require_supported_model"]
