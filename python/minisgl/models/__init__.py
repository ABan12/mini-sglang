from __future__ import annotations

from typing import TYPE_CHECKING
from .config import ModelConfig, RotaryConfig

if TYPE_CHECKING:
    from .base import BaseLLMModel

def create_model(model_config: ModelConfig) -> BaseLLMModel:
    from .register import get_model_class
    return get_model_class(model_config.architectures[0], model_config)

def __getattr__(name: str):
    if name == "BaseLLMModel":
        from .base import BaseLLMModel
        return BaseLLMModel
    if name == "load_weight":
        from .weight import load_weight
        return load_weight
    if name == "get_model_class":
        from .register import get_model_class
        return get_model_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    

__all__ = ["create_model", "load_weight", "RotaryConfig"]
