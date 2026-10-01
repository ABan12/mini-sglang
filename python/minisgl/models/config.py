from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None
    rope_type: str = "default"
    partial_rotary_factor: float = 1.0
    mrope_interleaved: bool = False
    mrope_section: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    layer_types: tuple[str, ...] = ()
    attn_output_gate: bool = False
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0
    mamba_ssm_dtype: str | None = None

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        top = config
        # 读取配置里的纯文本模型参数
        text = getattr(top, "text_config", None)
        config = deepcopy(text if text is not None else top)
        # 如果配置是dict字典 就转化为config配置对象
        if isinstance(config, dict):
            config = PretrainedConfig(**config)
        for name in ("rope_theta", "rope_scaling"):
            if getattr(config, name, None) is None and getattr(top, name, None) is not None:
                setattr(config, name, deepcopy(getattr(top, name)))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        is_qwen35 = model_type == "qwen3_5_text"
        architectures = (
            getattr(top, "architectures", None)
            or getattr(config, "architectures", None)
            or ["LlamaForCausalLM"]
        )
        # 优先使用文本配置中的共享权重设置；属性不存在时，再取外层值；外层也没有时，默认 False
        tie_word_embeddings = getattr(config, "tie_word_embeddings", getattr(top, "tie_word_embeddings", False))
        # Read RoPE parameters from Transformers 4.x and 5.x configurations.
        rope_scaling = deepcopy(getattr(config, "rope_scaling", None))
        rope_parameters = deepcopy(getattr(config, "rope_parameters", None) or {})
        rope_theta = (
            rope_parameters.get("rope_theta")
            or getattr(config, "rope_theta", None)
            or (rope_scaling or {}).get("rope_theta")
        )
        partial_factor = rope_parameters.get("partial_rotary_factor", 1.0) if is_qwen35 else 1.0
        # Transformers 5.x also exposes the default parameters as rope_scaling.
        if rope_scaling is not None and rope_scaling.get("rope_type") == "default":
            rope_scaling = None


        result = cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=int(head_dim * partial_factor),
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
                rope_type=rope_parameters.get("rope_type", "default"),
                partial_rotary_factor=partial_factor,
                mrope_interleaved=rope_parameters.get("mrope_interleaved", False),
                mrope_section=tuple(rope_parameters.get("mrope_section", [])),
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=list(architectures),
            layer_types=tuple(getattr(config, "layer_types", None) or []),
            attn_output_gate=getattr(config, "attn_output_gate", False),
            linear_num_key_heads=getattr(config, "linear_num_key_heads", 0),
            linear_num_value_heads=getattr(config, "linear_num_value_heads", 0),
            linear_key_head_dim=getattr(config, "linear_key_head_dim", 0),
            linear_value_head_dim=getattr(config, "linear_value_head_dim", 0),
            linear_conv_kernel_dim=getattr(config, "linear_conv_kernel_dim", 0),
            mamba_ssm_dtype=getattr(config, "mamba_ssm_dtype", None),
        )
        return result

