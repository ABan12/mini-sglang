from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from minisgl.core import get_global_ctx
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.deltanet import GDNState, Qwen35GatedDeltaNet
from minisgl.layers.linear import LinearReplicated
from minisgl.layers.norm import Qwen35RMSNorm

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig, RotaryConfig


@dataclass
class KVState:
    key: torch.Tensor
    value: torch.Tensor


def text_position_embeddings(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    config: RotaryConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Default partial RoPE for text, whose three MRoPE axes share positions."""
    dim = config.rotary_dim
    inv_freq = 1.0 / (
        config.base
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim)
    )
    freqs = (
        inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        @ position_ids[:, None, :].float()
    ).transpose(1, 2)
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1).to(x.dtype)
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1).to(x.dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_partial_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = cos.shape[-1]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q_rot, k_rot = q[..., :dim], k[..., :dim]
    q = torch.cat((q_rot * cos + _rotate_half(q_rot) * sin, q[..., dim:]), dim=-1)
    k = torch.cat((k_rot * cos + _rotate_half(k_rot) * sin, k[..., dim:]), dim=-1)
    return q, k


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch, heads, length, dim = x.shape
    return x[:, :, None, :, :].expand(batch, heads, groups, length, dim).reshape(
        batch, heads * groups, length, dim
    )


class Qwen35Attention(BaseOP):
    """Single-request eager attention with an explicit append-only K/V state."""

    def __init__(self, config: ModelConfig):
        self._q_heads = config.num_qo_heads
        self._kv_heads = config.num_kv_heads
        self._head_dim = config.head_dim
        self._groups = self._q_heads // self._kv_heads
        self._scaling = self._head_dim**-0.5
        self._rotary_config = config.rotary_config
        self.q_proj = LinearReplicated(
            config.hidden_size, self._q_heads * self._head_dim * 2, has_bias=False
        )
        self.k_proj = LinearReplicated(
            config.hidden_size, self._kv_heads * self._head_dim, has_bias=False
        )
        self.v_proj = LinearReplicated(
            config.hidden_size, self._kv_heads * self._head_dim, has_bias=False
        )
        self.o_proj = LinearReplicated(
            self._q_heads * self._head_dim, config.hidden_size, has_bias=False
        )
        self.q_norm = Qwen35RMSNorm(self._head_dim, config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self._head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        state: KVState | None = None,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVState]:
        batch, length, _ = x.shape
        past_length = 0 if state is None else state.key.shape[2]
        q, gate = self.q_proj.forward(x).reshape(
            batch, length, self._q_heads, self._head_dim * 2
        ).chunk(2, dim=-1)
        gate = gate.reshape(batch, length, self._q_heads * self._head_dim)
        q = self.q_norm.forward(q).transpose(1, 2)
        k = self.k_norm.forward(
            self.k_proj.forward(x).reshape(batch, length, self._kv_heads, self._head_dim)
        ).transpose(1, 2)
        v = self.v_proj.forward(x).reshape(
            batch, length, self._kv_heads, self._head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            if position_ids is None:
                position_ids = torch.arange(
                    past_length, past_length + length, device=x.device
                )[None, :].expand(batch, -1)
            position_embeddings = text_position_embeddings(x, position_ids, self._rotary_config)
        q, k = apply_partial_rotary(q, k, *position_embeddings)
        if state is not None:
            k = torch.cat((state.key, k), dim=2)
            v = torch.cat((state.value, v), dim=2)
        new_state = KVState(k, v)

        keys = _repeat_kv(k, self._groups)
        values = _repeat_kv(v, self._groups)
        scores = torch.matmul(q, keys.transpose(2, 3)) * self._scaling
        if attention_mask is None:
            query_positions = torch.arange(length, device=x.device) + past_length
            key_positions = torch.arange(k.shape[2], device=x.device)
            causal = key_positions[None, :] <= query_positions[:, None]
            attention_mask = torch.zeros(
                (length, k.shape[2]), dtype=q.dtype, device=x.device
            ).masked_fill(~causal, torch.finfo(q.dtype).min)
        scores = scores + attention_mask
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probabilities, values).transpose(1, 2).contiguous()
        output = output.reshape(batch, length, self._q_heads * self._head_dim)
        output = output * torch.sigmoid(gate)
        return self.o_proj.forward(output), new_state


class Qwen35MLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearReplicated(
            config.hidden_size, config.intermediate_size * 2, has_bias=False
        )
        self.down_proj = LinearReplicated(
            config.intermediate_size, config.hidden_size, has_bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj.forward(x).chunk(2, dim=-1)
        return self.down_proj.forward(F.silu(gate) * up)


class Qwen35DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self._block_type = config.layer_types[layer_id]
        if self._block_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config)
        else:
            self.self_attn = Qwen35Attention(config)
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        state: GDNState | KVState | None = None,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, GDNState | KVState]:
        normalized = self.input_layernorm.forward(x)
        if self._block_type == "linear_attention":
            mixed, state = self.linear_attn.forward(normalized, state)
        else:
            mixed, state = self.self_attn.forward(
                normalized,
                position_ids,
                state,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        x = x + mixed
        return x + self.mlp.forward(self.post_attention_layernorm.forward(x)), state


class Qwen35Embedding(BaseOP):
    def __init__(self, config: ModelConfig):
        self.weight = torch.empty(config.vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class Qwen35TextModel(BaseOP):
    """Text decoder with one history state per layer."""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = Qwen35Embedding(config)
        self.layers = OPList(
            [Qwen35DecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._rotary_config = config.rotary_config

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        states: list[GDNState | KVState | None] | None = None,
    ) -> tuple[torch.Tensor, list[GDNState | KVState]]:
        x = self.embed_tokens.forward(input_ids)
        if states is None:
            states = [None] * len(self.layers.op_list)
        if position_ids is None:
            past_length = next(
                (state.key.shape[2] for state in states if isinstance(state, KVState)), 0
            )
            position_ids = torch.arange(
                past_length, past_length + input_ids.shape[1], device=input_ids.device
            )[None, :].expand(input_ids.shape[0], -1)
        position_embeddings = text_position_embeddings(x, position_ids, self._rotary_config)
        new_states: list[GDNState | KVState] = []
        for layer, state in zip(self.layers.op_list, states):
            x, state = layer.forward(
                x, position_ids, state, position_embeddings=position_embeddings
            )
            new_states.append(state)
        return self.norm.forward(x), new_states


class Qwen35TextSkeleton(BaseOP):
    """All text checkpoint parameters, with the embedding also used by the tied head."""

    def __init__(self, config: ModelConfig):
        self.model = Qwen35TextModel(config)

    def forward_tokens(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        states: list[GDNState | KVState | None] | None = None,
    ) -> tuple[torch.Tensor, list[GDNState | KVState]]:
        hidden, states = self.model.forward(input_ids, position_ids, states)
        return F.linear(hidden, self.model.embed_tokens.weight), states


class Qwen35ForCausalLM(BaseLLMModel):
    """Eager engine adapter; each request owns its GDN history slot."""

    def __init__(self, config: ModelConfig):
        self.model = Qwen35TextModel(config)
        super().__init__()

    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        last_hidden = []
        offset = 0
        for req in batch.reqs:
            length = req.extend_len
            if req.hybrid_state is None:
                req.hybrid_state = ctx.kv_cache.allocate(req.uid)
            past_locations = ctx.page_table[req.table_idx, : req.cached_len]
            states = ctx.kv_cache.read_states(req.hybrid_state, past_locations)
            input_ids = batch.input_ids[offset : offset + length].unsqueeze(0)
            positions = batch.positions[offset : offset + length].unsqueeze(0)
            hidden, states = self.model.forward(input_ids, positions, states)
            new_locations = batch.out_loc[offset : offset + length]
            ctx.kv_cache.write_states(req.hybrid_state, states, new_locations)
            last_hidden.append(hidden[0, -1])
            offset += length
        return F.linear(torch.stack(last_hidden), self.model.embed_tokens.weight)


__all__ = [
    "KVState",
    "Qwen35Attention",
    "Qwen35MLP",
    "Qwen35DecoderLayer",
    "Qwen35TextModel",
    "Qwen35TextSkeleton",
    "Qwen35ForCausalLM",
    "text_position_embeddings",
    "apply_partial_rotary",
]
