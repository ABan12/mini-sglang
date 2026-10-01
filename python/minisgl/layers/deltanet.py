from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .base import BaseOP
from .linear import LinearReplicated

if TYPE_CHECKING:
    from minisgl.models.config import ModelConfig


@dataclass
class GDNState:
    conv: torch.Tensor
    recurrent: torch.Tensor


class Qwen35GatedNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size, dtype=torch.float32)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        normalized = self.weight * normalized.to(input_dtype)
        output = normalized * F.silu(z.float())
        return output.to(input_dtype)


class CausalConv1d(BaseOP):
    def __init__(self, channels: int, kernel_size: int) -> None:
        self.weight = torch.empty(channels, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight, groups=self.weight.shape[0])


def recurrent_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_dtype = query.dtype
    batch_size, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query = query / (key_dim ** 0.5)

    recurrent = (
        value.new_zeros(batch_size, heads, key_dim, value_dim)
        if initial_state is None
        else initial_state.to(value)
    )
    output = torch.zeros_like(value)
    
    for token in range(length):
        q_t = query[:, :, token]
        k_t = key[:, :, token]
        v_t = value[:, :, token]
        recurrent = recurrent * decay[:, :, token].exp()[..., None, None]
        prediction = (recurrent * k_t.unsqueeze(-1)).sum(-2)
        correction = (v_t - prediction) * beta[:, :, token].unsqueeze(-1)
        recurrent = recurrent + k_t.unsqueeze(-1) * correction.unsqueeze(-2)
        output[:, :, token] = (recurrent * q_t.unsqueeze(-1)).sum(-2)
    return output.transpose(1, 2).contiguous().to(input_dtype), recurrent


class Qwen35GatedDeltaNet(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel_size = config.linear_conv_kernel_dim
        self.in_proj_qkv = LinearReplicated(config.hidden_size, self.conv_dim, False)
        self.in_proj_z = LinearReplicated(config.hidden_size, self.value_dim, False)
        self.in_proj_b = LinearReplicated(config.hidden_size, self.num_value_heads, False)
        self.in_proj_a = LinearReplicated(config.hidden_size, self.num_value_heads, False)
        self.conv1d = CausalConv1d(self.conv_dim, self.kernel_size)
        self.dt_bias = torch.empty(self.num_value_heads)
        self.A_log = torch.empty(self.num_value_heads, dtype=torch.float32)
        self.norm = Qwen35GatedNorm(self.value_head_dim, config.rms_norm_eps)
        self.out_proj = LinearReplicated(self.value_dim, config.hidden_size, False)

    def forward(
        self, x: torch.Tensor, state: GDNState | None = None
    ) -> tuple[torch.Tensor, GDNState]:
        batch_size, length, _ = x.shape
        projected = self.in_proj_qkv.forward(x).transpose(1, 2)
        z = self.in_proj_z.forward(x).reshape(
            batch_size, length, self.num_value_heads, self.value_head_dim
        )
        b = self.in_proj_b.forward(x)
        a = self.in_proj_a.forward(x)
        conv = (
            projected.new_zeros(batch_size, self.conv_dim, self.kernel_size)
            if state is None
            else state.conv
        )
        projected_history = torch.cat((conv, projected), dim=-1)
        conv = projected_history[:, :, -self.kernel_size :].contiguous()
        mixed = self.conv1d.forward(projected_history)[:, :, -length:]
        mixed = F.silu(mixed).transpose(1, 2)
        query, key, value = torch.split(
            mixed, (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        query = query.reshape(batch_size, length, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch_size, length, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch_size, length, self.num_value_heads, self.value_head_dim)
        beta = b.sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        repeats = self.num_value_heads // self.num_key_heads
        if repeats > 1:
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        output, recurrent = recurrent_delta_rule(
            query, key, value, decay, beta,
            None if state is None else state.recurrent,
        )
        output = self.norm.forward(output, z).reshape(batch_size, length, self.value_dim)
        output = self.out_proj.forward(output)
        return output, GDNState(conv=conv, recurrent=recurrent)
