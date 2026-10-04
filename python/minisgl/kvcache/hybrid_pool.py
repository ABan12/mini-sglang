from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minisgl.layers.deltanet import GDNState

from .base import BaseKVCachePool

if TYPE_CHECKING:
    from minisgl.models.config import ModelConfig
    from minisgl.models.qwen3_5 import KVState


@dataclass(frozen=True)
class HybridStateHandle:
    slot: int
    uid: int


def gdn_state_bytes(config: ModelConfig, max_running_req: int, dtype: torch.dtype) -> int:
    conv_dim = (
        2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    conv_bytes = conv_dim * config.linear_conv_kernel_dim * dtype.itemsize
    recurrent_bytes = (
        config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * torch.float32.itemsize
    )
    return (
        config.layer_types.count("linear_attention")
        * max_running_req
        * (conv_bytes + recurrent_bytes)
    )


class HybridKVCache(BaseKVCachePool):
    """Paged K/V for full attention, and fixed request slots for GDN state."""

    def __init__(
        self,
        model_config: ModelConfig,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        max_running_req: int,
    ) -> None:
        self.layer_types = model_config.layer_types
        self.full_attention_layers = tuple(
            i for i, kind in enumerate(self.layer_types) if kind == "full_attention"
        )
        self.gdn_layers = tuple(
            i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"
        )
        self._full_indices = {
            layer_id: index for index, layer_id in enumerate(self.full_attention_layers)
        }
        self._gdn_indices = {
            layer_id: index for index, layer_id in enumerate(self.gdn_layers)
        }
        self._kv_buffer = torch.empty(
            2,
            len(self.full_attention_layers),
            num_pages,
            page_size,
            model_config.num_kv_heads,
            model_config.head_dim,
            dtype=dtype,
            device=device,
        )
        conv_dim = (
            2 * model_config.linear_num_key_heads * model_config.linear_key_head_dim
            + model_config.linear_num_value_heads * model_config.linear_value_head_dim
        )
        self._conv_buffer = torch.empty(
            len(self.gdn_layers),
            max_running_req,
            conv_dim,
            model_config.linear_conv_kernel_dim,
            dtype=dtype,
            device=device,
        )
        self._recurrent_buffer = torch.empty(
            len(self.gdn_layers),
            max_running_req,
            model_config.linear_num_value_heads,
            model_config.linear_key_head_dim,
            model_config.linear_value_head_dim,
            dtype=torch.float32,
            device=device,
        )
        self._storage_shape = (
            num_pages * page_size,
            model_config.num_kv_heads,
            model_config.head_dim,
        )
        self._free_slots = list(reversed(range(max_running_req)))

    def allocate(self, uid: int) -> HybridStateHandle:
        if not self._free_slots:
            raise RuntimeError("No free GDN request state slots")
        slot = self._free_slots.pop()
        self._conv_buffer[:, slot].zero_()
        self._recurrent_buffer[:, slot].zero_()
        return HybridStateHandle(slot=slot, uid=uid)

    def release(self, handle: HybridStateHandle) -> None:
        self._free_slots.append(handle.slot)

    def read_states(
        self, handle: HybridStateHandle, past_locations: torch.Tensor
    ) -> list[GDNState | KVState | None]:
        from minisgl.models.qwen3_5 import KVState

        locations = past_locations.to(torch.long)
        if len(locations) == 0:
            return [None] * self.num_layers
        states = []
        for layer_id, kind in enumerate(self.layer_types):
            if kind == "linear_attention":
                index = self._gdn_indices[layer_id]
                states.append(
                    GDNState(
                        conv=self._conv_buffer[index, handle.slot].unsqueeze(0),
                        recurrent=self._recurrent_buffer[index, handle.slot].unsqueeze(0),
                    )
                )
            else:
                key = self.k_cache(layer_id).view(self._storage_shape)[locations]
                value = self.v_cache(layer_id).view(self._storage_shape)[locations]
                states.append(
                    KVState(
                        key=key.transpose(0, 1).unsqueeze(0),
                        value=value.transpose(0, 1).unsqueeze(0),
                    )
                )
        return states

    def write_states(
        self,
        handle: HybridStateHandle,
        states: list[GDNState | KVState],
        new_locations: torch.Tensor,
    ) -> None:
        length = len(new_locations)
        for layer_id, (kind, state) in enumerate(zip(self.layer_types, states)):
            if kind == "linear_attention":
                index = self._gdn_indices[layer_id]
                self._conv_buffer[index, handle.slot].copy_(state.conv[0])
                self._recurrent_buffer[index, handle.slot].copy_(state.recurrent[0])
            else:
                self.store_kv(
                    state.key[0, :, -length:].transpose(0, 1),
                    state.value[0, :, -length:].transpose(0, 1),
                    new_locations,
                    layer_id,
                )

    def k_cache(self, index: int) -> torch.Tensor:
        return self._kv_buffer[0, self._full_indices[index]]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._kv_buffer[1, self._full_indices[index]]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        locations = out_loc.to(torch.long)
        self.k_cache(layer_id).view(self._storage_shape)[locations] = k
        self.v_cache(layer_id).view(self._storage_shape)[locations] = v

    @property
    def device(self) -> torch.device:
        return self._kv_buffer.device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)
