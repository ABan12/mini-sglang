from __future__ import annotations

import torch

from .base import BaseKVCachePool


def quantize_per_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantization along the last (head) dimension.

    Each token/head has an FP32 scale. Zero vectors use scale 1; NaN and
    infinities are rejected. This reference implementation synchronizes to
    check finite inputs. Finite outliers remain part of the scale maximum.
    """
    values = x.float()
    if not torch.isfinite(values).all():
        raise ValueError("INT8 KV quantization requires finite input values")
    maximum = values.abs().amax(dim=-1, keepdim=True)
    nonzero_scale = (maximum / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.where(maximum > 0, nonzero_scale, torch.ones_like(maximum))
    quantized = (values / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_per_head(
    quantized: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    return (quantized.float() * scale).to(dtype)


def int8_kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int) -> int:
    """Both INT8 K/V and their separate FP32 token/head scales."""
    return 2 * num_layers * num_kv_heads * (head_dim + 4)


class QuantizedKVCache(BaseKVCachePool):
    """Paged INT8 K/V storage with transient dequantization on read.

    The initial engine integration uses tensor parallel size 1. No complete
    BF16 copy of the cache is retained.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.compute_dtype = dtype
        self._device = device
        self._num_layers = num_layers
        self._storage_shape = (num_pages * page_size, num_kv_heads, head_dim)
        self._scale_shape = (num_pages * page_size, num_kv_heads, 1)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, num_kv_heads, head_dim),
            device=device,
            dtype=torch.int8,
        )
        self._scale_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, num_kv_heads, 1),
            device=device,
            dtype=torch.float32,
        )

    def k_cache(self, index: int) -> torch.Tensor:
        return self._kv_buffer[0, index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._kv_buffer[1, index]

    def k_scale(self, index: int) -> torch.Tensor:
        return self._scale_buffer[0, index]

    def v_scale(self, index: int) -> torch.Tensor:
        return self._scale_buffer[1, index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        locations = out_loc.to(torch.long)
        quantized_k, k_scale = quantize_per_head(k)
        quantized_v, v_scale = quantize_per_head(v)
        self.k_cache(layer_id).view(self._storage_shape)[locations] = quantized_k
        self.v_cache(layer_id).view(self._storage_shape)[locations] = quantized_v
        self.k_scale(layer_id).view(self._scale_shape)[locations] = k_scale
        self.v_scale(layer_id).view(self._scale_shape)[locations] = v_scale

    def read_kv(
        self, layer_id: int, locations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        locations = locations.to(torch.long)
        k = dequantize_per_head(
            self.k_cache(layer_id).view(self._storage_shape)[locations],
            self.k_scale(layer_id).view(self._scale_shape)[locations],
            self.compute_dtype,
        )
        v = dequantize_per_head(
            self.v_cache(layer_id).view(self._storage_shape)[locations],
            self.v_scale(layer_id).view(self._scale_shape)[locations],
            self.compute_dtype,
        )
        return k, v

    @property
    def storage_nbytes(self) -> int:
        return sum(
            buffer.numel() * buffer.element_size()
            for buffer in (self._kv_buffer, self._scale_buffer)
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
