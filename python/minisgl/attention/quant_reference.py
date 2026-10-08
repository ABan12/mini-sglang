from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.kvcache.quant_pool import QuantizedKVCache
from minisgl.utils import div_even

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class QuantReferenceMetadata(BaseAttnMetadata):
    # Each entry is (request table row, cached length, query length).
    requests: tuple[tuple[int, int, int], ...]
    last_indices: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QuantReferenceBackend(BaseAttnBackend):
    """Eager SDPA attention with the same arithmetic for BF16 and INT8 caches.

    INT8 K/V are dequantized one request and one layer at a time. The pool never
    retains a BF16 shadow cache. Workspace counters describe only the returned
    dequantized K/V tensors, excluding temporary quantization and SDPA buffers.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.quantized = isinstance(self.kvcache, QuantizedKVCache)
        self.last_dequantized_kv_bytes = 0
        self.peak_dequantized_kv_bytes = 0

    def prepare_metadata(self, batch: Batch) -> None:
        requests = tuple((r.table_idx, r.cached_len, r.extend_len) for r in batch.reqs)
        last_indices = []
        offset = 0
        for _, _, query_len in requests:
            offset += query_len
            last_indices.append(offset - 1)
        batch.attn_metadata = QuantReferenceMetadata(
            requests=requests,
            last_indices=torch.tensor(last_indices, dtype=torch.long, device=self.device),
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, QuantReferenceMetadata)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        if self.quantized:
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        else:
            self.kvcache.store_kv(k.flatten(1), v.flatten(1), batch.out_loc, layer_id)
        output = torch.empty_like(q, memory_format=torch.contiguous_format)
        page_table = get_global_ctx().page_table
        offset = 0
        self.last_dequantized_kv_bytes = 0

        for table_idx, cached_len, query_len in metadata.requests:
            sequence_len = cached_len + query_len
            locations = page_table[table_idx, :sequence_len].to(torch.long)
            if self.quantized:
                cached_k, cached_v = self.kvcache.read_kv(layer_id, locations)
                workspace_bytes = (cached_k.numel() + cached_v.numel()) * cached_k.element_size()
                self.last_dequantized_kv_bytes = max(
                    self.last_dequantized_kv_bytes, workspace_bytes
                )
                self.peak_dequantized_kv_bytes = max(
                    self.peak_dequantized_kv_bytes, workspace_bytes
                )
            else:
                shape = (-1, self.num_kv_heads, self.head_dim)
                cached_k = self.kvcache.k_cache(layer_id).view(shape)[locations]
                cached_v = self.kvcache.v_cache(layer_id).view(shape)[locations]

            query = q[offset : offset + query_len].transpose(0, 1).unsqueeze(0)
            key = cached_k.transpose(0, 1).unsqueeze(0)
            value = cached_v.transpose(0, 1).unsqueeze(0)
            mask = None
            # SDPA's is_causal mask is aligned to the upper left. A chunk with
            # cached history instead needs query i to see keys 0..cached_len+i.
            if cached_len > 0 and query_len > 1:
                query_positions = cached_len + torch.arange(query_len, device=q.device)
                key_positions = torch.arange(sequence_len, device=q.device)
                mask = key_positions[None, :] <= query_positions[:, None]

            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=cached_len == 0,
                enable_gqa=self.num_qo_heads != self.num_kv_heads,
            )
            output[offset : offset + query_len] = attended.squeeze(0).transpose(0, 1)
            offset += query_len
            # Release this request's transient buffers before gathering the next.
            del cached_k, cached_v, query, key, value, attended, mask

        return output

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError("quant-reference attention uses eager execution only")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError("quant-reference attention does not support CUDA Graph capture")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError("quant-reference attention does not support CUDA Graph replay")
