from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.kernel.quant_decode import create_decode_workspace, decode_attention, quant_store
from minisgl.kernel.quant_prefill import paged_prefill_attention
from minisgl.kvcache.quant_pool import QuantizedKVCache

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class QuantTritonMetadata(BaseAttnMetadata):
    page_table: torch.Tensor
    table_indices: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_query_len: int

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : bs + 1] - 1


class QuantTritonBackend(BaseAttnBackend):
    """Paged INT8 attention with transient on-chip dequantization.

    Metadata belongs to each scheduled batch. Only graph replay copies it into
    stable capture buffers, on the engine stream after the preceding forward.
    This allows the scheduler to prepare the next batch concurrently.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.kvcache = get_global_ctx().kv_cache
        if not isinstance(self.kvcache, QuantizedKVCache):
            raise ValueError("quant-triton requires an INT8 KV cache")
        self.device = self.kvcache.device
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.capture_metadata: QuantTritonMetadata | None = None
        self.capture_bs: List[int] = []
        self.decode_workspaces: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.workspace_buffer: tuple[torch.Tensor, torch.Tensor] | None = None

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        lengths = [r.extend_len for r in reqs]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        def device_tensor(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, pin_memory=True).to(
                self.device, non_blocking=True
            )

        rows = device_tensor([r.table_idx for r in reqs])
        # Snapshot addresses before a later overlap iteration can recycle a
        # finished/aborted request's global table row.
        page_table = get_global_ctx().page_table.index_select(0, rows)
        batch.attn_metadata = QuantTritonMetadata(
            page_table=page_table,
            table_indices=torch.arange(len(reqs), dtype=torch.int32, device=self.device),
            # Graph padding requests must neither read nor write cache slots.
            seq_lens=device_tensor([r.device_len if r.uid != -1 else 0 for r in reqs]),
            cu_seqlens_q=device_tensor(offsets),
            max_query_len=max(lengths),
        )

    def _workspace(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = q.shape[0]
        if bs not in self.decode_workspaces:
            if self.workspace_buffer is None or self.workspace_buffer[0].shape[0] < bs:
                capacity = 1 << (bs - 1).bit_length()
                reserve_q = torch.empty(
                    capacity, self.num_qo_heads, self.head_dim,
                    dtype=q.dtype, device=q.device,
                )
                self.workspace_buffer = create_decode_workspace(reserve_q)
            self.decode_workspaces[bs] = tuple(buffer[:bs] for buffer in self.workspace_buffer)
        return self.decode_workspaces[bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, QuantTritonMetadata)
        pool = self.kvcache
        k_cache, v_cache = pool.k_cache(layer_id), pool.v_cache(layer_id)
        k_scale, v_scale = pool.k_scale(layer_id), pool.v_scale(layer_id)
        locations = batch.out_loc
        if batch.is_decode:
            locations = torch.where(metadata.seq_lens > 0, locations, -1)
        # Model activations are expected to be finite, as in native GPU kernels.
        # quant-reference retains the synchronous diagnostic validation path.
        quant_store(k, v, locations, k_cache, v_cache, k_scale, v_scale)
        page_table = metadata.page_table
        if batch.is_decode:
            return decode_attention(
                q, k_cache, v_cache, k_scale, v_scale, page_table,
                metadata.table_indices, metadata.seq_lens, workspace=self._workspace(q),
            )
        return paged_prefill_attention(
            q, k_cache, v_cache, k_scale, v_scale, page_table,
            metadata.table_indices, metadata.cu_seqlens_q, metadata.seq_lens,
            metadata.max_query_len,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        max_bs = max(bs_list)
        self.capture_bs = list(bs_list)
        self.capture_metadata = QuantTritonMetadata(
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=self.device),
            table_indices=torch.arange(max_bs, dtype=torch.int32, device=self.device),
            seq_lens=torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            cu_seqlens_q=torch.arange(max_bs + 1, dtype=torch.int32, device=self.device),
            max_query_len=1,
        )
        q = torch.empty(max_bs, self.num_qo_heads, self.head_dim,
                        dtype=self.kvcache.compute_dtype, device=self.device)
        self._workspace(q)
        assert self.workspace_buffer is not None
        for bs in bs_list:
            self.decode_workspaces[bs] = tuple(buffer[:bs] for buffer in self.workspace_buffer)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert self.capture_metadata is not None and batch.size in self.capture_bs
        self.prepare_metadata(batch)
        self.prepare_for_replay(batch)
        capture = self.capture_metadata
        bs = batch.size
        batch.attn_metadata = QuantTritonMetadata(
            page_table=capture.page_table[:bs],
            table_indices=capture.table_indices[:bs],
            seq_lens=capture.seq_lens[:bs],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            max_query_len=1,
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        capture, metadata = self.capture_metadata, batch.attn_metadata
        assert capture is not None and isinstance(metadata, QuantTritonMetadata)
        bs = batch.padded_size
        capture.page_table[:bs].copy_(metadata.page_table)
        capture.table_indices[:bs].copy_(metadata.table_indices)
        capture.seq_lens[:bs].copy_(metadata.seq_lens)
