"""Metadata and buffer lifetime checks for overlapping quantized attention."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention import quant_triton
from minisgl.core import Batch
from minisgl.kvcache.quant_pool import QuantizedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.fixture
def backend_env(monkeypatch):
    pool = QuantizedKVCache(
        num_kv_heads=2,
        num_layers=1,
        head_dim=64,
        num_pages=16,
        page_size=16,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    page_table = torch.arange(5 * 128, dtype=torch.int32, device="cuda").view(5, 128) % 256
    ctx = SimpleNamespace(kv_cache=pool, page_table=page_table)
    monkeypatch.setattr(quant_triton, "get_global_ctx", lambda: ctx)
    backend = quant_triton.QuantTritonBackend(
        SimpleNamespace(num_qo_heads=8, num_kv_heads=2, head_dim=64)
    )
    return backend, ctx


def _request(table_idx, cached_len, extend_len=1, uid=1):
    return SimpleNamespace(
        table_idx=table_idx,
        cached_len=cached_len,
        device_len=cached_len + extend_len,
        extend_len=extend_len,
        uid=uid,
    )


def _batch(requests, phase="decode", padding=()):
    batch = Batch(reqs=list(requests), phase=phase)
    batch.padded_reqs = batch.reqs + list(padding)
    return batch


def test_overlapping_batches_own_addresses_before_global_row_reuse(backend_env):
    backend, ctx = backend_env
    first = _batch([_request(4, 7), _request(1, 3, 3)], phase="prefill")
    expected_first = ctx.page_table[[4, 1]].clone()
    backend.prepare_metadata(first)
    metadata = first.attn_metadata
    ctx.page_table[4].fill_(71)
    second = _batch([_request(4, 16), _request(2, 7, 2)], phase="prefill")
    backend.prepare_metadata(second)
    ctx.page_table[1].fill_(89)
    assert torch.equal(metadata.page_table, expected_first)
    assert torch.equal(second.attn_metadata.page_table[0], torch.full_like(ctx.page_table[4], 71))
    assert metadata.page_table.data_ptr() != second.attn_metadata.page_table.data_ptr()
    assert metadata.table_indices.tolist() == [0, 1]
    assert metadata.seq_lens.tolist() == [8, 6]
    assert metadata.cu_seqlens_q.tolist() == [0, 1, 4]
    assert metadata.get_last_indices(2).tolist() == [0, 3]


def test_capture_buffers_change_only_when_replay_is_prepared(backend_env):
    backend, ctx = backend_env
    backend.init_capture_graph(128, [1, 2, 4])
    capture = backend.capture_metadata
    capture_addresses = capture.page_table.data_ptr()
    dummy = _request(4, 0, uid=-1)
    batch = _batch([_request(2, 9)], padding=[dummy])
    backend.prepare_metadata(batch)
    assert torch.count_nonzero(capture.page_table).item() == 0
    assert capture.seq_lens.tolist() == [0, 0, 0, 0]
    backend.prepare_for_replay(batch)
    expected = batch.attn_metadata.page_table.clone()
    assert torch.equal(capture.page_table[:2], expected)
    assert capture.seq_lens.tolist() == [10, 0, 0, 0]
    assert capture.table_indices.tolist() == [0, 1, 2, 3]
    ctx.page_table[2].fill_(99)
    following = _batch([_request(2, 12)], padding=[dummy])
    backend.prepare_metadata(following)
    assert torch.equal(capture.page_table[:2], expected)
    assert torch.equal(batch.attn_metadata.page_table, expected)
    backend.prepare_for_replay(following)
    assert capture.page_table.data_ptr() == capture_addresses
    assert torch.equal(capture.page_table[:2], following.attn_metadata.page_table)
    assert capture.seq_lens.tolist() == [13, 0, 0, 0]


def test_capture_metadata_binds_stable_views_and_zero_length_padding(backend_env):
    backend, _ = backend_env
    backend.init_capture_graph(128, [1, 2, 4])
    batch = _batch([_request(4, 0, uid=-1), _request(4, 0, uid=-1)])
    backend.prepare_for_capture(batch)
    capture = backend.capture_metadata
    metadata = batch.attn_metadata
    assert metadata.page_table.data_ptr() == capture.page_table.data_ptr()
    assert metadata.table_indices.data_ptr() == capture.table_indices.data_ptr()
    assert metadata.seq_lens.data_ptr() == capture.seq_lens.data_ptr()
    assert metadata.seq_lens.tolist() == [0, 0]
    assert metadata.cu_seqlens_q.tolist() == [0, 1, 2]


def test_eager_workspace_growth_preserves_captured_buffer_addresses(backend_env):
    backend, _ = backend_env
    backend.init_capture_graph(128, [1, 2, 4])

    def workspace(batch_size):
        q = torch.empty((batch_size, 8, 64), dtype=torch.bfloat16, device="cuda")
        return backend._workspace(q)

    captured = workspace(4)
    pointers = tuple(buffer.data_ptr() for buffer in captured)
    captured[0].fill_(3)
    captured[1].fill_(5)
    grown = workspace(9)
    assert backend.workspace_buffer[0].shape[0] == 16
    assert tuple(buffer.data_ptr() for buffer in grown) != pointers
    assert tuple(buffer.data_ptr() for buffer in workspace(4)) == pointers
    assert tuple(buffer.data_ptr() for buffer in workspace(1)) == pointers
    assert torch.all(captured[0] == 3)
    assert torch.all(captured[1] == 5)
    assert tuple(buffer.data_ptr() for buffer in workspace(12)) == tuple(
        buffer.data_ptr() for buffer in grown
    )
    workspace(33)
    assert backend.workspace_buffer[0].shape[0] == 64
    assert tuple(buffer.data_ptr() for buffer in workspace(4)) == pointers
    assert tuple(buffer.data_ptr() for buffer in workspace(9)) == tuple(
        buffer.data_ptr() for buffer in grown
    )
