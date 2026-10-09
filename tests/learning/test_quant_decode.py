"""CUDA correctness checks for paged INT8 store and split-K decode."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from minisgl.kernel.quant_decode import (
    create_decode_workspace,
    decode_attention,
    quant_store,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _quantize(x):
    values = x.float()
    maximum = values.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(
        maximum > 0,
        (maximum / 127.0).clamp_min(torch.finfo(torch.float32).tiny),
        torch.ones_like(maximum),
    )
    return (values / scale).round().clamp(-127, 127).to(torch.int8), scale


def _case(batch, query_heads, kv_heads, dim, length):
    torch.manual_seed(23)
    page_size = 16
    cache_tokens = ((batch * length + page_size - 1) // page_size + 1) * page_size
    k_cache = torch.full(
        (cache_tokens // page_size, page_size, kv_heads, dim),
        -41,
        dtype=torch.int8,
        device="cuda",
    )
    v_cache = torch.full_like(k_cache, -39)
    k_scale = torch.full(
        (cache_tokens // page_size, page_size, kv_heads, 1),
        17.0,
        dtype=torch.float32,
        device="cuda",
    )
    v_scale = torch.full_like(k_scale, 19.0)
    # Include noncontiguous source token strides and a sentinel that must not
    # write to the last physical cache token via Python-style negative indexing.
    qkv = torch.randn(
        (batch * length + 1, 3 * kv_heads, dim), dtype=torch.bfloat16, device="cuda"
    )
    k, v = qkv[:, :kv_heads], qkv[:, kv_heads : 2 * kv_heads]
    k[0].zero_()
    k[1, 0, :7] = torch.tensor(
        [127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device="cuda"
    )
    k[-1].fill_(float("nan"))
    v[-1].fill_(float("inf"))
    locations = torch.randperm(cache_tokens - 1, device="cuda")[: batch * length].to(
        torch.int32
    )
    out_loc = torch.cat((locations, torch.tensor([-1], device="cuda", dtype=torch.int32)))
    q = torch.randn((batch, query_heads, dim), dtype=torch.bfloat16, device="cuda")
    page_table = locations.view(batch, length).clone()
    table_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    lengths = [length, length - 1, 1, 0]
    lengths += [max(1, length - 31 * i) for i in range(batch - 4)]
    seq_lens = torch.tensor(lengths[:batch], device="cuda", dtype=torch.int32)
    return k, v, out_loc, q, k_cache, v_cache, k_scale, v_scale, page_table, table_indices, seq_lens


def _reference(q, k_cache, v_cache, k_scale, v_scale, page_table, table_indices, seq_lens):
    batch, query_heads, dim = q.shape
    kv_heads = k_cache.shape[-2]
    k_cache, v_cache = k_cache.view(-1, kv_heads, dim), v_cache.view(-1, kv_heads, dim)
    k_scale, v_scale = k_scale.view(-1, kv_heads, 1), v_scale.view(-1, kv_heads, 1)
    output = torch.zeros_like(q)
    for b, (row, length) in enumerate(zip(table_indices.tolist(), seq_lens.tolist())):
        if length:
            locations = page_table[row, :length].long()
            k = (k_cache[locations].float() * k_scale[locations]).to(q.dtype)
            v = (v_cache[locations].float() * v_scale[locations]).to(q.dtype)
            output[b] = F.scaled_dot_product_attention(
                q[b].unsqueeze(0).unsqueeze(2),
                k.permute(1, 0, 2).unsqueeze(0),
                v.permute(1, 0, 2).unsqueeze(0),
                enable_gqa=query_heads != kv_heads,
            ).squeeze(0).squeeze(1)
    return output


def test_store_matches_torch_rounding_scales_and_skips_padding():
    k, v, locations, _, kc, vc, ks, vs, *_ = _case(4, 16, 8, 128, 7)
    quant_store(k, v, locations, kc, vc, ks, vs)
    valid_locations = locations[:-1].long()
    for source, cache, scales in ((k, kc, ks), (v, vc, vs)):
        expected, expected_scale = _quantize(source[:-1])
        assert torch.equal(cache.view(-1, 8, 128)[valid_locations], expected)
        assert torch.equal(scales.view(-1, 8, 1)[valid_locations], expected_scale)
    assert torch.all(kc.view(-1, 8, 128)[-1] == -41)
    assert torch.all(vc.view(-1, 8, 128)[-1] == -39)
    assert torch.all(ks.view(-1, 8, 1)[-1] == 17)
    assert torch.all(vs.view(-1, 8, 1)[-1] == 19)


@pytest.mark.parametrize("batch,length", [(4, 193), (16, 9216)])
def test_decode_matches_sdpa_with_gqa_physical_pages_and_zero_rows(batch, length):
    k, v, locations, q, kc, vc, ks, vs, page, rows, lengths = _case(
        batch, 16, 8, 128, length
    )
    quant_store(k, v, locations, kc, vc, ks, vs)
    output = decode_attention(q, kc, vc, ks, vs, page, rows, lengths)
    expected = _reference(q, kc, vc, ks, vs, page, rows, lengths)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.002)
    assert torch.count_nonzero(output[3]).item() == 0


def test_graph_replay_reads_changed_lengths_table_rows_and_physical_locations():
    k, v, locations, q, kc, vc, ks, vs, page, rows, lengths = _case(4, 16, 8, 128, 193)
    workspace = create_decode_workspace(q)
    # Compile and allocate before capture. The graph then reuses all addresses.
    quant_store(k, v, locations, kc, vc, ks, vs)
    decode_attention(q, kc, vc, ks, vs, page, rows, lengths, workspace)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quant_store(k, v, locations, kc, vc, ks, vs)
        output = decode_attention(q, kc, vc, ks, vs, page, rows, lengths, workspace)
    graph.replay()
    torch.testing.assert_close(
        output, _reference(q, kc, vc, ks, vs, page, rows, lengths), rtol=0.02, atol=0.002
    )

    lengths.copy_(torch.tensor([97, 3, 0, 193], device="cuda", dtype=torch.int32))
    rows.copy_(rows.flip(0))
    page.copy_(page.roll(1, dims=1))
    q.copy_(torch.randn_like(q))
    k.mul_(0.5)
    v.mul_(0.25)
    graph.replay()
    torch.testing.assert_close(
        output, _reference(q, kc, vc, ks, vs, page, rows, lengths), rtol=0.02, atol=0.002
    )
    assert torch.count_nonzero(output[2]).item() == 0

    lengths.zero_()
    graph.replay()
    assert torch.count_nonzero(output).item() == 0
