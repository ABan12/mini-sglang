"""GPU checks for paged INT8 prefill against FP32 attention arithmetic."""

import pytest
import torch

pytest.importorskip("triton")

from minisgl.kernel.quant_prefill import paged_prefill_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def _fixture(query_lens, history_lens, num_q_heads, num_kv_heads, head_dim, quantized):
    torch.manual_seed(42)
    sequence_lens = [q + h for q, h in zip(query_lens, history_lens)]
    capacity = max(sum(sequence_lens) + 71, 128)
    page_table = torch.zeros(
        (len(query_lens) + 3, max(sequence_lens) + 13), dtype=torch.int32, device="cuda"
    )
    table_indices = torch.arange(2, len(query_lens) + 2, dtype=torch.int32, device="cuda")
    locations = torch.randperm(capacity, device="cuda").int()
    offset = 0
    for request, length in enumerate(sequence_lens):
        page_table[request + 2, :length] = locations[offset : offset + length]
        offset += length
    k = torch.randn((capacity, num_kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    k_scale = v_scale = None
    if quantized:
        k_scale = k.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
        v_scale = v.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
        k = (k.float() / k_scale).round().clamp(-127, 127).to(torch.int8)
        v = (v.float() / v_scale).round().clamp(-127, 127).to(torch.int8)
    # Noncontiguous query features and a token stride unlike the output layout.
    q = torch.randn(
        (sum(query_lens), num_q_heads, head_dim * 2), dtype=torch.bfloat16, device="cuda"
    )[..., ::2]
    cu_seqlens = [0]
    for length in query_lens:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return [
        q,
        k,
        v,
        k_scale,
        v_scale,
        page_table,
        table_indices,
        torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda"),
        torch.tensor(sequence_lens, dtype=torch.int32, device="cuda"),
        max(query_lens),
    ]


def _reference(q, k, v, k_scale, v_scale, page_table, table_indices, cu_seqlens, seq_lens):
    output = []
    for request, table_idx in enumerate(table_indices.tolist()):
        start, end = cu_seqlens[request : request + 2].tolist()
        sequence_len = int(seq_lens[request])
        query_len = end - start
        locations = page_table[table_idx, :sequence_len].long()
        key, value = k[locations], v[locations]
        if k_scale is not None:
            key = (key.float() * k_scale[locations]).bfloat16()
            value = (value.float() * v_scale[locations]).bfloat16()
        group_size = q.shape[1] // k.shape[1]
        key = key.repeat_interleave(group_size, dim=1).transpose(0, 1).float()
        value = value.repeat_interleave(group_size, dim=1).transpose(0, 1).float()
        query = q[start:end].transpose(0, 1).float()
        scores = query @ key.transpose(-1, -2) * q.shape[-1] ** -0.5
        query_positions = sequence_len - query_len + torch.arange(query_len, device=q.device)
        key_positions = torch.arange(sequence_len, device=q.device)
        scores.masked_fill_(key_positions[None, :] > query_positions[:, None], float("-inf"))
        output.append((scores.softmax(-1) @ value).transpose(0, 1).bfloat16())
    return torch.cat(output)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    "query_lens,history_lens,num_q_heads,num_kv_heads,head_dim",
    [
        ([1], [0], 4, 1, 64),
        ([1], [97], 8, 2, 128),
        ([63], [0], 4, 1, 80),
        ([65], [5], 8, 2, 128),
        ([17, 65, 127], [33, 0, 97], 6, 2, 64),
        ([3, 7], [0, 13], 4, 4, 37),
        ([129], [191], 8, 2, 256),
        ([129, 17], [0, 129], 4, 1, 128),
        ([17], [8191], 16, 8, 128),
        ([65], [4095], 16, 8, 128),
    ],
)
def test_prefill_matches_float32_attention(
    query_lens, history_lens, num_q_heads, num_kv_heads, head_dim, quantized
):
    args = _fixture(query_lens, history_lens, num_q_heads, num_kv_heads, head_dim, quantized)
    actual = paged_prefill_attention(*args)
    expected = _reference(*args[:-1])
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("quantized", [False, True])
def test_chunk_boundaries_preserve_causal_attention(quantized):
    args = _fixture([129], [0], 8, 2, 128, quantized)
    full = paged_prefill_attention(*args)
    chunk_outputs = []
    start = 0
    for end in (17, 81, 129):
        chunk_args = args.copy()
        chunk_args[0] = args[0][start:end]
        chunk_args[7] = torch.tensor([0, end - start], dtype=torch.int32, device="cuda")
        chunk_args[8] = torch.tensor([end], dtype=torch.int32, device="cuda")
        chunk_args[9] = end - start
        chunk_outputs.append(paged_prefill_attention(*chunk_args))
        start = end
    torch.testing.assert_close(torch.cat(chunk_outputs), full, atol=0.01, rtol=0.01)


def test_graph_replay_reads_changed_request_metadata():
    args = _fixture([17, 23], [37, 19], 8, 2, 128, True)
    for _ in range(3):
        paged_prefill_attention(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = paged_prefill_attention(*args)
    graph.replay()
    torch.testing.assert_close(captured, _reference(*args[:-1]), atol=0.01, rtol=0.01)
    args[6].copy_(args[6].flip(0))
    args[7].copy_(torch.tensor([0, 23, 40], dtype=torch.int32, device="cuda"))
    args[8].copy_(torch.tensor([42, 54], dtype=torch.int32, device="cuda"))
    args[0].normal_()
    graph.replay()
    torch.testing.assert_close(captured, _reference(*args[:-1]), atol=0.01, rtol=0.01)
