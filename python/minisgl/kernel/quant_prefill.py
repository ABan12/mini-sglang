"""Paged FlashAttention with INT8 K/V dequantization inside each tile."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_prefill_attention(
    Q,
    K,
    V,
    KScale,
    VScale,
    PageTable,
    TableIndices,
    CuSeqLensQ,
    SeqLens,
    O,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_kst: tl.constexpr,
    stride_ksh: tl.constexpr,
    stride_vst: tl.constexpr,
    stride_vsh: tl.constexpr,
    stride_pt: tl.constexpr,
    stride_ps: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    QUANTIZED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_q = tl.program_id(0)
    head_q = tl.program_id(1)
    request = tl.program_id(2)
    start_q = tl.load(CuSeqLensQ + request)
    end_q = tl.load(CuSeqLensQ + request + 1)
    query_len = end_q - start_q
    if block_q * BLOCK_M >= query_len:
        return

    sequence_len = tl.load(SeqLens + request)
    history_len = sequence_len - query_len
    table_row = tl.load(TableIndices + request)
    head_kv = head_q // GROUP_SIZE
    offs_m = block_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    query = tl.load(
        Q
        + (start_q + offs_m[:, None]) * stride_qt
        + head_q * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=(offs_m[:, None] < query_len) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    # A query tile never needs keys beyond its last query's absolute position.
    key_end = tl.minimum(sequence_len, history_len + (block_q + 1) * BLOCK_M)
    for start_n in range(0, key_end, BLOCK_N):
        logical_tokens = start_n + offs_n
        locations = tl.load(
            PageTable + table_row * stride_pt + logical_tokens * stride_ps,
            mask=logical_tokens < sequence_len,
            other=0,
        ).to(tl.int64)
        key = tl.load(
            K
            + locations[None, :] * stride_kt
            + head_kv * stride_kh
            + offs_d[:, None] * stride_kd,
            mask=(logical_tokens[None, :] < sequence_len)
            & (offs_d[:, None] < HEAD_DIM),
            other=0.0,
        )
        if QUANTIZED:
            key_scale = tl.load(
                KScale + locations * stride_kst + head_kv * stride_ksh,
                mask=logical_tokens < sequence_len,
                other=0.0,
            )
            key = (key.to(tl.float32) * key_scale[None, :]).to(query.dtype)
        scores = tl.dot(query, key) * SCALE
        causal = logical_tokens[None, :] <= history_len + offs_m[:, None]
        scores = tl.where(
            (logical_tokens[None, :] < sequence_len) & causal, scores, float("-inf")
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        probabilities = tl.exp2(scores - new_maximum[:, None])
        correction = tl.exp2(maximum - new_maximum)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        value = tl.load(
            V
            + locations[:, None] * stride_vt
            + head_kv * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=(logical_tokens[:, None] < sequence_len)
            & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        if QUANTIZED:
            value_scale = tl.load(
                VScale + locations * stride_vst + head_kv * stride_vsh,
                mask=logical_tokens < sequence_len,
                other=0.0,
            )
            value = (value.to(tl.float32) * value_scale[:, None]).to(query.dtype)
        accumulator = tl.dot(probabilities.to(query.dtype), value, accumulator)
        maximum = new_maximum

    result = accumulator / denominator[:, None]
    tl.store(
        O
        + (start_q + offs_m[:, None]) * NUM_Q_HEADS * HEAD_DIM
        + head_q * HEAD_DIM
        + offs_d[None, :],
        result.to(O.dtype.element_ty),
        mask=(offs_m[:, None] < query_len) & (offs_d[None, :] < HEAD_DIM),
    )


def paged_prefill_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    page_table: torch.Tensor,
    table_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
) -> torch.Tensor:
    """Attend packed query tokens to paged history without a BF16 cache copy.

    ``page_table`` stores physical token locations, including for page sizes
    above one. The request metadata stays on the device. Causal positions are
    right aligned: query ``i`` attends through ``seq_len - query_len + i``.
    Passing BF16 caches and no scales provides the same attention algorithm
    for comparison with the INT8 storage path.
    """
    if q.ndim != 3 or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("q must have shape [tokens, heads, dim] and BF16/FP16 dtype")
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[-2]
    if head_dim != k_cache.shape[-1] or num_q_heads % num_kv_heads:
        raise ValueError("query and KV head dimensions must match with integral GQA groups")
    if k_cache.shape != v_cache.shape or k_cache.dtype != v_cache.dtype:
        raise ValueError("K/V caches must have matching shapes and dtypes")
    quantized = k_cache.dtype == torch.int8
    if quantized != (k_scale is not None and v_scale is not None):
        raise ValueError("INT8 caches require both FP32 scales; floating caches require no scales")
    if not quantized and (k_scale is not None or v_scale is not None or k_cache.dtype != q.dtype):
        raise ValueError("floating caches must match the query dtype and have no scales")
    if quantized and (k_scale.dtype != torch.float32 or v_scale.dtype != torch.float32):
        raise ValueError("INT8 cache scales must use FP32")
    batch_size = table_indices.numel()
    if cu_seqlens_q.numel() != batch_size + 1 or seq_lens.numel() != batch_size:
        raise ValueError("request metadata has inconsistent lengths")
    output = torch.empty((num_tokens, num_q_heads, head_dim), dtype=q.dtype, device=q.device)
    if num_tokens == 0 or batch_size == 0:
        return output
    if max_query_len <= 0:
        raise ValueError("max_query_len must be positive for a nonempty batch")
    k = k_cache.view(-1, num_kv_heads, head_dim)
    v = v_cache.view(-1, num_kv_heads, head_dim)
    if quantized:
        ks = k_scale.view(-1, num_kv_heads)
        vs = v_scale.view(-1, num_kv_heads)
        scale_strides = (*ks.stride(), *vs.stride())
    else:
        ks, vs = k, v  # unused pointer arguments in the floating specialization
        scale_strides = (0, 0, 0, 0)
    block_m, block_n = 64, 64
    _paged_prefill_attention[(triton.cdiv(max_query_len, block_m), num_q_heads, batch_size)](
        q,
        k,
        v,
        ks,
        vs,
        page_table,
        table_indices,
        cu_seqlens_q,
        seq_lens,
        output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *scale_strides,
        *page_table.stride(),
        NUM_Q_HEADS=num_q_heads,
        GROUP_SIZE=num_q_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        SCALE=head_dim**-0.5 * 1.4426950408889634,
        QUANTIZED=quantized,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        num_warps=4,
        num_stages=1 if head_dim > 128 else 2,
    )
    return output
