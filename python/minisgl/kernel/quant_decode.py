from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_store_kernel(
    K,
    V,
    LOCATIONS,
    K_CACHE,
    V_CACHE,
    K_SCALE,
    V_SCALE,
    K_STRIDE_T: tl.constexpr,
    K_STRIDE_H: tl.constexpr,
    K_STRIDE_D: tl.constexpr,
    V_STRIDE_T: tl.constexpr,
    V_STRIDE_H: tl.constexpr,
    V_STRIDE_D: tl.constexpr,
    LOCATION_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    location = tl.load(LOCATIONS + token * LOCATION_STRIDE).to(tl.int64)
    if location >= 0:
        d = tl.arange(0, BLOCK_D)
        k = tl.load(
            K + token * K_STRIDE_T + head * K_STRIDE_H + d * K_STRIDE_D,
            d < DIM,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V + token * V_STRIDE_T + head * V_STRIDE_H + d * V_STRIDE_D,
            d < DIM,
            other=0.0,
        ).to(tl.float32)
        k_max = tl.max(tl.abs(k), 0)
        v_max = tl.max(tl.abs(v), 0)
        tiny = tl.full((), 1.1754943508222875e-38, tl.float32)
        k_scale = tl.where(k_max > 0.0, tl.maximum(k_max / 127.0, tiny), 1.0)
        v_scale = tl.where(v_max > 0.0, tl.maximum(v_max / 127.0, tiny), 1.0)
        # cvt.rni implements the same ties-to-even rounding as torch.round.
        # IEEE division avoids reciprocal approximation changing boundary ties.
        k_quant = tl.inline_asm_elementwise(
            "cvt.rni.s32.f32 $0, $1;",
            constraints="=r,f",
            args=[tl.div_rn(k, k_scale)],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        v_quant = tl.inline_asm_elementwise(
            "cvt.rni.s32.f32 $0, $1;",
            constraints="=r,f",
            args=[tl.div_rn(v, v_scale)],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        cache_offset = (location * HEADS + head) * DIM + d
        tl.store(K_CACHE + cache_offset, tl.minimum(tl.maximum(k_quant, -127), 127), d < DIM)
        tl.store(V_CACHE + cache_offset, tl.minimum(tl.maximum(v_quant, -127), 127), d < DIM)
        tl.store(K_SCALE + location * HEADS + head, k_scale)
        tl.store(V_SCALE + location * HEADS + head, v_scale)


@triton.jit
def _decode_partial_kernel(
    Q,
    K_CACHE,
    V_CACHE,
    K_SCALE,
    V_SCALE,
    PAGE_TABLE,
    TABLE_INDICES,
    SEQ_LENS,
    PARTIAL_OUTPUT,
    PARTIAL_STATS,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    Q_STRIDE_D: tl.constexpr,
    TABLE_STRIDE_R: tl.constexpr,
    TABLE_STRIDE_S: tl.constexpr,
    TABLE_INDEX_STRIDE: tl.constexpr,
    SEQ_LEN_STRIDE: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = head // (QUERY_HEADS // KV_HEADS)
    table_index = tl.load(TABLE_INDICES + batch * TABLE_INDEX_STRIDE)
    seq_len = tl.load(SEQ_LENS + batch * SEQ_LEN_STRIDE)
    split_size = tl.cdiv(seq_len, NUM_SPLITS * BLOCK_N) * BLOCK_N
    start = split * split_size
    end = tl.minimum(start + split_size, seq_len)
    d = tl.arange(0, BLOCK_D)
    q = tl.load(
        Q + batch * Q_STRIDE_B + head * Q_STRIDE_H + d * Q_STRIDE_D,
        d < DIM,
        other=0.0,
    ).to(tl.float32)
    accumulator = tl.full((BLOCK_D,), 0.0, tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    for block_start in range(start, end, BLOCK_N):
        n = block_start + tl.arange(0, BLOCK_N)
        valid_n = n < end
        location = tl.load(
            PAGE_TABLE + table_index * TABLE_STRIDE_R + n * TABLE_STRIDE_S,
            valid_n,
            other=0,
        ).to(tl.int64)
        scale_offset = location * KV_HEADS + kv_head
        k_scale = tl.load(K_SCALE + scale_offset, valid_n, other=1.0)
        v_scale = tl.load(V_SCALE + scale_offset, valid_n, other=1.0)
        offset = scale_offset[:, None] * DIM + d[None, :]
        valid = valid_n[:, None] & (d[None, :] < DIM)
        k = tl.load(K_CACHE + offset, valid, other=0).to(tl.float32)
        v = tl.load(V_CACHE + offset, valid, other=0).to(tl.float32)
        # Materialize the same BF16 values as the reference cache reader before
        # accumulating QK and PV in FP32. No full history tensor is allocated.
        k = (k * k_scale[:, None]).to(Q.dtype.element_ty).to(tl.float32)
        v = (v * v_scale[:, None]).to(Q.dtype.element_ty).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * SOFTMAX_SCALE
        scores = tl.where(valid_n, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 0))
        correction = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum)
        denominator = denominator * correction + tl.sum(probabilities, 0)
        accumulator = accumulator * correction + tl.sum(probabilities[:, None] * v, 0)
        maximum = next_maximum
    output_base = ((batch * QUERY_HEADS + head) * NUM_SPLITS + split) * DIM
    stats_base = ((batch * QUERY_HEADS + head) * NUM_SPLITS + split) * 2
    tl.store(PARTIAL_OUTPUT + output_base + d, accumulator, d < DIM)
    tl.store(PARTIAL_STATS + stats_base, maximum)
    tl.store(PARTIAL_STATS + stats_base + 1, denominator)


@triton.jit
def _decode_reduce_kernel(
    PARTIAL_OUTPUT,
    PARTIAL_STATS,
    OUTPUT,
    OUTPUT_STRIDE_B: tl.constexpr,
    OUTPUT_STRIDE_H: tl.constexpr,
    OUTPUT_STRIDE_D: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.arange(0, BLOCK_SPLITS)
    d = tl.arange(0, BLOCK_D)
    stats_base = ((batch * QUERY_HEADS + head) * NUM_SPLITS + split) * 2
    maximum = tl.load(PARTIAL_STATS + stats_base, split < NUM_SPLITS, other=-float("inf"))
    denominator = tl.load(PARTIAL_STATS + stats_base + 1, split < NUM_SPLITS, other=0.0)
    global_maximum = tl.max(maximum, 0)
    correction = tl.where(denominator > 0.0, tl.exp(maximum - global_maximum), 0.0)
    total_denominator = tl.sum(denominator * correction, 0)
    output_base = ((batch * QUERY_HEADS + head) * NUM_SPLITS + split) * DIM
    partial = tl.load(
        PARTIAL_OUTPUT + output_base[:, None] + d[None, :],
        (split[:, None] < NUM_SPLITS) & (d[None, :] < DIM),
        other=0.0,
    )
    result = tl.sum(partial * correction[:, None], 0)
    result = result / tl.where(total_denominator > 0.0, total_denominator, 1.0)
    tl.store(
        OUTPUT + batch * OUTPUT_STRIDE_B + head * OUTPUT_STRIDE_H + d * OUTPUT_STRIDE_D,
        result,
        d < DIM,
    )


DecodeWorkspace = tuple[torch.Tensor, torch.Tensor]


def quant_store(
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Quantize and store K/V; negative output locations are padding sentinels.

    Cache layer slices must be contiguous, with their final dimensions [Hkv, D].
    Inputs must be finite. This asynchronous kernel intentionally performs no
    device-to-host finite-value check, including during CUDA Graph capture.
    """
    heads, dim = k_cache.shape[-2:]
    k = k.reshape(-1, heads, dim)
    v = v.reshape(-1, heads, dim)
    if k.shape != v.shape or out_loc.numel() != k.shape[0]:
        raise ValueError("K/V token shapes must agree with out_loc")
    if k.shape[0] == 0:
        return
    _quant_store_kernel[(k.shape[0], heads)](
        k,
        v,
        out_loc,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        *k.stride(),
        *v.stride(),
        out_loc.stride(0),
        heads,
        dim,
        triton.next_power_of_2(dim),
        num_warps=4,
        enable_fp_fusion=False,
    )


def create_decode_workspace(q: torch.Tensor, num_splits: int = 32) -> DecodeWorkspace:
    """Allocate reusable split-K buffers before capture for q shaped [B, Hq, D]."""
    if q.ndim != 3 or num_splits <= 0:
        raise ValueError("Expected q [B, Hq, D] and a positive split count")
    batch, heads, dim = q.shape
    return (
        torch.empty((batch, heads, num_splits, dim), dtype=torch.float32, device=q.device),
        torch.empty((batch, heads, num_splits, 2), dtype=torch.float32, device=q.device),
    )


def decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    table_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace: DecodeWorkspace | None = None,
) -> torch.Tensor:
    """Paged single-token GQA decode with inline INT8 dequantization.

    q is [B, Hq, D]. page_table stores physical token locations; seq_lens and
    table_indices contain one entry per batch row. A zero length emits zeros.
    Reuse workspace to avoid split-K allocations in the graph capture path.
    All host branches depend only on tensor metadata, never device values.
    """
    if q.ndim != 3:
        raise ValueError("decode_attention expects q [B, Hq, D]")
    batch, query_heads, dim = q.shape
    kv_heads, cache_dim = k_cache.shape[-2:]
    if cache_dim != dim or query_heads % kv_heads != 0:
        raise ValueError("Query dimensions must match the cache and Hq must be divisible by Hkv")
    if table_indices.numel() != batch or seq_lens.numel() != batch:
        raise ValueError("Expected one table index and sequence length per query")
    if workspace is None:
        workspace = create_decode_workspace(q)
    partial_output, partial_stats = workspace
    num_splits = partial_output.shape[2]
    if (
        partial_output.shape != (batch, query_heads, num_splits, dim)
        or partial_stats.shape != (batch, query_heads, num_splits, 2)
    ):
        raise ValueError("Decode workspace shape does not match q")
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    if batch == 0:
        return output
    block_d = triton.next_power_of_2(dim)
    _decode_partial_kernel[(batch, query_heads, num_splits)](
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        page_table,
        table_indices,
        seq_lens,
        partial_output,
        partial_stats,
        *q.stride(),
        *page_table.stride(),
        table_indices.stride(0),
        seq_lens.stride(0),
        query_heads,
        kv_heads,
        dim,
        num_splits,
        64,
        block_d,
        1.0 / math.sqrt(dim),
        num_warps=4,
        num_stages=2,
    )
    _decode_reduce_kernel[(batch, query_heads)](
        partial_output,
        partial_stats,
        output,
        *output.stride(),
        query_heads,
        dim,
        num_splits,
        triton.next_power_of_2(num_splits),
        block_d,
        num_warps=4,
    )
    return output
