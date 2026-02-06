"""
Attention with Column-wise Softmax Attention Weight Sum
Efficient Triton kernel implementation with fused QK^T + softmax + col_sum
"""

import math
import torch
import triton
import triton.language as tl
try:
    from .flash import maybe_contiguous, attention, get_fwd_config
    from .dropout import philox_cuda_seed_offset
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from flash import maybe_contiguous, attention, get_fwd_config
    from dropout import philox_cuda_seed_offset

__all__ = ["attention_cross_token_softmax_sum", "attention_cross_token_softmax_sum_buffered"]


@triton.jit
def _cross_token_weights_sum_kernel_fused(
    Q, K, V, sm_scale,
    dropout_p,
    seed,
    offset,
    L, O, CrossTokenWeightsSum,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_cross_token_weights_z, stride_cross_token_weights_h, stride_cross_token_weights_n,
    Z, H, M, N, P_SEQ, split_pos,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr, LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    """
    Fused kernel for computing attention output and cross-token softmax weights sum

    Optimized version that combines attention computation with cross-token sum in a single kernel.
    This avoids the overhead of calling attention() separately and reduces QK^T computations from 3 to 2.

    Strategy:
    1. Pass 1: Compute attention output (O) and normalization statistics (m_i, l_i)
    2. Pass 2: Re-compute QK^T for keys < split, apply softmax, accumulate cross-token sum

    Only queries >= split_pos contribute to keys < split_pos in the cross-token sum.
    """
    input_dtype = Q.dtype.element_ty

    # Grid IDs
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # Calculate corresponding head_k index
    off_hk = off_h // num_groups

    # Offset pointers
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M
    CrossTokenWeightsSum += off_z * stride_cross_token_weights_z + off_hk * stride_cross_token_weights_h

    # Query block offsets
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # Determine if current query block has queries after split
    is_query_after_split = offs_m >= split_pos
    block_start_m = start_m * BLOCK_M
    block_end_m = block_start_m + BLOCK_M
    has_query_after_split = block_end_m > split_pos

    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]

    # Initialize pointers
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m

    # Initialize accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load queries
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    # Dot I trick for small head_dim
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k,
                    tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
                    tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I).to(input_dtype)

    # Determine loop bound
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    max_blocks = (hi + BLOCK_N - 1) // BLOCK_N

    # -----------------------------------------------------------
    # Pass 1: Standard Flash Attention Calculation
    # Compute acc, m_i (max), and l_i (denominator)
    # -----------------------------------------------------------
    for block_offset in range(max_blocks):
        block_idx = (start_m % max_blocks + block_offset) % max_blocks
        start_n = block_idx * BLOCK_N

        if start_n < hi:
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + offs_n_base

            # Load K, V
            k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
            v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

            if DIVISIBLE_N:
                k = tl.load(k_ptrs, cache_modifier=".cg")
                v = tl.load(v_ptrs, cache_modifier=".cg")
            else:
                mask_n = offs_n < N
                k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
                v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

            # Compute QK^T
            s = tl.dot(q, k)

            # Apply masks
            if not DIVISIBLE_N:
                mask_n = offs_n < N
                s = tl.where(mask_n[None, :], s, float("-inf"))

            if IS_CAUSAL:
                causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                s = tl.where(causal_mask, s, float("-inf"))

            # Update Statistics & Accumulator
            m_i_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
            p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)
            p_sum = tl.sum(p, 1)

            # Dropout
            if IS_DROPOUT:
                offs_rng = start_n + offs_rng_base
                pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
                p *= pmask.to(tl.float32)

            acc *= alpha[:, None]
            acc += tl.dot(p.to(input_dtype), v)
            l_i = l_i * alpha + p_sum
            m_i = m_i_new

    # -----------------------------------------------------------
    # Post-Processing 1: Finalize L and O
    # -----------------------------------------------------------
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * sm_scale + tl.log(l_i)

    # Store L and O
    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")

    # -----------------------------------------------------------
    # Pass 2: Compute softmax weights and accumulate cross-token column sums
    # Only process if there are queries after split
    # -----------------------------------------------------------
    if has_query_after_split:
        # Only process key blocks before split
        num_split_blocks = (split_pos + BLOCK_N - 1) // BLOCK_N

        for n_block_idx in range(num_split_blocks):
            start_n = n_block_idx * BLOCK_N

            if start_n < hi:
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + offs_n_base

                # Determine which keys in this block are before split
                mask_n_before_split = offs_n < split_pos

                # Load keys (no need for V in Pass 2)
                k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
                if DIVISIBLE_N:
                    k = tl.load(k_ptrs, cache_modifier=".cg")
                else:
                    mask_n = offs_n < N
                    k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")

                # Compute QK^T
                s = tl.dot(q, k)
                    
                # Apply masks
                if not DIVISIBLE_N:
                    mask_n = offs_n < N
                    s = tl.where(mask_n[None, :], s, float("-inf"))
                if IS_CAUSAL:
                    causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                    s = tl.where(causal_mask, s, float("-inf"))

                # Compute final softmax weights using m_i and l_i from Pass 1
                p = tl.math.exp2(s * qk_scale - m_i[:, None] * qk_scale)
                softmax_weights = p / (l_i[:, None] + 1e-12)

                # Apply split mask: only queries >= split_pos contribute to keys < split_pos
                split_mask = is_query_after_split[:, None] & mask_n_before_split[None, :]

                if not DIVISIBLE_M:
                    mask_m = offs_m < M
                    split_mask = split_mask & mask_m[:, None]

                if not DIVISIBLE_N:
                    mask_n = offs_n < N
                    split_mask = split_mask & mask_n[None, :]
                    
                if IS_CAUSAL:
                    causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                    split_mask = split_mask & causal_mask

                # Apply mask: set non-split contributions to 0
                softmax_weights_split = tl.where(split_mask, softmax_weights, 0.0)

                # Sum over queries to get column sum contribution
                col_sum_block = tl.sum(softmax_weights_split, axis=0)  # (BLOCK_N,)

                # Accumulate to global column sum (using atomic add)
                col_sum_ptrs = CrossTokenWeightsSum + offs_n * stride_cross_token_weights_n
                if DIVISIBLE_N:
                    tl.atomic_add(col_sum_ptrs, col_sum_block, mask=mask_n_before_split, sem="relaxed")
                else:
                    mask_n = offs_n < N
                    mask_n_atomic = mask_n & mask_n_before_split
                    tl.atomic_add(col_sum_ptrs, col_sum_block, mask=mask_n_atomic, sem="relaxed")


@triton.jit
def _cross_token_weights_sum_kernel(
    Q, K, sm_scale,
    CrossTokenWeightsSum,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_cross_token_weights_z, stride_cross_token_weights_h, stride_cross_token_weights_n,
    Z, H, M, N, P_SEQ, split_pos,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    """
    Fused kernel for computing column-wise softmax attention weight sum with split support

    Computes cross-token softmax weights sum: only queries >= split_pos contribute to keys < split_pos
    Output shape: (B, Hk, split_pos) - only keys before split position
    """
    # Grid IDs
    start_m = tl.program_id(0)  # Query block index
    off_h = tl.program_id(1)    # Head_q index
    off_z = tl.program_id(2)    # Batch index

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # Calculate corresponding head_k index
    off_hk = off_h // num_groups

    # Offset pointers
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh

    # Query block offsets
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # Determine if current query block has queries after split
    is_query_after_split = offs_m >= split_pos
    block_start_m = start_m * BLOCK_M
    block_end_m = block_start_m + BLOCK_M
    has_query_after_split = block_end_m > split_pos

    # Load queries
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    # Determine loop bound
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if hi < 0:
            hi = 0
    else:
        hi = N

    # Pass 1: Compute max and sum for softmax normalization
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    num_n_blocks = tl.cdiv(hi, BLOCK_N)
    for n_block_idx in range(num_n_blocks):
        start_n = n_block_idx * BLOCK_N
        if start_n < hi:
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + offs_n_base

            # Load keys
            k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
            if DIVISIBLE_N:
                k = tl.load(k_ptrs, cache_modifier=".cg")
            else:
                mask_n = offs_n < N
                k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")

            # Compute QK^T
            s = tl.dot(q, k)

            # Apply masks
            if not DIVISIBLE_N:
                mask_n = offs_n < N
                s = tl.where(mask_n[None, :], s, float("-inf"))
            if IS_CAUSAL:
                causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                s = tl.where(causal_mask, s, float("-inf"))

            # Update max and sum for softmax
            m_i_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
            p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

    # Pass 2: Compute softmax weights and accumulate column sums (only for split region)
    # Only accumulate if there are queries after split
    if has_query_after_split:
        for n_block_idx in range(num_n_blocks):
            start_n = n_block_idx * BLOCK_N
            if start_n < hi:
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + offs_n_base

                # Only process key blocks before split
                key_block_in_split_range = start_n < split_pos
                if key_block_in_split_range:
                    # Determine which keys in this block are before split
                    mask_n_before_split = offs_n < split_pos

                    # Load keys
                    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
                    if DIVISIBLE_N:
                        k = tl.load(k_ptrs, cache_modifier=".cg")
                    else:
                        mask_n = offs_n < N
                        k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")

                    # Compute QK^T
                    s = tl.dot(q, k)

                    # Apply masks
                    if not DIVISIBLE_N:
                        mask_n = offs_n < N
                        s = tl.where(mask_n[None, :], s, float("-inf"))
                    if IS_CAUSAL:
                        causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                        s = tl.where(causal_mask, s, float("-inf"))

                    # Compute final softmax weights
                    p = tl.math.exp2(s * qk_scale - m_i[:, None] * qk_scale)
                    softmax_weights = p / (l_i[:, None] + 1e-12)

                    # Apply split mask: only queries >= split_pos contribute to keys < split_pos
                    split_mask = is_query_after_split[:, None] & mask_n_before_split[None, :]
                    if not DIVISIBLE_N:
                        mask_n = offs_n < N
                        split_mask = split_mask & mask_n[None, :]
                    if IS_CAUSAL:
                        causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                        split_mask = split_mask & causal_mask

                    # Apply mask: set non-split contributions to 0
                    softmax_weights_split = tl.where(split_mask, softmax_weights, 0.0)

                    # Sum over queries to get column sum contribution
                    col_sum_block = tl.sum(softmax_weights_split, axis=0)  # (BLOCK_N,)

                    # Accumulate to global column sum (using atomic add)
                    col_sum_ptrs = CrossTokenWeightsSum + off_z * stride_cross_token_weights_z + off_hk * stride_cross_token_weights_h + offs_n * stride_cross_token_weights_n
                    if DIVISIBLE_N:
                        tl.atomic_add(col_sum_ptrs, col_sum_block, mask=mask_n_before_split, sem="relaxed")
                    else:
                        mask_n = offs_n < N
                        mask_n_atomic = mask_n & mask_n_before_split
                        tl.atomic_add(col_sum_ptrs, col_sum_block, mask=mask_n_atomic, sem="relaxed")


@triton.jit
def _cross_token_weights_sum_kernel_buffered(
    Q, K, V, sm_scale,
    O,
    QK_Buffer,
    CrossTokenWeightsSum,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_bufz, stride_bufh, stride_bufm, stride_bufn,
    stride_cross_token_weights_z, stride_cross_token_weights_h, stride_cross_token_weights_n,
    Z, H, M, N, P_SEQ, split_pos,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    """
    Buffered version: stores QK^T in Pass 1, loads it in Pass 2
    Also computes attention output in the same kernel to save memory
    Supports split-based cross-token sum

    Uses global memory buffer to store QK^T scores between passes.
    This trades compute for memory bandwidth.
    """
    # Grid IDs
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # Calculate corresponding head_k index
    off_hk = off_h // num_groups

    # Offset pointers
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    QK_Buffer += off_z * stride_bufz + off_h * stride_bufh

    # Query block offsets
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # Determine if current query block has queries after split
    is_query_after_split = offs_m >= split_pos
    block_start_m = start_m * BLOCK_M
    block_end_m = block_start_m + BLOCK_M
    has_query_after_split = block_end_m > split_pos

    # Load queries
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    # Determine loop bound
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if hi < 0:
            hi = 0
    else:
        hi = N

    # Pass 1: Compute QK^T, track max and sum, store QK^T to buffer, and accumulate attention output
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    num_n_blocks = tl.cdiv(hi, BLOCK_N)
    for n_block_idx in range(num_n_blocks):
        start_n = n_block_idx * BLOCK_N
        if start_n < hi:
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + offs_n_base

            # Load keys and values
            k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
            v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
            if DIVISIBLE_N:
                k = tl.load(k_ptrs, cache_modifier=".cg")
                v = tl.load(v_ptrs, cache_modifier=".cg")
            else:
                mask_n = offs_n < N
                k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
                v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

            # Compute QK^T
            s = tl.dot(q, k)

            # Apply masks
            if not DIVISIBLE_N:
                mask_n = offs_n < N
                s = tl.where(mask_n[None, :], s, float("-inf"))
            if IS_CAUSAL:
                causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                s = tl.where(causal_mask, s, float("-inf"))

            # Store QK^T to buffer
            buf_ptrs = QK_Buffer + (offs_m[:, None] * stride_bufm + offs_n[None, :] * stride_bufn)
            if DIVISIBLE_M and DIVISIBLE_N:
                tl.store(buf_ptrs, s)
            elif DIVISIBLE_M:
                mask_n = offs_n < N
                tl.store(buf_ptrs, s, mask=mask_n[None, :])
            elif DIVISIBLE_N:
                mask_m = offs_m < M
                tl.store(buf_ptrs, s, mask=mask_m[:, None])
            else:
                mask_m = offs_m < M
                mask_n = offs_n < N
                tl.store(buf_ptrs, s, mask=mask_m[:, None] & mask_n[None, :])

            # Update max and sum for softmax
            m_i_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
            p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

            # Update accumulator for attention output
            acc *= alpha[:, None]
            acc += tl.dot(p.to(Q.dtype.element_ty), v)

            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

    # Pass 2: Load QK^T from buffer, compute softmax weights, and accumulate column sums (only for split region)
    # Only accumulate if there are queries after split
    if has_query_after_split:
        for n_block_idx in range(num_n_blocks):
            start_n = n_block_idx * BLOCK_N
            if start_n < hi:
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + offs_n_base

                # Only process key blocks before split
                key_block_in_split_range = start_n < split_pos
                if key_block_in_split_range:
                    # Determine which keys in this block are before split
                    mask_n_before_split = offs_n < split_pos

                    # Load QK^T from buffer
                    buf_ptrs = QK_Buffer + (offs_m[:, None] * stride_bufm + offs_n[None, :] * stride_bufn)
                    if DIVISIBLE_M and DIVISIBLE_N:
                        s = tl.load(buf_ptrs)
                    elif DIVISIBLE_M:
                        mask_n = offs_n < N
                        s = tl.load(buf_ptrs, mask=mask_n[None, :])
                    elif DIVISIBLE_N:
                        mask_m = offs_m < M
                        s = tl.load(buf_ptrs, mask=mask_m[:, None])
                    else:
                        mask_m = offs_m < M
                        mask_n = offs_n < N
                        s = tl.load(buf_ptrs, mask=mask_m[:, None] & mask_n[None, :])

                    # Compute final softmax weights
                    p = tl.math.exp2(s * qk_scale - m_i[:, None] * qk_scale)
                    softmax_weights = p / (l_i[:, None] + 1e-12)

                    # Apply split mask: only queries >= split_pos contribute to keys < split_pos
                    split_mask = is_query_after_split[:, None] & mask_n_before_split[None, :]
                    if not DIVISIBLE_N:
                        mask_n = offs_n < N
                        split_mask = split_mask & mask_n[None, :]

                    if not DIVISIBLE_M:
                        mask_m = offs_m < M
                        split_mask = split_mask & mask_m[:,None]

                    if IS_CAUSAL:
                        causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                        split_mask = split_mask & causal_mask

                    # Apply mask: set non-split contributions to 0
                    softmax_weights_split = tl.where(split_mask, softmax_weights, 0.0)

                    # Sum over queries to get column sum contribution
                    col_sum_block = tl.sum(softmax_weights_split, axis=0)  # (BLOCK_N,)

                    # Accumulate to global column sum (using atomic add)
                    col_sum_ptrs = CrossTokenWeightsSum + off_z * stride_cross_token_weights_z + off_hk * stride_cross_token_weights_h + offs_n * stride_cross_token_weights_n
                    if DIVISIBLE_N:
                        tl.atomic_add(col_sum_ptrs, col_sum_block, mask=mask_n_before_split, sem="relaxed")
                    else:
                        mask_n = offs_n < N
                        mask_n_atomic = mask_n & mask_n_before_split
                        tl.atomic_add(col_sum_ptrs, col_sum_block, mask=mask_n_atomic, sem="relaxed")

    # Write attention output
    acc = acc * (1.0 / l_i[:, None])
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    if DIVISIBLE_M:
        tl.store(o_ptrs, acc.to(Q.dtype.element_ty), cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=mask_m[:, None], cache_modifier=".cg")


def attention_cross_token_softmax_sum_buffered(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    split: int,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Buffered version of attention_cross_token_softmax_sum.

    This version stores QK^T in a global memory buffer during Pass 1
    and loads it in Pass 2, avoiding recomputation.
    Also computes attention output in the same fused kernel to reduce memory overhead.

    Trade-off: Saves compute but requires extra memory bandwidth and storage.
    Memory requirement: B × H × M × N × 4 bytes for QK^T buffer.

    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        split: Split position - queries >= split contribute to keys < split
        causal: Whether to use causal mask
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability

    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        cross_token_softmax_sum: Cross-token softmax weight sum, shape (batch_size, num_heads_k, split)
    """
    # Parameter validation
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv, "Q, K, V must have the same head_dim"
    assert Dk in {16, 32, 64, 128}

    B, H, M, D = q.shape
    N = k.shape[2]
    Hk = k.shape[1]
    Hv = v.shape[1]
    assert Hk == Hv, "K and V must have the same number of heads"
    assert H % Hk == 0, "num_heads_q must be divisible by num_heads_k"
    num_groups = H // Hk

    assert 0 < split <= N, f"Split position must be in valid range (0, {N}]"

    P_SEQ = N - M

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    # Ensure contiguity
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

    # Get optimal configuration
    config = get_fwd_config(B, H, M, N, D, causal)
    BLOCK_M, BLOCK_N, num_stages, num_warps = config

    divisible_m = M % BLOCK_M == 0
    divisible_n = N % BLOCK_N == 0

    # Allocate outputs
    output = torch.empty_like(q)

    # Allocate QK^T buffer (B, H, M, N)
    qk_buffer = torch.empty((B, H, M, N), device=q.device, dtype=torch.float32)

    # Allocate output for cross-token sum (only keys before split)
    cross_token_weights_sum = torch.zeros((B, Hk, split), device=q.device, dtype=torch.float32)

    # Launch kernel - one program per (query_block, head_q, batch)
    grid = (triton.cdiv(M, BLOCK_M), H, B)

    _cross_token_weights_sum_kernel_buffered[grid](
        q, k, v, sm_scale,
        output,
        qk_buffer,
        cross_token_weights_sum,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        qk_buffer.stride(0), qk_buffer.stride(1), qk_buffer.stride(2), qk_buffer.stride(3),
        cross_token_weights_sum.stride(0), cross_token_weights_sum.stride(1), cross_token_weights_sum.stride(2),
        B, H, M, N, P_SEQ, split,
        num_groups,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
        num_warps=num_warps, num_stages=num_stages,
    )

    return output, cross_token_weights_sum


def attention_cross_token_softmax_sum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    split: int,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output using Flash Attention, while also returning cross-token softmax attention weight sum.

    Computes split-based cross-token softmax weights: only queries >= split contribute to keys < split.
    This is useful for analyzing attention patterns between different segments of the sequence.

    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        split: Split position - queries >= split contribute to keys < split
        causal: Whether to use causal mask
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability

    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        cross_token_softmax_sum: Cross-token softmax weight sum, shape (batch_size, num_heads_k, split)
                                For each key position j < split:
                                cross_token_softmax_sum[b, h_k, j] = sum over i >= split of softmax(QK^T)[b, h_q, i, j]

    Note:
        Optimized implementation that fuses attention output computation with cross-token sum.
        This reduces QK^T computations from 3 to 2 compared to the previous implementation.
    """
    # Parameter validation
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv, "Q, K, V must have the same head_dim"
    assert Dk in {16, 32, 64, 128}

    B, H, M, D = q.shape
    N = k.shape[2]
    Hk = k.shape[1]
    assert Hk == v.shape[1]
    assert H % Hk == 0
    num_groups = H // Hk

    assert 0 < split <= N, f"Split position must be in valid range (0, {N}]"

    P_SEQ = N - M
    larger_m = M > N

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    # Ensure contiguity
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

    # Dropout preparation
    device = torch.cuda.device_of(q)
    with torch.cuda.device(device):
        is_dropout = dropout_p > 0
        if is_dropout:
            offset_increment = B * H * M * N
            seed, offset = philox_cuda_seed_offset(offset_increment)
        else:
            seed, offset = 0, 0

        # Get optimal configuration
        config = get_fwd_config(B, H, M, N, D, causal)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config

        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0

        # Allocate outputs
        output = torch.empty_like(q)
        L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

        # Allocate output for cross-token sum (only keys before split)
        cross_token_weights_sum = torch.zeros((B, Hk, split), device=q.device, dtype=torch.float32)

        # Launch fused kernel - one program per (query_block, head_q, batch)
        grid = (triton.cdiv(M, BLOCK_M), H, B)

        # Dynamically adjust configuration to handle shared memory insufficiency
        num_stages_adjusted = num_stages
        BLOCK_N_adjusted = BLOCK_N
        divisible_n_adjusted = divisible_n

        for attempt in range(5):
            try:
                _cross_token_weights_sum_kernel_fused[grid](
                    q, k, v, sm_scale,
                    dropout_p, seed, offset,
                    L, output, cross_token_weights_sum,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                    cross_token_weights_sum.stride(0), cross_token_weights_sum.stride(1), cross_token_weights_sum.stride(2),
                    B, H, M, N, P_SEQ, split,
                    num_groups,
                    BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N_adjusted,
                    IS_CAUSAL=causal, IS_DROPOUT=is_dropout, LARGER_M=larger_m,
                    DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n_adjusted,
                    num_warps=num_warps, num_stages=num_stages_adjusted,
                )
                break
            except Exception as e:
                if attempt < 4:
                    error_msg = str(e).lower()
                    if "shared memory" in error_msg or "outofresources" in error_msg:
                        if num_stages_adjusted > 1:
                            num_stages_adjusted = max(1, num_stages_adjusted - 1)
                        elif BLOCK_N_adjusted > 32:
                            BLOCK_N_adjusted = max(32, BLOCK_N_adjusted // 2)
                            divisible_n_adjusted = N % BLOCK_N_adjusted == 0
                        else:
                            raise RuntimeError(
                                f"Cannot resolve shared memory insufficiency by adjusting configuration. "
                                f"Current config: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N_adjusted}, "
                                f"num_stages={num_stages_adjusted}, num_warps={num_warps}"
                            ) from e
                    else:
                        raise
                else:
                    raise

    return output, cross_token_weights_sum
