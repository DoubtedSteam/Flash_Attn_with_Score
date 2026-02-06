"""
Attention with Column-wise Score Sum
Fused computation of column-wise attention score sum during Flash Attention computation
"""

import math
import torch
import triton
import triton.language as tl
try:
    from .flash import maybe_contiguous, get_fwd_config
    from .dropout import philox_cuda_seed_offset
except ImportError:
    # Handle case when running as standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from flash import maybe_contiguous, get_fwd_config
    from dropout import philox_cuda_seed_offset

@triton.jit
def _fwd_kernel_with_col_sum(
    Q, K, V, sm_scale,
    dropout_p,
    seed,
    offset,
    L, O, ColSum,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_colsum_z, stride_colsum_h, stride_colsum_n,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr, LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    ):
    """
    Flash Attention forward kernel with column-wise score sum computation
    Accumulates score sum for each key position using atomic operations while computing attention
    
    Key points:
    - Multiple query blocks write to the same column positions, requiring atomic operations
    - Within the key block loop, compute local col_sum_block for each key block
    - Use tl.atomic_add to accumulate to global col_sum
    
    Optimization: In causal mode, use reverse-order processing of key blocks to reduce contention
    - With forward-order processing, multiple query blocks process the same key columns simultaneously, causing contention on col_sum
    - With reverse-order processing, different query blocks process different key columns simultaneously, reducing contention
    - Example: query block 0 processes key column N-1 first, query block 1 processes key column N-2 first, and so on
    """
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M
    ColSum += off_z * stride_colsum_z + off_hk * stride_colsum_h

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    # Dot I trick: to place q in registers, it saves shared memory
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k,
                    tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
                    tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I).to(input_dtype)

    # Loop bound for N
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    # In causal mode, process keys in reverse order to reduce col_sum contention
    # Reverse-order processing allows different query blocks to process different key columns simultaneously, reducing write contention
    if IS_CAUSAL:
        # Calculate the number of blocks to process (round up)
        # Note: In Triton, range loops are unrolled at compile time, so we need to know the maximum number of blocks
        # Use N as the upper bound since hi <= N
        max_blocks = (N + BLOCK_N - 1) // BLOCK_N
        
        # Use forward loop but calculate reverse-order start_n
        for block_idx in range(max_blocks):
            # Calculate reverse-order start_n: start from the last block and decrement forward
            # First calculate the index of the last valid block
            last_valid_block_idx = (hi - 1) // BLOCK_N
            # Calculate the current block's index in reverse order: block_idx=0 corresponds to the last block
            reverse_block_idx = last_valid_block_idx - block_idx
            # Calculate the corresponding start_n
            start_n = reverse_block_idx * BLOCK_N
            
            # Check if block is valid: only process blocks where start_n >= 0 and start_n < hi
            # Use conditional to wrap all computations, avoiding continue
            if start_n >= 0 and start_n < hi:
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + offs_n_base
                
                # Calculate pointer positions (recompute each time instead of incrementing)
                k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)
                v_ptrs = V + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
                
                # -- Load k, v --
                if DIVISIBLE_N:
                    k = tl.load(k_ptrs, cache_modifier=".cg")
                    v = tl.load(v_ptrs, cache_modifier=".cg")
                else:
                    mask_n = offs_n < N
                    k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
                    v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

                # -- Compute QK^T once (only once per key block) --
                s = tl.dot(q, k)
                
                # -- Apply mask and scaling --
                # 1. Apply boundary mask
                if not DIVISIBLE_N:
                    mask_n = offs_n < N
                    s = tl.where(mask_n[None, :], s, float("-inf"))
                
                # 2. Apply causal mask
                causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                s = tl.where(causal_mask, s, float("-inf"))

                # -- Accumulate col sum (sum(scores)) --
                # Compute scaled scores for col sum
                s_for_sum = s * sm_scale
                # Apply masks (set masked positions to 0 for sum)
                if not DIVISIBLE_N:
                    mask_n = offs_n < N
                    s_for_sum = tl.where(mask_n[None, :], s_for_sum, 0.0)
                s_for_sum = tl.where(causal_mask, s_for_sum, 0.0)
                # Sum over queries dimension (dim=0), get (BLOCK_N,) - local sum for each key position
                col_sum_block = tl.sum(s_for_sum, 0)  # (BLOCK_N,)

                # -- Numerically stable softmax --
                m_i_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
                p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)
                p_sum = tl.sum(p, 1)
                
                # With reverse-order processing, different query blocks process different key columns, reducing contention
                # Still need atomic operations to ensure correctness (there may still be some contention)
                col_sum_ptrs = ColSum + offs_n
                if DIVISIBLE_N:
                    tl.atomic_add(col_sum_ptrs, col_sum_block, sem="relaxed")
                else:
                    mask_n = offs_n < N
                    tl.atomic_add(col_sum_ptrs, col_sum_block, mask=mask_n, sem="relaxed")

                # -- dropout --
                if IS_DROPOUT:
                    offs_rng = start_n + offs_rng_base
                    pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
                    p *= pmask.to(tl.float32)

                # -- Update accumulators --
                acc *= alpha[:, None]
                acc += tl.dot(p.to(input_dtype), v)
                l_i = l_i * alpha + p_sum
                m_i = m_i_new
    else:
        # Non-causal case: use Round-Robin processing to reduce atomic contention
        # Round-Robin allows different query blocks to process different key columns simultaneously
        max_blocks = (hi + BLOCK_N - 1) // BLOCK_N
        
        for block_offset in range(max_blocks):
            # Calculate Round-Robin key block index
            # Each query block starts from a different key block to reduce contention
            block_idx = (start_m % max_blocks + block_offset) % max_blocks
            start_n = block_idx * BLOCK_N
            
            # Check if block is valid
            if start_n < hi:
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + offs_n_base
                
                # Recompute pointer positions (non-sequential access)
                k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)
                v_ptrs = V + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)

                # -- Load k, v --
                if DIVISIBLE_N:
                    k = tl.load(k_ptrs, cache_modifier=".cg")
                    v = tl.load(v_ptrs, cache_modifier=".cg")
                else:
                    mask_n = offs_n < N
                    k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
                    v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

                # -- Compute QK^T once (only once per key block) --
                s = tl.dot(q, k)
                
                # -- Apply mask and scaling --
                # 1. Apply boundary mask
                if not DIVISIBLE_N:
                    mask_n = offs_n < N
                    s = tl.where(mask_n[None, :], s, float("-inf"))

                # -- Accumulate col sum (sum(scores)) --
                # Compute scaled scores for col sum
                s_for_sum = s * sm_scale
                # Apply masks (set masked positions to 0 for sum)
                if not DIVISIBLE_N:
                    mask_n = offs_n < N
                    s_for_sum = tl.where(mask_n[None, :], s_for_sum, 0.0)
                # Sum over queries dimension (dim=0), get (BLOCK_N,) - local sum for each key position
                col_sum_block = tl.sum(s_for_sum, 0)  # (BLOCK_N,)

                # -- Numerically stable softmax --
                m_i_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
                p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)
                p_sum = tl.sum(p, 1)
                
                # Use atomic operations to accumulate to global col_sum
                # Round-Robin reduces contention by distributing writes across different key positions
                col_sum_ptrs = ColSum + offs_n
                if DIVISIBLE_N:
                    tl.atomic_add(col_sum_ptrs, col_sum_block, sem="relaxed")
                else:
                    mask_n = offs_n < N
                    tl.atomic_add(col_sum_ptrs, col_sum_block, mask=mask_n, sem="relaxed")

                # -- dropout --
                if IS_DROPOUT:
                    offs_rng = start_n + offs_rng_base
                    pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
                    p *= pmask.to(tl.float32)

                # -- Update accumulators --
                acc *= alpha[:, None]
                acc += tl.dot(p.to(input_dtype), v)
                l_i = l_i * alpha + p_sum
                m_i = m_i_new

    # write back l & o
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * sm_scale + tl.log(l_i)

    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


def attention_with_col_sum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output using Flash Attention, while also returning column-wise attention score sum
    
    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        causal: Whether to use causal mask
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability

    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        col_sum: Column-wise attention score sum, shape (batch_size, num_heads_k, seq_len_k)
                 Each value is the sum of attention scores for that key position across all queries
    
    Note:
        The column sum computation uses atomic operations which may introduce numerical errors.
        Relative error of the sum can be around 1e-3. Use with caution if high precision is required.
    """
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
    assert Dk in {16, 32, 64, 128}

    B, H, M, D = q.shape
    N = k.shape[2]
    Hk, Hv = k.shape[1], v.shape[1]
    assert Hk == Hv, "num of heads in k and v should be equal"
    assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
    num_groups = H // Hk

    P_SEQ = N - M
    larger_m = M > N

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    # contiguity
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

    # to work around https://github.com/openai/triton/issues/2441
    device = torch.cuda.device_of(q)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    with torch.cuda.device(device):
        # Dropout preparation
        is_dropout = dropout_p > 0
        if is_dropout:
            offset_increment = B * H * M * N
            seed, offset = philox_cuda_seed_offset(offset_increment)
        else:
            seed, offset = 0, 0

        # Only support non-split_kv case (simplified implementation)
        config = get_fwd_config(B, H, M, N, D, causal)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config

        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0
        
        grid = (triton.cdiv(M, BLOCK_M), H, B)
        o = torch.empty_like(q)
        L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
        
        # Create col sum output tensor
        col_sum = torch.zeros((B, Hk, N), device=q.device, dtype=torch.float32)
        
        # Dynamically adjust configuration to handle shared memory insufficiency
        num_stages_adjusted = num_stages
        BLOCK_N_adjusted = BLOCK_N
        divisible_n_adjusted = divisible_n
        
        for attempt in range(5):  # Try up to 5 times
            try:
                _fwd_kernel_with_col_sum[grid](
                    q, k, v, sm_scale,
                    dropout_p, seed, offset,
                    L, o, col_sum,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    col_sum.stride(0), col_sum.stride(1), col_sum.stride(2),
                    B, H, M, N, P_SEQ, num_groups,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_adjusted, BLOCK_DMODEL=D,
                    IS_CAUSAL=causal, IS_DROPOUT=is_dropout, LARGER_M=larger_m,
                    DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n_adjusted,
                    num_warps=num_warps, num_stages=num_stages_adjusted,
                )
                break
            except Exception as e:
                if attempt < 4:  # Still have retry opportunities
                    error_msg = str(e).lower()
                    if "shared memory" in error_msg or "outofresources" in error_msg:
                        # First try reducing num_stages
                        if num_stages_adjusted > 1:
                            num_stages_adjusted = max(1, num_stages_adjusted - 1)
                        # If num_stages is already 1, reduce BLOCK_N
                        elif BLOCK_N_adjusted > 32:
                            BLOCK_N_adjusted = max(32, BLOCK_N_adjusted // 2)
                            divisible_n_adjusted = N % BLOCK_N_adjusted == 0
                        else:
                            # If already at minimum configuration, raise exception
                            raise RuntimeError(
                                f"Cannot resolve shared memory insufficiency by adjusting configuration. "
                                f"Current config: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N_adjusted}, "
                                f"num_stages={num_stages_adjusted}, num_warps={num_warps}"
                            ) from e
                    else:
                        raise
                else:
                    # Last attempt failed, raise exception
                    raise

    return o, col_sum