import math
import torch
import triton
import triton.language as tl
try:
    from .flash import maybe_contiguous, get_fwd_config
    from .dropout import philox_cuda_seed_offset
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from flash import maybe_contiguous, get_fwd_config
    from dropout import philox_cuda_seed_offset


@triton.jit
def _fwd_kernel_cross_token_qk(
    Q, K, V, sm_scale,
    dropout_p,
    seed,
    offset,
    L, O, CrossSegmentSum,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_cross_segment_z, stride_cross_segment_h, stride_cross_segment_n,
    Z, H, M, N, split_pos,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    ):
    """
    Cross-segment attention kernel - optimized version

    Main optimizations:
    1. Use single sequential loop to simplify control flow
    2. Execute atomic operations only when necessary
    3. Optimize loop unrolling to avoid invalid iterations
    """
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e
    
    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M
    CrossSegmentSum += off_z * stride_cross_segment_z + off_hk * stride_cross_segment_h
    
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # Determine if the current query block is after the split position
    is_query_after_split = offs_m >= split_pos
    
    # Prepare random numbers for dropout
    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]
    
    # Load query
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
    
    # Initialize accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop upper bound: use simple sequential loop similar to attention_with_scores
    if IS_CAUSAL:
        hi = tl.minimum(N, (start_m + 1) * BLOCK_M)
        if hi < 0:
            hi = 0
    else:
        hi = N
    
    # Initialize pointers
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    
    # Optimization: check if the current query block has any query after split
    # If the entire block is before split, can skip cross_token_qk computation
    block_start_m = start_m * BLOCK_M
    block_end_m = block_start_m + BLOCK_M
    has_query_after_split = block_end_m > split_pos
    
    # Note: For attention output computation, all key blocks must be processed in order
    # Because the online softmax algorithm (using m_i and l_i accumulation) depends on processing order
    # Round-Robin is only used for cross_token_qk atomic operations, but does not change key blocks processing order
    # Calculate the number of key blocks before split
    num_split_blocks = (split_pos + BLOCK_N - 1) // BLOCK_N
    max_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    
    # Process all key blocks in sequential order (for attention output computation)
    for block_offset in range(max_blocks):
        # Calculate key block index sequentially (ensures correct attention output computation)
        start_n = block_offset * BLOCK_N

        # Check if block is valid
        if start_n < hi:
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + offs_n_base
            
            # Recompute pointer positions (because it's no longer sequential access)
            k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
            v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

            # Determine which keys in the current key block are before split_pos
            mask_n_before_split = offs_n < split_pos

            # Boundary mask
            if not DIVISIBLE_N:
                mask_n = offs_n < N
            else:
                mask_n = None
            
            # Causal mask
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
            else:
                causal_mask = None
            
            # Load key, value
            if DIVISIBLE_N:
                k = tl.load(k_ptrs, cache_modifier=".cg")
                v = tl.load(v_ptrs, cache_modifier=".cg")
            else:
                k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
                v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")
            
            # Compute QK^T
            s = tl.dot(q, k)

            # Apply boundary mask
            if not DIVISIBLE_N:
                s = tl.where(mask_n[None, :], s, float("-inf"))
            
            # Apply causal mask
            if IS_CAUSAL:
                s = tl.where(causal_mask, s, float("-inf"))
            
            # ---------------------------------------------------------
            # Compute CrossSegmentSum: only keep contributions from queries after split to keys before split
            # Optimization: only compute when there are queries after split
            # Solution 2+3 optimizations:
            # 1. Round-Robin process key blocks before split to distribute atomic operation contention
            # 2. Optimize mask computation to reduce redundant calculations
            # 3. Use precise masks to reduce actual number of atomic operations
            # ---------------------------------------------------------
            if has_query_after_split:
                # Fast check: is the current key block before split
                # If the entire key block is after split (start_n >= split_pos), skip all computation
                key_block_in_split_range = start_n < split_pos

                # Only compute when key block overlaps with the part before split
                if key_block_in_split_range:
                    # Compute scaled scores for split sum (reuse already computed s to avoid redundant calculation)
                    s_for_sum = s * sm_scale

                    # Optimization: build complete split mask at once to reduce redundant mask calculations
                    # split_mask = (query >= split) & (key < split) & (boundary mask) & (causal mask)
                    split_mask = is_query_after_split[:, None] & mask_n_before_split[None, :]

                    # Apply boundary mask (if key exceeds N, set to 0)
                    if not DIVISIBLE_N:
                        split_mask = split_mask & mask_n[None, :]

                    if not DIVISIBLE_M:
                        split_mask = split_mask & mask_m[:,None]

                    # Apply causal mask (if query < key, set to 0)
                    if IS_CAUSAL:
                        split_mask = split_mask & causal_mask

                    # Apply split mask: only keep valid contributions, set invalid positions to 0
                    s_for_sum = tl.where(split_mask, s_for_sum, 0.0)

                    # First sum over query dimension within block to get contribution for each key position
                    # This way each key position needs only one atomic operation instead of one per query-key pair
                    # This is the key optimization to reduce atomic operation frequency
                    cross_token_qk_block = tl.sum(s_for_sum, 0)  # (BLOCK_N,)

                    # Atomic addition (only for the part before split)
                    # Use precise mask to ensure only valid key positions are written, reducing unnecessary atomic operations
                    # Mask ensures:
                    # 1. Only execute atomic operations for keys before split (mask_n_before_split)
                    # 2. Only execute for valid key positions (mask_n, if not divisible)
                    cross_token_qk_ptrs = CrossSegmentSum + offs_n
                    if DIVISIBLE_N:
                        # Use mask_n_before_split to ensure only keys before split execute atomic operations
                        # If key >= split_pos, mask is False and atomic operation won't actually write
                        tl.atomic_add(cross_token_qk_ptrs, cross_token_qk_block, mask=mask_n_before_split, sem="relaxed")
                    else:
                        # Combine masks: boundary mask & split mask
                        # Only execute atomic operations for keys both within N range and before split
                        mask_n_atomic = mask_n & mask_n_before_split
                        tl.atomic_add(cross_token_qk_ptrs, cross_token_qk_block, mask=mask_n_atomic, sem="relaxed")
            
            # ---------------------------------------------------------
            # Compute softmax and attention output (all queries participate)
            # ---------------------------------------------------------
            m_i_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
            p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)
            p_sum = tl.sum(p, 1)
            
            # dropout
            if IS_DROPOUT:
                offs_rng = start_n + offs_rng_base
                pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
                p *= pmask.to(tl.float32)
            
            # Update accumulators
            acc *= alpha[:, None]
            acc += tl.dot(p.to(Q.dtype.element_ty), v)
            l_i = l_i * alpha + p_sum
            m_i = m_i_new
    
    # -- Write attention output and logsumexp --
    acc = acc * (1.0 / l_i[:, None])
    l = m_i * sm_scale + tl.log(l_i)

    # Write O and L
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m
    
    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(Q.dtype.element_ty), cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=mask_m[:, None], cache_modifier=".cg")
        

def attention_cross_token_qk_sum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    split: int,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
    *,
    out: torch.Tensor | None = None,
    cross_token_qk_out: torch.Tensor | None = None,
    reuse_buffers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cross-segment attention with fused score sum computation - optimized version

    Computes attention output and cross_token_qk, where cross_token_qk[j] represents the score sum
    from all queries after split to key j before split.

    Main optimizations:
    1. Use single sequential loop to simplify control flow
    2. Execute atomic operations only when necessary
    3. Optimize loop unrolling to avoid invalid iterations

    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        split: Split position, cross_token_qk computes score sum from queries in [split:] to keys in [:split]
        causal: Whether to use causal mask
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability
        out: Optional output buffer, shape must match q
        cross_token_qk_out: Optional cross_token_qk output buffer, shape must be (B, Hk, split)
        reuse_buffers: Whether to reuse buffers (currently unused)

    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        cross_token_qk: Split QK Sum, shape (batch_size, num_heads_k, split)
                        Represents the score sum for each key position j (j < split), accumulated from all query positions i (i >= split)
    """
    # Parameter checks
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv, "Feature dimensions of q, k, v must be equal"
    assert Dk in {16, 32, 64, 128}, f"Unsupported head_dim: {Dk}"
    
    B, H, M, D = q.shape
    N = k.shape[2]
    Hk, Hv = k.shape[1], v.shape[1]
    
    assert Hk == Hv, "Number of heads in k and v must be equal"
    assert H % Hk == 0, "Number of heads in q must be a multiple of that in k"
    assert 0 < split <= N, f"Split position must be within valid range (0, {N}]"
    
    num_groups = H // Hk
    
    # Compute scaling factor
    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    # Ensure memory contiguity
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

    # Device information
    device = torch.cuda.device_of(q)

    with torch.cuda.device(device):
        # Dropout preparation
        is_dropout = dropout_p > 0
        if is_dropout:
            offset_increment = B * H * M * N
            seed, offset = philox_cuda_seed_offset(offset_increment)
        else:
            seed, offset = 0, 0
        
        # Configuration parameters
        config = get_fwd_config(B, H, M, N, D, causal)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config

        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0

        # Compute grid size
        grid = (triton.cdiv(M, BLOCK_M), H, B)

        # Allocate output buffers
        if out is not None:
            o = out
            assert o.shape == q.shape, f"out.shape must equal q.shape, current out={o.shape}, q={q.shape}"
            assert o.device == q.device, "out.device must match q.device"
            assert o.dtype == q.dtype, "out.dtype must match q.dtype"
        else:
            o = torch.empty_like(q)

        if cross_token_qk_out is not None:
            cross_token_qk = cross_token_qk_out
            assert cross_token_qk.shape == (B, Hk, split), (
                f"cross_token_qk_out.shape must be {(B, Hk, split)}, current is {cross_token_qk.shape}"
            )
            assert cross_token_qk.device == q.device, "cross_token_qk_out.device must match q.device"
            assert cross_token_qk.dtype == torch.float32, "cross_token_qk_out.dtype must be torch.float32"
        else:
            cross_token_qk = torch.empty((B, Hk, split), device=q.device, dtype=torch.float32)

        # L buffer
        L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

        # Zero out cross_token_qk
        cross_token_qk.zero_()

        # Dynamically adjust configuration
        num_stages_adjusted = num_stages
        BLOCK_N_adjusted = BLOCK_N
        divisible_n_adjusted = divisible_n
        
        for attempt in range(5):
            try:
                _fwd_kernel_cross_token_qk[grid](
                    q, k, v, sm_scale,
                    dropout_p, seed, offset,
                    L, o, cross_token_qk,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    cross_token_qk.stride(0), cross_token_qk.stride(1), cross_token_qk.stride(2),
                    B, H, M, N, split, num_groups,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_adjusted, BLOCK_DMODEL=D,
                    IS_CAUSAL=causal, IS_DROPOUT=is_dropout,
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
    
    return o, cross_token_qk

