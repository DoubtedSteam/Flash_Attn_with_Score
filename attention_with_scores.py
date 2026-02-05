"""
Attention 模块：使用 Flash Attention 计算输出，同时返回 Attention Scores
融合实现：在一次 kernel 调用中同时计算 attention 输出和 scores，避免重复计算 Q @ K^T
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
def _fwd_kernel_with_scores(
    Q, K, V, sm_scale,
    dropout_p,
    seed,
    offset,
    L, O, Scores,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_sz, stride_sh, stride_sm, stride_sn,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr, LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    """
    融合的 Flash Attention forward kernel，同时计算 attention 输出和 scores
    避免重复计算 Q @ K^T
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
    Scores += off_z * stride_sz + off_h * stride_sh

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

    # loop over k, v and update accumulators
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    scores_ptrs = Scores + (offs_m[:, None] * stride_sm + offs_n_init[None, :] * stride_sn)
    
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)

        # 应用边界 mask（用于 attention 计算）
        if not DIVISIBLE_N:
            mask_n = offs_n < N
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # -- 保存 scores（应用缩放，mask 位置设为 0）--
        # 计算用于保存的 scores：应用缩放，mask 位置设为 0
        s_for_scores = s * sm_scale
        # 将 -inf 位置设为 0（用于 scores 输出）
        if not DIVISIBLE_N:
            mask_n = offs_n < N
            s_for_scores = tl.where(mask_n[None, :], s_for_scores, 0.0)
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s_for_scores = tl.where(causal_mask, s_for_scores, 0.0)
        
        # 保存 scores
        if DIVISIBLE_M and DIVISIBLE_N:
            tl.store(scores_ptrs, s_for_scores.to(input_dtype), cache_modifier=".cg")
        elif DIVISIBLE_M:
            mask_n = offs_n < N
            tl.store(scores_ptrs, s_for_scores.to(input_dtype), mask=mask_n[None, :], cache_modifier=".cg")
        elif DIVISIBLE_N:
            mask_m = offs_m < M
            tl.store(scores_ptrs, s_for_scores.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")
        else:
            mask_m = offs_m < M
            mask_n = offs_n < N
            tl.store(scores_ptrs, s_for_scores.to(input_dtype), mask=mask_m[:, None] & mask_n[None, :], cache_modifier=".cg")

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)

        # -- apply dropout --
        if IS_DROPOUT:
            offs_rng = start_n + offs_rng_base
            pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
            p *= pmask.to(tl.float32)

        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        scores_ptrs += BLOCK_N * stride_sn

    # write back l & o
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * sm_scale + tl.log(l_i)

    # -- scale o due to dropout
    if IS_DROPOUT:
        scale = 1.0 / (1.0 - dropout_p)
        acc *= scale

    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


def attention_with_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用融合的 Flash Attention kernel 计算 attention 输出和 scores
    在一次 kernel 调用中完成，避免重复计算 Q @ K^T
    
    Args:
        q: Query 张量，形状 (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key 张量，形状 (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value 张量，形状 (batch_size, num_heads_k, seq_len_k, head_dim)
        causal: 是否使用因果 mask。如果为 True，scores 中被 mask 的位置会被设为 0
        sm_scale: 缩放因子，如果为 None 则使用 1/sqrt(head_dim)
        dropout_p: Dropout 概率（仅用于 attention 计算，不影响 scores）
    
    Returns:
        output: Attention 输出，形状 (batch_size, num_heads_q, seq_len_q, head_dim)
        scores: Attention scores，形状 (batch_size, num_heads_q, seq_len_q, seq_len_k)
                如果 causal=True，被 mask 的位置值为 0（而不是 -inf）
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
        
        # Create scores output tensor, initialized to 0 (mask positions will remain 0)
        scores = torch.zeros((B, H, M, N), device=q.device, dtype=q.dtype)
        
        # Dynamically adjust configuration to handle shared memory insufficiency
        num_stages_adjusted = num_stages
        BLOCK_N_adjusted = BLOCK_N
        divisible_n_adjusted = divisible_n
        
        for attempt in range(5):  # Try up to 5 times
            try:
                _fwd_kernel_with_scores[grid](
                    q, k, v, sm_scale,
                    dropout_p, seed, offset,
                    L, o, scores,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
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

    return o, scores

