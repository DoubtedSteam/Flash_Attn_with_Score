"""
Efficient Attention Score Computation Module
Implemented using Triton, computes only Q @ K^T * sm_scale, without softmax
"""

import math
import torch
import triton
import triton.language as tl


def maybe_contiguous(x):
    """Ensure the last dimension is contiguous for LDGSTS instruction usage"""
    return x.contiguous() if x.stride(-1) != 1 else x


def get_fwd_config(B, H, M, N, D, causal):
    """
    Select optimal configuration based on GPU architecture and parameters
    Reference implementation from flash.py
    """
    if torch.cuda.get_device_capability() == (8, 0):  # A100
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
        else:  # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
    elif torch.cuda.get_device_capability() == (8, 6):  # RTX-3090
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else:  # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _attention_scores_kernel(
    Q, K, sm_scale,
    Scores,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_sz, stride_sh, stride_sm, stride_sn,
    Z, H, M, N,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    """
    Compute Attention Scores: Q @ K^T * sm_scale
    
    Args:
        Q: Query tensor, shape (B, H, M, D)
        K: Key tensor, shape (B, Hk, N, D), where Hk = H / num_groups
        sm_scale: Scaling factor
        Scores: Output tensor, shape (B, H, M, N)
        Other parameters: stride information and dimension information
    """
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    Scores += off_z * stride_sz + off_h * stride_sh

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to Q
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)  # (BLOCK_M, BLOCK_DMODEL)

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

    # loop over k and compute scores
    # Initialize pointers (initialized outside loop, updated inside loop)
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)  # (BLOCK_DMODEL, BLOCK_N)
    scores_ptrs = Scores + (offs_m[:, None] * stride_sm + offs_n_init[None, :] * stride_sn)  # (BLOCK_M, BLOCK_N)

    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")

        # -- compute qk: Q @ K^T --
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)

        # Apply scaling
        s = s * sm_scale

        # -- write back scores --
        if DIVISIBLE_M and DIVISIBLE_N:
            tl.store(scores_ptrs, s.to(input_dtype), cache_modifier=".cg")
        elif DIVISIBLE_M:
            mask_n = offs_n < N
            tl.store(scores_ptrs, s.to(input_dtype), mask=mask_n[None, :], cache_modifier=".cg")
        elif DIVISIBLE_N:
            mask_m = offs_m < M
            tl.store(scores_ptrs, s.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")
        else:
            mask_m = offs_m < M
            mask_n = offs_n < N
            tl.store(scores_ptrs, s.to(input_dtype), mask=mask_m[:, None] & mask_n[None, :], cache_modifier=".cg")

        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        scores_ptrs += BLOCK_N * stride_sn


def compute_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Efficiently compute Attention Scores: Q @ K^T * sm_scale
    
    This function only computes attention scores, without softmax and multiplication with V.
    Suitable for scenarios where access to attention scores is needed.
    
    Note: Current version doesn't support causal mask, only performs pure matrix multiplication.
    
    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        causal: Whether to use causal mask (currently ignored)
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
    
    Returns:
        scores: Attention scores, shape (batch_size, num_heads_q, seq_len_q, seq_len_k)
    """
    Dq, Dk = q.shape[-1], k.shape[-1]
    assert Dq == Dk, "Q and K must have the same head_dim"
    assert Dk in {16, 32, 64, 128}, f"head_dim must be one of 16, 32, 64, 128, got {Dk}"

    B, H, M, D = q.shape
    N = k.shape[2]
    Hk = k.shape[1]
    assert H % Hk == 0, "num_heads_q must be a multiple of num_heads_k (GQA support)"
    num_groups = H // Hk

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Ensure contiguity
    q, k = maybe_contiguous(q), maybe_contiguous(k)

    # Handle device
    device = torch.cuda.device_of(q)

    with torch.cuda.device(device):
        # Use non-causal config (simpler)
        config = get_fwd_config(B, H, M, N, D, causal=False)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config

        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0

        # Create output tensor
        scores = torch.empty((B, H, M, N), dtype=q.dtype, device=q.device)

        # Set grid
        grid = (triton.cdiv(M, BLOCK_M), H, B)

        # Launch kernel (don't pass causal-related parameters)
        _attention_scores_kernel[grid](
            q, k, sm_scale,
            scores,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            B, H, M, N, num_groups,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            num_stages=num_stages, num_warps=num_warps,
        )

    return scores

