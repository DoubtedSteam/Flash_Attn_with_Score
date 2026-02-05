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
def _fwd_kernel_split_qk_sum(
    Q, K, V, sm_scale,
    dropout_p,
    seed,
    offset,
    L, O, SplitQKSum,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_splitqk_z, stride_splitqk_h, stride_splitqk_n,
    Z, H, M, N, split_pos,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    ):
    """
    跨段注意力 kernel - 优化版本
    
    主要优化：
    1. 使用单个顺序循环，简化控制流
    2. 只在必要时执行原子操作
    3. 优化循环展开，避免无效迭代
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
    SplitQKSum += off_z * stride_splitqk_z + off_hk * stride_splitqk_h
    
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # 判断当前处理的query block是否在split之后
    is_query_after_split = offs_m >= split_pos
    
    # 为dropout准备随机数
    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]
    
    # 加载 query
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
    
    # 初始化 accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # 循环上界：使用简单的顺序循环，类似 attention_with_scores
    if IS_CAUSAL:
        hi = tl.minimum(N, (start_m + 1) * BLOCK_M)
        if hi < 0:
            hi = 0
    else:
        hi = N
    
    # 初始化指针
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    
    # 优化：检查当前 query block 是否有任何 query 在 split 之后
    # 如果整个 block 都在 split 之前，可以跳过 split_qk_sum 计算
    block_start_m = start_m * BLOCK_M
    block_end_m = block_start_m + BLOCK_M
    has_query_after_split = block_end_m > split_pos
    
    # 注意：对于 attention output 的计算，所有 key blocks 必须按顺序处理
    # 因为 softmax 的在线算法（使用 m_i 和 l_i 累积）依赖于处理顺序
    # Round-Robin 只用于 split_qk_sum 的原子操作，但不改变 key blocks 的处理顺序
    # 计算 split 之前的 key blocks 数量
    num_split_blocks = (split_pos + BLOCK_N - 1) // BLOCK_N
    max_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    
    # 按顺序处理所有 key blocks（用于 attention output 计算）
    for block_offset in range(max_blocks):
        # 按顺序计算 key block 索引（确保 attention output 计算正确）
        start_n = block_offset * BLOCK_N
        
        # 检查 block 是否有效
        if start_n < hi:
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + offs_n_base
            
            # 重新计算指针位置（因为不再是顺序访问）
            k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
            v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
            
            # 判断当前 key block 中哪些 key 在 split_pos 之前
            mask_n_before_split = offs_n < split_pos
            
            # 边界 mask
            if not DIVISIBLE_N:
                mask_n = offs_n < N
            else:
                mask_n = None
            
            # 因果 mask
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
            else:
                causal_mask = None
            
            # 加载 key, value
            if DIVISIBLE_N:
                k = tl.load(k_ptrs, cache_modifier=".cg")
                v = tl.load(v_ptrs, cache_modifier=".cg")
            else:
                k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
                v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")
            
            # 计算 QK^T
            s = tl.dot(q, k)
            
            # 应用边界 mask
            if not DIVISIBLE_N:
                s = tl.where(mask_n[None, :], s, float("-inf"))
            
            # 应用 causal mask
            if IS_CAUSAL:
                s = tl.where(causal_mask, s, float("-inf"))
            
            # ---------------------------------------------------------
            # 计算 SplitQKSum：只保留 split 之后的 query 对 split 之前的 key 的贡献
            # 优化：只在有 split 之后的 query 时才计算
            # 方案2+3优化：
            # 1. Round-Robin 处理 split 之前的 key blocks，分散原子操作竞争
            # 2. 优化 mask 计算，减少重复计算
            # 3. 使用精确的 mask 减少实际原子操作次数
            # ---------------------------------------------------------
            if has_query_after_split:
                # 快速检查：当前 key block 是否在 split 之前
                # 如果整个 key block 都在 split 之后（start_n >= split_pos），跳过所有计算
                key_block_in_split_range = start_n < split_pos
                
                # 只在 key block 与 split 之前的部分重叠时才计算
                if key_block_in_split_range:
                    # 计算 scaled scores for split sum（复用已计算的 s，避免重复计算）
                    s_for_sum = s * sm_scale
                    
                    # 优化：一次性构建完整的 split mask，减少重复的 mask 计算
                    # split_mask = (query >= split) & (key < split) & (边界 mask) & (causal mask)
                    split_mask = is_query_after_split[:, None] & mask_n_before_split[None, :]
                    
                    # 应用边界 mask（如果 key 超出 N，设为 0）
                    if not DIVISIBLE_N:
                        split_mask = split_mask & mask_n[None, :]
                    
                    # 应用 causal mask（如果 query < key，设为 0）
                    if IS_CAUSAL:
                        split_mask = split_mask & causal_mask
                    
                    # 应用 split mask：只保留有效的贡献，无效位置设为 0
                    s_for_sum = tl.where(split_mask, s_for_sum, 0.0)
                    
                    # 在 block 内部先对 query 维度求和，得到每个 key 位置的贡献
                    # 这样每个 key 位置只需要一次原子操作，而不是每个 query-key 对一次
                    # 这是减少原子操作频率的关键优化
                    split_qk_sum_block = tl.sum(s_for_sum, 0)  # (BLOCK_N,)
                    
                    # 原子加法（只对 split 之前的部分）
                    # 使用精确的 mask 确保只写入有效的 key 位置，减少不必要的原子操作
                    # mask 确保：
                    # 1. 只对 split 之前的 key 执行原子操作（mask_n_before_split）
                    # 2. 只对有效的 key 位置执行（mask_n，如果不可整除）
                    split_qk_sum_ptrs = SplitQKSum + offs_n
                    if DIVISIBLE_N:
                        # 使用 mask_n_before_split 确保只对 split 之前的 key 执行原子操作
                        # 如果 key >= split_pos，mask 为 False，原子操作实际上不会写入
                        tl.atomic_add(split_qk_sum_ptrs, split_qk_sum_block, mask=mask_n_before_split, sem="relaxed")
                    else:
                        # 组合 mask：边界 mask & split mask
                        # 只对既在 N 范围内又在 split 之前的 key 执行原子操作
                        mask_n_atomic = mask_n & mask_n_before_split
                        tl.atomic_add(split_qk_sum_ptrs, split_qk_sum_block, mask=mask_n_atomic, sem="relaxed")
            
            # ---------------------------------------------------------
            # 计算 softmax 和 attention 输出（所有query都参与）
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
            
            # 更新 accumulators
            acc *= alpha[:, None]
            acc += tl.dot(p.to(Q.dtype.element_ty), v)
            l_i = l_i * alpha + p_sum
            m_i = m_i_new
    
    # -- 写入 attention 输出和 logsumexp --
    acc = acc * (1.0 / l_i[:, None])
    l = m_i * sm_scale + tl.log(l_i)
    
    # 写入 O 和 L
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m
    
    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(Q.dtype.element_ty), cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=mask_m[:, None], cache_modifier=".cg")
        

def attention_split_qk_sum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    split: int,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
    *,
    out: torch.Tensor | None = None,
    split_qk_sum_out: torch.Tensor | None = None,
    reuse_buffers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    跨段注意力与分数和融合计算 - 优化版本
    
    计算 attention 输出和 split_qk_sum，其中 split_qk_sum[j] 表示所有 split 之后的 query 
    对 split 之前的 key j 的分数和。
    
    主要优化：
    1. 使用单个顺序循环，简化控制流
    2. 只在必要时执行原子操作
    3. 优化循环展开，避免无效迭代
    
    Args:
        q: Query 张量，形状 (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key 张量，形状 (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value 张量，形状 (batch_size, num_heads_k, seq_len_k, head_dim)
        split: 分割位置，split_qk_sum 计算 [split:] 部分的 query 对 [:split] 部分的 key 的分数和
        causal: 是否使用因果 mask
        sm_scale: 缩放因子，如果为 None 则使用 1/sqrt(head_dim)
        dropout_p: Dropout 概率
        out: 可选的输出缓冲区，形状必须与 q 相同
        split_qk_sum_out: 可选的 split_qk_sum 输出缓冲区，形状必须为 (B, Hk, split)
        reuse_buffers: 是否重用缓冲区（当前未使用）
    
    Returns:
        output: Attention 输出，形状 (batch_size, num_heads_q, seq_len_q, head_dim)
        split_qk_sum: Split QK Sum，形状 (batch_size, num_heads_k, split)
                     表示每个 key 位置 j (j < split) 的分数和，累加自所有 query 位置 i (i >= split)
    """
    # 参数检查
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv, "q, k, v的特征维度必须相等"
    assert Dk in {16, 32, 64, 128}, f"不支持的head_dim: {Dk}"
    
    B, H, M, D = q.shape
    N = k.shape[2]
    Hk, Hv = k.shape[1], v.shape[1]
    
    assert Hk == Hv, "k和v的头数必须相等"
    assert H % Hk == 0, "q的头数必须是k头数的整数倍"
    assert 0 < split <= N, f"分割位置必须在有效范围内 (0, {N}]"
    
    num_groups = H // Hk
    
    # 计算缩放因子
    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)
    
    # 确保内存连续性
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
    
    # 设备信息
    device = torch.cuda.device_of(q)
    
    with torch.cuda.device(device):
        # Dropout准备
        is_dropout = dropout_p > 0
        if is_dropout:
            offset_increment = B * H * M * N
            seed, offset = philox_cuda_seed_offset(offset_increment)
        else:
            seed, offset = 0, 0
        
        # 配置参数
        config = get_fwd_config(B, H, M, N, D, causal)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config
        
        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0
        
        # 计算grid大小
        grid = (triton.cdiv(M, BLOCK_M), H, B)

        # 分配输出缓冲区
        if out is not None:
            o = out
            assert o.shape == q.shape, f"out.shape 必须等于 q.shape，当前 out={o.shape}, q={q.shape}"
            assert o.device == q.device, "out.device 必须与 q.device 相同"
            assert o.dtype == q.dtype, "out.dtype 必须与 q.dtype 相同"
        else:
            o = torch.empty_like(q)

        if split_qk_sum_out is not None:
            split_qk_sum = split_qk_sum_out
            assert split_qk_sum.shape == (B, Hk, split), (
                f"split_qk_sum_out.shape 必须为 {(B, Hk, split)}，当前为 {split_qk_sum.shape}"
            )
            assert split_qk_sum.device == q.device, "split_qk_sum_out.device 必须与 q.device 相同"
            assert split_qk_sum.dtype == torch.float32, "split_qk_sum_out.dtype 必须为 torch.float32"
        else:
            split_qk_sum = torch.empty((B, Hk, split), device=q.device, dtype=torch.float32)

        # L缓冲区
        L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
        
        # 清零split_qk_sum
        split_qk_sum.zero_()
        
        # 动态调整配置
        num_stages_adjusted = num_stages
        BLOCK_N_adjusted = BLOCK_N
        divisible_n_adjusted = divisible_n
        
        for attempt in range(5):
            try:
                _fwd_kernel_split_qk_sum[grid](
                    q, k, v, sm_scale,
                    dropout_p, seed, offset,
                    L, o, split_qk_sum,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    split_qk_sum.stride(0), split_qk_sum.stride(1), split_qk_sum.stride(2),
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
                                f"无法通过调整配置解决共享内存不足问题。"
                                f"当前配置: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N_adjusted}, "
                                f"num_stages={num_stages_adjusted}, num_warps={num_warps}"
                            ) from e
                    else:
                        raise
                else:
                    raise
    
    return o, split_qk_sum

