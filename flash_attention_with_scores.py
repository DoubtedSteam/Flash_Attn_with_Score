"""
Flash Attention with Scores Module
Efficient attention computation using Flash Attention, with independent score calculation
Combines high-performance Flash Attention computation with independent score calculation
"""

import torch
try:
    from .flash import attention as flash_attention
    from .attention_scores import compute_attention_scores
except ImportError:
    # Handle case when running as standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from flash import attention as flash_attention
    from attention_scores import compute_attention_scores


def attention_with_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output using Flash Attention, while also returning attention scores
    
    This function combines:
    1. Flash Attention: Efficient attention computation (optimized with Triton)
    2. Independent score computation: Separately computes Q @ K^T * sm_scale
    
    Aligned with naive methods, returns (attention_output, attention_scores)
    
    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        causal: Whether to use causal mask (note: current score computation doesn't support causal, this parameter is ignored for scores)
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability (only used for attention computation, doesn't affect scores)
    
    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        scores: Attention scores, shape (batch_size, num_heads_q, seq_len_q, seq_len_k)
                Note: Current version doesn't apply causal mask to scores, only performs pure matrix multiplication
    """
    # Compute output using Flash Attention
    output = flash_attention(q, k, v, causal=causal, sm_scale=sm_scale, dropout_p=dropout_p)
    
    # Independently compute attention scores
    # Note: Current compute_attention_scores doesn't support causal mask, so pass False
    # If causal mask for scores is needed, it should be implemented in a future version
    scores = compute_attention_scores(q, k, causal=False, sm_scale=sm_scale)
    
    return output, scores

