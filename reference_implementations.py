"""
Reference implementations for comparison
Includes naive PyTorch implementation and PyTorch SDPA implementation
"""

import math
import torch
import torch.nn.functional as F


def naive_attention_with_scores(q, k, v, causal, dropout_p=0.0, sm_scale=None):
    """
    Naive attention implementation that returns both output and scores
    Aligned interface with attention_with_scores
    """
    batch_size, num_heads_q, seq_len_q, head_dim = q.shape
    num_heads_k = k.shape[1]
    
    # Handle GQA
    if num_heads_q != num_heads_k:
        assert num_heads_q % num_heads_k == 0
        num_groups = num_heads_q // num_heads_k
        k = torch.repeat_interleave(k, repeats=num_groups, dim=1)
        v = torch.repeat_interleave(v, repeats=num_groups, dim=1)
    
    # Calculate scaling factor
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    
    # Apply causal mask
    if causal:
        seq_len_k = k.shape[-2]
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool), 
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
    
    # Compute output
    output = torch.matmul(attn_weights, v)
    
    return output, scores


def pytorch_sdpa_attention_with_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch SDPA attention implementation that returns both output and scores
    
    Uses PyTorch's optimized scaled_dot_product_attention (auto-dispatches to FlashAttention, 
    Memory-efficient attention, etc.), simulating real LLM attention computation.
    
    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        causal: Whether to use causal mask
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability
    
    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        scores: Attention scores (with causal mask applied but no softmax),
                shape (batch_size, num_heads_q, seq_len_q, seq_len_k)
    """
    batch_size, num_heads_q, seq_len_q, head_dim = q.shape
    num_heads_k = k.shape[1]
    
    # Handle GQA: if num_heads_q > num_heads_k, need to repeat k and v
    if num_heads_q != num_heads_k:
        assert num_heads_q % num_heads_k == 0, f"num_heads_q ({num_heads_q}) must be a multiple of num_heads_k ({num_heads_k})"
        num_groups = num_heads_q // num_heads_k
        k = torch.repeat_interleave(k, repeats=num_groups, dim=1)
        v = torch.repeat_interleave(v, repeats=num_groups, dim=1)
    
    # Calculate scaling factor
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Use PyTorch native scaled_dot_product_attention for output
    # This auto-dispatches to optimal implementation (FlashAttention, Memory-efficient attention, etc.)
    output = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=None,  # Use is_causal parameter instead of attn_mask
        dropout_p=dropout_p,  # PyTorch accepts 0.0 for no dropout
        is_causal=causal,
        scale=sm_scale,
    )
    
    # Independently compute attention scores (for returning and analysis)
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    
    # Apply causal mask (if enabled)
    if causal:
        seq_len_k = k.shape[-2]
        # Create causal mask: upper triangle is True (needs masking), lower triangle is False (keep)
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    return output, scores

