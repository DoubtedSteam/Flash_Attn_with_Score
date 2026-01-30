# Flash Attention with Scores

An efficient attention implementation that returns both attention output and attention scores, built on top of Flash Attention v2.

## Features

- **High Performance**: Uses Flash Attention v2 for efficient attention computation
- **Score Extraction**: Independently computes attention scores (Q @ K^T * sm_scale)
- **Multiple Implementations**: Includes Flash Attention, PyTorch SDPA, and naive reference implementations
- **Comprehensive Benchmarking**: Built-in benchmark suite for performance comparison
- **GQA Support**: Supports Grouped Query Attention (GQA)

## Installation

```bash
pip install torch triton
```

## Quick Start

```python
import torch
from flash_attention_with_scores import attention_with_scores

# Create input tensors
batch_size, num_heads, seq_len, head_dim = 1, 32, 1024, 128
q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

# Compute attention with scores
# Returns BOTH output and scores in a single call
output, scores = attention_with_scores(q, k, v, causal=True)

print(f"Output shape: {output.shape}")  # (1, 32, 1024, 128)
print(f"Scores shape: {scores.shape}")  # (1, 32, 1024, 1024)

# You can now use scores for visualization, analysis, or other purposes
# without needing to recompute them separately
```

## API Reference

### `attention_with_scores`

Main function that computes attention output and scores.

```python
def attention_with_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output using Flash Attention, while also returning attention scores.
    
    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        causal: Whether to use causal mask
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability
    
    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        scores: Attention scores, shape (batch_size, num_heads_q, seq_len_q, seq_len_k)
    """
```

### `compute_attention_scores`

Efficiently computes attention scores only (Q @ K^T * sm_scale).

```python
def compute_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Efficiently compute attention scores: Q @ K^T * sm_scale.
    
    Note: Current version doesn't apply causal mask, only performs pure matrix multiplication.
    """
```

## Replacing Standard Attention

This implementation can be used as a drop-in replacement for standard attention implementations. Here's how to integrate it:

### Basic Replacement

**Before (standard attention):**
```python
import torch.nn.functional as F

# Standard attention - only returns output
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**After (with scores):**
```python
from flash_attention_with_scores import attention_with_scores

# Flash attention with scores - returns both output and scores
output, scores = attention_with_scores(q, k, v, causal=True)
```

### Integration in Transformer Models

**Example: Replace attention in a Transformer block**

```python
import torch
import torch.nn as nn
from flash_attention_with_scores import attention_with_scores

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        # ... other layers ...
    
    def forward(self, x):
        # Reshape for multi-head attention
        B, T, C = x.shape
        q = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Replace standard attention with attention_with_scores
        # Old: attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output, attn_scores = attention_with_scores(q, k, v, causal=True)
        
        # attn_scores can be used for visualization, analysis, or other purposes
        # Shape: (B, n_heads, T, T)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # ... rest of the block ...
        return attn_output, attn_scores  # Optionally return scores
```

### Key Differences

1. **Return Value**: `attention_with_scores` returns `(output, scores)` tuple instead of just `output`
2. **Parameter Name**: Uses `causal` instead of `is_causal` (for consistency with Flash Attention API)
3. **Score Access**: You now have direct access to attention scores without recomputing

### Performance Considerations

- **Flash Attention**: Optimized for memory efficiency and speed
- **Score Computation**: Additional computation for scores, but still efficient
- **Memory**: Scores are stored in memory (shape: `(batch, heads, seq_q, seq_k)`)

## Benchmarking

Run the benchmark suite to compare different implementations:

```bash
python benchmark.py
```

The benchmark compares:
- **Naive**: PyTorch native implementation (reference) - **returns both output and scores**
- **PyTorch SDPA**: PyTorch's `scaled_dot_product_attention` (auto-optimized) - **returns both output and scores**
- **Flash Attention**: Triton-optimized Flash Attention - **returns both output and scores**

> **Note**: All implementations in this benchmark return both attention output and attention scores, making it easy to analyze attention patterns, visualize attention weights, or use scores for downstream tasks.

## Architecture

- `flash_attention_with_scores.py`: Main module with `attention_with_scores` function
- `attention_scores.py`: Efficient score computation using Triton
- `flash.py`: Core Flash Attention v2 implementation
- `total.py`, `split_kv.py`, `dropout.py`: Supporting modules

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Triton >= 2.0
- CUDA-capable GPU

## Roadmap / TODO

Future enhancements planned for this project:

- [ ] **Row-wise and Column-wise Score Sums**: Implement versions that return attention score sums along rows and columns
  - Row sum: Sum of attention scores for each query position (shape: `(batch, heads, seq_q)`)
  - Column sum: Sum of attention scores for each key position (shape: `(batch, heads, seq_k)`)
  - Useful for analyzing attention distribution and identifying important tokens

## License

This project is based on FlagAttention and follows the same license.
