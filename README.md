# Flash Attention with Scores

An efficient attention implementation that returns both attention output and attention scores, built on top of Flash Attention v2.

## Features

- **High Performance**: Uses Flash Attention v2 for efficient attention computation
- **Score Extraction**: Independently computes attention scores (Q @ K^T * sm_scale)
- **Row-wise and Column-wise Score Sums**: Efficiently compute attention score sums along rows and columns
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

### `attention_with_row_sum`

Computes attention output and row-wise score sums (sum of scores for each query position).

```python
def attention_with_row_sum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output using Flash Attention, while also returning row-wise attention score sums.
    
    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        causal: Whether to use causal mask
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability
    
    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        row_sum: Row-wise attention score sum, shape (batch_size, num_heads_q, seq_len_q)
                 Each value is the sum of attention scores for that query position across all keys
    
    ⚠️ Precision Warning:
        The row sum computation may introduce numerical errors due to floating-point accumulation.
        Relative error of the sum can be around 1e-3.
    """
```

### `attention_with_col_sum`

Computes attention output and column-wise score sums (sum of scores for each key position).

```python
def attention_with_col_sum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output using Flash Attention, while also returning column-wise attention score sums.
    
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
    
    Implementation Note:
        In causal attention mode, the kernel uses a reverse-order processing trick to reduce
        atomic operation contention. Instead of processing key blocks sequentially from start
        to end, different query blocks process different key columns simultaneously by starting
        from different positions. This reduces write conflicts when multiple query blocks update
        the same column sum locations.
    
    ⚠️ Precision Warning:
        The col sum computation may introduce numerical errors due to floating-point accumulation.
        Relative error of the sum can be around 1e-3.
    """
```

### `attention_with_col_sum_sequential`

Computes attention output and column-wise score sums using sequential (forward-order) processing for all cases.

```python
def attention_with_col_sum_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output using Flash Attention, while also returning column-wise attention score sums.
    Uses sequential (forward-order) processing for all cases, including causal attention.
    
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
        This version uses sequential processing for all cases. For causal attention, consider using
        `attention_with_col_sum` which uses reverse-order processing to reduce atomic contention.
    
    ⚠️ Precision Warning:
        The col sum computation may introduce numerical errors due to floating-point accumulation.
        Relative error of the sum can be around 1e-3.
    """
```

**Comparison with `attention_with_col_sum`:**

- **`attention_with_col_sum`**: Uses reverse-order processing in causal mode to reduce atomic contention. Recommended for causal attention.
- **`attention_with_col_sum_sequential`**: Uses sequential processing for all cases. Simpler implementation, but may have more atomic contention in causal mode.

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
- **Row Sum**: Flash Attention with row-wise score sum computation
- **Col Sum (Reverse-Order)**: Flash Attention with column-wise score sum using reverse-order optimization (recommended for causal attention)
- **Col Sum (Sequential)**: Flash Attention with column-wise score sum using sequential processing

By default, the benchmark includes all operators. Use `--no-sum-ops` to disable sum operators benchmarking.

### Understanding Benchmark Results

The benchmark output shows all operators with their execution times and speedup relative to Naive:
- **Speedup (vs Naive)**: Performance speedup multiplier relative to Naive implementation
  - Formula: `naive_time / operator_time`
  - Example: If Naive takes 100ms and Flash takes 10ms, speedup is `100/10 = 10.0x`
  - Higher speedup is better
  - All speedups are calculated relative to Naive as the baseline
- **Operators compared**:
  - **Naive**: PyTorch native implementation (baseline, 1.0x)
  - **SDPA**: PyTorch SDPA speedup vs Naive
  - **Flash**: Flash Attention speedup vs Naive
  - **RowSum**: Row sum operator speedup vs Naive
  - **ColSum-R**: Column sum (reverse-order) speedup vs Naive
  - **ColSum-S**: Column sum (sequential) speedup vs Naive

> **Note**: All implementations in this benchmark return both attention output and attention scores, making it easy to analyze attention patterns, visualize attention weights, or use scores for downstream tasks.

## Architecture

- `flash_attention_with_scores.py`: Main module with `attention_with_scores` function
- `attention_scores.py`: Efficient score computation using Triton
- `attention_with_row_sum.py`: Row-wise score sum computation
- `attention_with_col_sum.py`: Column-wise score sum computation with reverse-order optimization (recommended for causal attention)
- `attention_with_col_sum_sequential.py`: Column-wise score sum computation with sequential processing (simpler, for all cases)
- `flash.py`: Core Flash Attention v2 implementation
- `total.py`, `split_kv.py`, `dropout.py`: Supporting modules

### Implementation Details: Reverse-Order Processing in Column Sum

The `attention_with_col_sum` implementation includes an important optimization for causal attention: **reverse-order key block processing**.

**Problem**: In causal attention, multiple query blocks need to write to the same column sum locations using atomic operations, causing contention and performance degradation.

**Solution**: Instead of processing key blocks sequentially (0, 1, 2, ...), different query blocks start from different key positions:
- Query block 0 processes keys in reverse: N-1, N-2, N-3, ...
- Query block 1 processes keys in reverse: N-2, N-3, N-4, ...
- Query block 2 processes keys in reverse: N-3, N-4, N-5, ...

This reduces write conflicts because different query blocks are updating different column locations simultaneously, while still maintaining correctness through atomic operations.

**Benefits**:
- Reduced atomic operation contention
- Better GPU utilization
- Improved performance in causal attention scenarios

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Triton >= 2.0
- CUDA-capable GPU

## Roadmap / TODO

### Completed Features

- [x] **Row-wise and Column-wise Score Sums**: Implemented versions that return attention score sums along rows and columns
  - Row sum: Sum of attention scores for each query position (shape: `(batch, heads, seq_q)`)
  - Column sum: Sum of attention scores for each key position (shape: `(batch, heads, seq_k)`)
  - Useful for analyzing attention distribution and identifying important tokens
  - **Optimization**: Column sum uses reverse-order processing in causal mode to reduce atomic contention

### Future Enhancements

- [ ] **Experimental Results**: Add benchmark results and performance comparisons
  - Performance comparison between standard attention and score sum variants
  - Memory usage analysis
  - Speedup measurements for different sequence lengths and batch sizes
  - Comparison of reverse-order vs sequential processing in column sum

## Citation

If you use this project in your research, please cite it as follows:

**BibTeX:**
```bibtex
@software{flash_attention_with_scores,
  title = {Flash Attention with Scores},
  author = {DoubtedSteam},
  url = {https://github.com/DoubtedSteam/Flash_Attn_with_Score},
  year = {2026}
}
```

## License

This project is based on FlagAttention and follows the same license.
