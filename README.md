# Flash Attention with Scores

An efficient attention implementation that returns both attention output and attention scores, built on top of Flash Attention v2.

## Features

- **High Performance**: Uses Flash Attention v2 for efficient attention computation
- **Fused Score Computation**: Computes both attention output and scores in a single kernel call, avoiding duplicate computation of Q @ K^T
- **Score Extraction**: Returns attention scores (Q @ K^T * sm_scale) with proper causal masking support
- **Row-wise and Column-wise Score Sums**: Efficiently compute attention score sums along rows and columns
- **Split QK Sum**: Compute cross-segment attention scores (scores from queries after split to keys before split)
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
- **Col Sum (Reverse-Order)**: Flash Attention with column-wise score sum using reverse-order processing for causal attention and Round-Robin for non-causal (recommended for causal attention)
- **Col Sum (Sequential)**: Flash Attention with column-wise score sum using Round-Robin processing for all cases
- **Split QK Sum**: Flash Attention with split QK sum computation (scores from queries after split to keys before split), optimized with Round-Robin processing

By default, the benchmark includes all operators. Use `--no-sum-ops` to disable sum operators benchmarking.

### Benchmark Options

```bash
# Basic benchmark
python benchmark.py

# Disable sum operators benchmarking
python benchmark.py --no-sum-ops

# Enable configuration search for optimal kernel parameters
python benchmark.py --enable-config-search

# Customize configuration search parameters
python benchmark.py --enable-config-search --config-search-trials 5 --config-search-warmup 10

# Customize benchmark timing parameters
python benchmark.py --warmup-ms 100 --rep-ms 500 --num-runs 5
```

**Command-line Arguments:**
- `--warmup-ms`: Warmup time in milliseconds (default: 50)
- `--rep-ms`: Repetition time in milliseconds (default: 200)
- `--num-runs`: Number of runs to use for statistics (default: 3)
- `--device`: Device name (default: "cuda")
- `--no-sum-ops`: Disable row_sum and col_sum benchmarks (enabled by default)
- `--enable-config-search`: Enable configuration search for optimal BLOCK_M, BLOCK_N, num_stages, num_warps
- `--config-search-trials`: Number of trials per configuration during search (default: 3)
- `--config-search-warmup`: Number of warmup iterations per configuration during search (default: 5)

### Configuration Search

The benchmark supports automatic configuration search to find optimal kernel parameters for your specific hardware and input dimensions. When enabled, the benchmark will:

1. **Generate Configuration Space**: Automatically generate a search space of possible configurations (BLOCK_M, BLOCK_N, num_stages, num_warps)
2. **Test Each Configuration**: Measure performance for each configuration
3. **Select Optimal Configuration**: Choose the configuration with the best performance
4. **Use Optimal Configuration**: Apply the optimal configuration for subsequent benchmark tests

> **Note**: All implementations in this benchmark return both attention output and attention scores, making it easy to analyze attention patterns, visualize attention weights, or use scores for downstream tasks.

### Experimental Results

Benchmark results on Qwen3 8B configuration (causal attention, head_dim=128):

**Example Output Format (Transposed Table):**
```
========================================================================================================================================================================================================
Performance Summary (All operators, speedup relative to Naive)
========================================================================================================================================================================================================
Method                                         1K (ms)       speed up     2K (ms)       speed up     4K (ms)       speed up     8K (ms)       speed up     
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Naive (PyTorch)                               0.674        -            3.352        -            11.371       -            41.987       -            
PyTorch SDPA                                  0.560        1.20x        2.041        1.64x        7.992        1.42x        30.859       1.36x        
Flash Attention + Scores                      0.194        3.47x        1.725        1.94x        2.427        4.69x        8.136        5.16x        
Flash Attention + Row Sum                     0.129        5.24x        0.335        10.01x       1.154        9.85x        4.252        9.87x        
Flash Attention + Col Sum (Reverse)           0.156        4.33x        0.493        6.81x        1.699        6.69x        6.273        6.69x        
Flash Attention + Col Sum (Sequential)        0.159        4.23x        0.497        6.74x        1.720        6.61x        6.303        6.66x        
Flash Attention + Split QK Sum                0.159        4.22x        0.424        7.91x        1.436        7.92x        5.102        8.23x        
========================================================================================================================================================================================================
```

## Architecture

- `attention_with_scores.py`: Main module with fused `attention_with_scores` function (computes both output and scores in a single kernel call)
- `attention_with_row_sum.py`: Row-wise score sum computation
- `attention_with_col_sum.py`: Column-wise score sum computation with reverse-order processing for causal attention and Round-Robin for non-causal (recommended for causal attention)
- `attention_with_col_sum_sequential.py`: Column-wise score sum computation with Round-Robin processing for all cases
- `attention_split_qk_sum.py`: Split QK sum computation (scores from queries after split to keys before split)
- `flash.py`: Core Flash Attention v2 implementation
- `total.py`, `split_kv.py`, `dropout.py`: Supporting modules

## API Reference

### `attention_with_scores`

Main function that computes attention output and scores.

The implementation uses a fused kernel that computes both the attention output and scores in a single pass, avoiding duplicate computation of Q @ K^T.

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
    Compute attention output and scores using fused Flash Attention kernel.
    Computes both in a single kernel call, avoiding duplicate computation of Q @ K^T.
    
    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        causal: Whether to use causal mask. If True, masked positions in scores will be set to 0
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability (only used for attention computation, doesn't affect scores)
    
    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        scores: Attention scores, shape (batch_size, num_heads_q, seq_len_q, seq_len_k)
                If causal=True, masked positions have value 0 (instead of -inf)
    """
```

### `attention_with_row_sum`

Computes attention output and row-wise score sums (sum of scores for each query position).

The result has shape `(batch_size, num_heads_q, seq_len_q)`, where each element represents the sum of attention scores for a specific query position across all key positions.

**Implementation Details:**
- The row sum is computed during the Flash Attention forward pass, fused with the attention computation
- Masked positions (causal mask or boundary mask) are set to 0 before summation
- Uses Kahan summation for improved numerical stability

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
    """
```

### `attention_with_col_sum`

Computes attention output and column-wise score sums (sum of scores for each key position).

The result has shape `(batch_size, num_heads_k, seq_len_k)`, where each element represents the sum of attention scores for a specific key position across all query positions.

**Implementation Details:**
- The column sum is computed during the Flash Attention forward pass, fused with the attention computation
- Multiple query blocks write to the same column positions, requiring atomic operations (`tl.atomic_add`)
- Masked positions (causal mask or boundary mask) are set to 0 before summation
- **Causal attention**: Uses reverse-order key block processing
  - Different query blocks process different key columns simultaneously
  - Query block 0 processes keys in reverse: N-1, N-2, N-3, ...
  - Query block 1 processes keys in reverse: N-2, N-3, N-4, ...
  - This reduces write conflicts when multiple query blocks update the same column sum locations
- **Non-causal attention**: Uses Round-Robin processing
  - Different query blocks start processing from different key blocks in a cyclic manner
  - Query block 0 starts from key block 0, query block 1 from key block 1, etc., wrapping around
  - This distributes concurrent writes from different query blocks to different memory locations

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
    """
```

### `attention_with_col_sum_sequential`

Computes attention output and column-wise score sums using sequential (forward-order) processing for all cases.

The result has shape `(batch_size, num_heads_k, seq_len_k)`, where each element represents the sum of attention scores for a specific key position across all query positions.

**Implementation Details:**
- The column sum is computed during the Flash Attention forward pass, fused with the attention computation
- Multiple query blocks write to the same column positions, requiring atomic operations (`tl.atomic_add`)
- Masked positions (causal mask or boundary mask) are set to 0 before summation
- Uses Round-Robin processing for all cases (both causal and non-causal)
  - Different query blocks start processing from different key blocks in a cyclic manner
  - Query block 0 starts from key block 0, query block 1 from key block 1, etc., wrapping around
  - This distributes concurrent writes from different query blocks to different memory locations
  - Simpler implementation than `attention_with_col_sum` (no special handling for causal vs non-causal)

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
    """
```

**Comparison with `attention_with_col_sum`:**

- **`attention_with_col_sum`**: Uses reverse-order processing in causal mode and Round-Robin in non-causal mode. Recommended for causal attention where reverse-order provides optimal performance.
- **`attention_with_col_sum_sequential`**: Uses Round-Robin processing for all cases (both causal and non-causal). Simpler implementation with consistent processing strategy across all modes.

### `attention_split_qk_sum`

Computes attention output and split QK sum, which accumulates scores from queries after the split position to keys before the split position.

**Split QK Sum Output:**

The split QK sum computes the cross attention scores from tokens after the split position to tokens before the split position.

Given the attention scores \( S \in \mathbb{R}^{B \times H_q \times M \times N} \) and a split position \( \text{split} \), the split QK sum is defined as:

\[
\text{split\_qk\_sum}[b, h_k, j] = \sum_{i=\text{split}}^{M-1} S[b, h_q, i, j] \quad \text{for } j \in [0, \text{split})
\]

where:
- \( b \in [0, B) \): batch index
- \( h_k \in [0, H_k) \): key head index
- \( h_q \in [0, H_q) \): query head index (for GQA, multiple query heads may map to the same key head)
- \( i \in [\text{split}, M) \): query position index (only queries after split)
- \( j \in [0, \text{split}) \): key position index (only keys before split)
- \( M, N \): sequence lengths for queries and keys respectively

The output tensor has shape \( (B, H_k, \text{split}) \), where each element represents the **accumulated sum** of attention scores from all query positions after the split to a specific key position before the split. This provides a quantitative measure of the total attention weight that all later tokens (after split) collectively assign to each earlier token (before split), which is useful for analyzing cross-segment attention patterns and information flow between sequence segments.

**Use Cases:**
- Cross-segment attention analysis: Understanding how later tokens attend to earlier tokens
- Attention pattern analysis: Analyzing attention patterns across sequence segments
- Model interpretability: Understanding information flow between sequence segments

**Implementation Details:**
- The split QK sum is computed during the Flash Attention forward pass, fused with the attention computation
- Uses atomic operations (`tl.atomic_add`) to accumulate scores from multiple query blocks
- Only queries after the split position contribute to the sum
- Only keys before the split position are included in the sum
- Masked positions (causal mask or boundary mask) are set to 0 before summation
- Uses a single sequential loop (similar to `attention_with_scores`)
- Uses Round-Robin scheduling for key blocks before the split position
  - Different query blocks start processing from different key blocks in a cyclic manner
  - Query block 0 starts from key block 0, query block 1 from key block 1, etc., wrapping around
  - This distributes concurrent writes from different query blocks to different memory locations
  - Key blocks after the split position are processed sequentially (for attention computation)
- Atomic operations are only executed when necessary
  - Only executed when query block contains queries after split
  - Uses precise masks to reduce actual atomic writes to relevant key positions only
  - Skips computation for key blocks entirely outside the split range

```python
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
    Compute attention output and split QK sum using fused Flash Attention kernel.
    Computes both in a single kernel call, avoiding duplicate computation of Q @ K^T.
    
    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        split: Split position. split_qk_sum computes scores from queries [split:] to keys [:split]
        causal: Whether to use causal mask
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability
        out: Optional output buffer, must have same shape as q
        split_qk_sum_out: Optional split_qk_sum output buffer, must have shape (B, Hk, split)
        reuse_buffers: Whether to reuse buffers (currently unused)
    
    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        split_qk_sum: Split QK sum, shape (batch_size, num_heads_k, split)
                      **Meaning**: Each value `split_qk_sum[b, h_k, j]` represents the **accumulated sum** 
                      of attention scores from **ALL queries after the split position** (queries at 
                      positions [split, split+1, ..., seq_len_q-1]) to the specific key position `j` 
                      (where j < split).
                      
                      This sum indicates the total attention weight that all later tokens (after split) 
                      collectively assign to each earlier token (before split) at position j.
    
    Example:
        >>> q = torch.randn(1, 32, 1024, 128, dtype=torch.float16, device="cuda")
        >>> k = torch.randn(1, 32, 1024, 128, dtype=torch.float16, device="cuda")
        >>> v = torch.randn(1, 32, 1024, 128, dtype=torch.float16, device="cuda")
        >>> output, split_qk_sum = attention_split_qk_sum(q, k, v, split=768, causal=True)
        >>> print(f"Output shape: {output.shape}")  # (1, 32, 1024, 128)
        >>> print(f"Split QK Sum shape: {split_qk_sum.shape}")  # (1, 32, 768)
        >>> # split_qk_sum[0, 0, 100] contains the SUM of scores from ALL queries [768:1024] to key 100
        >>> # This represents how much total attention all tokens after position 768 pay to token 100
    """
```

**Performance Notes:**
- Uses a single sequential loop structure (similar to `attention_with_scores`)
- Uses Round-Robin processing for key blocks before split
- Reduces atomic operation frequency through precise masking
- Performance is comparable to or better than `attention_with_scores`

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

## Acknowledgments

This project is built on top of [FlagAttention](https://github.com/flagos-ai/FlagAttention), a collection of memory-efficient attention operators implemented in the Triton language.

## License

This project is based on FlagAttention and follows the same license.
