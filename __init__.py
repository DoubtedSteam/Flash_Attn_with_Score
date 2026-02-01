"""
Flash Attention with Scores
An efficient attention implementation that returns both attention output and scores
"""

try:
    from .flash_attention_with_scores import attention_with_scores
    from .attention_scores import compute_attention_scores
    from .flash import attention
    from .attention_with_col_sum import attention_with_col_sum
    from .attention_with_col_sum_sequential import attention_with_col_sum_sequential
    from .attention_with_row_sum import attention_with_row_sum
except ImportError:
    # Handle case when running as standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from flash_attention_with_scores import attention_with_scores
    from attention_scores import compute_attention_scores
    from flash import attention
    from attention_with_col_sum import attention_with_col_sum
    from attention_with_col_sum_sequential import attention_with_col_sum_sequential
    from attention_with_row_sum import attention_with_row_sum

__all__ = [
    "attention",
    "compute_attention_scores",
    "attention_with_scores",
    "attention_with_col_sum",
    "attention_with_col_sum_sequential",
    "attention_with_row_sum",
]

