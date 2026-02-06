"""
Flash Attention with Scores
An efficient attention implementation that returns both attention output and scores
"""

try:
    from .attention_with_scores import attention_with_scores
    from .flash import attention
    from .attention_with_col_sum import attention_with_col_sum
    from .attention_with_col_sum_sequential import attention_with_col_sum_sequential
    from .attention_with_col_softmax_sum import attention_with_softmax_col_sum
    from .attention_cross_token_softmax_sum import attention_cross_token_softmax_sum, attention_cross_token_softmax_sum_buffered
    from .attention_with_row_sum import attention_with_row_sum
    from .attention_cross_token_qk_sum import attention_cross_token_qk_sum
except ImportError:
    # Handle case when running as standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from attention_with_scores import attention_with_scores
    from flash import attention
    from attention_with_col_sum import attention_with_col_sum
    from attention_with_col_sum_sequential import attention_with_col_sum_sequential
    from attention_with_col_softmax_sum import attention_with_softmax_col_sum
    from attention_cross_token_softmax_sum import attention_cross_token_softmax_sum, attention_cross_token_softmax_sum_buffered
    from attention_with_row_sum import attention_with_row_sum
    from attention_cross_token_qk_sum import attention_cross_token_qk_sum

__all__ = [
    "attention",
    "attention_with_scores",
    "attention_with_col_sum",
    "attention_with_col_sum_sequential",
    "attention_with_softmax_col_sum",
    "attention_cross_token_softmax_sum",
    "attention_cross_token_softmax_sum_buffered",
    "attention_with_row_sum",
    "attention_cross_token_qk_sum",
]

