"""
Flash Attention with Scores
An efficient attention implementation that returns both attention output and scores
"""

from .flash_attention_with_scores import attention_with_scores
from .attention_scores import compute_attention_scores
from .flash import attention

__all__ = ["attention", "compute_attention_scores", "attention_with_scores"]

