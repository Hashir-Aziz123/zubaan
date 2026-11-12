"""Metrics that stay fair when tokenizers differ.

Tokens per second and per-token perplexity both change meaning when the vocabulary
changes, because each token carries a different amount of text. Compare models with
words per second and bits per character instead.
"""

from __future__ import annotations

import math


def words_per_second(tokens_per_second: float, tokens_per_word: float) -> float:
    """User-visible output speed: tokens per second divided by tokens per word."""
    if tokens_per_word <= 0:
        raise ValueError("tokens_per_word must be positive")
    return tokens_per_second / tokens_per_word


def bits_per_character(total_loss_nats: float, n_chars: int) -> float:
    """Total cross-entropy loss over a text (in nats) normalized by its character count.

    bits per character = total loss / (characters x ln 2)
    """
    if n_chars <= 0:
        raise ValueError("n_chars must be positive")
    return total_loss_nats / (n_chars * math.log(2))
