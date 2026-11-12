"""Stage 1: measure the language tax (tokenizer fertility and speculative decoding acceptance)."""

from zubaan.audit.acceptance import acceptance_rate, expected_tokens_per_pass, speculative_speedup
from zubaan.audit.fertility import FertilityResult, compute_fertility, cost_multiplier

__all__ = [
    "FertilityResult",
    "acceptance_rate",
    "compute_fertility",
    "cost_multiplier",
    "expected_tokens_per_pass",
    "speculative_speedup",
]
