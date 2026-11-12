"""Speculative decoding acceptance rate and expected speedup.

The acceptance rate at one position is the overlap between the target (verifier)
and draft next-token distributions:

    alpha = sum over vocabulary of min(p_target(x), p_draft(x))

Averaging it over the positions of real text gives the per-token acceptance rate,
which sets the achievable speedup. Model-level helpers that produce these
distributions live in later modules; this file stays dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence


def acceptance_rate(p_target: Sequence[float], p_draft: Sequence[float]) -> float:
    """Probability that a token drafted from p_draft is accepted by p_target."""
    if len(p_target) != len(p_draft):
        raise ValueError("distributions must cover the same vocabulary")
    return float(sum(min(p, q) for p, q in zip(p_target, p_draft, strict=True)))


def expected_tokens_per_pass(alpha: float, gamma: int) -> float:
    """Expected tokens produced per target forward pass with gamma drafted tokens.

    (1 - alpha^(gamma + 1)) / (1 - alpha), assuming independent acceptances.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speculative_speedup(alpha: float, gamma: int, draft_cost: float) -> float:
    """Approximate decode speedup: expected tokens per pass / (1 + gamma x draft_cost).

    draft_cost is one draft step's time relative to one target step.
    """
    return expected_tokens_per_pass(alpha, gamma) / (1.0 + gamma * draft_cost)
