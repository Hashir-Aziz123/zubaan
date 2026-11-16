"""Speculative decoding acceptance rate and expected speedup.

The acceptance rate at one position is the overlap between the target (verifier)
and draft next-token distributions:

    alpha = sum over vocabulary of min(p_target(x), p_draft(x))

Averaging it over the positions of real text gives the per-token acceptance rate,
which sets the achievable speedup. Model-level helpers that produce these
distributions live in later modules; this file stays dependency-free.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass

# Per-position results for one text: acceptance rates and whether the top tokens agree.
PairFn = Callable[[list[int]], tuple[list[float], list[bool]]]


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


def argmax(p: Sequence[float]) -> int:
    """Index of the most likely token (the first one on ties)."""
    if not p:
        raise ValueError("empty distribution")
    return max(range(len(p)), key=p.__getitem__)


def python_pair_fn(
    target_probs: Callable[[list[int]], list[Sequence[float]]],
    draft_probs: Callable[[list[int]], list[Sequence[float]]],
) -> PairFn:
    """Per-position acceptance from two functions that return next-token distributions.

    Each function maps token ids to one distribution per predicted position. This is the
    reference implementation; model code uses a vectorized version tested against it.
    """

    def pair(ids: list[int]) -> tuple[list[float], list[bool]]:
        pt, pd = target_probs(ids), draft_probs(ids)
        if len(pt) != len(pd):
            raise ValueError("target and draft returned different numbers of positions")
        alphas = [acceptance_rate(p, q) for p, q in zip(pt, pd, strict=True)]
        greedy = [argmax(p) == argmax(q) for p, q in zip(pt, pd, strict=True)]
        return alphas, greedy

    return pair


def measure_acceptance(
    token_ids: Iterable[list[int]], pair_fn: PairFn
) -> tuple[list[list[float]], list[list[bool]]]:
    """Acceptance rate and greedy agreement at every position of every text.

    Texts with fewer than two tokens have no predicted position and are skipped.
    """
    alphas: list[list[float]] = []
    greedy: list[list[bool]] = []
    for ids in token_ids:
        if len(ids) < 2:
            continue
        a, g = pair_fn(ids)
        alphas.append(a)
        greedy.append(g)
    return alphas, greedy


def pooled_mean(groups: Sequence[Sequence[float]]) -> float:
    """Mean over all positions of all texts: sum of every value / number of values."""
    n = sum(len(g) for g in groups)
    if n == 0:
        raise ValueError("no positions to average")
    return sum(sum(g) for g in groups) / n


def bootstrap_ci(
    groups: Sequence[Sequence[float]], n_boot: int = 1000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Confidence interval of the pooled mean by resampling whole texts with replacement.

    Positions inside one sentence are not independent, so texts (not positions) are
    resampled. Returns the (1 - level) / 2 and (1 + level) / 2 quantiles of the means.
    """
    if not groups:
        raise ValueError("no texts to resample")
    rng = random.Random(seed)
    sums = [sum(g) for g in groups]
    counts = [len(g) for g in groups]
    k = len(groups)
    means = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        n = sum(counts[i] for i in idx)
        if n:
            means.append(sum(sums[i] for i in idx) / n)
    means.sort()
    lo = means[int((1 - level) / 2 * (len(means) - 1))]
    hi = means[int((1 + level) / 2 * (len(means) - 1))]
    return lo, hi


@dataclass(frozen=True)
class AcceptanceSummary:
    """Acceptance for one language. alpha is the mean of sum(min(p_target, p_draft))."""

    n_texts: int
    n_positions: int
    alpha: float
    alpha_ci_low: float
    alpha_ci_high: float
    alpha_median: float
    greedy_match: float
    greedy_ci_low: float
    greedy_ci_high: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_acceptance(
    alphas: Sequence[Sequence[float]], greedy: Sequence[Sequence[bool]], seed: int = 0
) -> AcceptanceSummary:
    """Pooled means, medians and 95% bootstrap intervals for one language."""
    g_float = [[float(x) for x in g] for g in greedy]
    a_lo, a_hi = bootstrap_ci(alphas, seed=seed)
    g_lo, g_hi = bootstrap_ci(g_float, seed=seed)
    return AcceptanceSummary(
        n_texts=len(alphas),
        n_positions=sum(len(a) for a in alphas),
        alpha=pooled_mean(alphas),
        alpha_ci_low=a_lo,
        alpha_ci_high=a_hi,
        alpha_median=statistics.median(x for a in alphas for x in a),
        greedy_match=pooled_mean(g_float),
        greedy_ci_low=g_lo,
        greedy_ci_high=g_hi,
    )


MAX_GAMMA = 6


def best_speedup(table: Sequence[dict], max_gamma: int = MAX_GAMMA) -> dict:
    """The row of a speedup table with the highest speedup among draft lengths <= max_gamma.

    Drafting more tokens raises expected tokens per pass but also the draft cost, so the best
    draft length differs by language: a lower alpha favors fewer drafted tokens.
    """
    rows = [r for r in table if r["gamma"] <= max_gamma]
    if not rows:
        raise ValueError(f"no speedup rows with gamma <= {max_gamma}")
    return max(rows, key=lambda r: r["speedup"])


def speedup_table(
    alpha: float, draft_cost: float, gammas: Iterable[int] = range(1, MAX_GAMMA + 1)
) -> list:
    """Expected tokens per target pass and modelled speedup for each draft length gamma."""
    return [
        {
            "gamma": g,
            "tokens_per_pass": expected_tokens_per_pass(alpha, g),
            "speedup": speculative_speedup(alpha, g, draft_cost),
        }
        for g in gammas
    ]
