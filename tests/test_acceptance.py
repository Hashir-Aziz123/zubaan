import pytest

from zubaan.audit.acceptance import (
    acceptance_rate,
    argmax,
    bootstrap_ci,
    expected_tokens_per_pass,
    measure_acceptance,
    pooled_mean,
    python_pair_fn,
    speculative_speedup,
    speedup_table,
    summarize_acceptance,
)


def test_acceptance_worked_example():
    # min(0.6, 0.4) + min(0.3, 0.2) + min(0.1, 0.4) = 0.7
    assert acceptance_rate([0.6, 0.3, 0.1], [0.4, 0.2, 0.4]) == pytest.approx(0.7)


def test_identical_distributions_always_accept():
    assert acceptance_rate([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0)


def test_mismatched_vocab_raises():
    with pytest.raises(ValueError):
        acceptance_rate([1.0], [0.5, 0.5])


def test_expected_tokens_worked_example():
    # (1 - 0.7^5) / (1 - 0.7) = 2.77
    assert expected_tokens_per_pass(0.7, 4) == pytest.approx(2.7731, abs=1e-3)


def test_perfect_acceptance():
    assert expected_tokens_per_pass(1.0, 4) == 5.0


def test_speedup_worked_example():
    # 2.77 / (1 + 4 x 0.05) = 2.31
    assert speculative_speedup(0.7, 4, 0.05) == pytest.approx(2.311, abs=1e-3)


def test_argmax():
    assert argmax([0.1, 0.7, 0.2]) == 1


def test_python_pair_fn_worked_example():
    # Position 1 is the docs example: alpha 0.7, and both top choices are token 0
    # (the draft ties 0 and 2; argmax takes the first).
    # Position 2: min(0.2, 0.7) + min(0.8, 0.3) = 0.5, and the top choices differ.
    target = [[0.6, 0.3, 0.1], [0.2, 0.8, 0.0]]
    draft = [[0.4, 0.2, 0.4], [0.7, 0.3, 0.0]]
    pair = python_pair_fn(lambda ids: target, lambda ids: draft)
    alphas, greedy = pair([1, 2, 3])
    assert alphas == pytest.approx([0.7, 0.5])
    assert greedy == [True, False]


def test_measure_acceptance_self_test_is_perfect():
    # A model compared with itself must always be accepted.
    def probs(ids):
        return [[0.5, 0.25, 0.25] for _ in ids[1:]]

    alphas, greedy = measure_acceptance([[1, 2, 3], [4, 5]], python_pair_fn(probs, probs))
    assert pooled_mean(alphas) == pytest.approx(1.0)
    assert all(all(g) for g in greedy)


def test_measure_acceptance_skips_one_token_texts():
    alphas, _ = measure_acceptance([[1]], python_pair_fn(lambda i: [], lambda i: []))
    assert alphas == []


def test_pooled_mean_weights_every_position():
    # (0.9 + 0.7 + 0.5 + 0.8) / 4 = 0.725, regardless of how positions group into texts.
    assert pooled_mean([[0.9, 0.7], [0.5, 0.8]]) == pytest.approx(0.725)
    assert pooled_mean([[0.9], [0.7, 0.5, 0.8]]) == pytest.approx(0.725)


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    groups = [[0.9, 0.8], [0.5], [0.7, 0.6, 0.4], [0.95]]
    lo, hi = bootstrap_ci(groups, seed=1)
    assert lo <= pooled_mean(groups) <= hi
    assert bootstrap_ci(groups, seed=1) == (lo, hi)


def test_summarize_acceptance():
    s = summarize_acceptance([[0.9, 0.7], [0.5, 0.8]], [[True, False], [True, True]])
    assert s.n_texts == 2
    assert s.n_positions == 4
    assert s.alpha == pytest.approx(0.725)
    assert s.greedy_match == pytest.approx(0.75)


def test_speedup_table_matches_formula():
    table = speedup_table(0.7, 0.05, gammas=[4])
    assert table[0]["tokens_per_pass"] == pytest.approx(expected_tokens_per_pass(0.7, 4))
    assert table[0]["speedup"] == pytest.approx(2.311, abs=1e-3)


def test_slow_drafter_can_make_decoding_slower():
    # Illustrative Urdu numbers from the plan: alpha 0.55, gamma 4, c 0.35 -> 0.88x.
    assert speculative_speedup(0.55, 4, 0.35) == pytest.approx(0.879, abs=1e-3)
