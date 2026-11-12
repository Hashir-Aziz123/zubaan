import pytest

from zubaan.audit.acceptance import (
    acceptance_rate,
    expected_tokens_per_pass,
    speculative_speedup,
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
