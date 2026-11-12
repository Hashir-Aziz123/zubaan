import pytest

from zubaan.audit.fertility import FertilityResult, compute_fertility, cost_multiplier


def test_counts_with_fake_tokenizer(char_tokenizer):
    r = compute_fertility(char_tokenizer, ["ab cd", "efg"])
    assert (r.n_texts, r.n_words, r.n_tokens) == (2, 3, 7)
    assert r.tokens_per_word == pytest.approx(7 / 3)


def test_special_tokens_are_excluded(char_tokenizer):
    r = compute_fertility(char_tokenizer, ["ab"])
    assert r.n_tokens == 2


def test_cost_multiplier():
    urdu = FertilityResult(n_texts=2, n_words=20, n_chars=100, n_tokens=80)
    eng = FertilityResult(n_texts=2, n_words=20, n_chars=100, n_tokens=25)
    assert cost_multiplier(urdu, eng) == pytest.approx(3.2)


def test_cost_multiplier_requires_parallel_corpora():
    a = FertilityResult(1, 1, 1, 1)
    b = FertilityResult(2, 2, 2, 2)
    with pytest.raises(ValueError):
        cost_multiplier(a, b)
