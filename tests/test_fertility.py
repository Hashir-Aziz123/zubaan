import pytest

from zubaan.audit.fertility import (
    FertilityResult,
    check_tokenizer,
    compute_fertility,
    cost_multiplier,
    fertility_report,
)


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


def test_check_tokenizer_lossless(space_tokenizer):
    c = check_tokenizer(space_tokenizer, ["ایک دو", "abc"])
    # "ایک دو" is 5 Urdu letters (2 bytes each) plus a space; "abc" is 3 bytes.
    assert c.n_bytes == 5 * 2 + 1 + 3
    assert c.n_tokens == 6 + 3
    assert c.lossless


class _UnkTokenizer:
    """Maps every non-ASCII character to one unknown token, like a tokenizer without bytes."""

    unk_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return [ord(c) if ord(c) < 128 else 1 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) if i != 1 else "?" for i in ids)


def test_check_tokenizer_flags_unknown_tokens():
    c = check_tokenizer(_UnkTokenizer(), ["ایک"])
    assert c.n_unk == 3
    assert c.n_roundtrip_failures == 1
    assert not c.lossless


def test_fertility_report_is_plain_data(space_tokenizer):
    r = fertility_report(space_tokenizer, "fake", ["ایک دو"], ["one two"], n_examples=1)
    assert r["cost_multiplier"] == pytest.approx(6 / 7)
    assert r["target_checks"]["lossless"]
    assert r["examples"]["target"][0][:3] == ["ا", "ی", "ک"]


@pytest.mark.slow
def test_qwen3_tokenizer_on_fixtures(fixtures_dir):
    from transformers import AutoTokenizer

    from zubaan.io import read_lines

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    urdu = read_lines(fixtures_dir / "urdu_sample.txt")
    r = fertility_report(tok, "qwen3", urdu, read_lines(fixtures_dir / "english_sample.txt"))
    checks = r["target_checks"]
    assert checks["lossless"]
    # Byte-level BPE: never more tokens than UTF-8 bytes.
    assert checks["n_tokens"] <= checks["n_bytes"]
    assert r["cost_multiplier"] > 1
