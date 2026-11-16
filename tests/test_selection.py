import random
import sys

import pytest

from zubaan.extend.bytelevel import bytes_to_token
from zubaan.extend.selection import (
    ARABIC_LEAD_BYTES,
    contains_arabic_block,
    is_urdu_safe,
    pair_is_safe,
    select_merges,
    validate_merges,
)
from zubaan.extend.sweep import learn_urdu_merges
from zubaan.extend.train_bpe import Merge


def tok(text: str) -> str:
    """Byte-level display form of a string, as it appears inside a vocabulary."""
    return bytes_to_token(text.encode("utf-8"))


def test_lead_bytes_are_exactly_d8_to_db():
    assert sorted(ARABIC_LEAD_BYTES) == [0xD8, 0xD9, 0xDA, 0xDB]


def test_the_safety_argument_holds_for_every_code_point():
    """A character's UTF-8 bytes contain D8 to DB if and only if it is in U+0600 to U+06FF.

    This is the fact the English-unchanged guarantee rests on, checked for every character.
    """
    for cp in range(sys.maxunicode + 1):
        if 0xD800 <= cp <= 0xDFFF:
            continue  # surrogates are not valid text
        encoded = chr(cp).encode("utf-8")
        has_lead = any(b in ARABIC_LEAD_BYTES for b in encoded)
        assert has_lead == (0x0600 <= cp <= 0x06FF), hex(cp)


@pytest.mark.parametrize("text", ["ا", "ہے", "اردو", "aا", "1ب", "۱۲"])
def test_tokens_with_arabic_block_letters_are_safe(text):
    assert is_urdu_safe(tok(text))


@pytest.mark.parametrize(
    "text",
    [
        "a",
        "the",
        " the",
        "123",
        ".,",
        "é",  # Latin-1 accent
        "д",  # Cyrillic
        "ש",  # Hebrew
        "है",  # Devanagari
        "日本",
        "😀",
        "‌",  # zero width non-joiner is not in the Arabic block
        "ﭐ",  # Arabic presentation forms are outside U+0600 to U+06FF
        "ݐ",  # Arabic Supplement
        "",
    ],
)
def test_other_tokens_are_not_safe(text):
    assert not is_urdu_safe(tok(text))


def test_a_token_of_only_continuation_bytes_is_not_safe():
    # 0x99 0x8A are continuation bytes of the Arabic yeh (D9 8A), but other scripts use them too.
    assert not is_urdu_safe(bytes_to_token(bytes([0x99, 0x8A])))
    assert is_urdu_safe(bytes_to_token(bytes([0xD9, 0x8A])))


def test_a_partial_character_with_the_lead_byte_is_safe():
    # Just the lead byte D8 can only start an Arabic-block character.
    assert is_urdu_safe(bytes_to_token(bytes([0xD8])))
    assert not is_urdu_safe(bytes_to_token(bytes([0xD7])))  # Hebrew lead byte
    assert not is_urdu_safe(bytes_to_token(bytes([0xDC])))  # Syriac lead byte


def test_non_byte_level_strings_are_not_safe():
    assert not is_urdu_safe("a b")  # a literal space is not in the byte table
    assert not is_urdu_safe("<|im_start|>")  # printable ASCII only


def test_pair_is_safe_uses_the_merged_token():
    # Neither half has a lead byte, but together they could still be a continuation pair only.
    assert pair_is_safe(bytes_to_token(b"\xd8"), bytes_to_token(b"\xa7"))
    assert not pair_is_safe(bytes_to_token(b"\x99"), bytes_to_token(b"\x8a"))
    assert not pair_is_safe("a", "b")


def test_contains_arabic_block():
    assert contains_arabic_block("hello ا")
    assert not contains_arabic_block("hello world, é д ש 日本 😀")
    assert not contains_arabic_block("")


# validate_merges / select_merges ----------------------------------------------------------------

A, B, C = tok("ا"), tok("ب"), tok("ت")
BASE = {A, B, C, "a", "b"}


def test_validate_accepts_a_chain_of_merges():
    validate_merges([Merge(A, B, 5), Merge(A + B, C, 3)], BASE)
    validate_merges([], BASE)


@pytest.mark.parametrize(
    ("merge", "message"),
    [
        (Merge("zz", A, 1), "left part"),
        (Merge(A, "zz", 1), "right part"),
        (Merge("a", "b", 1), "could occur in non-Urdu text"),
        (Merge(A, B, 1), "already exists"),
    ],
)
def test_validate_rejects_each_problem(merge, message):
    base = BASE | ({A + B} if message == "already exists" else set())
    with pytest.raises(ValueError, match=message):
        validate_merges([merge], base)


def test_validate_rejects_dependency_on_a_later_merge():
    with pytest.raises(ValueError, match="does not exist yet"):
        validate_merges([Merge(A + B, C, 1), Merge(A, B, 1)], BASE)


def test_validate_rejects_duplicate_token_in_list():
    with pytest.raises(ValueError, match="already exists"):
        validate_merges([Merge(A, B, 1), Merge(A, B, 1)], BASE)


def test_validate_can_skip_the_safety_rule():
    validate_merges([Merge("a", "b", 1)], BASE, require_safe=False)


def test_select_drops_unsafe_missing_and_existing_but_keeps_order():
    cands = [
        Merge("a", "b", 100),  # unsafe
        Merge(A, B, 90),  # ok
        Merge(A + C, B, 80),  # missing part (A+C was never created)
        Merge(A + B, C, 70),  # ok, built on the first
        Merge(A, B, 60),  # duplicate token
        Merge(B, C, 50),  # ok
    ]
    got = select_merges(cands, BASE, 10)
    assert [m.count for m in got] == [90, 70, 50]
    validate_merges(got, BASE)


def test_select_respects_n_and_is_a_prefix_of_larger_n():
    cands = [Merge(A, B, 9), Merge(A + B, C, 8), Merge(B, C, 7)]
    assert select_merges(cands, BASE, 0) == []
    assert select_merges(cands, BASE, 2) == cands[:2]
    for n in range(4):
        assert select_merges(cands, BASE, n) == select_merges(cands, BASE, 3)[:n]
    with pytest.raises(ValueError, match="non-negative"):
        select_merges(cands, BASE, -1)


def test_select_is_idempotent():
    cands = [Merge("a", "b", 5), Merge(A, B, 4), Merge(A + B, C, 3)]
    once = select_merges(cands, BASE, 10)
    assert select_merges(once, BASE, 10) == once


def test_learned_merges_are_all_valid_and_safe(tiny_base_json, urdu_lines):
    merges = learn_urdu_merges(urdu_lines, tiny_base_json, 300, min_count=2)
    assert len(merges) > 50
    vocab = tiny_base_json["model"]["vocab"].keys()
    validate_merges(merges, vocab)
    assert all(is_urdu_safe(m.token) for m in merges)
    # Selecting from the learner's own output changes nothing: it is already a fixed point.
    assert select_merges(merges, vocab, 10_000) == merges


def test_learner_never_proposes_english_merges_even_when_given_english_text(
    tiny_base_json, urdu_lines, english_lines
):
    # English text mixed into the corpus must not produce any merge: every pair is ASCII.
    merges = learn_urdu_merges(english_lines * 5 + urdu_lines, tiny_base_json, 300, min_count=2)
    assert merges and all(is_urdu_safe(m.token) for m in merges)
    english_only = learn_urdu_merges(english_lines * 5, tiny_base_json, 300, min_count=1)
    assert english_only == []


def test_random_tokens_safe_iff_they_contain_a_lead_byte():
    rng = random.Random(0)
    for _ in range(2000):
        data = bytes(rng.randrange(256) for _ in range(rng.randint(1, 6)))
        expected = any(0xD8 <= b <= 0xDB for b in data)
        assert is_urdu_safe(bytes_to_token(data)) == expected
