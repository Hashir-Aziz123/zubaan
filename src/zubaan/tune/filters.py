"""Automatic checks on generated replies (Stage 4, chunks 0 and 4).

Each check answers one yes or no question about one reply, so every dropped conversation can
be counted under a reason. Pure Python, no dependencies.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from zubaan.text.normalize import split_words

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class SplitReply:
    """A reply cut into its reasoning block and its answer.

    ``think`` is None when the reply has no ``<think>`` block (thinking off: Qwen3's template
    already wrote the empty block into the prompt). ``closed`` is False when the block was
    opened and never closed; the answer is then empty.
    """

    think: str | None
    answer: str
    closed: bool


def split_reply(text: str) -> SplitReply:
    """Split a raw reply into reasoning and answer."""
    stripped = text.lstrip()
    if not stripped.startswith(THINK_OPEN):
        return SplitReply(think=None, answer=text.strip(), closed=True)
    body = stripped[len(THINK_OPEN) :]
    if THINK_CLOSE not in body:
        return SplitReply(think=body.strip(), answer="", closed=False)
    think, answer = body.split(THINK_CLOSE, 1)
    return SplitReply(think=think.strip(), answer=answer.strip(), closed=True)


def max_ngram_repeats(text: str, n: int = 8) -> int:
    """How often the most frequent run of ``n`` consecutive words occurs in the text.

    Natural prose almost never repeats an 8-word run three times; a model stuck in a loop
    repeats it many times. Text shorter than ``n`` words returns 0.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    words = split_words(text)
    if len(words) < n:
        return 0
    counts = Counter(
        hashlib.blake2b("\x1f".join(words[i : i + n]).encode("utf-8"), digest_size=8).digest()
        for i in range(len(words) - n + 1)
    )
    return max(counts.values())


def has_loop(text: str, n: int = 8, max_repeats: int = 2) -> bool:
    """True if some run of ``n`` words occurs more than ``max_repeats`` times."""
    return max_ngram_repeats(text, n) > max_repeats


# A reply to a standalone question must not reveal that the teacher was shown a passage.
# Heuristic word lists, used to count and to flag replies for reading, not as proof.
_SOURCE_EN = re.compile(
    r"\b(the |this |given |provided )(passage|text|article|background|reference)\b"
    r"|\bbackground (information|text)\b|\baccording to the (passage|text|article|background)\b",
    re.I,
)
_SOURCE_UR = re.compile(
    r"اقتباس|دیے گئے متن|دیا گیا متن|متن کے مطابق|اس متن|مضمون کے مطابق|اس مضمون|پس منظر کے مطابق"
)


def mentions_source(text: str) -> bool:
    """True if the text refers to a passage, text, article or background it was given."""
    return bool(_SOURCE_EN.search(text) or _SOURCE_UR.search(text))
