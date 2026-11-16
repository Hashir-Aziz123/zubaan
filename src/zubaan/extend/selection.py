"""Which merges are safe to add, and checks that a merge list is valid.

The English-unchanged guarantee
-------------------------------
New merges are appended after every base merge, so they have the lowest priority. A new merge
(left, right) fires wherever the pair ``left right`` is adjacent after the base merges. That
could include English text, if the pair can occur there. We rule it out with a byte argument:

* Arabic-script letters (U+0600 to U+06FF) are encoded in UTF-8 with a lead byte from
  0xD8 to 0xDB. Those four byte values never occur in valid UTF-8 for anything else.
* If the merged token ``left + right`` contains one of those bytes, it can only appear in
  input that contains an Arabic-block character.

So any text with no character in U+0600 to U+06FF tokenizes exactly as before. That covers
English, and every other script outside the Arabic block. This is a proof, not a heuristic,
and the tests check it on real text anyway.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from functools import lru_cache

from zubaan.extend.bytelevel import token_to_bytes
from zubaan.extend.train_bpe import Merge

# UTF-8 lead bytes of U+0600 to U+06FF: D8 (0600-063F), D9, DA, DB (06C0-06FF).
ARABIC_LEAD_BYTES = frozenset(range(0xD8, 0xDC))


@lru_cache(maxsize=1_000_000)
def is_urdu_safe(token: str) -> bool:
    """True if a byte-level token contains a lead byte of the Arabic block.

    Tokens made only of ASCII, digits, Latin or other scripts are not safe, and neither are
    tokens that are only continuation bytes (which other scripts also use).
    """
    try:
        data = token_to_bytes(token)
    except ValueError:
        return False
    return any(b in ARABIC_LEAD_BYTES for b in data)


def pair_is_safe(left: str, right: str) -> bool:
    """Pair filter for :func:`zubaan.extend.train_bpe.learn_merges`."""
    return is_urdu_safe(left + right)


def contains_arabic_block(text: str) -> bool:
    """True if the text has any character in U+0600 to U+06FF."""
    return any("؀" <= c <= "ۿ" for c in text)


def validate_merges(
    merges: Sequence[Merge], base_vocab: Collection[str], require_safe: bool = True
) -> None:
    """Raise ValueError unless every merge can be appended to a tokenizer with ``base_vocab``.

    Each merge needs both parts to exist already (in the base vocabulary or created by an
    earlier merge), must create a token that does not exist yet, and, if ``require_safe``,
    must keep English unchanged (see the module docstring).
    """
    available = set(base_vocab)
    for i, m in enumerate(merges):
        if m.left not in available:
            raise ValueError(f"merge {i}: left part {m.left!r} does not exist yet")
        if m.right not in available:
            raise ValueError(f"merge {i}: right part {m.right!r} does not exist yet")
        if m.token in available:
            raise ValueError(f"merge {i}: token {m.token!r} already exists")
        if require_safe and not is_urdu_safe(m.token):
            raise ValueError(f"merge {i}: token {m.token!r} could occur in non-Urdu text")
        available.add(m.token)


def select_merges(
    candidates: Iterable[Merge], base_vocab: Collection[str], n: int, require_safe: bool = True
) -> list[Merge]:
    """The first ``n`` usable merges of an ordered candidate list.

    A candidate is dropped if its token already exists, it is unsafe for English, or one of
    its parts is missing (for example because the merge that makes it was dropped). Order is
    kept, so the result always passes :func:`validate_merges`.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    available = set(base_vocab)
    out: list[Merge] = []
    for m in candidates:
        if len(out) >= n:
            break
        if m.left not in available or m.right not in available or m.token in available:
            continue
        if require_safe and not is_urdu_safe(m.token):
            continue
        available.add(m.token)
        out.append(m)
    return out
