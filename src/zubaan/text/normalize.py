"""Urdu text normalization and word splitting.

Urdu text on the web mixes visually identical Unicode characters (Arabic forms of
yeh, kaf and heh next to the Urdu forms). Counting words or tokens without
normalizing first distorts every metric, so all measurements go through here.

Code points are written as escapes on purpose: diacritics and tatweel are
invisible or confusable when written literally.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic code points that look identical to the Urdu letters but are different characters.
_CHAR_MAP = {
    chr(0x064A): chr(0x06CC),  # ARABIC YEH -> FARSI YEH (Urdu choti ye)
    chr(0x0649): chr(0x06CC),  # ALEF MAKSURA -> FARSI YEH
    chr(0x0643): chr(0x06A9),  # ARABIC KAF -> KEHEH
    chr(0x0647): chr(0x06C1),  # ARABIC HEH -> HEH GOAL
}
_TRANSLATION = str.maketrans(_CHAR_MAP)


def _chars(*code_points: int) -> str:
    return "".join(chr(c) for c in code_points)


# Diacritics (aerab, U+064B to U+0652), superscript alef (U+0670), tatweel (U+0640).
_DIACRITICS = re.compile("[" + chr(0x064B) + "-" + chr(0x0652) + _chars(0x0670, 0x0640) + "]")

# Arabic-script punctuation: comma, semicolon, question mark, full stop (U+060C, U+061B,
# U+061F, U+06D4) and percent/decimal/thousands separators (U+066A to U+066C).
_ARABIC_PUNCT = _chars(0x060C, 0x061B, 0x061F, 0x06D4) + chr(0x066A) + "-" + chr(0x066C)
# Guillemets, ellipsis, en dash, em dash.
_TYPOGRAPHIC_PUNCT = _chars(0x00AB, 0x00BB, 0x2026, 0x2013, 0x2014)

# Word boundaries: whitespace plus the punctuation above and ASCII punctuation.
# ZWNJ (U+200C) is deliberately not a boundary: it joins parts of one Urdu word.
_WORD_SPLIT = re.compile(
    r"[\s" + _ARABIC_PUNCT + _TYPOGRAPHIC_PUNCT + r".,!?;:\"'()\[\]{}\-/\\|*#@%&+=<>~`]+"
)


def normalize_urdu(text: str, strip_diacritics: bool = True) -> str:
    """Return text in a canonical form for counting.

    Applies Unicode NFC, maps Arabic look-alike letters to their Urdu forms,
    optionally strips diacritics, and collapses runs of whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TRANSLATION)
    if strip_diacritics:
        text = _DIACRITICS.sub("", text)
    return " ".join(text.split())


def split_words(text: str) -> list[str]:
    """Split text into words on whitespace and punctuation. Empty pieces are dropped."""
    return [w for w in _WORD_SPLIT.split(text) if w]
