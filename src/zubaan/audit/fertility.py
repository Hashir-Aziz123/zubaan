"""Tokenizer fertility: how many tokens a language needs per word and per character."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Protocol

from zubaan.text.normalize import normalize_urdu, split_words


class TokenizerLike(Protocol):
    """Anything with a Hugging Face style encode method."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


@dataclass(frozen=True)
class FertilityResult:
    n_texts: int
    n_words: int
    n_chars: int
    n_tokens: int

    @property
    def tokens_per_word(self) -> float:
        return self.n_tokens / self.n_words if self.n_words else 0.0

    @property
    def chars_per_token(self) -> float:
        return self.n_chars / self.n_tokens if self.n_tokens else 0.0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "tokens_per_word": self.tokens_per_word,
            "chars_per_token": self.chars_per_token,
        }


def compute_fertility(
    tokenizer: TokenizerLike, texts: Iterable[str], normalize: bool = True
) -> FertilityResult:
    """Count words, characters and tokens over a corpus.

    Special tokens are excluded so the numbers reflect the text itself.
    """
    n_texts = n_words = n_chars = n_tokens = 0
    for text in texts:
        if normalize:
            text = normalize_urdu(text)
        n_texts += 1
        n_words += len(split_words(text))
        n_chars += len(text)
        n_tokens += len(tokenizer.encode(text, add_special_tokens=False))
    return FertilityResult(n_texts, n_words, n_chars, n_tokens)


def cost_multiplier(target: FertilityResult, reference: FertilityResult) -> float:
    """Token cost of the target language relative to a reference language.

    Only meaningful on a parallel corpus (same sentences in both languages, such as FLORES-200).
    """
    if reference.n_tokens == 0:
        raise ValueError("reference corpus produced no tokens")
    if target.n_texts != reference.n_texts:
        raise ValueError("target and reference must be parallel corpora of equal length")
    return target.n_tokens / reference.n_tokens
