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


@dataclass(frozen=True)
class TokenizerCheck:
    """Sanity checks that catch a tokenizer hiding text loss behind a low token count.

    A tokenizer that maps unknown characters to one ``<unk>`` token looks cheap but
    destroys the text, so a fair comparison needs ``n_unk == 0`` and lossless round trips.
    """

    n_texts: int
    n_bytes: int
    n_tokens: int
    n_unk: int
    n_roundtrip_failures: int

    @property
    def bytes_per_token(self) -> float:
        """UTF-8 bytes per token. Byte-level tokenizers can never go below 1."""
        return self.n_bytes / self.n_tokens if self.n_tokens else 0.0

    @property
    def lossless(self) -> bool:
        return self.n_unk == 0 and self.n_roundtrip_failures == 0

    def to_dict(self) -> dict:
        return {**asdict(self), "bytes_per_token": self.bytes_per_token, "lossless": self.lossless}


def check_tokenizer(tokenizer, texts: Iterable[str], normalize: bool = True) -> TokenizerCheck:
    """Count UTF-8 bytes and unknown tokens, and check that decode(encode(text)) == text.

    Leading and trailing whitespace is ignored in the round trip, because some tokenizers
    add a prefix space by design.
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    n_texts = n_bytes = n_tokens = n_unk = n_fail = 0
    for text in texts:
        if normalize:
            text = normalize_urdu(text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        n_texts += 1
        n_bytes += len(text.encode("utf-8"))
        n_tokens += len(ids)
        if unk_id is not None:
            n_unk += sum(1 for i in ids if i == unk_id)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded.strip() != text.strip():
            n_fail += 1
    return TokenizerCheck(n_texts, n_bytes, n_tokens, n_unk, n_fail)


def token_pieces(tokenizer, text: str, normalize: bool = True, limit: int = 20) -> list[str]:
    """The first tokens of a text, each decoded on its own, to show how words are split.

    A piece shown as U+FFFD is part of a character that was split across tokens.
    """
    if normalize:
        text = normalize_urdu(text)
    ids = tokenizer.encode(text, add_special_tokens=False)[:limit]
    return [tokenizer.decode([i]) for i in ids]


def fertility_report(
    tokenizer,
    name: str,
    target_texts: list[str],
    reference_texts: list[str] | None = None,
    normalize: bool = True,
    n_examples: int = 0,
) -> dict:
    """Fertility, sanity checks and optional examples for one tokenizer, as plain data.

    The target corpus is normalized (``normalize_urdu``); the reference is not.
    """
    target = compute_fertility(tokenizer, target_texts, normalize=normalize)
    out: dict = {
        "tokenizer": name,
        "vocab_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
        "normalize": normalize,
        "target": target.to_dict(),
        "target_checks": check_tokenizer(tokenizer, target_texts, normalize).to_dict(),
    }
    if reference_texts is not None:
        ref = compute_fertility(tokenizer, reference_texts, normalize=False)
        out["reference"] = ref.to_dict()
        out["reference_checks"] = check_tokenizer(tokenizer, reference_texts, False).to_dict()
        out["cost_multiplier"] = cost_multiplier(target, ref)
    if n_examples:
        out["examples"] = {
            "target": [token_pieces(tokenizer, t, normalize) for t in target_texts[:n_examples]],
            "reference": [
                token_pieces(tokenizer, t, False) for t in (reference_texts or [])[:n_examples]
            ],
        }
    return out


def cost_multiplier(target: FertilityResult, reference: FertilityResult) -> float:
    """Token cost of the target language relative to a reference language.

    Only meaningful on a parallel corpus (same sentences in both languages, such as FLORES-200).
    """
    if reference.n_tokens == 0:
        raise ValueError("reference corpus produced no tokens")
    if target.n_texts != reference.n_texts:
        raise ValueError("target and reference must be parallel corpora of equal length")
    return target.n_tokens / reference.n_tokens
