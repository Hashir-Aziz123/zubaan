"""Urdu training text for tokenizer extension.

New merges must be learned from text that is (a) Urdu, (b) cleaned the same way the audit
cleans Urdu, and (c) never contains the FLORES sentences used for evaluation. This module
turns raw Wikipedia articles into a deduplicated training file, a small held-out file, and a
metadata record with the licence and a hash of every output.

Pure Python: only reading the Wikipedia parquet file needs ``pyarrow`` (installed with the
``audit`` extra).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from zubaan.text.normalize import normalize_urdu, split_words

WIKIPEDIA_REPO = "wikimedia/wikipedia"
WIKIPEDIA_FILE = "20231101.ur/train-00000-of-00001.parquet"
WIKIPEDIA_LICENSE = "CC BY-SA 3.0 and GFDL (Wikipedia contributors)"
SHINGLE_WORDS = 8


def is_arabic_block(ch: str) -> bool:
    """True for the Arabic block U+0600 to U+06FF, where the Urdu letters live."""
    return "؀" <= ch <= "ۿ"


def urdu_fraction(text: str) -> float:
    """Share of non-space characters that are in the Arabic block.

    Text with Latin, digits or other scripts mixed in scores lower. Spaces are ignored so the
    number does not depend on how many words there are.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if is_arabic_block(c)) / len(chars)


def split_paragraphs(article: str) -> list[str]:
    """One candidate line per paragraph of an article (articles are newline separated)."""
    return [p.strip() for p in article.splitlines() if p.strip()]


def clean_line(line: str, min_words: int = 5, min_urdu_fraction: float = 0.7) -> str | None:
    """Normalize a line and return it, or None if it should not be used for training.

    Keeps only lines that are mostly Urdu script and long enough to be a real sentence,
    which drops headings, lists, tables and English or code fragments.
    """
    text = normalize_urdu(line)
    if len(split_words(text)) < min_words:
        return None
    if urdu_fraction(text) < min_urdu_fraction:
        return None
    return text


def _shingle_hashes(words: list[str], n: int) -> Iterator[int]:
    for i in range(len(words) - n + 1):
        key = "\x1f".join(words[i : i + n]).encode("utf-8")
        yield int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def flores_shingles(sentences: Iterable[str], n: int = SHINGLE_WORDS) -> set[int]:
    """Hashes of every run of ``n`` consecutive words in the evaluation sentences.

    A sentence shorter than ``n`` words is hashed whole, so short sentences are protected
    too. Used to find training lines that copy any part of the evaluation data.
    """
    out: set[int] = set()
    for s in sentences:
        words = split_words(normalize_urdu(s))
        if not words:
            continue
        out.update(_shingle_hashes(words, min(n, len(words))))
    return out


def overlaps_flores(line: str, shingles: set[int], n: int = SHINGLE_WORDS) -> bool:
    """True if the line shares ``n`` consecutive words with an evaluation sentence.

    A line shorter than ``n`` words is compared whole against the short evaluation sentences.
    """
    words = split_words(normalize_urdu(line))
    if not words:
        return False
    return any(h in shingles for h in _shingle_hashes(words, min(n, len(words))))


def is_heldout(line: str, percent: float) -> bool:
    """Deterministic split: about ``percent`` of lines go to the held-out file.

    Decided by a hash of the line, so the split does not change between runs or depend on
    the order of the lines.
    """
    if not 0 <= percent < 100:
        raise ValueError("heldout percent must be in [0, 100)")
    h = int.from_bytes(hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest(), "big")
    return (h % 10_000) < percent * 100


@dataclass
class CorpusStats:
    """What happened to the input lines. Every dropped line is counted under one reason."""

    articles: int = 0
    paragraphs: int = 0
    dropped_filter: int = 0
    dropped_duplicate: int = 0
    dropped_flores: int = 0
    train_lines: int = 0
    heldout_lines: int = 0
    train_bytes: int = 0
    heldout_bytes: int = 0
    stopped_at_cap: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {**self.__dict__}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_corpus(
    articles: Iterable[str],
    out_dir: Path,
    flores_sentences: Iterable[str],
    max_mb: float | None = None,
    heldout_percent: float = 1.0,
    min_words: int = 5,
    min_urdu_fraction: float = 0.7,
    source: dict | None = None,
) -> dict:
    """Write ``urdu_train.txt``, ``urdu_heldout.txt`` and ``urdu_meta.json`` into ``out_dir``.

    Each article is split into paragraphs; each paragraph is normalized, filtered, checked
    against the FLORES sentences, deduplicated, and routed to train or held-out. Stops
    adding text once ``max_mb`` megabytes of training UTF-8 have been written. Returns the
    metadata that was saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shingles = flores_shingles(flores_sentences)
    if not shingles:
        raise ValueError("no FLORES sentences given; the contamination guard would do nothing")
    cap = None if max_mb is None else int(max_mb * 1_000_000)
    stats = CorpusStats()
    seen: set[bytes] = set()
    train_path = out_dir / "urdu_train.txt"
    held_path = out_dir / "urdu_heldout.txt"
    with (
        train_path.open("w", encoding="utf-8", newline="\n") as train,
        held_path.open("w", encoding="utf-8", newline="\n") as held,
    ):
        for article in articles:
            stats.articles += 1
            for para in split_paragraphs(article):
                stats.paragraphs += 1
                line = clean_line(para, min_words, min_urdu_fraction)
                if line is None:
                    stats.dropped_filter += 1
                    continue
                if overlaps_flores(line, shingles):
                    stats.dropped_flores += 1
                    continue
                key = hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()
                if key in seen:
                    stats.dropped_duplicate += 1
                    continue
                seen.add(key)
                size = len(line.encode("utf-8")) + 1
                if is_heldout(line, heldout_percent):
                    held.write(line + "\n")
                    stats.heldout_lines += 1
                    stats.heldout_bytes += size
                else:
                    train.write(line + "\n")
                    stats.train_lines += 1
                    stats.train_bytes += size
            if cap is not None and stats.train_bytes >= cap:
                stats.stopped_at_cap = True
                break
    if stats.train_lines == 0:
        raise ValueError("no training lines survived cleaning; check the input and filters")
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source or {},
        "filters": {
            "min_words": min_words,
            "min_urdu_fraction": min_urdu_fraction,
            "heldout_percent": heldout_percent,
            "max_mb": max_mb,
            "flores_shingle_words": SHINGLE_WORDS,
            "normalization": "zubaan.text.normalize_urdu",
        },
        "stats": stats.to_dict(),
        "files": {
            "train": {"path": train_path.name, "sha256": _sha256(train_path)},
            "heldout": {"path": held_path.name, "sha256": _sha256(held_path)},
        },
    }
    with (out_dir / "urdu_meta.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return meta


def iter_parquet_text(path: Path, column: str = "text", batch_size: int = 1000) -> Iterator[str]:
    """Stream one string per row from a parquet file without loading it all into memory."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[column]):
        yield from (t for t in batch.column(0).to_pylist() if t)


def download_wikipedia(out_dir: Path) -> Path:
    """Fetch the Urdu Wikipedia parquet from the Hub (about 170 MB). Returns its path."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(WIKIPEDIA_REPO, WIKIPEDIA_FILE, repo_type="dataset", local_dir=str(out_dir))
    )
