"""Token blocks for continued pretraining (stage 3).

Training reads fixed-size blocks of token ids. Each document (one corpus line) is written as the
end-of-text token followed by its tokens, and documents are packed back to back and cut into
blocks of ``block_size`` ids. The end-of-text token comes first because that is how the quality
harness frames every text it scores (``bench.quality.text_bits_per_character`` gives each text
a one-token end-of-text context), so training sees the same shape as evaluation.

Urdu blocks and English replay blocks are kept in separate files. The English replay keeps the
model from drifting on English; both languages use the same tokenizer, because the extension
leaves English tokenization byte-identical.

Everything here is dependency-free (``array`` from the standard library holds the ids), apart from
the helpers that talk to a Hugging Face tokenizer or dataset, which import lazily.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from zubaan.data.urdu import flores_shingles, overlaps_flores

EOS_TOKEN = "<|endoftext|>"
DEFAULT_BLOCK_SIZE = 1024
URDU_FILE = "urdu_blocks.u32"
ENGLISH_FILE = "english_blocks.u32"
META_FILE = "blocks_meta.json"
WIKIPEDIA_EN_CONFIG = "20231101.en"
WIKIPEDIA_LICENSE = "CC BY-SA 3.0 and GFDL (Wikipedia contributors)"


def _check_array_format() -> None:
    """Block files are raw little-endian uint32. Refuse to run where ``array('I')`` is not."""
    if array("I").itemsize != 4 or sys.byteorder != "little":
        raise RuntimeError("block files need 4-byte unsigned ints and a little-endian machine")


@dataclass
class PackStats:
    """What packing did to the documents. Tokens include one end-of-text token per document."""

    n_docs: int = 0
    n_tokens: int = 0
    n_blocks: int = 0
    dropped_tokens: int = 0

    def to_dict(self) -> dict:
        return {**self.__dict__}


class PackedBlocks:
    """A table of ``len(self)`` blocks of ``block_size`` token ids, stored flat."""

    def __init__(self, block_size: int, flat: array | None = None) -> None:
        _check_array_format()
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.block_size = block_size
        self.flat = flat if flat is not None else array("I")
        if len(self.flat) % block_size:
            raise ValueError("flat ids are not a whole number of blocks")

    def __len__(self) -> int:
        return len(self.flat) // self.block_size

    def block(self, i: int) -> list[int]:
        if not 0 <= i < len(self):
            raise IndexError(f"block {i} out of range for {len(self)} blocks")
        s = i * self.block_size
        return self.flat[s : s + self.block_size].tolist()

    def save(self, path: Path) -> None:
        with Path(path).open("wb") as f:
            self.flat.tofile(f)

    @classmethod
    def load(cls, path: Path, block_size: int) -> PackedBlocks:
        _check_array_format()
        flat = array("I")
        size = Path(path).stat().st_size
        if size % 4:
            raise ValueError(f"{path} is not a whole number of 4-byte ids")
        with Path(path).open("rb") as f:
            flat.fromfile(f, size // 4)
        return cls(block_size, flat)


def pack_documents(
    docs: Iterable[Sequence[int]], block_size: int, eos_id: int
) -> tuple[PackedBlocks, PackStats]:
    """Pack documents into blocks: each document is ``[eos] + tokens``, joined end to end.

    With documents of lengths l1, l2, ... the stream has sum(l_i + 1) ids, and the number of
    blocks is floor(stream / block_size). The last partial block is dropped and counted, never
    padded, so every block is full of real ids. Empty documents are skipped.
    """
    out = PackedBlocks(block_size)
    stats = PackStats()
    buf = array("I")
    for doc in docs:
        if not len(doc):
            continue
        stats.n_docs += 1
        buf.append(eos_id)
        buf.extend(doc)
        stats.n_tokens += len(doc) + 1
        if len(buf) >= block_size:
            whole = (len(buf) // block_size) * block_size
            out.flat.extend(buf[:whole])
            buf = buf[whole:]
    stats.dropped_tokens = len(buf)
    stats.n_blocks = len(out)
    return out, stats


def encode_lines(
    lines: Iterable[str], encode_batch: Callable[[list[str]], list[list[int]]], batch: int = 1000
) -> Iterator[list[int]]:
    """Token ids for each line, ``batch`` lines at a time (fast tokenizers are batch oriented)."""
    chunk: list[str] = []
    for line in lines:
        chunk.append(line)
        if len(chunk) >= batch:
            yield from encode_batch(chunk)
            chunk = []
    if chunk:
        yield from encode_batch(chunk)


def hf_batch_encoder(tokenizer) -> Callable[[list[str]], list[list[int]]]:
    """Wrap a ``tokenizers.Tokenizer`` so it encodes a list of texts without special tokens."""

    def encode(texts: list[str]) -> list[list[int]]:
        return [e.ids for e in tokenizer.encode_batch(texts, add_special_tokens=False)]

    return encode


# contamination guard ---------------------------------------------------------------------------


def find_overlaps(lines: Iterable[str], eval_texts: Iterable[str]) -> list[int]:
    """Indices of ``lines`` that share 8 consecutive words with any evaluation text.

    Uses the same shingle test as the Urdu corpus builder (``data.urdu.overlaps_flores``).
    """
    shingles = flores_shingles(eval_texts)
    if not shingles:
        raise ValueError("no evaluation texts given; the contamination guard would do nothing")
    return [i for i, line in enumerate(lines) if overlaps_flores(line, shingles)]


def require_no_overlap(lines: Sequence[str], eval_texts: Iterable[str], label: str) -> int:
    """Raise if any training line copies evaluation text; return how many lines were checked."""
    hits = find_overlaps(lines, eval_texts)
    if hits:
        raise ValueError(
            f"{len(hits)} {label} training lines overlap evaluation text "
            f"(first at line index {hits[0]}); remove them before training"
        )
    return len(lines)


# mixing and sampling ---------------------------------------------------------------------------


def english_block_count(n_urdu: int, english_ratio: float) -> int:
    """How many English blocks make English ``english_ratio`` of all blocks.

    With u Urdu blocks and share r for English, e / (u + e) = r, so e = u x r / (1 - r).
    Example: u = 90 and r = 0.1 give e = 90 x 0.1 / 0.9 = 10 blocks, 10 of 100 in all.
    """
    if not 0 <= english_ratio < 1:
        raise ValueError("english_ratio must be in [0, 1)")
    return round(n_urdu * english_ratio / (1 - english_ratio))


def mix_plan(n_urdu: int, n_english: int, english_ratio: float, seed: int) -> list[tuple[int, int]]:
    """One epoch of ``(source, index)`` pairs, source 0 = Urdu and 1 = English, shuffled.

    Every Urdu block appears once. English blocks are drawn without replacement, so a
    different seed shows different English text. Raises if there is not enough English.
    """
    need = english_block_count(n_urdu, english_ratio)
    if need > n_english:
        raise ValueError(
            f"English replay of {english_ratio:.0%} needs {need} blocks but only {n_english} exist"
        )
    rng = random.Random(seed)
    plan = [(0, i) for i in range(n_urdu)]
    plan += [(1, j) for j in rng.sample(range(n_english), need)]
    rng.shuffle(plan)
    return plan


class BlockSampler:
    """Deterministic stream of blocks over any number of epochs.

    Position ``p`` is the p-th block of the infinite stream: epoch ``p // epoch_length``,
    offset ``p % epoch_length``. Epoch ``e`` uses seed ``seed + e``. A run that stopped at
    position ``p`` therefore resumes with exactly the data it would have seen next.
    """

    def __init__(
        self,
        urdu: PackedBlocks,
        english: PackedBlocks | None,
        english_ratio: float,
        seed: int,
    ) -> None:
        if english is not None and english.block_size != urdu.block_size:
            raise ValueError("Urdu and English blocks must have the same block size")
        if not len(urdu):
            raise ValueError("no Urdu blocks")
        self.urdu, self.english = urdu, english
        self.ratio = english_ratio if english is not None else 0.0
        self.seed = seed
        self.block_size = urdu.block_size
        n_en = len(english) if english is not None else 0
        self._n_en = n_en
        self.epoch_length = len(urdu) + english_block_count(len(urdu), self.ratio)
        if english_block_count(len(urdu), self.ratio) > n_en:
            raise ValueError("not enough English blocks for the requested replay ratio")
        self._plans: dict[int, list[tuple[int, int]]] = {}

    def _plan(self, epoch: int) -> list[tuple[int, int]]:
        if epoch not in self._plans:
            self._plans = {
                epoch: mix_plan(len(self.urdu), self._n_en, self.ratio, self.seed + epoch)
            }
        return self._plans[epoch]

    def blocks_at(self, position: int, count: int) -> list[list[int]]:
        """``count`` consecutive blocks starting at ``position`` (may cross an epoch boundary)."""
        if position < 0 or count < 1:
            raise ValueError("position must be >= 0 and count >= 1")
        out = []
        for p in range(position, position + count):
            source, idx = self._plan(p // self.epoch_length)[p % self.epoch_length]
            out.append((self.urdu if source == 0 else self.english).block(idx))
        return out

    def epoch_of(self, position: int) -> float:
        return position / self.epoch_length


# building the files ----------------------------------------------------------------------------


def clean_english_line(
    line: str, min_words: int = 30, min_ascii_fraction: float = 0.9
) -> str | None:
    """Keep an English paragraph if it is long and mostly ASCII, else None.

    Drops headings, lists and tables (short) and paragraphs full of other scripts.
    """
    text = " ".join(line.split())
    if len(text.split()) < min_words:
        return None
    if sum(c.isascii() for c in text) / len(text) < min_ascii_fraction:
        return None
    return text


def iter_english_wikipedia() -> Iterator[str]:
    """Stream English Wikipedia articles from the Hub without downloading the whole dump."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", WIKIPEDIA_EN_CONFIG, split="train", streaming=True)
    for row in ds:
        if row["text"]:
            yield row["text"]


def build_english_replay(
    articles: Iterable[str],
    out_path: Path,
    eval_texts: Iterable[str],
    max_mb: float = 12.0,
    min_words: int = 30,
) -> dict:
    """Write cleaned English paragraphs (one per line) up to ``max_mb`` and return metadata.

    Paragraphs that overlap the English evaluation texts, and duplicates, are dropped.
    """
    shingles = flores_shingles(eval_texts)
    if not shingles:
        raise ValueError("no English evaluation texts given; the guard would do nothing")
    cap = int(max_mb * 1_000_000)
    seen: set[bytes] = set()
    written = n_para = dropped_filter = dropped_eval = dropped_dup = lines = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for article in articles:
            for para in (p for p in article.splitlines() if p.strip()):
                n_para += 1
                line = clean_english_line(para, min_words)
                if line is None:
                    dropped_filter += 1
                    continue
                if overlaps_flores(line, shingles):
                    dropped_eval += 1
                    continue
                key = hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()
                if key in seen:
                    dropped_dup += 1
                    continue
                seen.add(key)
                f.write(line + "\n")
                lines += 1
                written += len(line.encode("utf-8")) + 1
            if written >= cap:
                break
    if not lines:
        raise ValueError("no English lines survived cleaning")
    return {
        "source": {
            "dataset": "wikimedia/wikipedia",
            "config": WIKIPEDIA_EN_CONFIG,
            "licence": WIKIPEDIA_LICENSE,
            "note": "first articles of the streamed train split, in dataset order",
        },
        "paragraphs_seen": n_para,
        "dropped_filter": dropped_filter,
        "dropped_eval_overlap": dropped_eval,
        "dropped_duplicate": dropped_dup,
        "lines": lines,
        "bytes": written,
        "max_mb": max_mb,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _guard_lines(
    lines: Sequence[str], eval_texts: list[str], label: str, drop: bool
) -> tuple[Sequence[str], int]:
    """Return the lines without evaluation overlaps and how many were dropped.

    With ``drop`` False any overlap raises. With ``drop`` True the offending lines are removed
    and the rest is checked again, so a surviving overlap is still an error.
    """
    hits = find_overlaps(lines, eval_texts)
    if not hits:
        return lines, 0
    if not drop:
        require_no_overlap(lines, eval_texts, label)  # raises with the usual message
    bad = set(hits)
    kept = [line for i, line in enumerate(lines) if i not in bad]
    require_no_overlap(kept, eval_texts, label)
    return kept, len(hits)


def prepare_blocks(
    urdu_lines: Sequence[str],
    english_lines: Sequence[str],
    encode_batch: Callable[[list[str]], list[list[int]]],
    eos_id: int,
    out_dir: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    urdu_eval_texts: Iterable[str] = (),
    english_eval_texts: Iterable[str] = (),
    extra_meta: dict | None = None,
    drop_overlaps: bool = False,
) -> dict:
    """Tokenize, guard against evaluation text, pack, and write the block files and metadata.

    A training line that shares 8 consecutive words with an evaluation text raises before
    anything is written, or, with ``drop_overlaps``, is removed and counted in the metadata.
    Returns the metadata dict that was saved to ``blocks_meta.json``.
    """
    urdu_eval = list(urdu_eval_texts)
    english_eval = list(english_eval_texts)
    n_urdu_in, n_english_in = len(urdu_lines), len(english_lines)
    urdu_lines, urdu_dropped = _guard_lines(urdu_lines, urdu_eval, "Urdu", drop_overlaps)
    english_lines, english_dropped = _guard_lines(
        english_lines, english_eval, "English", drop_overlaps
    )
    guard = {
        "urdu_lines_checked": n_urdu_in,
        "urdu_lines_dropped": urdu_dropped,
        "urdu_eval_texts": len(urdu_eval),
        "english_lines_checked": n_english_in,
        "english_lines_dropped": english_dropped,
        "english_eval_texts": len(english_eval),
        "hits": 0,  # lines still overlapping after the guard; anything else raised
    }
    urdu, urdu_stats = pack_documents(encode_lines(urdu_lines, encode_batch), block_size, eos_id)
    english, english_stats = pack_documents(
        encode_lines(english_lines, encode_batch), block_size, eos_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    urdu.save(out_dir / URDU_FILE)
    english.save(out_dir / ENGLISH_FILE)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "block_size": block_size,
        "eos_id": eos_id,
        "urdu": {**urdu_stats.to_dict(), "sha256": _sha256(out_dir / URDU_FILE)},
        "english": {**english_stats.to_dict(), "sha256": _sha256(out_dir / ENGLISH_FILE)},
        "overlap_guard": guard,
        **(extra_meta or {}),
    }
    with (out_dir / META_FILE).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return meta


def load_blocks(data_dir: Path) -> tuple[PackedBlocks, PackedBlocks, dict]:
    """Read the files written by ``prepare_blocks``: Urdu blocks, English blocks, metadata."""
    meta = json.loads((data_dir / META_FILE).read_text(encoding="utf-8"))
    bs = meta["block_size"]
    return (
        PackedBlocks.load(data_dir / URDU_FILE, bs),
        PackedBlocks.load(data_dir / ENGLISH_FILE, bs),
        meta,
    )
