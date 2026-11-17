"""Tokenizer parity: Hugging Face and llama.cpp must give identical token ids.

A GGUF file carries its own copy of the tokenizer, and llama.cpp re-implements the
pre-tokenizer and BPE. If that implementation disagrees with Hugging Face on any text, the
model sees different ids at inference than it saw in training, and quality drops silently.
This module runs both on the same texts and reports every disagreement.

``llama-tokenize`` is run once per text on a file written byte for byte (``-f``, escapes off,
no BOS), so what is compared is the tokenizer and nothing else.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from zubaan.export.gguf import LlamaCpp, Runner


def stress_texts() -> list[str]:
    """Hand-picked texts that tend to expose pre-tokenizer differences."""
    return [
        "",
        " ",
        "\n",
        "\n\n\n",
        "   leading and trailing spaces   ",
        "tabs\tand\t\ttabs",
        "line one\nline two\r\nline three",
        "I've been told he's there, 'RE you sure? We'Ve a'lL",
        "3 33 333 3333 33333 3.3 3..3 3...3 1234567890",
        "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙",
        "你好，世界！我想在apple工作1314151天～",
        "нещо на Български ------======= '''''' ```````",
        "def f(x):\n    return x ** 2  # square\n",
        "<|im_start|>user\nHello<|im_end|>",
        "پاکستان ایک ملک ہے۔",
        "اردو میں 123 اور English کا مرکب mixed script ٹیسٹ",
        "۱۲۳۴۵ اور 12345",
        "کیا آپ ٹھیک ہیں؟ جی ہاں!",
        "ٹ" * 50,
        "اردو " * 400,
        ("The quick brown fox. " * 60) + ("یہ ایک لمبی لائن ہے۔ " * 60),
    ]


@dataclass
class ParityResult:
    """Outcome of comparing Hugging Face and llama.cpp token ids.

    ``n_mismatch`` counts every text whose ids differ. Some of those differ only because the
    text is not in Unicode NFC form (``n_nfc_only``, see ``check_parity``); the rest are real
    tokenizer disagreements. ``mismatches`` lists the real ones, ``nfc_only`` the explained ones.
    """

    n_texts: int
    n_mismatch: int
    mismatches: list[dict] = field(default_factory=list)
    n_nfc_only: int = 0
    nfc_only: list[dict] = field(default_factory=list)

    @property
    def n_real(self) -> int:
        """Mismatches that are not explained by missing NFC normalization."""
        return self.n_mismatch - self.n_nfc_only

    @property
    def passed(self) -> bool:
        return self.n_texts > 0 and self.n_real == 0


def describe_mismatch(text: str, a: Sequence[int], b: Sequence[int], pos: int) -> dict:
    """A small record of where two id lists first differ."""
    return {
        "text": text[:80],
        "position": pos,
        "hf_ids": list(a[max(0, pos - 2) : pos + 4]),
        "llama_ids": list(b[max(0, pos - 2) : pos + 4]),
        "hf_length": len(a),
        "llama_length": len(b),
    }


def first_difference(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first position where two id lists differ, or None if they are equal."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_ids(
    texts: Sequence[str],
    hf: Sequence[Sequence[int]],
    llama: Sequence[Sequence[int]],
    max_examples: int = 10,
) -> ParityResult:
    """Compare two id lists for every text; keep the first ``max_examples`` disagreements."""
    if not (len(texts) == len(hf) == len(llama)):
        raise ValueError("texts, hf ids and llama ids must have the same length")
    bad = []
    n_bad = 0
    for text, a, b in zip(texts, hf, llama, strict=True):
        pos = first_difference(a, b)
        if pos is None:
            continue
        n_bad += 1
        if len(bad) < max_examples:
            bad.append(describe_mismatch(text, a, b, pos))
    return ParityResult(len(texts), n_bad, bad)


def parse_ids(output: str) -> list[int]:
    """Parse the ``[1, 2, 3]`` line that ``llama-tokenize --ids`` prints."""
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            return json.loads(line)
    raise ValueError(f"no id list in llama-tokenize output: {output[-200:]!r}")


def tokenize_command(tools: LlamaCpp, gguf: Path, text_file: Path) -> list[str]:
    """``llama-tokenize`` for one text file: ids only, no BOS, escapes off, logging off."""
    return [
        str(tools.tool("llama-tokenize")),
        "-m",
        str(gguf),
        "-f",
        str(text_file),
        "--ids",
        "--no-bos",
        "--no-escape",
        "--log-disable",
    ]


def llama_ids(
    tools: LlamaCpp,
    gguf: Path,
    texts: Sequence[str],
    run: Runner = subprocess.run,
    workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> list[list[int]]:
    """Token ids from llama.cpp for each text, in order."""
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for i, text in enumerate(texts):
            f = Path(tmp) / f"t{i}.txt"
            f.write_bytes(text.encode("utf-8"))  # exact bytes, no newline added
            files.append(f)
        done = 0

        def one(f: Path) -> list[int]:
            nonlocal done
            proc = run(
                tokenize_command(tools, gguf, f),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"llama-tokenize failed ({proc.returncode}): {proc.stderr[-300:]}"
                )
            done += 1
            if progress and done % 200 == 0:
                progress(done, len(files))
            return parse_ids(proc.stdout)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(one, files))


def hf_ids(tokenizer, texts: Sequence[str]) -> list[list[int]]:
    """Token ids from a Hugging Face tokenizer, without special tokens added."""
    return [list(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def check_parity(
    tokenizer,
    tools: LlamaCpp,
    gguf: Path,
    texts: Sequence[str],
    run: Runner = subprocess.run,
    workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> ParityResult:
    """Tokenize ``texts`` with Hugging Face and with llama.cpp and compare every id.

    The Qwen tokenizer normalizes text to Unicode NFC before tokenizing; llama.cpp's BPE
    tokenizer does not. For a text that is not already NFC (for example Arabic diacritics in a
    different order than NFC puts them) the ids can therefore differ even though the
    vocabulary and merges are identical. A mismatch is counted as ``nfc_only`` when the text
    is not NFC and llama.cpp, given the NFC form of the text, gives exactly Hugging Face's ids.
    Every other mismatch is real and fails the check.
    """
    if not texts:
        raise ValueError("no texts to compare")
    a = hf_ids(tokenizer, texts)
    b = llama_ids(tools, gguf, texts, run, workers, progress)
    if not (len(a) == len(b) == len(texts)):
        raise ValueError("texts, hf ids and llama ids must have the same length")
    bad = [i for i in range(len(texts)) if first_difference(a[i], b[i]) is not None]
    candidates = [i for i in bad if not unicodedata.is_normalized("NFC", texts[i])]
    explained: set[int] = set()
    if candidates:
        nfc_texts = [unicodedata.normalize("NFC", texts[i]) for i in candidates]
        for i, ids in zip(candidates, llama_ids(tools, gguf, nfc_texts, run, workers), strict=True):
            if first_difference(a[i], ids) is None:
                explained.add(i)

    def record(i: int) -> dict:
        return describe_mismatch(texts[i], a[i], b[i], first_difference(a[i], b[i]))

    real = [i for i in bad if i not in explained]
    return ParityResult(
        len(texts),
        len(bad),
        [record(i) for i in real[:10]],
        len(explained),
        [record(i) for i in sorted(explained)[:10]],
    )
