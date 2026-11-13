"""FLORES parallel evaluation data.

FLORES is the same set of sentences professionally translated into 200+ languages,
which makes it the standard parallel corpus for comparing token costs across languages.

Two sources are supported:

- ``hf``: ``openlanguagedata/flores_plus`` on Hugging Face (maintained, gated: accept its
  terms and log in first). Rows are aligned by their sentence ``id``.
- ``meta``: the original FLORES-200 archive from Meta (no login). Files are already aligned
  line by line.

Text is written unnormalized; normalization happens when metrics are computed.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HF_REPO = "openlanguagedata/flores_plus"
META_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SPLITS = ("dev", "devtest")
SOURCES = ("auto", "hf", "meta")


@dataclass
class FloresDownload:
    """Where the files went and what produced them."""

    paths: dict[str, Path]
    n_lines: int
    meta: dict = field(default_factory=dict)


def clean_sentence(text: str) -> str:
    """Make a sentence safe for a one-sentence-per-line file.

    Line breaks inside a sentence would shift every later line, so they become spaces.
    """
    cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not cleaned:
        raise ValueError("empty sentence: parallel files would fall out of alignment")
    return cleaned


def align_by_id(rows_by_lang: Mapping[str, Iterable[Mapping]]) -> dict[str, list[str]]:
    """Align rows of the form ``{"id": ..., "text": ...}`` across languages by id.

    Returns one list of cleaned sentences per language, all in the same id order.
    """
    by_lang: dict[str, dict] = {}
    for lang, rows in rows_by_lang.items():
        table: dict = {}
        for row in rows:
            if row["id"] in table:
                raise ValueError(f"{lang}: duplicate id {row['id']!r}")
            table[row["id"]] = row["text"]
        by_lang[lang] = table
    id_sets = [set(t) for t in by_lang.values()]
    if not id_sets:
        raise ValueError("no languages given")
    if any(s != id_sets[0] for s in id_sets):
        raise ValueError("languages have different sentence ids; data is not parallel")
    ids = sorted(id_sets[0])
    return {lang: [clean_sentence(t[i]) for i in ids] for lang, t in by_lang.items()}


def check_parallel(texts_by_lang: Mapping[str, list[str]]) -> int:
    """Check every language has the same number of non-empty sentences. Returns that count."""
    lengths = {lang: len(t) for lang, t in texts_by_lang.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"parallel files differ in length: {lengths}")
    for lang, texts in texts_by_lang.items():
        for i, t in enumerate(texts):
            if not t.strip() or "\n" in t:
                raise ValueError(f"{lang} line {i + 1} is empty or contains a line break")
    return next(iter(lengths.values()))


def output_name(lang: str) -> str:
    """File name for a FLORES language code: ``urd_Arab`` -> ``flores_urd.txt``."""
    return f"flores_{lang.split('_')[0]}.txt"


def write_parallel(texts_by_lang: Mapping[str, list[str]], out_dir: Path) -> dict[str, Path]:
    """Write one UTF-8 file per language with ``\\n`` line endings, also on Windows."""
    check_parallel(texts_by_lang)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for lang, texts in texts_by_lang.items():
        path = out_dir / output_name(lang)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.writelines(t + "\n" for t in texts)
        paths[lang] = path
    return paths


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hf(langs: list[str], split: str) -> tuple[dict[str, list[str]], dict]:
    """Load FLORES+ from Hugging Face. Needs the dataset terms accepted and a login."""
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(HF_REPO).sha
    rows_by_lang = {}
    for lang in langs:
        path = hf_hub_download(
            HF_REPO, f"{split}/{lang}.jsonl", repo_type="dataset", revision=revision
        )
        with open(path, encoding="utf-8") as f:
            rows_by_lang[lang] = [json.loads(line) for line in f if line.strip()]
    meta = {"source": "hf", "repo": HF_REPO, "revision": revision, "license": "CC BY-SA 4.0"}
    return align_by_id(rows_by_lang), meta


def load_meta(langs: list[str], split: str, cache_dir: Path) -> tuple[dict[str, list[str]], dict]:
    """Load the original FLORES-200 archive from Meta. Files are already line-aligned.

    Members are read straight from the archive, never extracted to disk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "flores200_dataset.tar.gz"
    if not archive.exists():
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(META_URL, tmp)
        tmp.replace(archive)
    texts = {}
    with tarfile.open(archive, "r:gz") as tar:
        # The real archive stores names as "./flores200_dataset/...".
        members = {m.name.removeprefix("./"): m for m in tar.getmembers()}
        for lang in langs:
            info = members.get(f"flores200_dataset/{split}/{lang}.{split}")
            member = tar.extractfile(info) if info is not None else None
            if member is None:
                raise ValueError(f"{lang} not found in the FLORES-200 archive")
            lines = member.read().decode("utf-8").splitlines()
            texts[lang] = [clean_sentence(line) for line in lines]
    meta = {
        "source": "meta",
        "url": META_URL,
        "archive_sha256": _sha256(archive),
        "license": "CC BY-SA 4.0",
    }
    return texts, meta


def download_flores(
    langs: list[str],
    split: str = "devtest",
    out_dir: Path = Path("data"),
    source: str = "meta",
) -> FloresDownload:
    """Download FLORES for the given languages and write aligned one-sentence-per-line files.

    ``meta`` is the default: stable and needs no login. ``auto`` tries Hugging Face first
    and falls back to Meta's archive if access is denied.
    The chosen source is recorded in ``flores_meta.json`` next to the files.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if len(langs) < 2:
        raise ValueError("need at least two languages for a parallel corpus")

    fallback_reason = None
    if source in ("auto", "hf"):
        try:
            texts, meta = load_hf(langs, split)
        except Exception as e:  # gated, offline, or not logged in
            if source == "hf":
                raise
            fallback_reason = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            texts, meta = load_meta(langs, split, out_dir / ".cache")
    else:
        texts, meta = load_meta(langs, split, out_dir / ".cache")

    paths = write_parallel(texts, out_dir)
    n = check_parallel(texts)
    meta.update(
        {
            "split": split,
            "languages": langs,
            "n_lines": n,
            "files": {lang: p.name for lang, p in paths.items()},
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    with (out_dir / "flores_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return FloresDownload(paths=paths, n_lines=n, meta=meta)
