"""Provenance: record exactly what produced a result so it can be reproduced."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ("zubaan", "transformers", "tokenizers", "torch", "huggingface_hub", "datasets")


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    """Installed version of each package, or None when it is not installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_revision(name: str) -> str | None:
    """Commit hash of a Hugging Face repo, or None for local paths and offline runs."""
    if Path(name).exists():
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(name).sha
    except Exception:
        return None


def run_metadata(models: list[str] | None = None, data_meta: Path | None = None) -> dict:
    """Machine, software versions, model revisions and data source for one run."""
    out: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "packages": package_versions(),
    }
    if models:
        out["model_revisions"] = {m: model_revision(m) for m in models}
    if data_meta is not None and data_meta.exists():
        with data_meta.open(encoding="utf-8") as f:
            out["data"] = json.load(f)
    return out


def write_json(data: dict, path: Path) -> Path:
    """Write UTF-8 JSON with Urdu kept readable (not escaped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path
