"""Small input helpers."""

from __future__ import annotations

from pathlib import Path


def read_lines(path: str | Path, max_lines: int | None = None) -> list[str]:
    """Read non-empty, stripped lines from a UTF-8 text file (one text per line)."""
    lines: list[str] = []
    with Path(path).open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                lines.append(line)
                if max_lines is not None and len(lines) >= max_lines:
                    break
    return lines
