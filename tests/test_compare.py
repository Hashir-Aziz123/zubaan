import csv
import json

import pytest
from typer.testing import CliRunner

from zubaan import cli
from zubaan.audit.compare import COLUMNS, compare_tokenizers, write_csv


class _PairTokenizer:
    """Fake lossy tokenizer: one token per two characters, decodes to placeholders."""

    unk_token_id = None

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text[::2]]

    def decode(self, ids, skip_special_tokens=False):
        return "?" * len(ids)

    def __len__(self):
        return 10


def _loader(space_tokenizer):
    def load(name):
        if name == "gated/model":
            raise PermissionError("403 gated\nmore detail")
        return {"chars": space_tokenizer, "pairs": _PairTokenizer()}[name]

    return load


def test_skips_unavailable_models_and_continues(space_tokenizer):
    rows = compare_tokenizers(
        ["gated/model", "chars"], ["ایک دو"], ["one two"], _loader(space_tokenizer)
    )
    assert [r["model"] for r in rows] == ["chars", "gated/model"]
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "skipped"
    assert rows[1]["reason"] == "PermissionError: 403 gated"


def test_lossy_tokenizers_are_flagged_and_rows_sorted(space_tokenizer):
    rows = compare_tokenizers(["chars", "pairs"], ["ایک دو"], ["one two"], _loader(space_tokenizer))
    by_model = {r["model"]: r for r in rows}
    assert by_model["chars"]["cost_multiplier"] == pytest.approx(6 / 7)
    assert by_model["pairs"]["status"] == "lossy"
    # Word counts do not depend on the tokenizer.
    assert by_model["chars"]["target_words"] == by_model["pairs"]["target_words"] == 2
    costs = [r["cost_multiplier"] for r in rows]
    assert costs == sorted(costs)


def test_requires_parallel_texts(space_tokenizer):
    with pytest.raises(ValueError, match="parallel"):
        compare_tokenizers(["chars"], ["a", "b"], ["a"], _loader(space_tokenizer))


def test_write_csv_round_trips(space_tokenizer, tmp_path):
    rows = compare_tokenizers(["chars"], ["ایک"], ["one"], _loader(space_tokenizer))
    path = write_csv(rows, tmp_path / "compare.csv")
    with path.open(encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert tuple(read[0]) == COLUMNS
    assert read[0]["model"] == "chars"


def _compare_args(fixtures_dir, tmp_path, *models):
    args = ["audit", "compare"]
    for m in models:
        args += ["-t", m]
    return args + [
        "-c",
        str(fixtures_dir / "urdu_sample.txt"),
        "-r",
        str(fixtures_dir / "english_sample.txt"),
        "--out-dir",
        str(tmp_path),
    ]


def test_cli_compare_writes_reports(monkeypatch, space_tokenizer, fixtures_dir, tmp_path):
    monkeypatch.setattr(cli, "load_tokenizer", _loader(space_tokenizer))
    args = _compare_args(fixtures_dir, tmp_path, "chars", "gated/model")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in data["rows"]] == ["ok", "skipped"]
    assert (tmp_path / "compare.csv").exists()


def test_cli_compare_fails_when_nothing_loads(monkeypatch, space_tokenizer, fixtures_dir, tmp_path):
    monkeypatch.setattr(cli, "load_tokenizer", _loader(space_tokenizer))
    result = CliRunner().invoke(cli.app, _compare_args(fixtures_dir, tmp_path, "gated/model"))
    assert result.exit_code == 1


def test_example_config_loads():
    from zubaan.config import load_config

    cfg = load_config("configs/audit-compare.yaml").compare
    assert cfg is not None
    assert "Qwen/Qwen3-1.7B" in cfg.models


@pytest.mark.slow
def test_real_tokenizers_compare(fixtures_dir):
    from zubaan.io import read_lines

    rows = compare_tokenizers(
        ["Qwen/Qwen3-1.7B", "bigscience/bloom-560m"],
        read_lines(fixtures_dir / "urdu_sample.txt"),
        read_lines(fixtures_dir / "english_sample.txt"),
        cli.load_tokenizer,
    )
    assert all(r["status"] == "ok" for r in rows)
