import json

from typer.testing import CliRunner

from zubaan import cli

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("audit", "extend", "adapt", "draft", "export", "bench"):
        assert cmd in result.output


def test_planned_commands_exit_nonzero():
    result = runner.invoke(cli.app, ["extend"])
    assert result.exit_code == 1
    assert "stage 2" in result.output


def test_audit_fertility_json(monkeypatch, char_tokenizer, fixtures_dir):
    monkeypatch.setattr(cli, "load_tokenizer", lambda name: char_tokenizer)
    result = runner.invoke(
        cli.app,
        [
            "audit",
            "fertility",
            "--tokenizer",
            "fake",
            "--corpus",
            str(fixtures_dir / "urdu_sample.txt"),
            "--reference",
            str(fixtures_dir / "english_sample.txt"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["target"]["n_texts"] == 2
    assert data["cost_multiplier"] > 0


def test_audit_fertility_from_config_saves_json(
    monkeypatch, char_tokenizer, fixtures_dir, tmp_path
):
    monkeypatch.setattr(cli, "load_tokenizer", lambda name: char_tokenizer)
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "audit:\n"
        "  tokenizer: fake\n"
        f"  corpus: {(fixtures_dir / 'urdu_sample.txt').as_posix()}\n"
        f"  reference_corpus: {(fixtures_dir / 'english_sample.txt').as_posix()}\n",
        encoding="utf-8",
    )
    out = tmp_path / "result.json"
    result = runner.invoke(cli.app, ["audit", "fertility", "--config", str(cfg), "--out", str(out)])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tokenizer"] == "fake"
    assert data["reference"]["n_texts"] == 2
    assert "packages" in data["provenance"]


def test_audit_fertility_needs_tokenizer():
    result = runner.invoke(cli.app, ["audit", "fertility"])
    assert result.exit_code != 0
