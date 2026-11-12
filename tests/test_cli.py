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
