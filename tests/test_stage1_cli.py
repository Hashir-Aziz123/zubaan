"""End-to-end checks of the stage 1 commands with fake tokenizers and fake model runs."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zubaan import cli
from zubaan.audit.fertility import compute_fertility, fertility_report
from zubaan.config import load_config
from zubaan.text.normalize import normalize_urdu

runner = CliRunner()


def _fertility(fixtures_dir, *extra):
    return runner.invoke(
        cli.app,
        [
            "audit",
            "fertility",
            "-t",
            "fake",
            "-c",
            str(fixtures_dir / "urdu_sample.txt"),
            "-r",
            str(fixtures_dir / "english_sample.txt"),
            *extra,
        ],
    )


# fertility command ---------------------------------------------------------------------------


def test_fertility_max_texts_limits_both_corpora(monkeypatch, char_tokenizer, fixtures_dir):
    monkeypatch.setattr(cli, "load_tokenizer", lambda name: char_tokenizer)
    result = _fertility(fixtures_dir, "--max-texts", "1", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["target"]["n_texts"] == data["reference"]["n_texts"] == 1


def test_fertility_no_normalize_changes_the_count(monkeypatch, char_tokenizer, tmp_path):
    # Arabic yeh (U+064A) maps to Farsi yeh (U+06CC): same length, but a diacritic is stripped.
    monkeypatch.setattr(cli, "load_tokenizer", lambda name: char_tokenizer)
    corpus = tmp_path / "u.txt"
    corpus.write_text("کَتاب\n", encoding="utf-8")
    base = ["audit", "fertility", "-t", "fake", "-c", str(corpus), "--json"]
    on = json.loads(runner.invoke(cli.app, base).output)
    off = json.loads(runner.invoke(cli.app, [*base, "--no-normalize"]).output)
    assert on["target"]["n_chars"] == off["target"]["n_chars"] - 1
    assert on["normalize"] is True and off["normalize"] is False


def test_fertility_examples_are_printed(monkeypatch, char_tokenizer, fixtures_dir):
    monkeypatch.setattr(cli, "load_tokenizer", lambda name: char_tokenizer)
    result = _fertility(fixtures_dir, "--examples", "1")
    assert result.exit_code == 0, result.output
    assert "target:" in result.output and "reference:" in result.output


def test_fertility_flags_override_config(monkeypatch, char_tokenizer, fixtures_dir, tmp_path):
    monkeypatch.setattr(cli, "load_tokenizer", lambda name: char_tokenizer)
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "audit:\n"
        "  tokenizer: from-config\n"
        f"  corpus: {(fixtures_dir / 'urdu_sample.txt').as_posix()}\n"
        "  max_texts: 2\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        cli.app, ["audit", "fertility", "--config", str(cfg), "-t", "from-flag", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tokenizer"] == "from-flag"


def test_reference_is_not_normalized(char_tokenizer):
    # Normalization is for the target language only; reference text is counted as given.
    text = "aَb"
    report = fertility_report(char_tokenizer, "fake", [text], [text])
    assert report["target"]["n_chars"] == 2
    assert report["reference"]["n_chars"] == 3


def test_empty_or_wordless_corpus_raises(char_tokenizer):
    with pytest.raises(ValueError, match="no texts"):
        compute_fertility(char_tokenizer, [])
    with pytest.raises(ValueError, match="no words"):
        compute_fertility(char_tokenizer, ["۔ ،"])


# normalization --------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "hello world", "ایک دو", "کَتاب", "۱۲۳ 123"])
def test_normalize_is_idempotent(text):
    once = normalize_urdu(text)
    assert normalize_urdu(once) == once


def test_normalize_leaves_digits_and_ascii_alone():
    assert normalize_urdu("COVID-19 ۱۲۳ 123") == "COVID-19 ۱۲۳ 123"


# acceptance command ---------------------------------------------------------------------------


def _fake_language(alpha):
    return {
        "n_texts": 2,
        "n_positions": 10,
        "alpha": alpha,
        "alpha_ci_low": alpha - 0.01,
        "alpha_ci_high": alpha + 0.01,
        "greedy_match": alpha,
        "greedy_ci_low": alpha - 0.01,
        "greedy_ci_high": alpha + 0.01,
        "draft_cost": 0.35,
        "draft_cost_source": "parameter ratio",
        "speedup": [{"gamma": 4, "tokens_per_pass": 2.0, "speedup": 0.9}],
    }


def test_acceptance_command_passes_options_through_and_saves(monkeypatch, fixtures_dir, tmp_path):
    seen = {}

    def fake_report(draft, target, texts, **kwargs):
        seen.update(draft=draft, target=target, texts=texts, **kwargs)
        return {
            "draft": draft,
            "target": target,
            "languages": {"urd": _fake_language(0.55), "eng": _fake_language(0.65)},
        }

    monkeypatch.setattr("zubaan.audit.acceptance_lm.acceptance_report", fake_report)
    out = tmp_path / "acc.json"
    result = runner.invoke(
        cli.app,
        [
            "audit",
            "acceptance",
            "-d",
            "small",
            "-t",
            "big",
            "-c",
            str(fixtures_dir / "urdu_sample.txt"),
            "-r",
            str(fixtures_dir / "english_sample.txt"),
            "--max-texts",
            "1",
            "--dtype",
            "float32",
            "--mode",
            "sequential",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["normalize_lang"] == "urd"
    assert seen["dtype"] == "float32" and seen["mode"] == "sequential"
    assert set(seen["texts"]) == {"urd", "eng"}
    assert all(len(v) == 1 for v in seen["texts"].values())
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["draft"] == "small" and "provenance" in saved
    assert "0.550" in result.output and "not measured" in result.output


def test_acceptance_command_needs_models_and_corpus():
    result = runner.invoke(cli.app, ["audit", "acceptance", "-d", "small"])
    assert result.exit_code != 0


def test_acceptance_without_reference_measures_only_target(monkeypatch, fixtures_dir, tmp_path):
    seen = {}

    def fake_report(draft, target, texts, **kwargs):
        seen["langs"] = list(texts)
        return {"draft": draft, "target": target, "languages": {"urd": _fake_language(0.5)}}

    monkeypatch.setattr("zubaan.audit.acceptance_lm.acceptance_report", fake_report)
    result = runner.invoke(
        cli.app,
        [
            "audit",
            "acceptance",
            "-d",
            "a",
            "-t",
            "b",
            "-c",
            str(fixtures_dir / "urdu_sample.txt"),
            "--out",
            str(tmp_path / "o.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["langs"] == ["urd"]


# report command -------------------------------------------------------------------------------


def _compare_json(path: Path) -> Path:
    def row(model, cost):
        return {
            "model": model,
            "status": "ok",
            "vocab_size": 1000,
            "target_tokens_per_word": 3.0,
            "reference_tokens_per_word": 1.25,
            "target_chars_per_token": 1.5,
            "cost_multiplier": cost,
            "target_tokens": 300,
            "reference_tokens": 100,
            "reason": "",
        }

    data = {
        "rows": [row("org/a", 1.4), row("Qwen/Qwen3-1.7B", 3.0)],
        "provenance": {"python": "3.11", "packages": {"zubaan": "0.0.1"}, "data": {}},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_report_tax_command_writes_markdown_and_charts(tmp_path):
    pytest.importorskip("matplotlib")
    acc = tmp_path / "acc.json"
    acc.write_text(
        json.dumps(
            {
                "draft": "d",
                "target": "t",
                "languages": {
                    "urd": {
                        **_fake_language(0.55),
                        "run": {"dtype": "float32", "device": "cpu", "mode": "joint"},
                    },
                    "eng": {
                        **_fake_language(0.75),
                        "run": {"dtype": "float32", "device": "cpu", "mode": "joint"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report"
    result = runner.invoke(
        cli.app,
        [
            "report",
            "tax",
            "--compare",
            str(_compare_json(tmp_path / "compare.json")),
            "--acceptance",
            str(acc),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    md = (out / "index.md").read_text(encoding="utf-8")
    assert "**3.00x**" in md and "—" not in md
    for png in ("tokens_per_word.png", "cost_multiplier.png", "acceptance.png"):
        assert (out / png).stat().st_size > 0


def test_report_tax_command_warns_when_acceptance_is_missing(tmp_path):
    pytest.importorskip("matplotlib")
    result = runner.invoke(
        cli.app,
        [
            "report",
            "tax",
            "--compare",
            str(_compare_json(tmp_path / "compare.json")),
            "--acceptance",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not found" in result.output
    assert "Not run yet" in (tmp_path / "report" / "index.md").read_text(encoding="utf-8")


# configs and data -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(Path("configs").glob("*.yaml")))
def test_every_example_config_loads(path):
    load_config(path)


@pytest.mark.slow
def test_downloaded_flores_files_are_parallel_and_clean():
    urd, eng = Path("data/flores_urd.txt"), Path("data/flores_eng.txt")
    if not (urd.exists() and eng.exists()):
        pytest.skip("run `zubaan data flores` first")
    u = urd.read_bytes().decode("utf-8").split("\n")
    e = eng.read_bytes().decode("utf-8").split("\n")
    assert u[-1] == e[-1] == "" and len(u) == len(e) == 1013
    assert b"\r" not in urd.read_bytes() + eng.read_bytes()
    assert all(line.strip() for line in u[:-1] + e[:-1])
