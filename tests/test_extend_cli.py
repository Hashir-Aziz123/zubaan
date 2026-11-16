import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from zubaan import cli
from zubaan.config import ExtendConfig, UrduCorpusConfig, load_config
from zubaan.extend.merge import assert_only_model_changed, compare_encodings
from zubaan.extend.selection import validate_merges
from zubaan.extend.sweep import load_merges

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path, tiny_base_json, urdu_lines, english_lines):
    """A tiny base tokenizer folder, corpus files and a config that uses them."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text(json.dumps(tiny_base_json), encoding="utf-8")
    (base / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    n = min(len(urdu_lines), len(english_lines))
    (data / "urd.txt").write_text("\n".join(urdu_lines[:n]) + "\n", encoding="utf-8")
    (data / "eng.txt").write_text("\n".join(english_lines[:n]) + "\n", encoding="utf-8")
    (data / "train.txt").write_text("\n".join(urdu_lines) + "\n", encoding="utf-8")
    (data / "held.txt").write_text("\n".join(urdu_lines[:6]) + "\n", encoding="utf-8")
    cfg = {
        "extend": {
            "base_model": str(base),
            "corpus": str(data / "train.txt"),
            "heldout": str(data / "held.txt"),
            "eval_urdu": str(data / "urd.txt"),
            "eval_english": str(data / "eng.txt"),
            "n_new_tokens": 60,
            "sizes": [20, 100],
            "min_count": 2,
            "merges_file": str(tmp_path / "out" / "merges.json"),
            "output_dir": str(tmp_path / "out"),
        }
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path, tmp_path


def test_extend_help_lists_the_stage_2_commands():
    result = runner.invoke(cli.app, ["extend", "--help"])
    assert result.exit_code == 0 and "learn" in result.output and "sweep" in result.output


def test_extend_learn_writes_merges_and_a_valid_extended_tokenizer(workspace, tiny_base_json):
    cfg, root = workspace
    result = runner.invoke(cli.app, ["extend", "learn", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    merges = load_merges(root / "out" / "merges.json")
    assert 60 < len(merges) <= 100
    validate_merges(merges, tiny_base_json["model"]["vocab"].keys())
    out = root / "out" / "tokenizer"
    ext = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    assert_only_model_changed(tiny_base_json, ext)
    assert len(ext["model"]["merges"]) == len(tiny_base_json["model"]["merges"]) + 60
    assert (out / "vocab.json").exists() and (out / "merges.txt").exists()
    assert (out / "tokenizer_config.json").exists()
    eng = (root / "data" / "eng.txt").read_text(encoding="utf-8").splitlines()
    assert compare_encodings(tiny_base_json, ext, eng)["n_different"] == 0


def test_extend_sweep_writes_json_and_prints_the_table(workspace):
    cfg, root = workspace
    assert runner.invoke(cli.app, ["extend", "learn", "--config", str(cfg)]).exit_code == 0
    out = root / "sweep.json"
    result = runner.invoke(cli.app, ["extend", "sweep", "--config", str(cfg), "--out", str(out)])
    assert result.exit_code == 0, result.output
    saved = json.loads(out.read_text(encoding="utf-8"))
    rows = saved["rows"]
    assert [r["requested"] for r in rows] == [0, 20, 100]
    assert saved["n_merges_learned"] >= 60 and "provenance" in saved
    assert all(r["english_different"] == 0 and r["urdu_roundtrip_failures"] == 0 for r in rows)
    assert rows[0]["urdu_tokens"] > rows[-1]["urdu_tokens"]
    assert "not measured" in result.output and "stage 3" in result.output
    assert "heldout_tokens_per_word" in rows[0]


def test_extend_sweep_needs_learn_first(workspace):
    cfg, _ = workspace
    result = runner.invoke(cli.app, ["extend", "sweep", "--config", str(cfg)])
    assert result.exit_code != 0 and "extend learn" in result.output


def test_extend_requires_an_extend_section(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("audit:\n  tokenizer: x\n  corpus: y\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["extend", "learn", "--config", str(cfg)])
    assert result.exit_code != 0 and "no 'extend' section" in result.output


def test_base_model_flag_overrides_the_config(workspace, tmp_path):
    cfg, root = workspace
    result = runner.invoke(
        cli.app,
        ["extend", "learn", "--config", str(cfg), "--base-model", str(tmp_path / "missing")],
    )
    assert result.exit_code != 0  # the override is used, and that folder does not exist


def test_data_urdu_builds_the_corpus_from_a_parquet(tmp_path, urdu_lines):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    articles = ["\n".join(urdu_lines[i : i + 5]) for i in range(0, 30, 5)]
    parquet = tmp_path / "w.parquet"
    pq.write_table(pa.table({"text": articles}), parquet)
    flores = tmp_path / "flores.txt"
    flores.write_text(urdu_lines[0] + "\n", encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "urdu_corpus": {
                    "out_dir": str(tmp_path / "u"),
                    "flores": str(flores),
                    "heldout_percent": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        cli.app, ["data", "urdu", "--config", str(cfg), "--parquet", str(parquet)]
    )
    assert result.exit_code == 0, result.output
    meta = json.loads((tmp_path / "u" / "urdu_meta.json").read_text(encoding="utf-8"))
    assert meta["stats"]["dropped_flores"] == 1 and meta["stats"]["train_lines"] == 29
    assert meta["source"]["license"].startswith("CC BY-SA")
    assert urdu_lines[0] not in (tmp_path / "u" / "urdu_train.txt").read_text(encoding="utf-8")


def test_data_urdu_needs_flores_first(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        yaml.safe_dump({"urdu_corpus": {"flores": str(tmp_path / "nope.txt")}}), encoding="utf-8"
    )
    result = runner.invoke(cli.app, ["data", "urdu", "--config", str(cfg)])
    assert result.exit_code != 0 and "zubaan data flores" in result.output


def test_configs_have_sane_defaults_and_the_example_loads():
    assert UrduCorpusConfig().max_mb == 100.0 and UrduCorpusConfig().heldout_percent == 1.0
    ext = ExtendConfig(base_model="m", corpus=Path("c"))
    assert ext.sizes == [5_000, 10_000, 20_000, 32_000] and ext.min_count == 2
    cfg = load_config("configs/extend-qwen3.yaml")
    assert cfg.extend.base_model == "Qwen/Qwen3-1.7B" and cfg.urdu_corpus.max_mb == 100
    assert cfg.extend.n_new_tokens in cfg.extend.sizes
