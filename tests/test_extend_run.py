"""``zubaan extend run`` on tiny fixtures: learn, build, save and verify in seconds."""

import json

import pytest
import yaml
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from zubaan import cli  # noqa: E402
from zubaan.config import ExtendConfig  # noqa: E402
from zubaan.extend.pipeline import run_extend  # noqa: E402
from zubaan.extend.tokenizer_info import inspect_tokenizer_json  # noqa: E402

runner = CliRunner()


@pytest.fixture
def world(tmp_path, tiny_base_json, urdu_lines, english_lines, tiny_qwen3):
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
    cfg = ExtendConfig(
        base_model=str(base),
        corpus=data / "train.txt",
        eval_urdu=data / "urd.txt",
        eval_english=data / "eng.txt",
        n_new_tokens=50,
        sizes=[20, 80],
        merges_file=tmp_path / "out" / "merges.json",
        output_dir=tmp_path / "out",
    )
    rows = inspect_tokenizer_json(tiny_base_json).total_vocab_size + 8
    return cfg, lambda: tiny_qwen3(rows, seed=2), tmp_path


def test_run_extend_produces_a_verified_model_folder(world, tiny_base_json):
    cfg, factory, tmp = world
    summary = run_extend(cfg, log=lambda s: None, model_factory=factory)
    out = tmp / "out" / "model"
    assert summary["model_dir"] == str(out)
    assert (out / "model.safetensors").exists() and (out / "tokenizer.json").exists()
    assert (tmp / "out" / "tokenizer" / "tokenizer.json").exists()
    total = inspect_tokenizer_json(tiny_base_json).total_vocab_size + 50
    assert summary["verified"]["vocab_size"] == total
    assert summary["verified"]["embedding_rows"] == total
    cost = summary["cost"]
    assert cost["n_new_tokens"] == 50 and cost["english_different"] == 0
    assert cost["urdu_roundtrip_failures"] == 0
    assert summary["metadata"]["trained"] is False


def test_run_extend_reuses_merges_unless_relearn(world):
    cfg, factory, _ = world
    logs: list[str] = []
    run_extend(cfg, log=logs.append, model_factory=factory)
    assert any("Learning" in line for line in logs)
    logs.clear()
    run_extend(cfg, log=logs.append, model_factory=factory)
    assert any("Using" in line for line in logs)
    logs.clear()
    run_extend(cfg, relearn=True, log=logs.append, model_factory=factory)
    assert any("Learning" in line for line in logs)


def test_run_extend_overrides_and_errors(world):
    cfg, factory, _ = world
    summary = run_extend(cfg, n_new=30, method="random", log=lambda s: None, model_factory=factory)
    assert summary["metadata"]["n_new_tokens"] == 30
    assert summary["metadata"]["init_method"] == "random"
    with pytest.raises(ValueError, match="only"):
        run_extend(cfg, n_new=10**6, log=lambda s: None, model_factory=factory)


def test_extend_run_cli_end_to_end(world, monkeypatch):
    cfg, factory, tmp = world
    monkeypatch.setattr(
        "zubaan.audit.acceptance_lm.load_causal_lm", lambda name, dtype, device: factory()
    )
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump({"extend": json.loads(cfg.model_dump_json())}), encoding="utf-8")
    result = runner.invoke(cli.app, ["extend", "run", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert "untrained" in result.output and "English changed: 0" in result.output
    assert (tmp / "out" / "model" / "zubaan_extension.json").exists()

    result = runner.invoke(cli.app, ["extend", "run", "--config", str(path), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["verified"]["urdu_texts"] > 0
