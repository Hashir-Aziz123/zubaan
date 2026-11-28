"""Notebook 16 (Stage 4 re-export): structure, model discovery, and the chat check on the Q4_K_M
file with its packing step, run against a fake chat-check result."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

import zubaan.tune.chat_check

NB_DIR = Path(__file__).parent.parent / "notebooks"


@pytest.fixture(scope="module")
def nb16():
    spec = importlib.util.spec_from_file_location(
        "make_16_notebook", NB_DIR / "make_16_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(NB_DIR))
    spec.loader.exec_module(mod)
    return mod


def test_cells_compile_pin_the_commit_and_check_llama_server(nb16):
    nb = nb16.build("abc1234")
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any('COMMIT = "abc1234"' in c for c in cells)
    assert any('"llama-perplexity", "llama-server")' in c for c in cells)
    assert any("import quantized_checks as _q4_checks" in c for c in cells)
    for i, src in enumerate(cells):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")


def test_find_model_picks_the_stage4_folder(nb16, tmp_path):
    for name, meta in [
        ("s3", {"trained": True}),
        ("s4", {"trained": True, "stage4": {"step": 140}}),
    ]:
        d = tmp_path / name / "model"
        d.mkdir(parents=True)
        (d / "zubaan_extension.json").write_text(json.dumps(meta))
        for f in ("tokenizer.json", "config.json", "model.safetensors"):
            (d / f).write_text("{}")
        (d / "tokenizer_config.json").write_text('{"chat_template": "x"}')
    ns: dict = {}
    exec(nb16.FIND_MODEL.replace('Path("/kaggle/input")', f"Path({str(tmp_path)!r})"), ns)
    assert ns["MODEL_DIR"] == tmp_path / "s4" / "model"


def _summary(answered, hit):
    g = {
        "n": 60,
        "answered": answered,
        "answer_rate": answered / 60,
        "hit_limit": hit,
        "loops": 1,
        "right_language": 39,
        "language_due": 39,
        "right_language_rate": 1.0,
    }
    return {m: {"all": dict(g), "urd": dict(g)} for m in ("think", "no_think")}


def test_chat_check_and_pack_cells_run(nb16, tmp_path, monkeypatch):
    work = tmp_path / "working"
    (work / "export").mkdir(parents=True)
    (work / "export" / "qwen3-1.7b-urdu-chat-Q4_K_M.gguf").write_bytes(b"gguf")
    (work / "export" / "export_summary.json").write_text("{}")
    calls = []

    def fake_run(cfg, log=print):
        calls.append(cfg)
        result = {"seconds": 600.0, "summary": _summary(60, 3)}
        cfg.output.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(zubaan.tune.chat_check, "run_from_config", fake_run)
    ns: dict = {"json": json, "WORK": work, "NAME": "qwen3-1.7b-urdu-chat"}
    ns.update(LLAMA_DIR=tmp_path / "llama", BIN_DIR=tmp_path / "bin")
    exec(nb16.CHAT_Q4, ns)
    assert calls[0].gguf.name == "qwen3-1.7b-urdu-chat-Q4_K_M.gguf"
    assert calls[0].label == "stage4_q4"
    assert all(c["passed"] for c in ns["checks"])
    exec(nb16.CHAT_Q4, ns)  # a rerun reads the finished file
    assert len(calls) == 1
    pack = nb16.PACK.replace('"/kaggle/working/', f'{str(work)!r} + "/')
    exec(pack, ns)
    assert (work / "export_results.zip").exists()
    names = {p.name for p in (work / "export_results").iterdir()}
    assert {"export_summary.json", "chat_check_q4.json", "chat_check_q4_checks.md"} <= names
