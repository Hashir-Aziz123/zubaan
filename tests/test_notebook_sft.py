"""Notebook 14 (Stage 4 chat SFT): its own cells run end to end on a tiny model with a fake Hub,
in smoke mode and in main mode, so a mistake shows up here instead of on a Kaggle T4."""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from test_tune_data import make_chat_tokenizer, record  # noqa: E402

import zubaan.adapt.pipeline  # noqa: E402
import zubaan.tune.sft  # noqa: E402
from zubaan.extend.build import METADATA_FILE  # noqa: E402

NB_DIR = Path(__file__).parent.parent / "notebooks"


@pytest.fixture(scope="module")
def nb14():
    spec = importlib.util.spec_from_file_location(
        "make_14_notebook", NB_DIR / "make_14_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(NB_DIR))
    spec.loader.exec_module(mod)
    return mod


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_cells_compile_pin_the_commit_and_need_a_hub_repo(nb14):
    cells = code_cells(nb14.build("abc1234"))
    assert any('COMMIT = "abc1234"' in c for c in cells)
    assert any("uninstall -y -q torchao" in c for c in cells)
    for i, src in enumerate(cells):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")
    with pytest.raises(AssertionError, match="HUB_REPO"):
        exec(nb14.SETTINGS, {})


@pytest.fixture
def kaggle(tmp_path, tiny_base_json, tiny_qwen3, monkeypatch):
    """/kaggle/input with a Stage 3 model and a dataset, a working folder, and a fake Hub."""
    inputs = tmp_path / "input"
    model_dir = inputs / "07-output" / "adapted" / "model"
    tok = make_chat_tokenizer(tiny_base_json)
    tok.save_pretrained(model_dir)
    tiny_qwen3(len(tok), seed=1).save_pretrained(model_dir)
    (model_dir / METADATA_FILE).write_text(
        json.dumps({"first_new_id": len(tok) - 60, "trained": True}), encoding="utf-8"
    )
    data = inputs / "tune-data" / "conversations.jsonl"
    data.parent.mkdir(parents=True)
    recs = [record(i, i % 3 != 0, passage=f"urd-{i // 2}") for i in range(40)]
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    (data.parent / "curate_report.json").write_text(json.dumps({"slots": {}}), encoding="utf-8")

    hub = tmp_path / "hub"

    def uploader(repo, private, milestone_every, log):
        def up(path, step):
            dst = hub / repo / "last"
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(path, dst)

        return up

    def download(repo, directory, folder="last"):
        src = hub / repo / folder
        if not (src / "meta.json").exists():
            return None
        step = json.loads((src / "meta.json").read_text(encoding="utf-8"))["step"]
        final = directory / f"ckpt-{step:07d}"
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, final)
        return final

    monkeypatch.setattr(zubaan.adapt.pipeline, "make_hub_uploader", uploader)
    monkeypatch.setattr(zubaan.adapt.pipeline, "download_hub_checkpoint", download)
    work = tmp_path / "working"
    work.mkdir()
    monkeypatch.chdir(work)
    return {"inputs": inputs, "work": work, "hub": hub}


def run_cells(nb14, kaggle, mode: str) -> dict:
    ns: dict = {"gpu_name": "cpu"}
    settings = nb14.SETTINGS.replace('MODE = "smoke"', f'MODE = "{mode}"')
    exec(settings.replace('HUB_REPO = ""', 'HUB_REPO = "me/tune-ckpt"'), ns)
    exec(nb14.FIND_INPUTS.replace('Path("/kaggle/input")', f"Path({str(kaggle['inputs'])!r})"), ns)
    exec(nb14.CONFIG.replace('Path("/kaggle/working")', f"Path({str(kaggle['work'])!r})"), ns)
    # Shrink the T4 settings to the tiny model and the CPU, keeping everything else as written.
    tune = yaml.safe_load(Path("tune.yaml").read_text())["tune"]
    assert tune["hub_repo"] == ("me/tune-ckpt" if mode == "main" else "me/tune-ckpt-smoke")
    assert (tune["max_len"], tune["grad_accum"], tune["precision"]) == (4096, 16, "fp16")
    tune.update(
        device="cpu",
        precision="fp32",
        gradient_checkpointing=False,
        max_len=256,
        grad_accum=2,
        epochs=3,
        val_share=0.2,
        lora_rank=4,
        lora_alpha=8,
    )
    Path("tune.yaml").write_text(yaml.safe_dump({"tune": tune}))
    from zubaan.config import load_config

    ns["cfg"] = load_config("tune.yaml").tune
    exec(nb14.SMOKE, ns)
    exec(nb14.MAIN, ns)
    exec(nb14.MERGE, ns)
    return ns


def test_smoke_mode_resumes_from_the_hub_and_reports(nb14, kaggle):
    run_cells(nb14, kaggle, "smoke")
    report = json.loads((kaggle["work"] / "smoke_report.json").read_text(encoding="utf-8"))
    assert [s for s, _ in report["losses"]][-1] == 30
    assert report["total_steps"] >= 30 and report["projected_main_hours"] >= 0
    assert report["packing_max_logit_diff"] < 1e-4
    assert (kaggle["hub"] / "me/tune-ckpt-smoke" / "last" / "meta.json").exists()
    assert not (kaggle["work"] / "tuned").exists()  # main work is gated


def test_main_mode_trains_merges_and_probes(nb14, kaggle, monkeypatch):
    ns = run_cells(nb14, kaggle, "main")
    model_dir = Path(ns["merged"]["model_dir"])
    meta = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert "stage4" in meta and meta["trained"]
    assert not (kaggle["work"] / "tuned" / "checkpoints").exists()
    monkeypatch.setattr(
        zubaan.tune.sft,
        "quick_probe",
        functools.partial(zubaan.tune.sft.quick_probe, max_new_tokens=6, device="cpu"),
    )
    exec(nb14.PROBE, ns)
    probe = json.loads((kaggle["work"] / "probe.json").read_text(encoding="utf-8"))
    assert len(probe) == 8  # 2 Urdu + 2 English prompts, thinking on and off
    assert {p["thinking"] for p in probe} == {True, False}
    assert all("answered" in p and "reply" in p for p in probe)
