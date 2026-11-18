"""Notebooks 06 (speed trial) and 07 (main run): structure, config and their new cells."""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from zubaan.config import load_config

NB_DIR = Path(__file__).parent.parent / "notebooks"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, NB_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(NB_DIR))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def nb6():
    return _load("make_06_notebook").build("abc1234")


@pytest.fixture(scope="module")
def nb7():
    return _load("make_07_notebook").build("abc1234")


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("fixture", ["nb6", "nb7"])
def test_every_cell_compiles_pins_the_commit_and_removes_torchao(fixture, request):
    cells = code_cells(request.getfixturevalue(fixture))
    assert any('COMMIT = "abc1234"' in c for c in cells)
    assert any("pip uninstall -y -q torchao" in c for c in cells)
    for i, src in enumerate(cells):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")


def _run_config(nb, tmp_path, monkeypatch, **settings):
    cell = next(c for c in code_cells(nb) if 'Path("adapt.yaml").write_text' in c)
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    env = {"MODEL_DIR": Path("/kaggle/input/m"), **settings}
    exec(compile(cell.replace('Path("/kaggle/working")', f"Path({str(work)!r})"), "c", "exec"), env)
    return load_config(tmp_path / "adapt.yaml").adapt


MAIN_SETTINGS = {
    "MICRO_BATCH": 1, "GRAD_CHECKPOINT": False, "PHASE_A_STEPS": 100, "PHASE_B_STEPS": 700,
    "OUT_NAME": "adapted", "HUB": "me/ckpt",
}  # fmt: skip


def test_main_config_keeps_32_blocks_per_step_and_the_checkpoint_plan(nb7, tmp_path, monkeypatch):
    a = _run_config(nb7, tmp_path, monkeypatch, **MAIN_SETTINGS)
    assert (a.micro_batch_size, a.grad_accum, a.gradient_checkpointing) == (1, 32, False)
    assert (a.checkpoint_every, a.milestone_every, a.eval_every) == (50, 200, 100)
    assert (a.phase_a_steps, a.phase_b_steps) == (100, 700)
    assert (a.lora_rank, a.lora_alpha, a.precision) == (64, 128, "fp16")
    assert a.hub_repo == "me/ckpt" and a.hub_private and a.output_dir.name == "adapted"
    b = _run_config(nb7, tmp_path, monkeypatch, **{**MAIN_SETTINGS, "MICRO_BATCH": 4})
    assert b.micro_batch_size * b.grad_accum == 32


def test_main_notebook_refuses_to_run_without_a_hub_repo_or_with_a_bad_micro_batch(nb7):
    settings = code_cells(nb7)[0]
    with pytest.raises(AssertionError, match="HUB_REPO"):
        exec(settings, {})
    with pytest.raises(AssertionError, match="divide 32"):
        exec(
            settings.replace('HUB_REPO = ""', 'HUB_REPO = "a/b"').replace(
                "MICRO_BATCH = 1 ", "MICRO_BATCH = 3 "
            ),
            {},
        )


def test_main_notebook_checks_the_hub_before_training_and_trains_ungated(nb7):
    cells = code_cells(nb7)
    pre = next(i for i, c in enumerate(cells) if "_preflight" in c)
    train = next(i for i, c in enumerate(cells) if "run_adapt(cfg, log=print)" in c)
    data = next(i for i, c in enumerate(cells) if "!zubaan adapt prepare" in c)
    merge = next(i for i, c in enumerate(cells) if "!zubaan adapt merge" in c)
    assert data < pre < train < merge
    assert "assert api.repo_info(HUB_REPO).private" in cells[pre]
    assert not cells[train].startswith("if ")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", "\n".join(cells))  # no token in code


def test_speed_notebook_tests_the_hub_only_when_asked(nb6, tmp_path, monkeypatch):
    cells = code_cells(nb6)
    test = next(c for c in cells if "make_hub_uploader" in c)
    assert test.startswith("if HUB_REPO:") and '"-test"' in test  # skipped without a repo
    assert "milestones/step-0000200/meta.json" in test and "download_hub_checkpoint" in test
    check = next(i for i, c in enumerate(cells) if "5 fp16 steps" in c)
    speed = next(i for i, c in enumerate(cells) if "VARIANTS = [" in c)
    assert check < speed
    a = _run_config(
        nb6, tmp_path, monkeypatch, MICRO_BATCH=2, GRAD_CHECKPOINT=True, PHASE_A_STEPS=100,
        PHASE_B_STEPS=700, OUT_NAME="adapt_speed", HUB=None, HUB_REPO="",
    )  # fmt: skip
    assert a.hub_repo is None


def test_speed_trial_picks_the_fastest_setting_that_fits(nb6, tmp_path, monkeypatch):
    cell = next(c for c in code_cells(nb6) if "VARIANTS = [" in c)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adapt.yaml").write_text(yaml.safe_dump({"adapt": {"model_path": "m"}}))
    speeds = {(2, True): 600, (1, False): 900, (2, False): 800}

    def fake_run(cmd, capture_output, text):
        conf = yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())["adapt"]
        key = (conf["micro_batch_size"], conf["gradient_checkpointing"])
        if key not in speeds:  # micro batch 4 with checkpointing: pretend it ran out of memory
            return subprocess.CompletedProcess(cmd, 1, "", "OutOfMemoryError: CUDA out of memory")
        out = Path(conf["output_dir"])
        out.mkdir(parents=True)
        rows = [
            {"step": i, "phase": "A" if i <= 2 else "B", "loss": 5.0, "grad_norm": 1.0,
             "tokens_per_s": speeds[key], "gpu_mem_gb": 7.0 + key[0]}
            for i in range(1, 9)
        ]  # fmt: skip
        (out / "train_log.jsonl").write_text(chr(10).join(json.dumps(r) for r in rows))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    env = {"WORK": tmp_path, "json": json, "Path": Path, "yaml": yaml, "gpu_name": "T4"}
    exec(compile(cell, "speed", "exec"), env)
    report = json.loads((tmp_path / "speed_report.json").read_text())
    assert report["best"] == "mb1_ckpt_off"
    failed = next(r for r in report["results"] if r["name"] == "mb4_ckpt_on")
    assert not failed["ok"] and failed["why"] == "out of GPU memory"
    assert not any(p.is_dir() for p in tmp_path.glob("speed_mb*"))  # each trial's files are removed


def test_notebook_05_is_untouched_by_the_new_notebooks():
    text = (NB_DIR / "make_05_notebook.py").read_text(encoding="utf-8")
    assert 'MODE in ("check", "smoke", "main")' in text


def test_main_notebook_uses_the_speed_trial_winner_and_limits_fragmentation(nb7):
    settings = code_cells(nb7)[0]
    assert "MICRO_BATCH = 1 " in settings and "GRAD_CHECKPOINT = False" in settings
    assert "expandable_segments:True" in settings
