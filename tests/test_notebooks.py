"""The Kaggle export notebook is generated code: check that it is well formed and that the
config it writes is accepted by the package. The notebook itself only runs on Kaggle."""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from zubaan.config import load_config

SCRIPT = Path(__file__).parent.parent / "notebooks" / "make_03_notebook.py"


@pytest.fixture(scope="module")
def maker():
    spec = importlib.util.spec_from_file_location("make_03_notebook", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def notebook(maker):
    return maker.build("abc1234")


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_structure(notebook):
    assert notebook["nbformat"] == 4
    kinds = {c["cell_type"] for c in notebook["cells"]}
    assert kinds == {"markdown", "code"}
    for c in notebook["cells"]:
        assert isinstance(c["source"], list) and c["source"]


def test_install_cell_pins_the_commit_and_the_export_extra(notebook):
    src = code_cells(notebook)[0]
    assert 'COMMIT = "abc1234"' in src
    assert "zubaan[audit,export]" in src and "--force-reinstall --no-deps" in src


def test_every_code_cell_is_valid_python_once_shell_lines_are_removed(notebook):
    for i, src in enumerate(code_cells(notebook)):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")


def test_config_cell_writes_a_config_the_package_accepts(notebook, tmp_path, monkeypatch):
    cell = next(c for c in code_cells(notebook) if 'Path("export.yaml").write_text' in c)
    monkeypatch.chdir(tmp_path)
    env = {
        "BASE_MODEL": "Qwen/Qwen3-1.7B",
        "N_NEW": 10000,
        "METHOD": "mean",
        "NAME": "qwen3-1.7b-urdu",
        "LLAMA_DIR": Path("/tmp/tools/llama.cpp-b1"),
        "BIN_DIR": Path("/tmp/tools/bin"),
    }
    exec(compile(cell, "config cell", "exec"), env)
    cfg = load_config(tmp_path / "export.yaml")
    assert cfg.extend.n_new_tokens == 10000 and cfg.extend.init_method == "mean"
    assert cfg.extend.device == "cpu" and cfg.extend.model_dtype == "bfloat16"
    assert cfg.export.model_dir == Path("/kaggle/working/extended/model")
    assert cfg.export.model_dir == cfg.extend.output_dir / "model"  # export reads what extend wrote
    assert cfg.export.base_model == cfg.extend.base_model
    assert cfg.export.quants == ["Q4_K_M", "Q8_0"]
    assert cfg.urdu_corpus is not None
    assert yaml.safe_load((tmp_path / "export.yaml").read_text())["export"]["name"] == env["NAME"]


def test_the_notebook_runs_the_documented_commands_in_order(notebook):
    shell = [
        line.strip()
        for src in code_cells(notebook)
        for line in src.splitlines()
        if line.strip().startswith("!zubaan")
    ]
    names = [" ".join(line.split()[:3]) for line in shell]
    assert names.index("!zubaan extend learn") < names.index("!zubaan extend run")
    assert names.index("!zubaan extend run") < names.index("!zubaan export gguf")
    assert names.index("!zubaan export gguf") < names.index("!zubaan export ollama")


def test_the_vllm_check_is_off_by_default_and_the_output_stays_under_the_limit(notebook):
    joined = "\n".join(code_cells(notebook))
    assert "RUN_VLLM_CHECK = False" in joined
    assert "19e9" in joined  # Kaggle's output limit is 20 GB
    assert "*-F16.gguf" in joined  # the intermediate is deleted


# notebook 04: the vLLM check ------------------------------------------------------------------


SCRIPT4 = Path(__file__).parent.parent / "notebooks" / "make_04_notebook.py"


@pytest.fixture(scope="module")
def notebook4():
    spec = importlib.util.spec_from_file_location("make_04_notebook", SCRIPT4)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build("abc1234")


def test_notebook4_pins_the_commit_and_every_cell_compiles(notebook4):
    cells = code_cells(notebook4)
    assert 'COMMIT = "abc1234"' in cells[0]
    for i, src in enumerate(cells):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")


def test_notebook4_refuses_gpus_vllm_cannot_use(notebook4):
    cell = next(c for c in code_cells(notebook4) if "nvidia-smi" in c)
    assert "compute_cap" in cell and ">= 7.0" in cell and "P100" in cell


def test_notebook4_installs_vllm_in_its_own_environment(notebook4):
    cell = next(c for c in code_cells(notebook4) if "uv pip install" in c)
    assert "/tmp/vllm-env" in cell and "--python 3.12" in cell
    assert "pip install -q vllm" not in "\n".join(code_cells(notebook4))  # never into Kaggle's env


def test_notebook4_finds_the_model_by_its_marker_file(notebook4, tmp_path):
    cell = next(c for c in code_cells(notebook4) if "zubaan_extension.json" in c)
    inputs = tmp_path / "input" / "nb" / "extended" / "model"
    inputs.mkdir(parents=True)
    (inputs / "zubaan_extension.json").write_text('{"trained": false}', encoding="utf-8")
    (inputs / "model.safetensors").write_bytes(b"x")
    (inputs / "tokenizer.json").write_text("{}", encoding="utf-8")
    (inputs / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert "model_with_template" in cell  # fallback for exports that lack the chat template
    env = {"json": __import__("json")}
    exec(compile(cell.replace('"/kaggle/input"', repr(str(tmp_path / "input"))), "c", "exec"), env)
    assert env["MODEL_DIR"] == inputs
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AssertionError, match="Add Input"):
        exec(compile(cell.replace('"/kaggle/input"', repr(str(empty))), "c", "exec"), dict(env))


def test_notebook4_generates_the_vllm_config_with_half_precision(notebook4, tmp_path):
    cell = next(c for c in code_cells(notebook4) if "write_vllm_files" in c)
    model = tmp_path / "model"
    model.mkdir()
    env = {"Path": Path, "MODEL_DIR": model, "NAME": "qwen3-urdu", "MAX_MODEL_LEN": 4096}
    exec(compile(cell.replace('"/kaggle/working"', repr(str(tmp_path))), "c", "exec"), env)
    config = (tmp_path / "vllm" / "vllm.yaml").read_text(encoding="utf-8")
    assert "dtype: half" in config and "served-model-name: qwen3-urdu" in config
    assert str(model) in (tmp_path / "vllm" / "serve_vllm.sh").read_text(encoding="utf-8")


def test_notebook4_checks_tokenization_against_the_extended_tokenizer_json(notebook4):
    cell = next(c for c in code_cells(notebook4) if "/tokenize" in c)
    assert "tokenizer.json" in cell and "assert not mismatches" in cell
    assert "vllm_check.json" in cell


# notebook 05: stage 3 adapt -------------------------------------------------------------------


SCRIPT5 = Path(__file__).parent.parent / "notebooks" / "make_05_notebook.py"


@pytest.fixture(scope="module")
def notebook5():
    spec = importlib.util.spec_from_file_location("make_05_notebook", SCRIPT5)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build("abc1234")


def test_notebook5_pins_the_commit_and_every_cell_compiles(notebook5):
    cells = code_cells(notebook5)
    assert any('COMMIT = "abc1234"' in c for c in cells)
    for i, src in enumerate(cells):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")


def _run_config_cell(notebook5, tmp_path, monkeypatch, mode, hub):
    cell = next(c for c in code_cells(notebook5) if 'Path("adapt.yaml").write_text' in c)
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    env = {
        "MODE": mode,
        "HUB_REPO": hub,
        "MICRO_BATCH": 2,
        "GRAD_CHECKPOINT": True,
        "PHASE_A_STEPS": 100,
        "PHASE_B_STEPS": 700,
        "MODEL_DIR": Path("/kaggle/input/m"),
    }
    cell = cell.replace('Path("/kaggle/working")', f"Path({str(work)!r})")
    exec(compile(cell, "config cell", "exec"), env)
    return load_config(tmp_path / "adapt.yaml")


def test_notebook5_main_config_matches_the_decisions(notebook5, tmp_path, monkeypatch):
    cfg = _run_config_cell(notebook5, tmp_path, monkeypatch, "main", "me/ckpt")
    a = cfg.adapt
    assert (a.lora_rank, a.lora_alpha, a.precision, a.device) == (64, 128, "fp16", "cuda")
    assert (a.phase_a_steps, a.phase_b_steps, a.grad_accum) == (100, 700, 16)
    assert a.english_replay_ratio == 0.1 and a.block_size == 1024
    assert a.hub_repo == "me/ckpt" and a.hub_private is True
    assert (a.checkpoint_every, a.milestone_every, a.gradient_checkpointing) == (50, 200, True)
    assert a.output_dir.name == "adapted"
    assert cfg.urdu_corpus is not None  # 'zubaan data urdu --config adapt.yaml' needs this section


def test_notebook5_smoke_never_shares_a_hub_repo_with_the_main_run(
    notebook5, tmp_path, monkeypatch
):
    smoke = _run_config_cell(notebook5, tmp_path, monkeypatch, "smoke", "me/ckpt").adapt
    assert smoke.hub_repo == "me/ckpt-smoke" and smoke.output_dir.name == "adapt_smoke"
    assert smoke.phase_a_steps + smoke.phase_b_steps == 30 and smoke.checkpoint_every == 10
    assert smoke.milestone_every == 10
    assert _run_config_cell(notebook5, tmp_path, monkeypatch, "smoke", "").adapt.hub_repo is None


def test_notebook5_has_no_secret_and_main_work_is_gated(notebook5):
    joined = "\n".join(code_cells(notebook5))
    assert "hf_" not in joined  # no token in code
    assert 'get_secret("HF_TOKEN")' in joined
    assert 'HUB_REPO = ""' in joined  # nothing is pushed unless the user sets a repo
    for marker in ("run_adapt(cfg, log=print)\nprint({k", "!zubaan adapt merge"):
        cell = next(c for c in code_cells(notebook5) if marker in c.replace("    ", ""))
        assert cell.startswith("if MODE == ")


def test_notebook5_prepares_data_before_it_trains(notebook5):
    cells = code_cells(notebook5)
    prepare = next(i for i, c in enumerate(cells) if "!zubaan adapt prepare" in c)
    check = next(i for i, c in enumerate(cells) if "5 fp16 steps" in c)
    smoke = next(i for i, c in enumerate(cells) if "resumed_from_step" in c and "max_steps=20" in c)
    assert prepare < check < smoke


def test_notebook5_removes_the_torchao_that_kaggle_ships_before_peft_runs(notebook5):
    cells = code_cells(notebook5)
    install = next(i for i, c in enumerate(cells) if 'COMMIT = "abc1234"' in c)
    assert "pip uninstall -y -q torchao" in cells[install]
    first_peft = next(i for i, c in enumerate(cells) if "import peft" in c or "run_adapt" in c)
    assert install < first_peft


def test_notebook5_check_reads_the_log_file_not_a_key_run_adapt_does_not_return(notebook5):
    cell = next(c for c in code_cells(notebook5) if "5 fp16 steps" in c)
    assert "train_log.jsonl" in cell and 'res["history"]' not in cell
    from zubaan.adapt.pipeline import run_adapt  # noqa: F401  (the cell imports it)


def test_notebook5_micro_batch_and_checkpointing_settings_reach_the_config(
    notebook5, tmp_path, monkeypatch
):
    cell = next(c for c in code_cells(notebook5) if 'Path("adapt.yaml").write_text' in c)
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "MODE": "main",
        "HUB_REPO": "",
        "MICRO_BATCH": 1,
        "GRAD_CHECKPOINT": False,
        "PHASE_A_STEPS": 50,
        "PHASE_B_STEPS": 300,
        "MODEL_DIR": Path("/kaggle/input/m"),
    }
    exec(compile(cell.replace('Path("/kaggle/working")', f"Path({str(work)!r})"), "c", "exec"), env)
    a = load_config(tmp_path / "adapt.yaml").adapt
    assert (a.micro_batch_size, a.grad_accum, a.gradient_checkpointing) == (1, 32, False)
    assert a.phase_a_steps + a.phase_b_steps == 350


def test_notebook5_speed_trial_picks_the_fastest_setting_that_fits(
    notebook5, tmp_path, monkeypatch
):
    import json
    import subprocess

    cell = next(c for c in code_cells(notebook5) if "VARIANTS = [" in c)
    assert cell.startswith("if MODE == 'speed'")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adapt.yaml").write_text(yaml.safe_dump({"adapt": {"model_path": "m"}}))
    speeds = {(2, True): 600, (1, False): 900, (2, False): 800}

    def fake_run(cmd, capture_output, text):
        conf = yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())["adapt"]
        key = (conf["micro_batch_size"], conf["gradient_checkpointing"])
        if key not in speeds:  # micro batch 4 with checkpointing: pretend it ran out of memory
            return subprocess.CompletedProcess(
                cmd, 1, "", "torch.OutOfMemoryError: CUDA out of memory"
            )
        out = Path(conf["output_dir"])
        out.mkdir(parents=True)
        rows = [
            {
                "step": i,
                "phase": "A" if i <= 2 else "B",
                "loss": 5.0,
                "grad_norm": 1.0,
                "tokens_per_s": speeds[key],
                "gpu_mem_gb": 7.0 + key[0],
            }
            for i in range(1, 9)
        ]
        (out / "train_log.jsonl").write_text(chr(10).join(json.dumps(r) for r in rows))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    env = {"MODE": "speed", "WORK": tmp_path, "json": json, "Path": Path, "gpu_name": "T4"}
    exec(compile(cell, "speed", "exec"), env)
    report = json.loads((tmp_path / "speed_report.json").read_text())
    assert report["best"] == "mb1_ckpt_off"
    failed = next(r for r in report["results"] if r["name"] == "mb4_ckpt_on")
    assert not failed["ok"] and failed["why"] == "out of GPU memory"


def test_notebook5_smoke_checks_the_hub_holds_the_newest_and_the_milestone_copies(notebook5):
    cell = next(c for c in code_cells(notebook5) if "max_steps=20" in c)
    assert "milestones/step-0000010/meta.json" in cell and "last/meta.json" in cell
    assert '"milestones"' in cell  # an old smoke run's copies are cleared first
