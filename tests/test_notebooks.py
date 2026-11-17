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
