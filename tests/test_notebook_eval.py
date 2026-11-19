"""Notebook 08 (evaluation): structure, the config it writes, and its pinned commit."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from zubaan.config import load_config

NB_DIR = Path(__file__).parent.parent / "notebooks"


@pytest.fixture(scope="module")
def nb8():
    sys.path.insert(0, str(NB_DIR))
    spec = importlib.util.spec_from_file_location(
        "make_08_notebook", NB_DIR / "make_08_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build("abc1234")


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_every_cell_compiles_and_pins_the_commit(nb8):
    cells = code_cells(nb8)
    assert any('COMMIT = "abc1234"' in c for c in cells)
    assert any("import zubaan.adapt.evaluate" in c for c in cells)
    for i, src in enumerate(cells):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")
    json.dumps(nb8)


def test_config_cell_writes_a_valid_stage_2_protocol_config(nb8, tmp_path, monkeypatch):
    cell = next(c for c in code_cells(nb8) if 'Path("evaluate.yaml").write_text' in c)
    monkeypatch.chdir(tmp_path)
    env = {
        "EXTENDED_DIR": Path("/kaggle/input/a"),
        "ADAPTED_DIR": Path("/kaggle/input/b"),
        "Path": Path,
    }
    exec(compile(cell, "c", "exec"), env)
    cfg = load_config(tmp_path / "evaluate.yaml").evaluate
    assert list(cfg.models) == ["base", "extended-untrained", "adapted"]  # base first: reference
    assert cfg.models["base"] == "Qwen/Qwen3-1.7B"
    assert (cfg.untrained, cfg.adapted) == ("extended-untrained", "adapted")
    assert (cfg.quality_texts, cfg.max_chars, cfg.belebele_questions, cfg.dtype) == (
        1000,
        400,
        None,
        "float32",
    )


def test_data_is_checked_against_the_stage_2_heldout_hash_before_scoring(nb8):
    cells = code_cells(nb8)
    check = next(i for i, c in enumerate(cells) if "19c4f1b63b41b8a9" in c)
    rehearsal = next(i for i, c in enumerate(cells) if "REHEARSAL PASSED" in c)
    full = next(
        i for i, c in enumerate(cells) if c.strip().endswith("adapt eval --config evaluate.yaml")
    )
    reproduce = next(i for i, c in enumerate(cells) if "reproduces the Stage 2" in c)
    assert check < rehearsal < full < reproduce
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", "\n".join(cells))
