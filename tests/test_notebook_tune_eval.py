"""Notebook 15 (Stage 4 evaluation): structure, model discovery, and the chat-check and verdict
cells end to end against a fake vLLM server."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from zubaan.tune.client import ChatClient, parse_reply

NB_DIR = Path(__file__).parent.parent / "notebooks"


@pytest.fixture(scope="module")
def nb15():
    spec = importlib.util.spec_from_file_location(
        "make_15_notebook", NB_DIR / "make_15_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(NB_DIR))
    spec.loader.exec_module(mod)
    return mod


def test_cells_compile_and_pin_the_commit(nb15):
    nb = nb15.build("abc1234")
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any('COMMIT = "abc1234"' in c for c in cells)
    for i, src in enumerate(cells):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")


def test_find_models_tells_stage3_and_stage4_apart(nb15, tmp_path):
    for name, meta in [
        ("s3", {"trained": True}),
        ("s4", {"trained": True, "stage4": {}}),
        ("s2", {}),
    ]:
        d = tmp_path / name / "model"
        d.mkdir(parents=True)
        (d / "zubaan_extension.json").write_text(json.dumps(meta))
        (d / "tokenizer.json").write_text("{}")
    ns: dict = {"json": json}
    exec(
        nb15.SETTINGS.replace(
            'Path("/kaggle/working/tune_eval")', f"Path({str(tmp_path / 'out')!r})"
        ),
        ns,
    )
    exec(nb15.FIND_MODELS.replace('Path("/kaggle/input")', f"Path({str(tmp_path)!r})"), ns)
    assert ns["MODELS"]["stage3"].endswith(str(Path("s3/model")))
    assert ns["MODELS"]["stage4"].endswith(str(Path("s4/model")))
    assert ns["MODELS"]["base"] == "Qwen/Qwen3-1.7B"


def fake_chat(self, messages, sampling, enable_thinking=None, seed=None):
    urdu = any("؀" <= ch <= "ۿ" for ch in messages[-1]["content"])
    answer = "یہ ایک مختصر جواب ہے جو اردو میں لکھا گیا ہے۔" if urdu else "This is a short answer."
    if self.model == "stage3" and enable_thinking:
        content, finish = "<think>\nThinking without end", "stop"  # the Stage 3 failure
    else:
        content = f"<think>\nok\n</think>\n\n{answer}" if enable_thinking else answer
        finish = "stop"
    return parse_reply(
        {
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    )


def test_chat_and_verdict_cells_run_end_to_end(nb15, tmp_path, monkeypatch):
    monkeypatch.setattr(ChatClient, "chat", fake_chat)
    out = tmp_path / "tune_eval"
    ns: dict = {"json": json, "VLLM_VERSION": "0.0", "gpu_name": "T4"}
    exec(nb15.SETTINGS.replace('Path("/kaggle/working/tune_eval")', f"Path({str(out)!r})"), ns)
    ns["MODELS"] = {"base": "Qwen/Qwen3-1.7B", "stage3": "/s3", "stage4": "/s4"}
    served = []
    ns["start_teacher"] = lambda model, label: served.append(label) or object()
    ns["stop_teacher"] = lambda proc: None
    exec(nb15.CHAT, ns)
    assert served == ["base", "stage3", "stage4"]
    assert ns["chat"]["stage3"]["think"]["all"]["answered"] == 0
    assert ns["chat"]["stage4"]["think"]["all"]["answered"] == 60

    exec(nb15.CHAT, ns)  # a rerun reads the finished files and serves nothing
    assert served == ["base", "stage3", "stage4"]

    def row(urdu_flores):
        return {
            "bpc": {
                "urdu_heldout": {"bits_per_character": 1.486},
                "urdu_flores": {"bits_per_character": urdu_flores},
                "english_flores": {"bits_per_character": 1.066},
            },
            "belebele": {"urd_Arab": {"acc_char": 0.34, "acc_char_ci": [0.31, 0.37]}},
        }

    (out / "tune_quality.json").write_text(
        json.dumps({"variants": {"stage3": row(1.718), "stage4": row(1.73)}})
    )
    verdict = nb15.VERDICT.replace('"/kaggle/working/tune_eval"', repr(str(tmp_path / "tune_eval")))
    exec(verdict, ns)
    checks = json.loads((out / "stage4_checks.json").read_text(encoding="utf-8"))["checks"]
    assert [c["passed"] for c in checks] == [True] * 5
    assert (tmp_path / "tune_eval.zip").exists()
