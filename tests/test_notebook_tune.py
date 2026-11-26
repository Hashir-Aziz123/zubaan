"""Notebook 12 (Stage 4 teacher trial): structure, and its cells run end to end against a fake
teacher, so a mistake shows up here instead of after an hour on Kaggle."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

import zubaan.adapt.pipeline
from zubaan.tune.client import ChatClient, chat_body, parse_reply

NB_DIR = Path(__file__).parent.parent / "notebooks"


@pytest.fixture(scope="module")
def maker():
    spec = importlib.util.spec_from_file_location(
        "make_12_notebook", NB_DIR / "make_12_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(NB_DIR))
    spec.loader.exec_module(mod)
    return mod


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_cells_compile_and_pin_the_commit(maker):
    cells = code_cells(maker.build("abc1234"))
    assert 'COMMIT = "abc1234"' in cells[0]
    for i, src in enumerate(cells):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")


def fake_chat(self, messages, sampling, enable_thinking=None, seed=None):
    body = chat_body(self.model, messages, sampling, enable_thinking, seed)
    text = body["messages"][-1]["content"]
    if "JSON array" in text:
        n = int(text.split("JSON array of ")[1].split()[0])
        content = json.dumps([f"سوال {i}؟" for i in range(n)], ensure_ascii=False)
    elif enable_thinking:
        content = f"<think>\n{self.model} thinks.\n</think>\n\nلاہور ایک پرانا شہر ہے۔"
    else:
        content = "لاہور ایک پرانا شہر ہے۔"
    return parse_reply(
        {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    )


def test_trial_cells_run_end_to_end(maker, tmp_path, monkeypatch):
    m = maker
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ChatClient, "chat", fake_chat)
    monkeypatch.setattr(zubaan.adapt.pipeline, "belebele_passages", lambda lang: ["بیلیبیلی"])
    (tmp_path / "data/flores_dev").mkdir(parents=True)
    (tmp_path / "data/urdu").mkdir()
    (tmp_path / "data/flores_urd.txt").write_text("ایک جانچ جملہ\n", encoding="utf-8")
    (tmp_path / "data/flores_dev/flores_urd.txt").write_text("دوسرا جملہ\n", encoding="utf-8")
    lines = [" ".join(f"لفظ{i}x{j}" for j in range(70)) for i in range(6)]
    (tmp_path / "data/urdu/urdu_train.txt").write_text("\n".join(lines), encoding="utf-8")
    out = tmp_path / "trial"

    ns: dict = {"subprocess": __import__("subprocess"), "VLLM_VERSION": "0.0", "name": "T4"}
    exec(m.SETTINGS, ns)
    ns["N_PASSAGES"] = 3
    exec(m.PASSAGES.replace('"/kaggle/working/trial"', repr(str(out))), ns)
    exec(m.SERVER, ns)
    started = []
    ns["start_teacher"] = lambda model, served: started.append(served) or object()
    ns["stop_teacher"] = lambda proc: None
    exec(m.RUN, ns)
    exec(m.REPORT, ns)

    assert started == ["qwen3-8b", "qwen3-14b-awq"]
    questions = (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(questions) == 3 * 2 * 2  # passages x formats x per format
    for served in started:
        answers = (out / f"answers_{served}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(answers) == len(questions) * 2  # thinking on and off
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["questions"]["failed"] == 0 and len(summary["groups"]) == 8
    assert summary["groups"]["qwen3-14b-awq | grounded | think"]["think_closed"] == 6
    reading = (out / "reading.md").read_text(encoding="utf-8")
    assert "qwen3-8b, thinking on" in reading and "qwen3-14b-awq, thinking off" in reading


# notebook 13: generation ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def maker13():
    spec = importlib.util.spec_from_file_location(
        "make_13_notebook", NB_DIR / "make_13_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generation_cells_compile_and_pin_the_commit(maker13):
    cells = code_cells(maker13.build("abc1234"))
    assert 'COMMIT = "abc1234"' in cells[0]
    for i, src in enumerate(cells):
        py = "\n".join(re.sub(r"^(\s*)!.*", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")


@pytest.mark.parametrize("shard", [0, 1])
def test_generation_cells_run_end_to_end(maker13, tmp_path, monkeypatch, shard):
    import zubaan.adapt.data
    import zubaan.tune.generate as gen
    from zubaan.tune.generate import MixSlot

    m = maker13
    small = (
        MixSlot("urd", "grounded", True, 8),
        MixSlot("urd", "grounded", False, 4),
        MixSlot("urd", "standalone", False, 4),
        MixSlot("eng", "grounded", True, 4),
        MixSlot("eng", "standalone", False, 4),
    )
    monkeypatch.setattr(gen, "DEFAULT_MIX", small)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ChatClient, "chat", fake_chat)
    monkeypatch.setattr(zubaan.adapt.pipeline, "belebele_passages", lambda lang: ["بیلیبیلی"])

    def fake_replay(articles, out_path, eval_texts, max_mb):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [" ".join(f"word{i}x{j}" for j in range(70)) for i in range(40)]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return {"lines": 40, "source": {}}

    monkeypatch.setattr(zubaan.adapt.data, "build_english_replay", fake_replay)
    monkeypatch.setattr(zubaan.adapt.data, "iter_english_wikipedia", lambda: iter([]))
    (tmp_path / "data/flores_dev").mkdir(parents=True)
    (tmp_path / "data/urdu").mkdir()
    for f in ("data/flores_urd.txt", "data/flores_dev/flores_urd.txt"):
        (tmp_path / f).write_text("ایک جانچ جملہ\n", encoding="utf-8")
    for f in ("data/flores_eng.txt", "data/flores_dev/flores_eng.txt"):
        (tmp_path / f).write_text("a test sentence\n", encoding="utf-8")
    lines = [" ".join(f"لفظ{i}x{j}" for j in range(70)) for i in range(60)]
    (tmp_path / "data/urdu/urdu_train.txt").write_text("\n".join(lines), encoding="utf-8")

    ns: dict = {"subprocess": __import__("subprocess"), "VLLM_VERSION": "0.0", "name": "T4"}
    settings = m.SETTINGS.replace("SHARD = 0", f"SHARD = {shard}")
    exec(settings.replace('Path(f"/kaggle/working/data_shard{SHARD}")', "Path(f'out{SHARD}')"), ns)
    exec(m.ENGLISH, ns)
    exec(m.PASSAGES, ns)
    exec(m.n12.SERVER, ns)
    ns["start_teacher"] = lambda model, served: object()
    ns["stop_teacher"] = lambda proc: None
    exec(m.QUESTIONS, ns)
    exec(m.ANSWERS, ns)
    exec(m.REPORT.replace("/kaggle/working/", ""), ns)

    out = tmp_path / f"out{shard}"
    answers = [
        json.loads(x) for x in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    questions = (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(answers) == len(questions) > 0  # every question answered once
    assert not any(a["format"] == "standalone" and a["thinking"] for a in answers)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["shard"] == shard and summary["answers"] == len(answers)
    assert (tmp_path / f"tune_data_shard{shard}.zip").exists()
