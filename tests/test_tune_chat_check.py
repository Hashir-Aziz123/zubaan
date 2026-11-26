import json
import sys
from pathlib import Path

import pytest
import yaml

from zubaan.config import ChatCheckConfig, load_config
from zubaan.tune import chat_check
from zubaan.tune.chat_check import (
    ChatPrompt,
    check_markdown,
    check_thinking_switch,
    llama_server_command,
    load_prompts,
    run_chat_check,
    running_server,
    score_reply,
    summarize_check,
)
from zubaan.tune.client import ChatClient

URDU = "پاکستان کا دارالحکومت اسلام آباد ہے۔"


def test_packaged_prompt_set_is_valid():
    prompts, meta = load_prompts()
    assert meta["version"] == 1 and meta["n"] == len(prompts) == 60
    assert meta["by_lang"] == {"urd": 40, "eng": 20}
    multi = next(p for p in prompts if p.id == "ur-multi-01")
    assert [m["role"] for m in multi.messages()] == ["user", "assistant", "user"]
    assert next(p for p in prompts if p.id == "ur-translate-01").reply_lang is None
    assert next(p for p in prompts if p.id == "ur-fact-01").reply_lang == "urd"


def write_prompts(tmp_path: Path, items: list[dict]) -> Path:
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({"version": 9, "prompts": items}), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "item, msg",
    [
        ({"lang": "urd", "category": "chat", "text": "x"}, "without an id"),
        ({"id": "a", "lang": "fr", "category": "chat", "text": "x"}, "lang"),
        ({"id": "a", "lang": "urd", "category": "poem", "text": "x"}, "category"),
        ({"id": "a", "lang": "urd", "category": "chat"}, "either text or turns"),
        ({"id": "a", "lang": "urd", "category": "chat", "turns": ["x", "y"]}, "odd"),
        ({"id": "a", "lang": "urd", "category": "chat", "text": "x", "reply_lang": "fr"}, "reply"),
    ],
)
def test_bad_prompts_are_refused(tmp_path, item, msg):
    with pytest.raises(ValueError, match=msg):
        load_prompts(write_prompts(tmp_path, [item]))


def test_duplicate_ids_are_refused(tmp_path):
    item = {"id": "a", "lang": "urd", "category": "chat", "text": "x"}
    with pytest.raises(ValueError, match="duplicate"):
        load_prompts(write_prompts(tmp_path, [item, item]))


def test_score_a_good_urdu_answer_with_thinking():
    s = score_reply(f"<think>\nThe capital is Islamabad.\n</think>\n\n{URDU}", "stop", "urd")
    assert s["has_think"] and s["think_closed"] and s["answered"] and s["complete"]
    assert s["right_language"] is True and s["answer_urdu_share"] == 1.0
    assert not s["hit_limit"] and not s["loop"]


def test_score_the_stage3_failure():
    # Reasoning, then the turn ends: no answer, so the language is not judged.
    s = score_reply("<think>\nThe capital is Islamabad.", "stop", "urd")
    assert s["has_think"] and not s["think_closed"] and not s["answered"]
    assert s["right_language"] is None


def test_score_language_rules():
    assert score_reply("The capital is Islamabad.", "stop", "urd")["right_language"] is False
    assert score_reply("The capital is Islamabad.", "stop", "eng")["right_language"] is True
    assert score_reply(URDU, "stop", "eng")["right_language"] is False
    assert score_reply(URDU, "stop", None)["right_language"] is None
    assert score_reply("200", "stop", "urd")["right_language"] is None  # no letters


def test_score_limit_and_loop():
    looping = "یہ ایک بہت اچھا اور خوبصورت شہر ہے جو بہت پرانا ہے۔ " * 6
    s = score_reply(looping, "length", "urd")
    assert s["hit_limit"] and s["loop"] and s["answered"] and not s["complete"]


def fake_server(reply_for):
    """A post function answering like a chat server; ``reply_for(body)`` gives the text."""

    def post(url, body, timeout):
        text = reply_for(body)
        if isinstance(text, Exception):
            raise text
        return {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9},
        }

    return post


PROMPTS = [
    ChatPrompt("u1", "urd", "factual", ("سوال؟",), "urd"),
    ChatPrompt("u2", "urd", "translate", ("ترجمہ کریں",), None),
    ChatPrompt("e1", "eng", "factual", ("Question?",), "eng"),
]


def stage3_like(body):
    """Thinking on: reasoning only (Urdu) or a full answer (English). Off: always answers."""
    thinking = body["chat_template_kwargs"]["enable_thinking"]
    english = body["messages"][-1]["content"] == "Question?"
    if english:
        return "<think>\nok\n</think>\n\nAn answer." if thinking else "An answer."
    if body["messages"][-1]["content"] == "ترجمہ کریں":
        return TimeoutError("slow") if thinking else "Translation."
    return "<think>\nreasoning only" if thinking else URDU


def test_run_and_summarize():
    client = ChatClient("http://x", "m", post=fake_server(stage3_like))
    records = run_chat_check(client, PROMPTS, "stage3", log=lambda s: None)
    assert [(r["prompt_id"], r["thinking"]) for r in records][:3] == [
        ("u1", True),
        ("u2", True),
        ("e1", True),
    ]
    s = summarize_check(records)
    think, off = s["think"], s["no_think"]
    assert think["all"]["n"] == 3 and think["all"]["errors"] == 1
    assert think["all"]["answered"] == 1 and think["all"]["answer_rate"] == 0.3333
    assert think["urd"]["answered"] == 0 and think["eng"]["answered"] == 1
    assert think["urd"]["language_due"] == 1 and think["urd"]["right_language_rate"] == 0.0
    assert think["all"]["reasoned"] == 2 and think["all"]["reasoning_closed"] == 1
    assert off["all"]["answered"] == 3 and off["urd"]["right_language_rate"] == 1.0
    assert off["all"]["completion_tokens_median"] == 9

    md = check_markdown(records, PROMPTS)
    assert "NO ANSWER" in md and "reasoning (never closed)" in md and "request failed" in md


def test_requests_are_greedy_and_switch_thinking():
    bodies = []

    def record(body):
        bodies.append(body)
        return "ok"

    client = ChatClient("http://x", "m", post=fake_server(record))
    run_chat_check(client, PROMPTS[:1], "x", max_tokens_think=11, max_tokens_no_think=7, log=print)
    assert [b["temperature"] for b in bodies] == [0.0, 0.0]
    assert [b["max_tokens"] for b in bodies] == [11, 7]
    assert [b["chat_template_kwargs"]["enable_thinking"] for b in bodies] == [True, False]


def test_llama_server_keeps_raw_reasoning():
    cmd = llama_server_command(Path("llama-server"), Path("m.gguf"), "base", 8081, threads=4)
    assert cmd[cmd.index("--reasoning-format") + 1] == "none"
    assert "--jinja" in cmd and cmd[cmd.index("-np") + 1] == "1"
    assert cmd[cmd.index("--port") + 1] == "8081" and cmd[-2:] == ["-t", "4"]
    with pytest.raises(ValueError):
        llama_server_command(Path("s"), Path("m"), "a", ctx_size=10)


def test_running_server_starts_waits_and_stops(tmp_path):
    cmd = [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)"]
    calls = []

    def healthy(url):
        calls.append(url)
        return len(calls) >= 2

    with running_server(cmd, "http://h", tmp_path / "s.log", healthy=healthy) as proc:
        assert proc.poll() is None
    assert proc.poll() is not None and calls == ["http://h", "http://h"]


def test_running_server_reports_an_early_exit(tmp_path):
    cmd = [sys.executable, "-c", "raise SystemExit(3)"]
    server = running_server(cmd, "http://h", tmp_path / "s.log", healthy=lambda u: False)
    with pytest.raises(RuntimeError, match="exited with code 3"), server:
        pass


def test_run_from_config_with_an_external_server(tmp_path, monkeypatch):
    monkeypatch.setattr(
        chat_check,
        "ChatClient",
        lambda url, model: ChatClient(url, model, post=fake_server(stage3_like)),
    )
    cfg = ChatCheckConfig(label="stage3", max_prompts=5, output=tmp_path / "c.json")
    result = chat_check.run_from_config(cfg, log=lambda s: None)
    assert result["prompts"]["used"] == 5 and result["server"]["runtime"] == "external"
    saved = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert saved["summary"]["no_think"]["all"]["n"] == 5 and len(saved["records"]) == 10
    assert (tmp_path / "c.md").read_text(encoding="utf-8").startswith("# Chat check replies")


def test_example_config_is_accepted():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "chat-check.yaml").chat_check
    assert cfg.label == "base" and cfg.gguf.name == "Qwen3-1.7B-Q8_0.gguf"


def test_an_empty_think_block_is_not_reasoning():
    # llama-server echoes the template's pre-filled empty block with thinking off; vLLM does not.
    a = score_reply(f"<think>\n\n</think>\n\n{URDU}", "stop", "urd")
    b = score_reply(URDU, "stop", "urd")
    assert a == b and not a["has_think"] and a["answered"]


def template_server(off_prompt: str, on_prompt: str):
    def post(url, body, timeout):
        assert url == "http://h/apply-template"
        on = body["chat_template_kwargs"]["enable_thinking"]
        return {"prompt": on_prompt if on else off_prompt}

    return post


QWEN3_ON = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"


def test_thinking_switch_accepts_qwen3s_template():
    off = QWEN3_ON + "<think>\n\n</think>\n\n"
    assert check_thinking_switch("http://h", post=template_server(off, QWEN3_ON)) == off


def test_thinking_switch_refuses_plain_chatml():
    # The Stage 2 0.6B GGUF has no chat template: both modes render the same prompt.
    with pytest.raises(ValueError, match="ignores enable_thinking"):
        check_thinking_switch("http://h", post=template_server(QWEN3_ON, QWEN3_ON))


def test_an_interrupted_run_resumes_without_asking_again(tmp_path, monkeypatch):
    asked = []

    def counting(body):
        asked.append(body["messages"][-1]["content"])
        if len(asked) == 4:
            raise KeyboardInterrupt  # the run is stopped part way
        return stage3_like(body)

    monkeypatch.setattr(
        chat_check,
        "ChatClient",
        lambda url, model: ChatClient(url, model, post=fake_server(counting)),
    )
    cfg = ChatCheckConfig(label="stage3", max_prompts=3, output=tmp_path / "c.json")
    with pytest.raises(KeyboardInterrupt):
        chat_check.run_from_config(cfg, log=lambda s: None)
    partial = tmp_path / "c.partial.jsonl"
    saved = len(partial.read_text(encoding="utf-8").splitlines()) - 1  # minus the header
    assert 3 <= saved < 6  # the pool may finish one queued request after the stop
    asked.clear()
    result = chat_check.run_from_config(cfg, log=lambda s: None)
    assert len(asked) == 6 - saved  # only what was not saved is asked again
    assert len(result["records"]) == 6 and not partial.exists()


def test_a_progress_file_from_another_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        chat_check,
        "ChatClient",
        lambda url, model: ChatClient(url, model, post=fake_server(stage3_like)),
    )
    (tmp_path / "c.partial.jsonl").write_text('{"label": "other"}\n', encoding="utf-8")
    cfg = ChatCheckConfig(label="stage3", max_prompts=1, output=tmp_path / "c.json")
    with pytest.raises(ValueError, match="different run"):
        chat_check.run_from_config(cfg, log=lambda s: None)
