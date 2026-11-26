import threading
import time

import pytest

from zubaan.tune.client import (
    QWEN3_THINKING,
    ChatClient,
    Reply,
    Sampling,
    chat_body,
    parallel_map,
    parse_reply,
)

USER = [{"role": "user", "content": "سلام"}]


def response(content, reasoning=None, finish="stop", completion=5):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {
        "choices": [{"message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": completion},
    }


def test_body_carries_sampling_seed_and_thinking_switch():
    body = chat_body("m", USER, QWEN3_THINKING, enable_thinking=True, seed=7)
    assert body["model"] == "m" and body["messages"] == USER
    assert (body["temperature"], body["top_p"], body["top_k"], body["min_p"]) == (0.6, 0.95, 20, 0)
    assert body["max_tokens"] == 3072 and body["seed"] == 7
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_body_leaves_out_what_is_not_set():
    body = chat_body("m", USER, Sampling(temperature=0.0))
    for key in ("top_k", "min_p", "seed", "chat_template_kwargs"):
        assert key not in body


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "bot", "content": "x"}],
        [{"role": "user", "content": 3}],
        USER + [{"role": "assistant", "content": "x"}],
    ],
)
def test_bad_chats_are_refused(messages):
    with pytest.raises(ValueError):
        chat_body("m", messages, QWEN3_THINKING)


def test_bad_sampling_is_refused():
    with pytest.raises(ValueError):
        Sampling(temperature=-1)
    with pytest.raises(ValueError):
        Sampling(temperature=0.5, top_p=0)
    with pytest.raises(ValueError):
        Sampling(temperature=0.5, max_tokens=0)


def test_raw_reply_keeps_its_think_block():
    r = parse_reply(response("<think>\nhmm\n</think>\n\nجواب"))
    assert r.reasoning is None and r.text == "<think>\nhmm\n</think>\n\nجواب"
    assert (r.finish_reason, r.prompt_tokens, r.completion_tokens) == ("stop", 3, 5)


def test_split_out_reasoning_is_put_back_in_qwen3_format():
    r = parse_reply(response("جواب", reasoning=" hmm "))
    assert r.text == "<think>\nhmm\n</think>\n\nجواب"


def test_null_content_becomes_empty_text():
    r = parse_reply(response(None))
    assert r == Reply("", None, "stop", 3, 5)


def test_a_non_chat_response_raises():
    with pytest.raises(ValueError, match="not a chat completion"):
        parse_reply({"error": "boom"})


def test_client_posts_to_the_chat_endpoint():
    seen = {}

    def post(url, body, timeout):
        seen.update(url=url, body=body, timeout=timeout)
        return response("ok")

    client = ChatClient("http://host:8000/", "teacher", timeout=5, post=post)
    assert client.chat(USER, QWEN3_THINKING, enable_thinking=False).text == "ok"
    assert seen["url"] == "http://host:8000/v1/chat/completions"
    assert seen["body"]["model"] == "teacher" and seen["timeout"] == 5


def test_parallel_map_keeps_order_and_runs_concurrently():
    active, peak = 0, 0
    lock = threading.Lock()

    def slow(x):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return x * 2

    assert parallel_map(slow, list(range(10)), workers=4) == [x * 2 for x in range(10)]
    assert 1 < peak <= 4
    assert parallel_map(slow, [], workers=4) == []
    with pytest.raises(ValueError):
        parallel_map(slow, [1], workers=0)
