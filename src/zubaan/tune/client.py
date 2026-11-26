"""A small client for OpenAI-compatible chat servers (vLLM, llama.cpp's server).

Standard library only, so it works in the core install and in a Kaggle session whose vLLM
lives in a separate Python environment. Requests are sent concurrently so the server can
batch them; results keep the order of the inputs.

Thinking mode is switched per request with ``chat_template_kwargs: {"enable_thinking": ...}``,
which vLLM and llama.cpp's server pass to Qwen3's chat template. When the server runs without
a reasoning parser, the reply's ``content`` holds the raw ``<think>...</think>`` block, which is
what training data needs. With a parser, the reasoning arrives in a separate field and
``Reply.text`` puts it back in Qwen3's format.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class Sampling:
    """Sampling settings for one request. ``temperature`` 0 means greedy."""

    temperature: float
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be at least 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


# Qwen3's model card: thinking mode 0.6 / 0.95 / 20 / 0, non-thinking 0.7 / 0.8 / 20 / 0.
# Greedy decoding is discouraged for thinking mode (it can loop), so the teacher samples.
QWEN3_THINKING = Sampling(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=3072)
QWEN3_NO_THINKING = Sampling(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=1024)


@dataclass(frozen=True)
class Reply:
    """One chat reply. ``reasoning`` is set only when the server split it out."""

    content: str
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None

    @property
    def text(self) -> str:
        """The reply as the model wrote it, with any split-out reasoning put back in front."""
        if self.reasoning is None:
            return self.content
        return f"<think>\n{self.reasoning.strip()}\n</think>\n\n{self.content}"


Post = Callable[[str, dict, float], dict]


def http_post(url: str, body: dict, timeout: float) -> dict:
    """POST JSON and return the decoded JSON answer."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def check_messages(messages: Sequence[dict]) -> None:
    """Raise unless ``messages`` is a non-empty chat that ends with a user turn."""
    if not messages:
        raise ValueError("a chat needs at least one message")
    for m in messages:
        if m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            raise ValueError(f"bad message {m!r}: need a role in {ROLES} and string content")
    if messages[-1]["role"] != "user":
        raise ValueError("the last message must be the user's")


def chat_body(
    model: str,
    messages: Sequence[dict],
    sampling: Sampling,
    enable_thinking: bool | None = None,
    seed: int | None = None,
) -> dict:
    """The JSON body of one ``/v1/chat/completions`` request."""
    check_messages(messages)
    body: dict = {
        "model": model,
        "messages": list(messages),
        "max_tokens": sampling.max_tokens,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
    }
    if sampling.top_k is not None:
        body["top_k"] = sampling.top_k
    if sampling.min_p is not None:
        body["min_p"] = sampling.min_p
    if seed is not None:
        body["seed"] = seed
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return body


def parse_reply(response: dict) -> Reply:
    """Read the first choice of a chat completion response."""
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"not a chat completion response: {str(response)[:300]}") from e
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    usage = response.get("usage") or {}
    return Reply(
        content=message.get("content") or "",
        reasoning=reasoning or None,
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


@dataclass
class ChatClient:
    """Sends chat requests to one served model."""

    base_url: str
    model: str
    timeout: float = 900.0
    post: Post = http_post

    def chat(
        self,
        messages: Sequence[dict],
        sampling: Sampling,
        enable_thinking: bool | None = None,
        seed: int | None = None,
    ) -> Reply:
        body = chat_body(self.model, messages, sampling, enable_thinking, seed)
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        return parse_reply(self.post(url, body, self.timeout))


T = TypeVar("T")
R = TypeVar("R")


def parallel_map(fn: Callable[[T], R], items: Sequence[T], workers: int = 32) -> list[R]:
    """``[fn(x) for x in items]`` with up to ``workers`` calls in flight; order is kept."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(fn, items))
