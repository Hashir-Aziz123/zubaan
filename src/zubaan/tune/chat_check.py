"""Chat-behavior check (Stage 4, chunk 1): does a chat model still answer, and in which language?

Stage 3 passed every likelihood check and still broke thinking mode, which was found by hand.
This check makes that failure measurable. A fixed, hand-written prompt set is sent to a served
model twice, with thinking on and off, using greedy decoding so the result is reproducible on
one runtime. Per reply it records:

* ``answered``: the reply has an answer, that is, any ``<think>`` block is closed and text
  follows it. The Stage 3 failure is a reply that is only reasoning.
* ``complete``: answered and the model stopped by itself (not at the token limit).
* ``hit_limit``: generation stopped at ``max_tokens``, usually a loop or endless reasoning.
* ``loop``: some 8-word run occurs more than twice in the reply.
* ``right_language``: the answer is in the language the prompt asks for. Urdu means at least
  90% of its letters are Urdu script; English means at most 10%. An answer with no letters
  (only digits) is not judged.

It works with any OpenAI-compatible server (vLLM, llama.cpp's ``llama-server``). It measures
behavior, not quality: a fluent wrong answer counts as answered. Quality is read by a person.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from zubaan.config import ChatCheckConfig
from zubaan.export.check import urdu_letter_share
from zubaan.export.gguf import find_llama_cpp
from zubaan.provenance import run_metadata, write_json
from zubaan.tune.client import ChatClient, Post, Sampling, http_post, parallel_map
from zubaan.tune.filters import has_loop, split_reply

PROMPTS_FILE = Path(__file__).parent / "prompts" / "chat_check_v1.yaml"
LANGS = ("urd", "eng")
CATEGORIES = ("factual", "explain", "write", "reason", "chat", "translate", "multi")
URDU_MIN_SHARE = 0.9
ENGLISH_MAX_SHARE = 0.1


@dataclass(frozen=True)
class ChatPrompt:
    """One prompt: a single user turn or a short conversation that ends with the user."""

    id: str
    lang: str
    category: str
    turns: tuple[str, ...]
    reply_lang: str | None
    note: str = ""

    def messages(self) -> list[dict]:
        roles = ("user", "assistant")
        return [{"role": roles[i % 2], "content": t} for i, t in enumerate(self.turns)]


def _prompt(item: dict) -> ChatPrompt:
    pid = item.get("id")
    if not pid:
        raise ValueError(f"prompt without an id: {item!r}")
    if item.get("lang") not in LANGS:
        raise ValueError(f"{pid}: lang must be one of {LANGS}")
    if item.get("category") not in CATEGORIES:
        raise ValueError(f"{pid}: category must be one of {CATEGORIES}")
    if ("text" in item) == ("turns" in item):
        raise ValueError(f"{pid}: give either text or turns")
    turns = (item["text"],) if "text" in item else tuple(item["turns"])
    if len(turns) % 2 == 0 or not all(isinstance(t, str) and t.strip() for t in turns):
        raise ValueError(f"{pid}: turns must be non-empty text, odd in number, ending with user")
    reply_lang = item.get("reply_lang", item["lang"])
    if reply_lang is not None and reply_lang not in LANGS:
        raise ValueError(f"{pid}: reply_lang must be one of {LANGS} or null")
    return ChatPrompt(
        pid, item["lang"], item["category"], turns, reply_lang, item.get("note", "") or ""
    )


def load_prompts(path: Path | None = None) -> tuple[list[ChatPrompt], dict]:
    """The prompt set and a description of it (file, version, sha256, counts)."""
    path = path or PROMPTS_FILE
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict) or not data.get("prompts"):
        raise ValueError(f"{path} has no prompts")
    prompts = [_prompt(x) for x in data["prompts"]]
    ids = [p.id for p in prompts]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate prompt ids: {dupes}")
    meta = {
        "file": path.name,
        "version": data.get("version"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n": len(prompts),
        "by_lang": {lang: sum(p.lang == lang for p in prompts) for lang in LANGS},
    }
    return prompts, meta


def score_reply(text: str, finish_reason: str | None, reply_lang: str | None) -> dict:
    """The behavior flags of one reply (see the module docstring)."""
    parts = split_reply(text)
    answered = parts.closed and bool(parts.answer)
    share = urdu_letter_share(parts.answer) if parts.answer else None
    if reply_lang is None or not answered or share is None:
        right_language = None
    elif reply_lang == "urd":
        right_language = share >= URDU_MIN_SHARE
    else:
        right_language = share <= ENGLISH_MAX_SHARE
    # Only non-empty reasoning counts. With thinking off, Qwen3's template pre-fills an empty
    # block; llama-server echoes it into the reply and vLLM does not, so an empty block must
    # score the same as none for the two runtimes to agree.
    reasoned = bool(parts.think)
    return {
        "has_think": reasoned,
        "think_closed": reasoned and parts.closed,
        "answered": answered,
        "complete": answered and finish_reason == "stop",
        "hit_limit": finish_reason == "length",
        "loop": has_loop(text),
        "answer_urdu_share": share,
        "right_language": right_language,
    }


def greedy(max_tokens: int) -> Sampling:
    return Sampling(temperature=0.0, max_tokens=max_tokens)


def run_chat_check(
    client: ChatClient,
    prompts: Sequence[ChatPrompt],
    label: str,
    thinking_modes: Sequence[bool] = (True, False),
    max_tokens_think: int = 2048,
    max_tokens_no_think: int = 768,
    workers: int = 1,
    log: Callable[[str], None] = print,
    done: dict[tuple[str, bool], dict] | None = None,
    on_record: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Send every prompt in every thinking mode; one record per pair, in input order.

    A failed request becomes a record with ``error`` set and is counted, not retried.
    ``done`` maps ``(prompt_id, thinking)`` to records of an earlier, interrupted run, which
    are reused without a request. ``on_record`` is called with each new record as it arrives
    (from worker threads), so a caller can save progress.
    """
    jobs = [(p, t) for t in thinking_modes for p in prompts]
    done = done or {}
    counter = itertools.count(1)

    def ask(job: tuple[ChatPrompt, bool]) -> dict:
        p, thinking = job
        if (p.id, thinking) in done:
            next(counter)
            return done[(p.id, thinking)]
        rec = {
            "label": label,
            "prompt_id": p.id,
            "lang": p.lang,
            "category": p.category,
            "reply_lang": p.reply_lang,
            "thinking": thinking,
            "reply": None,
            "finish_reason": None,
            "completion_tokens": None,
            "seconds": None,
            "error": None,
        }
        t0 = time.perf_counter()
        try:
            reply = client.chat(
                p.messages(),
                greedy(max_tokens_think if thinking else max_tokens_no_think),
                enable_thinking=thinking,
            )
        except Exception as e:  # counted in the summary
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        else:
            rec.update(
                reply=reply.text,
                finish_reason=reply.finish_reason,
                completion_tokens=reply.completion_tokens,
                seconds=round(time.perf_counter() - t0, 2),
                **score_reply(reply.text, reply.finish_reason, p.reply_lang),
            )
        mode = "think" if thinking else "no think"
        status = rec["error"] or (
            f"answered={rec['answered']} finish={rec['finish_reason']} "
            f"tokens={rec['completion_tokens']} {rec['seconds']} s"
        )
        log(f"[{next(counter)}/{len(jobs)}] {label} {p.id} ({mode}): {status}")
        if on_record is not None:
            on_record(rec)
        return rec

    return parallel_map(ask, jobs, workers)


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def summarize_group(records: Sequence[dict]) -> dict:
    """Counts and rates for a group of records (rates are over all prompts in the group)."""
    ok = [r for r in records if r["error"] is None]
    judged = [r for r in ok if r["right_language"] is not None]
    lang_due = [r for r in records if r["reply_lang"] is not None]
    tokens = [r["completion_tokens"] for r in ok if r["completion_tokens"] is not None]
    n = len(records)
    answered = sum(r["answered"] for r in ok)
    right = sum(bool(r["right_language"]) for r in judged)
    return {
        "n": n,
        "errors": n - len(ok),
        "answered": answered,
        "answer_rate": _rate(answered, n),
        "complete": sum(r["complete"] for r in ok),
        "reasoned": sum(r["has_think"] for r in ok),
        "reasoning_closed": sum(r["think_closed"] for r in ok),
        "hit_limit": sum(r["hit_limit"] for r in ok),
        "loops": sum(r["loop"] for r in ok),
        "language_due": len(lang_due),
        "language_judged": len(judged),
        "right_language": right,
        # Over every prompt that asks for a language: an unanswered prompt counts as a miss.
        "right_language_rate": _rate(right, len(lang_due)),
        "completion_tokens_median": statistics.median(tokens) if tokens else None,
    }


def summarize_check(records: Sequence[dict]) -> dict:
    """Per thinking mode: all prompts, then Urdu and English prompts separately."""
    out: dict = {}
    for thinking, mode in ((True, "think"), (False, "no_think")):
        recs = [r for r in records if r["thinking"] == thinking]
        if not recs:
            continue
        out[mode] = {"all": summarize_group(recs)}
        for lang in LANGS:
            sub = [r for r in recs if r["lang"] == lang]
            if sub:
                out[mode][lang] = summarize_group(sub)
    return out


def check_markdown(records: Sequence[dict], prompts: Sequence[ChatPrompt]) -> str:
    """Every reply for reading, grouped by prompt; reasoning folded in ``<details>``."""
    by_id = {p.id: p for p in prompts}
    labels = sorted({r["label"] for r in records})
    out = [f"# Chat check replies: {', '.join(labels)}", ""]
    for p in prompts:
        out += [f"## {p.id} ({p.lang}, {p.category})", ""]
        if p.note:
            out += [f"*{p.note}*", ""]
        for i, t in enumerate(by_id[p.id].turns):
            out += [f"**{'User' if i % 2 == 0 else 'Assistant'}:** {t}", ""]
        for r in (r for r in records if r["prompt_id"] == p.id):
            mode = "thinking on" if r["thinking"] else "thinking off"
            head = f"### {r['label']}, {mode}"
            if r["error"] is not None:
                out += [head, "", f"request failed: {r['error']}", ""]
                continue
            flags = [f"{r['completion_tokens']} tokens", f"finish: {r['finish_reason']}"]
            if not r["answered"]:
                flags.append("NO ANSWER")
            if r["loop"]:
                flags.append("LOOP")
            if r["right_language"] is False:
                flags.append("WRONG LANGUAGE")
            out += [f"{head} ({', '.join(flags)})", ""]
            parts = split_reply(r["reply"])
            if parts.think is not None:
                state = "" if parts.closed else " (never closed)"
                out += [f"<details><summary>reasoning{state}</summary>", "", parts.think, ""]
                out += ["</details>", ""]
            out += [parts.answer or "*(no answer)*", ""]
    return "\n".join(out)


# llama.cpp server -------------------------------------------------------------------------------


def llama_server_command(
    server: Path,
    gguf: Path,
    alias: str,
    port: int = 8080,
    ctx_size: int = 4096,
    threads: int | None = None,
) -> list[str]:
    """``llama-server`` for the check: the GGUF's own chat template (``--jinja``), one slot, and
    ``--reasoning-format none`` so the reply keeps its raw ``<think>`` block. With a reasoning
    parser an unclosed block would come back looking like a closed one."""
    if ctx_size < 512:
        raise ValueError("ctx_size must be at least 512")
    cmd = [
        str(server),
        "-m",
        str(gguf),
        "-a",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(ctx_size),
        "-np",
        "1",
        "--jinja",
        "--reasoning-format",
        "none",
    ]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def check_thinking_switch(base_url: str, post: Post = http_post) -> str:
    """Refuse a server whose chat template ignores ``enable_thinking``.

    Asks llama-server to render a one-turn chat (``/apply-template``) with thinking off and on.
    Qwen3's template ends the thinking-off prompt with an empty ``<think></think>`` block and
    leaves it out with thinking on. A GGUF without Qwen3's template falls back to plain ChatML,
    which ignores the switch: both modes would then measure the same thing. Returns the
    thinking-off prompt.
    """
    url = base_url.rstrip("/") + "/apply-template"
    chat = [{"role": "user", "content": "Hi"}]
    rendered = {
        on: post(url, {"messages": chat, "chat_template_kwargs": {"enable_thinking": on}}, 60)[
            "prompt"
        ]
        for on in (False, True)
    }
    off_ok = rendered[False].rstrip().endswith("</think>")
    on_ok = "</think>" not in rendered[True]
    if not (off_ok and on_ok):
        raise ValueError(
            "the server's chat template ignores enable_thinking (rendered thinking-off prompt: "
            f"{rendered[False][-60:]!r}). The GGUF probably has no Qwen3 chat template: export "
            "it from a folder with tokenizer_config.json"
        )
    return rendered[False]


def _healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


@contextmanager
def running_server(
    cmd: Sequence[str],
    base_url: str,
    log_path: Path,
    timeout: float = 600.0,
    healthy: Callable[[str], bool] = _healthy,
) -> Iterator[subprocess.Popen]:
    """Start a server, wait until ``/health`` answers, and stop it on exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(list(cmd), stdout=log_file, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + timeout
            while not healthy(base_url):
                if proc.poll() is not None:
                    raise RuntimeError(f"the server exited with code {proc.returncode}: {log_path}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"the server did not start in {timeout:.0f} s: {log_path}")
                time.sleep(1)
            yield proc
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


def load_partial(path: Path, header: dict) -> dict[tuple[str, bool], dict]:
    """Finished replies of an interrupted run, keyed by ``(prompt_id, thinking)``.

    Failed requests are left out so they are asked again. Raises if the file belongs to a
    different run (label, prompt set, settings or model file).
    """
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    if json.loads(lines[0]) != header:
        raise ValueError(
            f"{path} belongs to a different run (label, prompts, settings or GGUF differ); "
            "delete it to start over"
        )
    done = {}
    for line in lines[1:]:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:  # a line cut off when the run was stopped
            continue
        if rec.get("error") is None:
            done[(rec["prompt_id"], rec["thinking"])] = rec
    return done


# running from a config ---------------------------------------------------------------------------


def run_from_config(cfg: ChatCheckConfig, log: Callable[[str], None] = print) -> dict:
    """Run the check (starting llama-server when ``cfg.gguf`` is set), then write the results
    JSON to ``cfg.output`` and the replies for reading next to it (``.md``)."""
    prompts, prompt_meta = load_prompts(cfg.prompts)
    prompts = prompts[: cfg.max_prompts]

    # Progress file: one JSON line per finished reply, so an interrupted run resumes. Its first
    # line names the run; a different label, prompt set or setting refuses to mix with it.
    partial = cfg.output.with_suffix(".partial.jsonl")
    header = {
        "label": cfg.label,
        "prompts_sha256": prompt_meta["sha256"],
        "max_prompts": cfg.max_prompts,
        "thinking_modes": cfg.thinking_modes,
        "max_tokens_think": cfg.max_tokens_think,
        "max_tokens_no_think": cfg.max_tokens_no_think,
        "gguf": str(cfg.gguf) if cfg.gguf else None,
    }
    done = load_partial(partial, header)
    if done:
        log(f"resuming: {len(done)} replies already in {partial}")
    else:
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(header) + "\n", encoding="utf-8", newline="\n")
    lock = threading.Lock()

    def save(rec: dict) -> None:
        with lock, partial.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run(model: str) -> list[dict]:
        client = ChatClient(cfg.server_url, model)
        return run_chat_check(
            client,
            prompts,
            cfg.label,
            thinking_modes=cfg.thinking_modes,
            max_tokens_think=cfg.max_tokens_think,
            max_tokens_no_think=cfg.max_tokens_no_think,
            workers=cfg.workers,
            log=log,
            done=done,
            on_record=save,
        )

    started = time.perf_counter()
    server: dict = {"url": cfg.server_url}
    if cfg.gguf is not None:
        if not cfg.gguf.exists():
            raise FileNotFoundError(f"{cfg.gguf} not found")
        tools = find_llama_cpp(cfg.llama_cpp_dir, cfg.llama_cpp_bin)
        port = int(cfg.server_url.rstrip("/").rsplit(":", 1)[-1])
        cmd = llama_server_command(
            tools.tool("llama-server"), cfg.gguf, cfg.label, port, cfg.ctx_size, cfg.threads
        )
        log_path = cfg.output.with_suffix(".server.log")
        log(f"starting llama-server on {cfg.gguf.name} (log: {log_path})")
        server.update(runtime="llama-server", gguf=str(cfg.gguf), command=cmd[1:])
        with running_server(cmd, cfg.server_url, log_path):
            check_thinking_switch(cfg.server_url)
            log("chat template: enable_thinking switch works")
            records = run(cfg.label)
    else:
        server["runtime"] = "external"
        records = run(cfg.model)
    result = {
        "label": cfg.label,
        "server": server,
        "settings": {
            "decoding": "greedy",
            "thinking_modes": cfg.thinking_modes,
            "max_tokens_think": cfg.max_tokens_think,
            "max_tokens_no_think": cfg.max_tokens_no_think,
            "urdu_min_share": URDU_MIN_SHARE,
            "english_max_share": ENGLISH_MAX_SHARE,
        },
        "prompts": {**prompt_meta, "used": len(prompts)},
        "seconds": round(time.perf_counter() - started, 1),
        "run": run_metadata(),
        "summary": summarize_check(records),
        "records": records,
    }
    write_json(result, cfg.output)
    md = cfg.output.with_suffix(".md")
    md.write_text(check_markdown(records, prompts), encoding="utf-8", newline="\n")
    partial.unlink()
    return result
