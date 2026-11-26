"""Chat SFT data for stage 4: render, mask, pack, and a deterministic row stream.

Conversations come from the filtered teacher data (one JSON record per line with ``id``,
``passage_id``, ``thinking`` and ``messages``; the last message is the assistant's raw reply,
``<think>...</think>`` included when thinking was on).

**Rendering uses the model's own chat template**, so the training text is exactly what a server
sends the model. For one conversation:

* ``prompt`` = the template applied to every message but the last, with the generation prompt
  and the same ``enable_thinking`` switch the server would use;
* ``full`` = the template applied to the whole conversation.

``full`` must start with ``prompt``; the rest is the reply. With thinking off Qwen3's template
pre-fills an empty ``<think>\\n\\n</think>\\n\\n`` block in the prompt, so the model is trained to
answer straight after it; with thinking on, the model is trained to write ``<think>`` itself.

**Loss only on the reply.** Labels are ``IGNORE`` (-100) on every prompt token. The reply ends
with the end-of-turn token (``<|im_end|>`` for Qwen3), which is trained: the model has to learn to
stop. Whitespace after it is dropped, since a server stops at that token.

**Packing.** Several conversations share one row of at most ``max_len`` tokens (first-fit
decreasing). Position ids restart at 0 for each conversation; recent Transformers versions read
that as packed sequences and keep attention inside each conversation (``sft.check_packing``
verifies this on the real model before training). Conversations longer than ``max_len`` are
dropped and counted, never cut, so no reply loses its ending.

Nothing here imports torch or transformers; ``hf_renderer`` wraps a tokenizer the caller loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

IGNORE = -100

# (messages, add_generation_prompt, enable_thinking) -> text
Render = Callable[[list[dict], bool, bool], str]
Encode = Callable[[str], list[int]]


def hf_renderer(tokenizer) -> tuple[Render, Encode]:
    """The chat template and the encoder of a Hugging Face tokenizer, as plain functions.

    Encoding adds no special tokens (Qwen3 has no BOS); the template already writes every
    special token the model sees.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "the tokenizer has no chat template; the model folder needs tokenizer_config.json "
            "with Qwen3's template (Stage 3 copies it from the base model)"
        )

    def render(messages: list[dict], add_generation_prompt: bool, enable_thinking: bool) -> str:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    return render, encode


@dataclass(frozen=True)
class Example:
    """One tokenized conversation: ``ids[:n_prompt]`` is context, the rest is trained."""

    id: str
    ids: tuple[int, ...]
    n_prompt: int

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def n_trained(self) -> int:
        return len(self.ids) - self.n_prompt


def check_record(record: dict) -> None:
    """Raise ValueError unless ``record`` is a usable training conversation."""
    for key in ("id", "messages", "thinking"):
        if key not in record:
            raise ValueError(f"record {record.get('id', '?')} has no '{key}'")
    messages = record["messages"]
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError(
            f"record {record['id']}: the last of at least 2 messages must be assistant"
        )
    if not any(m.get("role") == "user" for m in messages[:-1]):
        raise ValueError(f"record {record['id']}: no user message")
    reply = messages[-1].get("content") or ""
    if record["thinking"]:
        if "</think>" not in reply or not reply.split("</think>")[-1].strip():
            raise ValueError(
                f"record {record['id']}: thinking is on but the reply has no closed "
                "think block followed by an answer"
            )
    elif "<think>" in reply or "</think>" in reply:
        raise ValueError(f"record {record['id']}: thinking is off but the reply has think tags")


def encode_conversation(
    messages: list[dict], thinking: bool, render: Render, encode: Encode
) -> tuple[list[int], int]:
    """Token ids of a whole conversation and how many of them are the prompt.

    Raises ValueError if the template does not render the prompt as a prefix of the full text,
    or if tokenizing the two separately disagrees at the boundary (then masking by count would
    put the boundary in the wrong place).
    """
    prompt = render(messages[:-1], True, thinking)
    full = render(messages, False, thinking).rstrip()
    if not full.startswith(prompt) or len(full) == len(prompt):
        raise ValueError(
            "the chat template does not render the prompt as a prefix of the conversation; "
            f"prompt ends {prompt[-40:]!r}, conversation there reads "
            f"{full[len(prompt) - 40 : len(prompt)]!r}"
        )
    prompt_ids, full_ids = encode(prompt), encode(full)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenizing the prompt alone and inside the conversation disagrees")
    return full_ids, len(prompt_ids)


def load_conversations(path: Path) -> list[dict]:
    """Records from a JSONL file, each checked with ``check_record``; ids must be unique."""
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                check_record(rec)
                records.append(rec)
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate record ids")
    if not records:
        raise ValueError(f"{path} holds no conversations")
    return records


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_validation(record: dict, share: float) -> bool:
    """Whether a record is held out, decided by its passage so no passage is on both sides.

    crc32(passage id) mod 10,000 < share x 10,000. Records without a passage use their own id.
    """
    if not 0 <= share < 1:
        raise ValueError("val_share must be in [0, 1)")
    key = str(record.get("passage_id") or record["id"])
    return zlib.crc32(key.encode("utf-8")) % 10_000 < round(share * 10_000)


def encode_records(
    records: Iterable[dict], render: Render, encode: Encode, max_len: int
) -> tuple[list[Example], dict]:
    """Tokenize conversations; drop (and count) those longer than ``max_len`` tokens."""
    if max_len < 8:
        raise ValueError("max_len must be at least 8 tokens")
    examples, too_long, lengths = [], 0, []
    for rec in records:
        ids, n_prompt = encode_conversation(rec["messages"], bool(rec["thinking"]), render, encode)
        if len(ids) > max_len:
            too_long += 1
            continue
        examples.append(Example(rec["id"], tuple(ids), n_prompt))
        lengths.append(len(ids))
    lengths.sort()
    stats = {
        "conversations": len(examples),
        "dropped_too_long": too_long,
        "tokens": sum(lengths),
        "trained_tokens": sum(e.n_trained for e in examples),
        "median_tokens": lengths[len(lengths) // 2] if lengths else 0,
        "max_tokens": lengths[-1] if lengths else 0,
    }
    return examples, stats


def pack(lengths: Sequence[int], max_len: int) -> list[list[int]]:
    """Group item indices into rows whose lengths sum to at most ``max_len`` (first-fit decreasing).

    Items are placed longest first (ties by index, so the result is deterministic) into the
    first row with room. Every index appears exactly once.
    """
    if any(n > max_len or n < 1 for n in lengths):
        raise ValueError("every length must be between 1 and max_len")
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    rows: list[list[int]] = []
    room: list[int] = []
    for i in order:
        for r, free in enumerate(room):
            if lengths[i] <= free:
                rows[r].append(i)
                room[r] -= lengths[i]
                break
        else:
            rows.append([i])
            room.append(max_len - lengths[i])
    return rows


def row_arrays(examples: Sequence[Example]) -> tuple[list[int], list[int], list[int]]:
    """Input ids, labels and position ids of one packed row.

    Labels: ``IGNORE`` on prompt tokens, the token id itself on reply tokens (the model shifts
    labels by one inside its loss, as Hugging Face models do). Position ids restart at 0 for each
    conversation.
    """
    ids, labels, positions = [], [], []
    for ex in examples:
        ids += ex.ids
        labels += [IGNORE] * ex.n_prompt + list(ex.ids[ex.n_prompt :])
        positions += range(len(ex.ids))
    return ids, labels, positions


def pad_batch(
    rows: Sequence[tuple[list[int], list[int], list[int]]], pad_id: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Pad rows to the longest with ``pad_id``. Padding is untrained and is its own sequence
    (positions restart at 0), so no real token attends to it or is predicted from it."""
    width = max(len(r[0]) for r in rows)
    ids, labels, positions = [], [], []
    for i, lab, pos in rows:
        n = width - len(i)
        ids.append(list(i) + [pad_id] * n)
        labels.append(list(lab) + [IGNORE] * n)
        positions.append(list(pos) + list(range(n)))
    return ids, labels, positions


class RowSampler:
    """Deterministic stream of packed rows over any number of epochs.

    Row position ``p`` is row ``order_e[p mod n]`` of epoch ``e = p // n``, where ``order_e`` is
    the rows shuffled with seed ``seed + e``. Step ``s`` uses positions
    ``s x rows_per_step .. (s + 1) x rows_per_step - 1``, so the step number alone says which data
    comes next and a resumed run continues exactly where it stopped.
    """

    def __init__(self, rows: Sequence[Sequence[int]], seed: int) -> None:
        if not rows:
            raise ValueError("no rows to train on")
        self.rows = [list(r) for r in rows]
        self.seed = seed
        self._orders: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _order(self, epoch: int) -> list[int]:
        if epoch not in self._orders:
            order = list(range(len(self.rows)))
            random.Random(self.seed + epoch).shuffle(order)
            self._orders = {epoch: order}
        return self._orders[epoch]

    def rows_at(self, position: int, count: int) -> list[list[int]]:
        """``count`` consecutive rows (lists of example indices) from ``position``."""
        if position < 0 or count < 1:
            raise ValueError("position must be >= 0 and count >= 1")
        n = len(self.rows)
        return [self.rows[self._order(p // n)[p % n]] for p in range(position, position + count)]

    def epoch_of(self, position: int) -> float:
        return position / len(self.rows)


def total_steps(n_rows: int, rows_per_step: int, epochs: float) -> int:
    """Optimizer steps for ``epochs`` passes: ceil(epochs x n_rows / rows_per_step).

    Example: 2,900 rows, 16 rows per step, 2 epochs: ceil(2 x 2900 / 16) = ceil(362.5) = 363.
    """
    if n_rows < 1 or rows_per_step < 1 or epochs <= 0:
        raise ValueError("need n_rows >= 1, rows_per_step >= 1 and epochs > 0")
    return math.ceil(round(epochs * n_rows / rows_per_step, 9))
