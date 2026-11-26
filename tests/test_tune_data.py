"""Stage 4 chat data: rendering with Qwen3's real chat template, masking, packing, row order."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zubaan.tune.data import (
    IGNORE,
    Example,
    RowSampler,
    check_record,
    encode_conversation,
    encode_records,
    hf_renderer,
    is_validation,
    load_conversations,
    pack,
    pad_batch,
    row_arrays,
    total_steps,
)

FIXTURES = Path(__file__).parent / "fixtures"
URDU_Q = "پاکستان کا دارالحکومت کون سا شہر ہے؟"
URDU_A = "پاکستان کا دارالحکومت اسلام آباد ہے۔"


def make_chat_tokenizer(tiny_base_json: dict):
    """The tiny English BPE as a Transformers tokenizer with Qwen3's own chat template."""
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer

    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(tiny_base_json)),
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tok.add_tokens(["<think>", "</think>"])
    tok.chat_template = (FIXTURES / "qwen3_chat_template.jinja").read_text(encoding="utf-8")
    return tok


@pytest.fixture
def chat_tok(tiny_base_json):
    return make_chat_tokenizer(tiny_base_json)


def record(i: int, thinking: bool, passage: str = "p1") -> dict:
    reply = (
        f"<think>\nThe user asks about the capital {i}.\n</think>\n\n{URDU_A}"
        if thinking
        else URDU_A
    )
    return {
        "id": f"q{i}-{'think' if thinking else 'nothink'}",
        "passage_id": passage,
        "thinking": thinking,
        "messages": [
            {"role": "user", "content": f"{URDU_Q} {i}"},
            {"role": "assistant", "content": reply},
        ],
    }


def test_thinking_on_trains_the_think_block_and_the_end_token(chat_tok):
    render, encode = hf_renderer(chat_tok)
    ids, n_prompt = encode_conversation(record(1, True)["messages"], True, render, encode)
    prompt = chat_tok.decode(ids[:n_prompt])
    reply = chat_tok.decode(ids[n_prompt:])
    assert prompt.endswith("<|im_start|>assistant\n")  # the model writes <think> itself
    assert reply.startswith("<think>\nThe user asks")
    assert "</think>\n\n" + URDU_A in reply
    assert reply.endswith("<|im_end|>")  # trained to stop; nothing after it


def test_thinking_off_puts_the_empty_think_block_in_the_prompt(chat_tok):
    render, encode = hf_renderer(chat_tok)
    ids, n_prompt = encode_conversation(record(1, False)["messages"], False, render, encode)
    prompt = chat_tok.decode(ids[:n_prompt])
    # Exactly what a server sends with enable_thinking false, so the model answers after it.
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert chat_tok.decode(ids[n_prompt:]) == URDU_A + "<|im_end|>"


def test_two_turn_conversation_trains_only_the_last_reply(chat_tok):
    render, encode = hf_renderer(chat_tok)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": URDU_Q},
        {"role": "assistant", "content": URDU_A},
    ]
    ids, n_prompt = encode_conversation(messages, False, render, encode)
    assert "Hi there." in chat_tok.decode(ids[:n_prompt])
    assert chat_tok.decode(ids[n_prompt:]) == URDU_A + "<|im_end|>"


def test_a_template_that_does_not_prefix_is_refused():
    def render(messages, add_generation_prompt, enable_thinking):
        return "PROMPT" if add_generation_prompt else "OTHER TEXT"

    with pytest.raises(ValueError, match="prefix"):
        encode_conversation(record(1, False)["messages"], False, render, lambda s: list(s.encode()))


def test_tokenizer_without_template_is_refused(chat_tok):
    chat_tok.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        hf_renderer(chat_tok)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(thinking=True), "closed think block"),
        (lambda r: r["messages"].pop(), "last of at least 2"),
        (lambda r: r.pop("thinking"), "has no 'thinking'"),
    ],
)
def test_check_record_refuses_bad_records(change, message):
    rec = record(1, False)
    change(rec)
    with pytest.raises(ValueError, match=message):
        check_record(rec)


def test_check_record_refuses_think_tags_with_thinking_off():
    rec = record(1, True)
    rec["thinking"] = False
    with pytest.raises(ValueError, match="think tags"):
        check_record(rec)


def test_load_conversations_refuses_duplicate_ids(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(record(1, True)) for _ in range(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_conversations(path)


def test_encode_records_drops_and_counts_long_conversations(chat_tok):
    render, encode = hf_renderer(chat_tok)
    recs = [record(1, False), record(2, True)]
    full = [encode_conversation(r["messages"], r["thinking"], render, encode)[0] for r in recs]
    limit = min(len(f) for f in full)
    examples, stats = encode_records(recs, render, encode, max_len=limit)
    assert [e.id for e in examples] == ["q1-nothink"]
    assert stats["dropped_too_long"] == 1
    assert stats["trained_tokens"] == examples[0].n_trained


def test_pack_first_fit_decreasing_worked_example():
    # Longest first: 5 -> row 0 (3 left); 4 -> row 1 (4 left); 3 -> row 0 (full);
    # 3 -> row 1 (1 left); 2 -> new row 2.
    assert pack([5, 4, 3, 3, 2], 8) == [[0, 2], [1, 3], [4]]
    with pytest.raises(ValueError):
        pack([9], 8)


def test_pack_uses_every_item_once_and_respects_the_limit():
    lengths = [((i * 37) % 50) + 1 for i in range(200)]
    rows = pack(lengths, 64)
    assert sorted(i for r in rows for i in r) == list(range(200))
    assert all(sum(lengths[i] for i in r) <= 64 for r in rows)


def test_row_arrays_mask_prompts_and_restart_positions():
    a = Example("a", (1, 2, 3), 2)
    b = Example("b", (4, 5, 6, 7), 1)
    ids, labels, pos = row_arrays([a, b])
    assert ids == [1, 2, 3, 4, 5, 6, 7]
    assert labels == [IGNORE, IGNORE, 3, IGNORE, 5, 6, 7]
    assert pos == [0, 1, 2, 0, 1, 2, 3]


def test_pad_batch_pads_as_a_separate_untrained_sequence():
    rows = [row_arrays([Example("a", (1, 2, 3), 1)]), row_arrays([Example("b", (4,), 0)])]
    ids, labels, pos = pad_batch(rows, pad_id=0)
    assert ids == [[1, 2, 3], [4, 0, 0]]
    assert labels == [[IGNORE, 2, 3], [4, IGNORE, IGNORE]]
    assert pos == [[0, 1, 2], [0, 0, 1]]


def test_row_sampler_is_deterministic_and_reshuffles_each_epoch():
    rows = [[i] for i in range(10)]
    s1, s2 = RowSampler(rows, seed=3), RowSampler(rows, seed=3)
    first = s1.rows_at(0, 10)
    second = s1.rows_at(10, 10)
    assert sorted(first) == rows and sorted(second) == rows  # each epoch is a permutation
    assert first != second
    assert s2.rows_at(7, 6) == first[7:] + second[:3]  # crossing the epoch boundary
    assert s1.epoch_of(15) == 1.5


def test_total_steps_worked_example():
    assert total_steps(2900, 16, 2) == 363  # ceil(2 x 2900 / 16) = ceil(362.5)
    assert total_steps(32, 16, 1) == 2
    with pytest.raises(ValueError):
        total_steps(0, 16, 1)


def test_validation_split_keeps_a_passage_on_one_side():
    recs = [record(i, i % 2 == 0, passage=f"urd-{i // 4}") for i in range(4000)]
    held = [r for r in recs if is_validation(r, 0.1)]
    assert 0.06 < len(held) / len(recs) < 0.14
    by_passage: dict[str, set[bool]] = {}
    for r in recs:
        by_passage.setdefault(r["passage_id"], set()).add(is_validation(r, 0.1))
    assert all(len(v) == 1 for v in by_passage.values())
    assert not any(is_validation(r, 0.0) for r in recs)
