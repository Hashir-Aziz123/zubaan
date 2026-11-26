"""Stage 4, chunk 4: filtering, balancing and the review sample."""

from __future__ import annotations

import csv
import io
import json

import pytest
import yaml
from typer.testing import CliRunner

from zubaan import cli
from zubaan.tune.curate import (
    curate,
    jaccard,
    reply_problem,
    review_csv,
    review_sample,
    run_curate,
    word_set,
)
from zubaan.tune.filters import mixed_script_word
from zubaan.tune.generate import MixSlot

runner = CliRunner()
URDU = "پاکستان کا دارالحکومت اسلام آباد ہے اور یہ ایک خوبصورت شہر ہے۔"
ENGLISH = "The capital of Pakistan is Islamabad, a planned city in the north."


def rec(
    rid: str,
    answer: str = URDU,
    thinking: bool = False,
    lang: str = "urd",
    fmt: str = "grounded",
    question: str | None = None,
    think: str = "The user asks about the capital of Pakistan.",
    **extra,
) -> dict:
    content = f"<think>\n{think}\n</think>\n\n{answer}" if thinking else answer
    r = {
        "id": rid,
        "question_id": rid.rsplit("-", 1)[0],
        "passage_id": "urd-1",
        "lang": lang,
        "format": fmt,
        "kind": "factual",
        "thinking": thinking,
        "teacher": "t",
        "messages": [
            {"role": "user", "content": question or f"سوال نمبر {rid}"},
            {"role": "assistant", "content": content},
        ],
        "finish_reason": "stop",
        "error": None,
    }
    r.update(extra)
    return r


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (rec("a", error="timeout"), "failed_request"),
        (rec("a", finish_reason="length"), "hit_limit"),
        (rec("a", thinking=True, answer=""), "no_answer"),
        (rec("a", answer="ہاں"), "too_short"),
        (rec("a", answer=ENGLISH), "wrong_language"),
        (rec("a", lang="eng", answer=URDU), "wrong_language"),
        (rec("a", answer=f"**تómہ:** {URDU}"), "mixed_script"),
        (rec("a", answer=" ".join([URDU] * 4)), "loop"),
        (rec("a", fmt="standalone", answer="متن کے مطابق " + URDU), "source_mention"),
        (rec("a", lang="eng", answer=ENGLISH), None),
        (rec("a", thinking=True), None),
        (rec("a", fmt="grounded", answer="متن کے مطابق " + URDU), None),  # grounded may say so
    ],
)
def test_reply_problem(record, reason):
    assert reply_problem(record) == reason


def test_think_block_checks():
    unclosed = rec("a")
    unclosed.update(thinking=True)
    unclosed["messages"][-1]["content"] = "<think>\nstill thinking"
    assert reply_problem(unclosed) == "think_unclosed"
    no_block = rec("a")
    no_block["thinking"] = True
    assert reply_problem(no_block) == "think_mismatch"
    block_when_off = rec("a", thinking=True)
    block_when_off["thinking"] = False
    assert reply_problem(block_when_off) == "think_mismatch"


def test_mixed_script_allows_bracketed_english_terms():
    assert mixed_script_word("اس تبدیلی کو طفرہ (mutation) کہا جاتا ہے۔") is None
    assert mixed_script_word("**تómہ:** میثم") is not None
    assert mixed_script_word("امplitude 10 گنا") is not None


def test_jaccard_worked_example():
    assert jaccard(frozenset("abc"), frozenset("bcd")) == 0.5
    assert jaccard(frozenset(), frozenset()) == 0.0
    assert word_set("Hello, World") == frozenset({"hello", "world"})


MIX = (
    MixSlot("urd", "grounded", True, 2),
    MixSlot("urd", "grounded", False, 3),
)


def test_curate_drops_duplicates_and_chat_check_copies_then_balances():
    records = [rec(f"q{i}-think", thinking=True) for i in range(4)]
    records += [rec(f"r{i}-nothink") for i in range(2)]
    records.append(rec("r9-nothink", question="سوال نمبر q0-think"))  # same user turn as q0
    records.append(rec("s1-nothink", question="پاکستان کا دارالحکومت کون سا شہر ہے؟"))
    prompts = [("ur-fact-01", "پاکستان کا دارالحکومت کون سا شہر ہے؟")]
    kept, report = curate(records, prompts, MIX)
    assert report["dropped"] == {"duplicate_question": 1, "near_chat_check": 1}
    assert report["near_chat_check"] == [
        {"id": "s1-nothink", "prompt": "ur-fact-01", "jaccard": 1.0}
    ]
    think = report["slots"]["urd grounded think"]
    assert think == {
        "generated": 4,
        "passed": 4,
        "target": 2,
        "kept": 2,
        "short_by": 0,
        "dropped": {},
    }
    nothink = report["slots"]["urd grounded nothink"]
    assert (nothink["passed"], nothink["kept"], nothink["short_by"]) == (2, 2, 1)
    assert len(kept) == 4
    assert "finish_reason" not in kept[0] and "error" not in kept[0]
    # Same input in another order: same result.
    kept2, _ = curate(list(reversed(records)), prompts, MIX)
    assert kept2 == kept


def test_curate_refuses_a_shard_given_twice():
    with pytest.raises(ValueError, match="duplicate record ids"):
        curate([rec("a-nothink"), rec("a-nothink")], (), MIX)


def test_review_sample_covers_every_slot():
    kept = [rec(f"q{i}-think", thinking=True) for i in range(90)]
    kept += [rec(f"r{i}-nothink") for i in range(10)]
    sample = review_sample(kept, 20)
    slots = [r["thinking"] for r in sample]
    assert slots.count(True) == 18 and slots.count(False) == 2
    rows = list(csv.reader(io.StringIO(review_csv(sample))))
    assert rows[0] == ["n", "id", "slot", "verdict", "note"] and len(rows) == 21


def test_run_curate_and_cli_write_the_outputs(tmp_path):
    shard = tmp_path / "answers.jsonl"
    lines = [rec(f"q{i}-think", thinking=True) for i in range(3)] + [rec("x-nothink", error="boom")]
    shard.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in lines) + '\n{"id": "cut',
        encoding="utf-8",
    )
    report = run_curate([shard], tmp_path / "out", review_n=2)
    assert report["unreadable_lines"] == 1
    assert report["dropped"] == {"failed_request": 1}
    out = tmp_path / "out"
    assert len((out / "conversations.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert "## 1." in (out / "review.md").read_text(encoding="utf-8")

    cfg = tmp_path / "f.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"tune_filter": {"answers": [str(shard)], "out_dir": str(tmp_path / "cli")}}
        ),
        encoding="utf-8",
    )
    result = runner.invoke(cli.app, ["tune", "filter", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "3 kept" in result.output
    missing = runner.invoke(
        cli.app, ["tune", "filter", "--config", str(cfg), "--answers", str(tmp_path / "nope")]
    )
    assert missing.exit_code != 0
