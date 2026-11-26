import json

import pytest

from zubaan.tune.client import ChatClient
from zubaan.tune.generate import (
    BACKGROUND,
    Passage,
    Question,
    answer_messages,
    generate_answers,
    parse_questions,
    question_kinds,
    question_messages,
    reading_markdown,
    request_seed,
    sample_passages,
    summarize_answers,
    teacher_serve_command,
    to_jsonl,
    write_questions,
)

GUARD = ["یہ ایک جانچ کا جملہ ہے جو تربیت کے متن میں نہیں آنا چاہیے بالکل بھی"]


def words(n: int, word: str = "لفظ") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def corpus() -> list[str]:
    return [
        words(10),  # too short
        words(70),
        words(80) + " " + GUARD[0],  # copies the evaluation text
        words(90, "حرف"),
        words(300),  # too long
        words(100, "بات"),
    ]


def test_passages_respect_length_and_the_guard():
    passages, stats = sample_passages(corpus(), 3, GUARD, min_words=60, max_words=250)
    assert sorted(p.id for p in passages) == ["urd-1", "urd-3", "urd-5"]
    assert stats["candidates"] == 4 and stats["chosen"] == 3
    assert stats["skipped_overlap"] in (0, 1)  # depends on where the guarded line falls


def test_passages_are_deterministic_per_seed():
    a, _ = sample_passages(corpus(), 2, GUARD, seed=1)
    b, _ = sample_passages(corpus(), 2, GUARD, seed=1)
    assert a == b


def test_passages_refuse_bad_inputs():
    with pytest.raises(ValueError, match="only 3 usable"):
        sample_passages(corpus(), 4, GUARD)
    with pytest.raises(ValueError, match="guard"):
        sample_passages(corpus(), 1, [])
    with pytest.raises(ValueError, match="language"):
        sample_passages(corpus(), 1, GUARD, lang="xx")


def test_question_kinds_cycle():
    assert question_kinds("grounded", 5) == ["summarize", "explain", "list", "factual", "summarize"]
    with pytest.raises(ValueError):
        question_kinds("other", 1)


def test_question_request_names_language_kinds_and_format():
    p = Passage("urd-1", "urd", "لاہور ایک شہر ہے۔")
    grounded = question_messages(p, "grounded", 2)[0]["content"]
    assert "Urdu Wikipedia" in grounded and p.text in grounded
    assert "1. summarize" in grounded and "2. explain" in grounded and "JSON array of 2" in grounded
    standalone = question_messages(p, "standalone", 2)[0]["content"]
    assert "must not mention a passage" in standalone and "1. factual" in standalone


def test_parse_questions():
    reply = 'Here:\n["پہلا سوال؟", " دوسرا سوال؟ "]'
    assert parse_questions(reply, 2) == ["پہلا سوال؟", "دوسرا سوال؟"]
    assert parse_questions('<think>\n[1]\n</think>\n\n["a", "b"]', 2) == ["a", "b"]
    for bad, msg in [
        ("no array", "no JSON"),
        ("[1, 2]", "strings"),
        ('["a"]', "got 1"),
        ('["a", ', "no JSON"),
        ('["a",]', "parse"),
    ]:
        with pytest.raises(ValueError, match=msg):
            parse_questions(bad, 2)


def test_request_seed_is_stable():
    assert request_seed("q|urd-1|grounded") == request_seed("q|urd-1|grounded")
    assert request_seed("a") != request_seed("b")
    assert 0 <= request_seed("x") < 2**31


def test_grounded_prompt_is_the_same_for_teacher_and_training():
    p = Passage("urd-1", "urd", "متن")
    q = Question("urd-1-g0", "urd-1", "urd", "grounded", "summarize", "خلاصہ کریں")
    teacher, train = answer_messages(q, p)
    assert teacher == train == [{"role": "user", "content": "متن\n\nخلاصہ کریں"}]


def test_standalone_hides_the_passage_from_training():
    p = Passage("urd-1", "urd", "متن")
    q = Question("urd-1-s0", "urd-1", "urd", "standalone", "factual", "سوال؟")
    teacher, train = answer_messages(q, p)
    assert teacher[0] == {"role": "system", "content": BACKGROUND.format(passage="متن")}
    assert teacher[1:] == train == [{"role": "user", "content": "سوال؟"}]
    with pytest.raises(ValueError, match="not about"):
        answer_messages(q, Passage("urd-2", "urd", "x"))


class FakeTeacher:
    """Answers question requests with a JSON array and answer requests by thinking mode."""

    def __init__(self, fail_on: str | None = None):
        self.bodies = []
        self.fail_on = fail_on

    def __call__(self, url, body, timeout):
        self.bodies.append(body)
        text = body["messages"][-1]["content"]
        if self.fail_on and self.fail_on in text:
            raise TimeoutError("slow")
        if "JSON array" in text:
            n = int(text.split("JSON array of ")[1].split()[0])
            content = json.dumps([f"سوال {i}؟" for i in range(n)], ensure_ascii=False)
        elif body["chat_template_kwargs"]["enable_thinking"]:
            content = "<think>\nThe user asks about Lahore.\n</think>\n\nلاہور ایک پرانا شہر ہے۔"
        else:
            content = "لاہور ایک پرانا شہر ہے۔"
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }


def test_write_questions_and_answers_end_to_end():
    passages = [Passage("urd-1", "urd", "لاہور " + words(60)), Passage("urd-2", "urd", words(60))]
    teacher = FakeTeacher()
    client = ChatClient("http://x", "t", post=teacher)
    questions, stats = write_questions(client, passages, per_format=2, workers=2, log=lambda s: 0)
    assert stats == {"requests": 4, "failed": 0, "failures": []}
    assert len(questions) == 8
    assert questions[0].id == "urd-1-g0" and questions[0].kind == "summarize"
    assert {q.format for q in questions} == {"grounded", "standalone"}
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in teacher.bodies)

    records = generate_answers(client, questions, passages, "teacher-a", workers=4, log=lambda s: 0)
    assert len(records) == 16
    assert [r["thinking"] for r in records[:2]] == [True, False]
    rec = records[0]
    assert rec["id"] == "urd-1-g0-think" and rec["error"] is None
    assert rec["messages"][-1]["content"].startswith("<think>")
    standalone = next(r for r in records if r["format"] == "standalone")
    assert [m["role"] for m in standalone["messages"]] == ["user", "assistant"]

    summary = summarize_answers(records)
    think = summary["teacher-a | grounded | think"]
    assert think["n"] == 4 and think["think_closed"] == 4 and think["answered"] == 4
    assert think["urdu_answers"] == 4 and think["loops"] == 0
    assert summary["teacher-a | standalone | no think"]["with_think_block"] == 0

    md = reading_markdown(passages, questions, records, max_passages=1)
    assert "## Passage urd-1" in md and "urd-2" not in md
    assert "<details><summary>reasoning</summary>" in md and "teacher-a, thinking off" in md


def test_failures_are_counted_not_hidden():
    passages = [Passage("urd-1", "urd", words(60))]
    client = ChatClient("http://x", "t", post=FakeTeacher(fail_on="must not mention"))
    questions, stats = write_questions(client, passages, per_format=1, workers=1, log=lambda s: 0)
    assert stats["failed"] == 1 and stats["failures"][0]["format"] == "standalone"
    assert [q.format for q in questions] == ["grounded"]

    client = ChatClient("http://x", "t", post=FakeTeacher(fail_on="سوال"))
    records = generate_answers(client, questions, passages, "t", log=lambda s: 0)
    assert all(r["error"].startswith("TimeoutError") for r in records)
    assert summarize_answers(records)["t | grounded | think"]["failed_requests"] == 1
    assert "request failed" in reading_markdown(passages, questions, records)


def test_jsonl_keeps_urdu_readable():
    line = to_jsonl([Passage("urd-1", "urd", "اردو")])
    assert json.loads(line) == {"id": "urd-1", "lang": "urd", "text": "اردو"}
    assert "اردو" in line


def test_teacher_server_is_half_precision_without_a_reasoning_parser():
    cmd = teacher_serve_command("Qwen/Qwen3-8B", "qwen3-8b", tensor_parallel=2)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-8B"]
    assert cmd[cmd.index("--dtype") + 1] == "half"
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert not any("reasoning" in c for c in cmd)
    with pytest.raises(ValueError):
        teacher_serve_command("m", "m", tensor_parallel=0)


# the data mix ---------------------------------------------------------------------------------

from zubaan.tune.generate import (  # noqa: E402
    DEFAULT_MIX,
    MixSlot,
    assign_modes,
    check_mix,
    passage_counts,
    shard_of,
    split_by_format,
)


def test_passage_counts_worked_example():
    # 6,000 grounded Urdu conversations x 1.15 spare / 2 per passage = 3,450 passages.
    counts = passage_counts(DEFAULT_MIX)
    assert counts[("urd", "grounded")] == 3450
    assert counts[("urd", "standalone")] == 1150
    assert counts[("eng", "grounded")] == 863  # ceil(1,500 x 1.15 / 2) = ceil(862.5)
    assert counts[("eng", "standalone")] == 288  # ceil(287.5)
    assert sum(s.count for s in DEFAULT_MIX) == 10_000


def test_mix_refuses_standalone_thinking_and_bad_slots():
    with pytest.raises(ValueError, match="hidden background"):
        check_mix([MixSlot("urd", "standalone", True, 10)])
    for bad in (
        [MixSlot("xx", "grounded", True, 1)],
        [MixSlot("urd", "grounded", True, 0)],
        [MixSlot("urd", "grounded", True, 1)] * 2,
    ):
        with pytest.raises(ValueError):
            check_mix(bad)


def test_split_by_format_uses_each_passage_once():
    ps = [Passage(f"urd-{i}", "urd", "x") for i in range(5)]
    out = split_by_format(ps, {"grounded": 3, "standalone": 2})
    assert [p.id for p in out["grounded"]] == ["urd-0", "urd-1", "urd-2"]
    assert [p.id for p in out["standalone"]] == ["urd-3", "urd-4"]
    with pytest.raises(ValueError, match="need 6"):
        split_by_format(ps, {"grounded": 3, "standalone": 3})


def test_shards_cover_everything_once_and_are_stable():
    ids = [f"urd-{i}" for i in range(1000)]
    shards = [shard_of(i, 2) for i in ids]
    assert set(shards) == {0, 1} and 400 < shards.count(0) < 600
    assert shards == [shard_of(i, 2) for i in ids]
    with pytest.raises(ValueError):
        shard_of("x", 0)


def test_assign_modes_matches_the_mix_share_and_ignores_order():
    qs = [
        Question(f"urd-{i}-g0", f"urd-{i}", "urd", "grounded", "explain", "?") for i in range(3000)
    ]
    qs += [
        Question(f"urd-{i}-s0", f"urd-{i}", "urd", "standalone", "factual", "?") for i in range(500)
    ]
    modes = dict((q.id, t) for q, t in assign_modes(qs, DEFAULT_MIX))
    grounded_think = sum(modes[q.id] for q in qs if q.format == "grounded") / 3000
    assert abs(grounded_think - 2 / 3) < 0.03
    assert not any(modes[q.id] for q in qs if q.format == "standalone")
    assert dict((q.id, t) for q, t in assign_modes(qs[::-1], DEFAULT_MIX)) == modes
    with pytest.raises(ValueError, match="does not contain"):
        assign_modes([Question("e", "eng-1", "eng", "grounded", "x", "?")], DEFAULT_MIX[:3])


def test_given_jobs_done_records_and_progress_callback():
    passages = [Passage("urd-1", "urd", words(60))]
    qs = [
        Question("urd-1-g0", "urd-1", "urd", "grounded", "summarize", "خلاصہ؟"),
        Question("urd-1-g1", "urd-1", "urd", "grounded", "explain", "کیوں؟"),
    ]
    teacher = FakeTeacher()
    client = ChatClient("http://x", "t", post=teacher)
    earlier = {"urd-1-g0-think": {"id": "urd-1-g0-think", "error": None, "reused": True}}
    seen = []
    records = generate_answers(
        client,
        qs,
        passages,
        "t",
        jobs=[(qs[0], True), (qs[1], False)],
        done=earlier,
        on_record=seen.append,
        log=lambda s: 0,
    )
    assert records[0]["reused"] and records[1]["id"] == "urd-1-g1-nothink"
    assert [r["id"] for r in seen] == ["urd-1-g1-nothink"] and len(teacher.bodies) == 1
