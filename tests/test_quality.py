import math
from types import SimpleNamespace

import pytest

from zubaan.bench.quality import (
    belebele_choices,
    belebele_correct,
    belebele_prompt,
    belebele_scores,
    multiple_choice_accuracy,
    pick_answer,
    text_bits_per_character,
    token_logprobs,
    wilson_interval,
)
from zubaan.extend.merge import load_fast_tokenizer

# pure statistics ---------------------------------------------------------------------------------


def test_wilson_interval_worked_example():
    lo, hi = wilson_interval(225, 900)  # accuracy 0.25, exactly chance for 4 choices
    assert lo == pytest.approx(0.2228, abs=1e-3) and hi == pytest.approx(0.2793, abs=1e-3)


def test_wilson_interval_edges():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1
    lo, hi = wilson_interval(50, 50)
    assert hi == 1.0 and lo > 0.9
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)
    # More data gives a tighter interval.
    a, b = wilson_interval(50, 100), wilson_interval(500, 1000)
    assert b[1] - b[0] < a[1] - a[0]


def test_pick_answer_ties_and_empty():
    assert pick_answer([-3.0, -1.0, -1.0, -2.0]) == 1
    assert pick_answer([5]) == 0
    with pytest.raises(ValueError):
        pick_answer([])


def test_multiple_choice_accuracy_sum_versus_per_character():
    # Question 1: sums say answer 0 (-4 > -6); per character says answer 1 (-6/12 > -4/2).
    # Question 2: both say answer 2.
    lp = [[-4.0, -6.0, -9.0, -9.0], [-5.0, -5.0, -1.0, -5.0]]
    chars = [[2, 12, 3, 3], [4, 4, 4, 4]]
    out = multiple_choice_accuracy(lp, chars, correct=[1, 2])
    assert out["n"] == 2
    assert out["acc"] == 0.5  # q1 wrong by sum, q2 right
    assert out["acc_char"] == 1.0  # q1 right per character, q2 right
    assert out["chance"] == 0.25
    assert out["acc_ci"][0] <= out["acc"] <= out["acc_ci"][1]


def test_multiple_choice_accuracy_validates_input():
    with pytest.raises(ValueError, match="one entry per question"):
        multiple_choice_accuracy([[0, 0]], [[1, 1]], [])
    with pytest.raises(ValueError, match="no questions"):
        multiple_choice_accuracy([], [], [])
    with pytest.raises(ValueError, match="one character count"):
        multiple_choice_accuracy([[0, 0]], [[1]], [0])
    with pytest.raises(ValueError, match="at least one character"):
        multiple_choice_accuracy([[0, 0]], [[1, 0]], [0])


ITEM = {
    "flores_passage": "  The passage.  ",
    "question": " What? ",
    "mc_answer1": "one ",
    "mc_answer2": "two",
    "mc_answer3": "three",
    "mc_answer4": " four",
    "correct_answer_num": "3",
}


def test_belebele_item_helpers():
    assert belebele_prompt(ITEM) == "P: The passage.\nQ: What?\nA:"
    assert belebele_choices(ITEM) == ["one", "two", "three", "four"]
    assert belebele_correct(ITEM) == 2
    for bad in ("0", "5", "x"):
        with pytest.raises(ValueError):
            belebele_correct({**ITEM, "correct_answer_num": bad})


# fake models with known probabilities -----------------------------------------------------------


class UniformModel:
    """Every token is equally likely: log-probability is -ln(V) at every position."""

    def __init__(self, vocab: int):
        self.vocab = vocab

    def __call__(self, input_ids, attention_mask=None):
        import torch

        b, t = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(b, t, self.vocab))


class FavoriteModel:
    """Puts almost all probability on one token id at every position."""

    def __init__(self, vocab: int, favorite: int, strength: float = 20.0):
        self.vocab, self.fav, self.strength = vocab, favorite, strength

    def __call__(self, input_ids, attention_mask=None):
        import torch

        logits = torch.zeros(*input_ids.shape, self.vocab)
        logits[..., self.fav] = self.strength
        return SimpleNamespace(logits=logits)


@pytest.fixture
def tok(tiny_base_json):
    return load_fast_tokenizer(tiny_base_json)


def test_bits_per_character_of_a_uniform_model_by_hand(tok, tiny_base_json):
    pytest.importorskip("torch")
    vocab = 512
    text = "the country"
    n_tokens = len(tok.encode(text, add_special_tokens=False).ids)
    out = text_bits_per_character(UniformModel(vocab), tok, [text], context_id=0)
    # Each of the n tokens costs ln(V) nats; chars = 11. bits/char = n ln(V) / (11 ln 2).
    expected = n_tokens * math.log(vocab) / (len(text) * math.log(2))
    assert out["bits_per_character"] == pytest.approx(expected)
    assert out["n_tokens"] == n_tokens and out["n_chars"] == 11
    assert out["bits_per_token"] == pytest.approx(math.log2(vocab))
    assert out["tokens_per_char"] == pytest.approx(n_tokens / 11)


def test_bits_per_character_is_independent_of_batching_and_order(tok, english_lines):
    pytest.importorskip("torch")
    from conftest import make_tiny_qwen3

    model = make_tiny_qwen3(400, seed=3)
    texts = english_lines[:8]
    ref = text_bits_per_character(model, tok, texts, 0, batch_size=1)
    for bs in (3, 8):
        got = text_bits_per_character(model, tok, texts, 0, batch_size=bs)
        assert got["bits_per_character"] == pytest.approx(ref["bits_per_character"], rel=1e-4)
        assert got["n_tokens"] == ref["n_tokens"]
    shuffled = text_bits_per_character(model, tok, texts[::-1], 0, batch_size=3)
    assert shuffled["total_nll_nats"] == pytest.approx(ref["total_nll_nats"], rel=1e-4)


def test_padding_does_not_change_a_texts_score(tok):
    pytest.importorskip("torch")
    from conftest import make_tiny_qwen3

    model = make_tiny_qwen3(400, seed=1)
    short, long = "the city", "the government announced new projects to improve education"
    alone = text_bits_per_character(model, tok, [short], 0, batch_size=1)["total_nll_nats"]
    mixed = text_bits_per_character(model, tok, [short, long], 0, batch_size=2)["total_nll_nats"]
    alone_long = text_bits_per_character(model, tok, [long], 0, batch_size=1)["total_nll_nats"]
    assert mixed == pytest.approx(alone + alone_long, rel=1e-4)


def test_matches_a_direct_cross_entropy(tok):
    torch = pytest.importorskip("torch")
    from conftest import make_tiny_qwen3

    model = make_tiny_qwen3(400, seed=2)
    text = "heavy rain caused water to collect"
    ids = [0, *tok.encode(text, add_special_tokens=False).ids]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0, :-1]
        direct = torch.nn.functional.cross_entropy(
            logits.float(), torch.tensor(ids[1:]), reduction="sum"
        )
    got = text_bits_per_character(model, tok, [text], 0)["total_nll_nats"]
    assert got == pytest.approx(float(direct), rel=1e-5)


def test_bits_per_character_errors(tok):
    with pytest.raises(ValueError, match="no texts"):
        text_bits_per_character(UniformModel(10), tok, [], 0)
    with pytest.raises(ValueError, match="no tokens"):
        text_bits_per_character(UniformModel(10), tok, [""], 0)
    with pytest.raises(ValueError, match="max_tokens"):
        text_bits_per_character(UniformModel(10), tok, ["the country " * 50], 0, max_tokens=20)


def test_a_better_model_scores_fewer_bits(tok):
    pytest.importorskip("torch")
    text = "aaaa"
    fav = tok.encode("a", add_special_tokens=False).ids[0]
    good = text_bits_per_character(FavoriteModel(512, fav), tok, [text], 0)
    uniform = text_bits_per_character(UniformModel(512), tok, [text], 0)
    assert good["bits_per_character"] < 0.1 < uniform["bits_per_character"]


def test_token_logprobs_shape_and_values(tok):
    torch = pytest.importorskip("torch")
    out = token_logprobs(UniformModel(100), [[0, 5, 6], [0, 7]], pad_id=0)
    assert [len(x) for x in out] == [2, 1]
    assert all(torch.allclose(x, torch.full_like(x, -math.log(100))) for x in out)


# Belebele scoring --------------------------------------------------------------------------------


def make_items(answers, correct):
    return [
        {
            "flores_passage": "the government",
            "question": "what",
            "mc_answer1": answers[0],
            "mc_answer2": answers[1],
            "mc_answer3": answers[2],
            "mc_answer4": answers[3],
            "correct_answer_num": str(correct),
        }
    ]


def test_belebele_picks_the_answer_the_model_prefers(tok):
    pytest.importorskip("torch")
    fav = tok.encode("a", add_special_tokens=False).ids[0]
    items = make_items(["bbbbbb", "cccccc", "aaaaaa", "dddddd"], correct=3)  # answer 3 is "aaaaaa"
    out = belebele_scores(FavoriteModel(512, fav), tok, items, context_id=0)
    assert out["n"] == 1 and out["acc"] == 1.0 and out["acc_char"] == 1.0
    wrong = make_items(["bbbbbb", "cccccc", "aaaaaa", "dddddd"], correct=1)
    assert belebele_scores(FavoriteModel(512, fav), tok, wrong, 0)["acc"] == 0.0


def test_uniform_model_sum_prefers_few_tokens_but_per_char_does_not(tok):
    """The reason acc_char exists: under a uniform model the plain sum just counts tokens.

    Answer A has fewer tokens, answer B has fewer tokens per character. The sum picks A,
    the per-character score picks B, and a tokenizer that packs more characters into each token
    would otherwise get a free advantage.
    """
    pytest.importorskip("torch")
    pool = ["the", "the country", "the government announced", "cricket is the most popular sport"]

    def stats(text):
        n = len(tok.encode(" " + text, add_special_tokens=False).ids)
        return n, len(text) + 1

    pair = next(
        (
            (x, y)
            for x in pool
            for y in pool
            if stats(x)[0] < stats(y)[0] and stats(x)[0] / stats(x)[1] > stats(y)[0] / stats(y)[1]
        ),
        None,
    )
    assert pair is not None, "fixture tokenizer cannot produce the contrast"
    a_text, b_text = pair
    items = make_items([a_text, b_text, "zzzzzzzzzzzzzzzzzzzzzz", "zzzzzzzzzzzzzzzzzzzzzz"], 2)
    out = belebele_scores(UniformModel(512), tok, items, context_id=0)
    assert out["acc"] == 0.0  # the sum chose A (index 0), but B (index 1) is the right answer
    assert out["acc_char"] == 1.0  # per character chose B


def test_belebele_progress_callback(tok):
    pytest.importorskip("torch")
    seen = []
    items = make_items(["a", "b", "c", "d"], 1) * 3
    belebele_scores(UniformModel(512), tok, items, 0, on_progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_belebele_rejects_an_overlong_item(tok):
    pytest.importorskip("torch")
    items = make_items(["a", "b", "c", "d"], 1)
    with pytest.raises(ValueError, match="max_tokens"):
        belebele_scores(UniformModel(512), tok, items, 0, max_tokens=3)


@pytest.mark.slow
def test_real_belebele_loads_with_the_documented_fields():
    from zubaan.bench.quality import load_belebele

    for lang in ("urd_Arab", "eng_Latn"):
        items = load_belebele(lang, n=5)
        assert len(items) == 5
        assert {"flores_passage", "question", "correct_answer_num"} <= set(items[0])
        assert all(0 <= belebele_correct(i) < 4 for i in items)
    full = load_belebele("urd_Arab")
    assert len(full) == 900
    counts = [sum(belebele_correct(i) == k for i in full) for k in range(4)]
    assert min(counts) > 150  # answers are roughly balanced, so the chance rate is 0.25


# memory bounding ---------------------------------------------------------------------------------


def test_token_budget_batches_respect_both_limits():
    from zubaan.bench.quality import token_budget_batches

    lengths = [5, 5, 5, 10, 10, 40, 100]
    batches = token_budget_batches(lengths, max_items=4, max_tokens=40)
    assert [i for b in batches for i in b] == list(range(7))  # every item once, in order
    for b in batches:
        longest = max(lengths[i] for i in b)
        assert len(b) <= 4 and (len(b) == 1 or len(b) * longest <= 40)
    assert [lengths[i] for i in batches[-1]] == [100]  # an over-budget item stands alone
    assert token_budget_batches([], 4, 40) == []
    with pytest.raises(ValueError):
        token_budget_batches([1], 0, 10)
    with pytest.raises(ValueError):
        token_budget_batches([1], 1, 0)


def test_scores_do_not_depend_on_the_batch_token_budget(tok, english_lines):
    pytest.importorskip("torch")
    from conftest import make_tiny_qwen3

    model = make_tiny_qwen3(400, seed=5)
    texts = english_lines[:8]
    ref = text_bits_per_character(model, tok, texts, 0, batch_size=8, max_batch_tokens=10_000)
    tight = text_bits_per_character(model, tok, texts, 0, batch_size=8, max_batch_tokens=1)
    assert tight["total_nll_nats"] == pytest.approx(ref["total_nll_nats"], rel=1e-4)
    items = make_items(["the city", "the country", "heavy rain", "new projects"], 2) * 2
    a = belebele_scores(model, tok, items, 0, max_batch_tokens=10_000)
    b = belebele_scores(model, tok, items, 0, max_batch_tokens=1)
    assert (a["acc"], a["acc_char"]) == (b["acc"], b["acc_char"])


def test_memory_report_is_json_friendly():
    import json

    from zubaan.extend.quality_run import memory_report, release_gpu_memory

    json.dumps(memory_report())
    release_gpu_memory()  # must be a harmless no-op without a GPU
