import json

import pytest

torch = pytest.importorskip("torch")

from zubaan.extend.init_emb import resize_and_init  # noqa: E402
from zubaan.extend.quality_run import (  # noqa: E402
    build_variant,
    check_weights_match_tokenizer,
    context_id,
    evaluate_variant,
    load_results,
    run_quality,
    select_bpc_texts,
    summary_rows,
    variant_key,
)
from zubaan.extend.sweep import learn_urdu_merges  # noqa: E402
from zubaan.extend.tokenizer_info import inspect_tokenizer_json  # noqa: E402


@pytest.fixture(scope="module")
def merges(tiny_base_json):
    from pathlib import Path

    lines = (
        (Path(__file__).parent / "fixtures" / "urdu_train_small.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    return learn_urdu_merges(lines, tiny_base_json, 200, min_count=2)


def belebele_items(n=4):
    return [
        {
            "flores_passage": "پاکستان ایک ملک ہے",
            "question": "کون سا ملک",
            "mc_answer1": "پاکستان",
            "mc_answer2": "بھارت",
            "mc_answer3": "ایران",
            "mc_answer4": "چین",
            "correct_answer_num": str(1 + i % 4),
        }
        for i in range(n)
    ]


def test_variant_keys():
    assert variant_key(0, None) == "base"
    assert variant_key(10000, "mean") == "n10000-mean"


def test_select_bpc_texts_filters_by_characters_only():
    texts = ["اردو  متن", "x" * 50, "", "تیسرا"]
    assert select_bpc_texts(texts, None, 20, normalize=True) == ["اردو متن", "تیسرا"]
    assert select_bpc_texts(texts, 1, 20, normalize=True) == ["اردو متن"]
    assert select_bpc_texts(["a  b"], None, 20, normalize=False) == ["a  b"]  # untouched


def test_context_id(tiny_base_json):
    assert context_id(tiny_base_json) == 0
    with pytest.raises(ValueError, match="endoftext"):
        context_id({"added_tokens": []})


def test_check_weights_match_tokenizer(tiny_base_json):
    import copy

    check_weights_match_tokenizer(tiny_base_json, copy.deepcopy(tiny_base_json))
    other = copy.deepcopy(tiny_base_json)
    other["model"]["merges"].pop()
    with pytest.raises(ValueError, match="differs"):
        check_weights_match_tokenizer(tiny_base_json, other)


@pytest.mark.parametrize("method", ["mean", "random", "focus"])
def test_build_variant_makes_a_consistent_model_and_tokenizer(
    tiny_base_json, merges, tiny_qwen3, urdu_lines, method
):
    info = inspect_tokenizer_json(tiny_base_json)
    model, ext, meta = build_variant(
        lambda: tiny_qwen3(info.total_vocab_size + 8),
        tiny_base_json,
        merges,
        60,
        method,
        seed=1,
        focus_lines=urdu_lines * 4,
        focus_dim=8,
    )
    total = inspect_tokenizer_json(ext).total_vocab_size
    assert total == info.total_vocab_size + 60
    assert model.get_input_embeddings().weight.shape[0] == total
    assert meta["method"] == method and meta["n_new"] == 60
    if method == "focus":
        assert meta["focus"]["n_with_vector"] + meta["focus"]["n_fallback_to_mean"] == 60


def test_build_variant_errors(tiny_base_json, merges, tiny_qwen3):
    factory = lambda: tiny_qwen3(420)  # noqa: E731
    with pytest.raises(ValueError, match="method must be"):
        build_variant(factory, tiny_base_json, merges, 10, "bogus")
    with pytest.raises(ValueError, match="only"):
        build_variant(factory, tiny_base_json, merges, 10**6, "mean")
    with pytest.raises(ValueError, match="focus_lines"):
        build_variant(factory, tiny_base_json, merges, 10, "focus")


def test_evaluate_variant_shape(tiny_base_json, tiny_qwen3, english_lines):
    model = tiny_qwen3(420)
    sets = {"english": english_lines[:4]}
    out = evaluate_variant(model, tiny_base_json, sets, {"urd_Arab": belebele_items()})
    assert set(out) == {"bpc", "belebele"}
    assert out["bpc"]["english"]["bits_per_character"] > 0
    assert out["belebele"]["urd_Arab"]["n"] == 4


def test_run_quality_end_to_end_resume_and_summary(
    tmp_path, tiny_base_json, merges, tiny_qwen3, urdu_lines, english_lines
):
    info = inspect_tokenizer_json(tiny_base_json)
    calls = []

    def factory():
        calls.append(1)
        return tiny_qwen3(info.total_vocab_size + 8, seed=4)

    sets = {"urdu": urdu_lines[:6], "english": english_lines[:6]}
    bel = {"urd_Arab": belebele_items(), "eng_Latn": belebele_items()}
    out = tmp_path / "q.json"
    kw = dict(
        sets=sets,
        belebele=bel,
        output=out,
        focus_lines=urdu_lines * 3,
        focus_dim=8,
        log=lambda s: None,
    )
    res = run_quality("tiny", factory, tiny_base_json, merges, [40, 80], ["mean", "random"], **kw)
    assert list(res["variants"]) == [
        "base",
        "n40-mean",
        "n40-random",
        "n80-mean",
        "n80-random",
    ]
    assert len(calls) == 5
    base = res["variants"]["base"]
    assert base["n_new_tokens"] == 0 and base["method"] is None
    v = res["variants"]["n80-mean"]
    assert v["vocab_size"] == info.total_vocab_size + 80 and v["init"]["n_new"] == 80
    # English is the control: same characters, and the model sees new rows only as extra logits.
    assert v["bpc"]["english"]["n_chars"] == base["bpc"]["english"]["n_chars"]
    # A bigger vocabulary packs the same Urdu characters into fewer tokens.
    assert v["bpc"]["urdu"]["tokens_per_char"] < base["bpc"]["urdu"]["tokens_per_char"]

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["model"] == "tiny" and set(saved["variants"]) == set(res["variants"])

    calls.clear()  # resume: nothing left to do, so no model is loaded
    run_quality("tiny", factory, tiny_base_json, merges, [40, 80], ["mean", "random"], **kw)
    assert calls == []
    # Adding a method only runs the new variants.
    res3 = run_quality("tiny", factory, tiny_base_json, merges, [40], ["mean", "focus"], **kw)
    assert len(calls) == 1 and "n40-focus" in res3["variants"]

    rows = summary_rows(load_results(out))
    by = {r["variant"]: r for r in rows}
    assert by["base"]["bpc_english_delta"] == 0.0
    assert by["n40-mean"]["bpc_urdu_delta"] == pytest.approx(
        by["n40-mean"]["bpc_urdu"] - by["base"]["bpc_urdu"]
    )
    assert "belebele_urd_Arab_acc_char" in by["base"]


def test_old_token_logits_equal_so_english_only_changes_by_extra_mass(
    tiny_base_json, merges, tiny_qwen3, english_lines
):
    """On English text the extended model sees the same ids and the same hidden states.

    Only the softmax denominator can differ, because the new rows add probability mass; with
    a model trained to ignore them that mass is small, and here it can only lower the score.
    """
    from zubaan.bench.quality import text_bits_per_character
    from zubaan.extend.merge import load_fast_tokenizer

    info = inspect_tokenizer_json(tiny_base_json)
    factory = lambda: tiny_qwen3(info.total_vocab_size + 8, seed=9)  # noqa: E731
    base_model = factory()
    ext_model, ext, _ = build_variant(factory, tiny_base_json, merges, 100, "mean")
    tok_b, tok_e = load_fast_tokenizer(tiny_base_json), load_fast_tokenizer(ext)
    b = text_bits_per_character(base_model, tok_b, english_lines[:5], 0)
    e = text_bits_per_character(ext_model, tok_e, english_lines[:5], 0)
    assert b["n_tokens"] == e["n_tokens"]  # English tokenizes identically
    assert e["total_nll_nats"] >= b["total_nll_nats"] - 1e-6  # extra rows add mass, never remove


def test_resize_info_is_json_serializable(tiny_qwen3):
    info = resize_and_init(tiny_qwen3(48), [[1, 2], [3]], 40, "mean")
    json.dumps(info)


def test_run_quality_keeps_finished_variants_when_one_fails(
    tmp_path, tiny_base_json, merges, tiny_qwen3, urdu_lines, english_lines
):
    info = inspect_tokenizer_json(tiny_base_json)
    n_calls = []

    def factory():
        n_calls.append(1)
        if len(n_calls) == 2:  # the second variant fails while loading
            raise RuntimeError("boom")
        return tiny_qwen3(info.total_vocab_size + 8, seed=4)

    out = tmp_path / "q.json"
    kw = dict(
        sets={"english": english_lines[:4]},
        belebele={},
        output=out,
        log=lambda s: None,
    )
    with pytest.raises(RuntimeError, match="n40-mean"):
        run_quality("tiny", factory, tiny_base_json, merges, [40], ["mean", "random"], **kw)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert set(saved["variants"]) == {"base", "n40-random"}
