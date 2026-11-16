import json
from pathlib import Path

import pytest

from zubaan.audit.fertility import compute_fertility
from zubaan.extend.merge import extend_tokenizer_json
from zubaan.extend.sweep import (
    adapter_for,
    learn_urdu_merges,
    load_merges,
    measure_tokenizer,
    save_merges,
    sweep_vocab_sizes,
)
from zubaan.extend.tokenizer_info import inspect_tokenizer_json
from zubaan.extend.train_bpe import Merge


@pytest.fixture(scope="module")
def merges(tiny_base_json):
    path = Path(__file__).parent / "fixtures" / "urdu_train_small.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    return learn_urdu_merges(lines, tiny_base_json, 300, min_count=2)


@pytest.fixture
def parallel(urdu_lines, english_lines):
    n = min(len(urdu_lines), len(english_lines))
    return urdu_lines[:n], english_lines[:n]


def test_adapter_matches_the_interface_the_audit_expects(tiny_base_json):
    a = adapter_for(tiny_base_json)
    ids = a.encode("the country", add_special_tokens=False)
    assert a.decode(ids) == "the country"
    assert len(a) == inspect_tokenizer_json(tiny_base_json).total_vocab_size
    assert a.unk_token_id is None
    assert len(a.encode("the", add_special_tokens=True)) == len(
        a.encode("the", add_special_tokens=False)
    )  # this tokenizer adds no special tokens


def test_base_row_equals_a_direct_fertility_measurement(tiny_base_json, parallel):
    urdu, english = parallel
    row = measure_tokenizer(tiny_base_json, urdu, english)
    a = adapter_for(tiny_base_json)
    u = compute_fertility(a, urdu, normalize=True)
    e = compute_fertility(a, english, normalize=False)
    assert row["urdu_tokens"] == u.n_tokens and row["english_tokens"] == e.n_tokens
    assert row["cost_multiplier"] == pytest.approx(u.n_tokens / e.n_tokens)
    assert row["urdu_tokens_per_word"] == pytest.approx(u.n_tokens / u.n_words)
    assert "heldout_tokens" not in row


def test_cost_multiplier_worked_example(tiny_base_json):
    # One Urdu text and one English text; the multiplier is the ratio of their token counts.
    urdu, english = ["ا"], ["ab"]
    a = adapter_for(tiny_base_json)
    nu, ne = len(a.encode("ا", add_special_tokens=False)), len(a.encode("ab", False))
    row = measure_tokenizer(tiny_base_json, urdu, english)
    assert nu == 2 and ne == row["english_tokens"]  # alef is the 2 bytes D8 A7 in this base
    assert row["cost_multiplier"] == pytest.approx(nu / ne)


def test_measure_requires_parallel_texts(tiny_base_json):
    with pytest.raises(ValueError, match="parallel"):
        measure_tokenizer(tiny_base_json, ["ا ب"], ["a", "b"])


def test_sweep_rows_and_monotonic_cost(tiny_base_json, merges, parallel):
    urdu, english = parallel
    rows = sweep_vocab_sizes(tiny_base_json, merges, [20, 80, len(merges)], urdu, english)
    assert [r["requested"] for r in rows] == [0, 20, 80, len(merges)]
    assert [r["n_new_tokens"] for r in rows] == [0, 20, 80, len(merges)]
    base_vocab = inspect_tokenizer_json(tiny_base_json).total_vocab_size
    assert [r["vocab_size"] for r in rows] == [base_vocab + r["n_new_tokens"] for r in rows]
    tokens = [r["urdu_tokens"] for r in rows]
    assert tokens == sorted(tokens, reverse=True) and tokens[0] > tokens[-1]
    costs = [r["cost_multiplier"] for r in rows]
    assert costs == sorted(costs, reverse=True)
    # English is the control: same tokens at every size, and the safety checks are all zero.
    assert len({r["english_tokens"] for r in rows}) == 1
    assert all(r["english_different"] == 0 and r["urdu_roundtrip_failures"] == 0 for r in rows)


def test_sweep_dedupes_and_sorts_sizes(tiny_base_json, merges, parallel):
    urdu, english = parallel
    rows = sweep_vocab_sizes(tiny_base_json, merges, [50, 10, 50], urdu, english)
    assert [r["requested"] for r in rows] == [0, 10, 50]


def test_sweep_reports_the_real_count_when_fewer_merges_exist(tiny_base_json, merges, parallel):
    urdu, english = parallel
    rows = sweep_vocab_sizes(tiny_base_json, merges, [10**6], urdu, english)
    assert rows[-1]["requested"] == 10**6 and rows[-1]["n_new_tokens"] == len(merges)


def test_sweep_rejects_bad_sizes(tiny_base_json, merges, parallel):
    urdu, english = parallel
    for bad in ([0], [-5], [10, 0]):
        with pytest.raises(ValueError, match="positive"):
            sweep_vocab_sizes(tiny_base_json, merges, bad, urdu, english)


def test_sweep_adds_heldout_columns_only_when_given(tiny_base_json, merges, parallel, urdu_lines):
    urdu, english = parallel
    held = urdu_lines[:5]
    with_held = sweep_vocab_sizes(tiny_base_json, merges, [40], urdu, english, held)
    assert all("heldout_tokens_per_word" in r for r in with_held)
    assert with_held[1]["heldout_tokens"] < with_held[0]["heldout_tokens"]
    without = sweep_vocab_sizes(tiny_base_json, merges, [40], urdu, english)
    assert not any("heldout_tokens" in r for r in without)


def test_sweep_counts_a_broken_english_guarantee(tiny_base_json, parallel, merges):
    """The safety column is real: an extension that alters English shows up as a nonzero count."""
    urdu, english = parallel
    from collections import Counter

    from zubaan.extend.train_bpe import word_piece_counts

    vocab = tiny_base_json["model"]["vocab"]
    pairs: Counter = Counter()
    for pieces, n in word_piece_counts(english, tiny_base_json).items():
        for p in zip(pieces, pieces[1:], strict=False):
            pairs[p] += n
    a, b = next(p for p, _ in pairs.most_common() if p[0] + p[1] not in vocab)
    broken = json.loads(json.dumps(tiny_base_json))
    broken["model"]["vocab"][a + b] = max(vocab.values()) + 1
    broken["model"]["merges"].append([a, b])
    from zubaan.extend.merge import compare_encodings

    assert compare_encodings(tiny_base_json, broken, english)["n_different"] > 0
    assert (
        measure_tokenizer(broken, urdu, english)["english_tokens"]
        < measure_tokenizer(tiny_base_json, urdu, english)["english_tokens"]
    )


def test_prefix_of_learned_merges_equals_a_smaller_run(tiny_base_json, urdu_lines, merges):
    small = learn_urdu_merges(urdu_lines, tiny_base_json, 60, min_count=2)
    assert small == merges[:60]  # one long run serves every vocabulary size in the sweep
    a = extend_tokenizer_json(tiny_base_json, small)
    b = extend_tokenizer_json(tiny_base_json, merges[:60])
    assert a == b


def test_save_and_load_merges_roundtrip(tmp_path, merges):
    path = save_merges(merges, tmp_path / "sub" / "m.json")
    assert load_merges(path) == list(merges)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw[0] == [merges[0].left, merges[0].right, merges[0].count]


def test_learning_is_deterministic(tiny_base_json, urdu_lines, merges):
    again = learn_urdu_merges(list(reversed(urdu_lines)), tiny_base_json, 300, min_count=2)
    assert again == merges  # line order does not matter, only the counts


def test_learning_from_nothing_gives_no_merges(tiny_base_json):
    assert learn_urdu_merges([], tiny_base_json, 100) == []
    assert learn_urdu_merges(["only english here"], tiny_base_json, 100) == []


def test_merge_dataclass_token_and_equality():
    m = Merge("a", "b", 3)
    assert m.token == "ab" and m == Merge("a", "b", 3) and m != Merge("a", "b", 4)


@pytest.mark.slow
def test_real_base_row_matches_the_stage_1_report():
    """The adapter reproduces the stage 1 numbers computed with transformers (3.14x for Qwen3)."""
    from zubaan.extend.tokenizer_info import load_tokenizer_json
    from zubaan.io import read_lines

    compare = Path("reports/compare.json")
    if not (compare.exists() and Path("data/flores_urd.txt").exists()):
        pytest.skip("needs stage 1 outputs")
    rows = json.loads(compare.read_text(encoding="utf-8"))["rows"]
    stage1 = next(r for r in rows if r["model"] == "Qwen/Qwen3-1.7B")
    base = load_tokenizer_json("Qwen/Qwen3-1.7B")
    row = measure_tokenizer(
        base, read_lines("data/flores_urd.txt"), read_lines("data/flores_eng.txt")
    )
    assert row["urdu_tokens"] == stage1["target_tokens"]
    assert row["english_tokens"] == stage1["reference_tokens"]
    assert row["cost_multiplier"] == pytest.approx(stage1["cost_multiplier"])
    assert row["vocab_size"] == stage1["vocab_size"] == 151_669
