import copy
import json
import random
import unicodedata
from collections import Counter

import pytest

from zubaan.extend.merge import (
    assert_only_model_changed,
    check_roundtrip,
    compare_encodings,
    english_unchanged,
    extend_tokenizer_json,
    load_fast_tokenizer,
    merges_to_txt,
    save_extended,
)
from zubaan.extend.sweep import adapter_for, learn_urdu_merges
from zubaan.extend.tokenizer_info import inspect_tokenizer_json
from zubaan.extend.train_bpe import Merge, word_piece_counts


@pytest.fixture(scope="module")
def merges(tiny_base_json):
    from pathlib import Path

    lines = (
        (Path(__file__).parent / "fixtures" / "urdu_train_small.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    return learn_urdu_merges(lines, tiny_base_json, 400, min_count=2)


@pytest.fixture(scope="module")
def extended(tiny_base_json, merges):
    return extend_tokenizer_json(tiny_base_json, merges)


def test_fixture_learned_a_useful_number_of_merges(merges):
    assert 100 < len(merges) <= 400


# structure ---------------------------------------------------------------------------------------


def test_only_vocab_and_merges_change(tiny_base_json, extended):
    assert_only_model_changed(tiny_base_json, extended)
    for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder", "added_tokens"):
        assert extended[key] == tiny_base_json[key]


def test_base_is_not_mutated(tiny_base_json, merges):
    before = copy.deepcopy(tiny_base_json)
    extend_tokenizer_json(tiny_base_json, merges)
    assert tiny_base_json == before


def test_new_ids_are_sequential_after_the_last_used_id(tiny_base_json, merges, extended):
    start = inspect_tokenizer_json(tiny_base_json).next_free_id
    new = [t for t in extended["model"]["vocab"] if t not in tiny_base_json["model"]["vocab"]]
    assert new == [m.token for m in merges]  # in merge order
    assert [extended["model"]["vocab"][t] for t in new] == list(range(start, start + len(merges)))
    n = len(tiny_base_json["model"]["merges"])
    assert extended["model"]["merges"][n:] == [[m.left, m.right] for m in merges]
    assert extended["model"]["merges"][:n] == tiny_base_json["model"]["merges"]
    info = inspect_tokenizer_json(extended)
    assert info.total_vocab_size == start + len(merges)


def test_ids_start_after_added_tokens_not_inside_them(tiny_base_json, merges):
    """Qwen3 style: added tokens sit above the model vocabulary; new ids must clear them."""
    base = copy.deepcopy(tiny_base_json)
    top = max(base["model"]["vocab"].values())
    base["added_tokens"] = base["added_tokens"] + [
        {
            "id": top + 1 + i,
            "content": f"<|extra_{i}|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for i in range(6)
    ]
    ext = extend_tokenizer_json(base, merges[:10])
    ids = [ext["model"]["vocab"][m.token] for m in merges[:10]]
    assert ids == list(range(top + 7, top + 17))
    vocab_ids = list(ext["model"]["vocab"].values())
    assert len(vocab_ids) == len(set(vocab_ids))  # no id used twice in the vocabulary
    assert not set(ids) & {a["id"] for a in ext["added_tokens"]}  # none inside added tokens
    tok = load_fast_tokenizer(ext)
    # The library would move these to len(vocab) + i if they were not listed in model.vocab.
    for i in range(6):
        assert tok.token_to_id(f"<|extra_{i}|>") == top + 1 + i
    assert_only_model_changed(base, ext)


def test_the_library_really_moves_added_tokens_that_are_not_in_the_vocabulary(tiny_base_json):
    """Documents why extend_tokenizer_json lists added tokens in model.vocab.

    The tokenizers library ignores the stored id of an added token that is missing from
    model.vocab and gives it len(vocab) + position. If this ever stops being true, the
    workaround in merge.py can be reviewed.
    """
    base = copy.deepcopy(tiny_base_json)
    top = max(base["model"]["vocab"].values())
    base["added_tokens"].append(
        {
            "id": top + 1,
            "content": "<|only_added|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    assert load_fast_tokenizer(base).token_to_id("<|only_added|>") == top + 1
    naive = copy.deepcopy(base)
    naive["model"]["vocab"]["ab"] = top + 2  # grow the vocabulary the naive way
    naive["model"]["vocab"]["cd"] = top + 3
    assert load_fast_tokenizer(naive).token_to_id("<|only_added|>") != top + 1


def test_special_token_ids_never_move(tiny_base_json, extended):
    before, after = load_fast_tokenizer(tiny_base_json), load_fast_tokenizer(extended)
    for a in tiny_base_json["added_tokens"]:
        assert before.token_to_id(a["content"]) == after.token_to_id(a["content"]) == a["id"]
    text = "<|im_start|>hello<|endoftext|>"
    assert before.encode(text).ids == after.encode(text).ids


def test_string_format_merges_are_supported(tiny_base_json, merges):
    base = copy.deepcopy(tiny_base_json)
    base["model"]["merges"] = [f"{a} {b}" for a, b in base["model"]["merges"]]
    ext = extend_tokenizer_json(base, merges[:5])
    assert ext["model"]["merges"][-1] == f"{merges[4].left} {merges[4].right}"
    ref = extend_tokenizer_json(tiny_base_json, merges[:5])
    text = "پاکستان ایک ملک ہے"
    a = load_fast_tokenizer(ext).encode(text).ids
    assert a == load_fast_tokenizer(ref).encode(text).ids


def test_empty_merge_list_is_the_identity(tiny_base_json):
    assert extend_tokenizer_json(tiny_base_json, []) == tiny_base_json


# invalid input is refused -----------------------------------------------------------------------


def test_refuses_an_unsafe_english_merge(tiny_base_json):
    vocab = tiny_base_json["model"]["vocab"]
    pair = next((a, b) for a in vocab for b in vocab if len(a) == 1 and a + b not in vocab)
    with pytest.raises(ValueError, match="non-Urdu"):
        extend_tokenizer_json(tiny_base_json, [Merge(pair[0], pair[1], 1)])


def test_refuses_missing_parts_and_existing_tokens(tiny_base_json, merges):
    with pytest.raises(ValueError, match="does not exist yet"):
        extend_tokenizer_json(tiny_base_json, [Merge(merges[0].token, merges[0].left, 1)])
    with pytest.raises(ValueError, match="already exists"):
        extend_tokenizer_json(tiny_base_json, [merges[0], merges[0]])  # same token twice
    vocab = tiny_base_json["model"]["vocab"]
    first = merges[0]
    base = copy.deepcopy(tiny_base_json)
    base["model"]["vocab"][first.token] = max(vocab.values()) + 1
    with pytest.raises(ValueError, match="already exists"):
        extend_tokenizer_json(base, [first])


def test_refuses_a_token_that_equals_an_added_token(tiny_base_json, merges):
    base = copy.deepcopy(tiny_base_json)
    base["added_tokens"].append(
        {"id": 9999, "content": merges[0].token, "special": False, "normalized": False}
    )
    with pytest.raises(ValueError, match="equals an added token"):
        extend_tokenizer_json(base, merges[:3])


# the two required guarantees ---------------------------------------------------------------------


def encode_ids(tokenizer_json, text):
    return load_fast_tokenizer(tokenizer_json).encode(text, add_special_tokens=False).ids


def test_english_fixture_tokenizes_identically(tiny_base_json, extended, english_lines):
    report = english_unchanged(tiny_base_json, extended, english_lines)
    assert report["n_texts"] == len(english_lines) and report["n_different"] == 0


def test_non_arabic_text_of_every_kind_is_unchanged(tiny_base_json, extended):
    """Fuzz: random strings over scripts other than the Arabic block must not change at all."""
    pool = (
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        + list(" \t\n.,;:!?'\"()-_/\\@#$%&*+=<>[]{}|~`^")
        + list("éèàüöñçßøåÆŒ")  # Latin
        + list("дфжщыьюя")  # Cyrillic
        + list("שלוםעברית")  # Hebrew
        + list("हिन्दीभाषा")  # Devanagari
        + list("日本語中文한국어")  # CJK
        + ["😀", "🚀", "‌", "‍", " ", "’", "“", "—", "ﭐ"]
    )
    rng = random.Random(1234)
    texts = ["".join(rng.choice(pool) for _ in range(rng.randint(0, 60))) for _ in range(1500)]
    report = english_unchanged(tiny_base_json, extended, texts)
    assert report["n_different"] == 0, report["examples"]


def test_english_unchanged_refuses_text_with_arabic_letters(tiny_base_json, extended):
    with pytest.raises(ValueError, match="Arabic-block"):
        english_unchanged(tiny_base_json, extended, ["hello ا"])


def test_the_english_test_can_fail(tiny_base_json, english_lines):
    """Mutation check: sneak one English-capable merge past validation and the test must fail."""
    vocab = tiny_base_json["model"]["vocab"]
    pairs = Counter()
    for pieces, n in word_piece_counts(english_lines, tiny_base_json).items():
        for pair in zip(pieces, pieces[1:], strict=False):
            pairs[pair] += n  # only pairs inside one word can ever be merged
    a, b = next(p for p, _ in pairs.most_common() if p[0] + p[1] not in vocab)
    broken = copy.deepcopy(tiny_base_json)
    broken["model"]["vocab"][a + b] = max(vocab.values()) + 1
    broken["model"]["merges"].append([a, b])
    report = compare_encodings(tiny_base_json, broken, english_lines)
    assert report["n_different"] > 0
    with pytest.raises(ValueError):
        extend_tokenizer_json(tiny_base_json, [Merge(a, b, 1)])  # and the validator refuses it


URDU_RAW = [
    "پاکستان ایک خوبصورت ملک ہے۔",
    "ايک كتاب هے",  # Arabic yeh, kaf and heh, not normalized
    "یہ سب کچھ َُِ diacritics کے ساتھ ہے",
    "قیمت 1,500 روپے یا ۱۵۰۰ روپے ہے",
    "Mixed اردو and English کا مرکب 123 ہے!",
    "ای میل: name@example.com اور ویب http://urdu.org/page؟",
    "تصویر 😀 اور 🚀 والے جملے",
    "پہلی سطر\nدوسری سطر\r\nتیسری سطر",
    "  شروع اور آخر میں خالی جگہ  ",
    "ہم‌سفر کا ZWNJ والا لفظ",
    "",
    "ا",
    "ٹ" * 500,
    "لاہور " * 300,
    "أ اور ایک decomposed یٔ مثال",
    "ﷲ اور ﷺ جیسی علامات",
]


def test_urdu_roundtrips_on_fixture_and_raw_text(extended, urdu_lines):
    report = check_roundtrip(extended, urdu_lines + URDU_RAW)
    assert report["n_failures"] == 0, report["examples"]


def test_roundtrip_returns_the_nfc_form(extended):
    """decode(encode(x)) is x for NFC text; for other text it is NFC(x), the tokenizer's rule."""
    decomposed = "أ"  # alef + hamza above, canonically composes to U+0623
    assert unicodedata.normalize("NFC", decomposed) != decomposed
    tok = load_fast_tokenizer(extended)
    out = tok.decode(tok.encode(decomposed, add_special_tokens=False).ids)
    assert out == unicodedata.normalize("NFC", decomposed)


def test_roundtrip_check_can_fail(extended, urdu_lines):
    """Mutation check: remove the byte-level decoder and the check must report failures."""
    broken = copy.deepcopy(extended)
    broken["decoder"] = None
    assert check_roundtrip(broken, urdu_lines)["n_failures"] > 0


def test_extension_matches_the_learning_segmentation_exactly(
    tiny_base_json, extended, merges, urdu_lines
):
    """Replaying the learned merges by hand gives the same tokens the library produces."""

    def replay(word):
        pieces = list(word)
        for m in merges:
            out, i = [], 0
            while i < len(pieces):
                if i + 1 < len(pieces) and pieces[i] == m.left and pieces[i + 1] == m.right:
                    out.append(m.token)
                    i += 2
                else:
                    out.append(pieces[i])
                    i += 1
            pieces = out
        return pieces

    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(json.dumps(extended))
    base_tok = Tokenizer.from_str(json.dumps(tiny_base_json))
    for line in urdu_lines:
        expected = []
        for word, _ in base_tok.pre_tokenizer.pre_tokenize_str(
            base_tok.normalizer.normalize_str(line)
        ):
            expected += replay([t.value for t in base_tok.model.tokenize(word)])
        assert tok.encode(line, add_special_tokens=False).tokens == expected


def test_extension_reduces_urdu_tokens_and_never_increases_them(
    tiny_base_json, extended, urdu_lines
):
    base_a, ext_a = adapter_for(tiny_base_json), adapter_for(extended)
    base_total = ext_total = 0
    for line in urdu_lines + URDU_RAW:
        nb = len(base_a.encode(line, add_special_tokens=False))
        ne = len(ext_a.encode(line, add_special_tokens=False))
        assert ne <= nb  # a merge can only fuse two tokens into one
        base_total += nb
        ext_total += ne
    assert ext_total < 0.8 * base_total


def test_fewer_merges_never_beat_more_merges(tiny_base_json, merges, urdu_lines):
    sizes = [0, 10, 50, 100, len(merges)]
    totals = []
    for n in sizes:
        ext = extend_tokenizer_json(tiny_base_json, merges[:n])
        a = adapter_for(ext)
        totals.append(sum(len(a.encode(line, add_special_tokens=False)) for line in urdu_lines))
    assert totals == sorted(totals, reverse=True)
    assert totals[0] > totals[-1]


def test_word_pieces_of_the_extended_tokenizer_are_all_known_tokens(extended, urdu_lines):
    vocab = extended["model"]["vocab"]
    counts = word_piece_counts(urdu_lines, extended)
    assert all(p in vocab for word in counts for p in word)


# saving ----------------------------------------------------------------------------------------


def test_save_extended_writes_a_loadable_consistent_folder(tmp_path, tiny_base_json, extended):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "tokenizer_config.json").write_text('{"model_max_length": 8}', encoding="utf-8")
    (base_dir / "model.safetensors").write_bytes(b"weights")
    out = save_extended(extended, base_dir, tmp_path / "out")

    assert json.loads((out / "tokenizer.json").read_text(encoding="utf-8")) == extended
    assert (
        json.loads((out / "vocab.json").read_text(encoding="utf-8")) == extended["model"]["vocab"]
    )
    assert (out / "tokenizer_config.json").read_text(encoding="utf-8") == '{"model_max_length": 8}'
    assert not (out / "model.safetensors").exists()  # weights are never copied
    txt = (out / "merges.txt").read_text(encoding="utf-8")
    assert txt.splitlines()[0] == "#version: 0.2"
    assert len(txt.splitlines()) - 1 == len(extended["model"]["merges"])
    assert txt == merges_to_txt(extended)


def test_vocab_and_merges_files_give_the_same_ids_as_tokenizer_json(tmp_path, extended, urdu_lines):
    """The converter and slow tokenizers read vocab.json and merges.txt; they must agree."""
    from tokenizers import Tokenizer, models

    out = save_extended(extended, tmp_path, tmp_path / "out")
    tok = Tokenizer.from_str(json.dumps(extended))
    alt = Tokenizer(models.BPE.from_file(str(out / "vocab.json"), str(out / "merges.txt")))
    alt.normalizer = tok.normalizer
    alt.pre_tokenizer = tok.pre_tokenizer
    for line in urdu_lines + URDU_RAW[:8]:
        assert (
            alt.encode(line, add_special_tokens=False).ids
            == tok.encode(line, add_special_tokens=False).ids
        )


def test_transformers_reloads_the_extended_tokenizer(
    tmp_path, tiny_base_json, extended, urdu_lines
):
    transformers = pytest.importorskip("transformers")
    out = save_extended(extended, tmp_path, tmp_path / "out")
    hf = transformers.PreTrainedTokenizerFast(tokenizer_file=str(out / "tokenizer.json"))
    ref = load_fast_tokenizer(extended)
    for line in urdu_lines:
        assert hf.encode(line, add_special_tokens=False) == ref.encode(line).ids
        assert hf.decode(hf.encode(line, add_special_tokens=False)) == line
    assert len(hf) == inspect_tokenizer_json(extended).total_vocab_size
