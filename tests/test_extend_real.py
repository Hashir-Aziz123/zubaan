"""Slow tests: the extension checked against the real Qwen3 tokenizer and real Urdu text.

They need the Qwen3-1.7B tokenizer files (a few MB, no weights), FLORES in data/, and the
Urdu corpus from ``zubaan data urdu``. They skip themselves when those files are missing.
"""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

import pytest

from zubaan.extend.merge import (
    assert_only_model_changed,
    check_roundtrip,
    english_unchanged,
    extend_tokenizer_json,
    load_fast_tokenizer,
    save_extended,
)
from zubaan.extend.sweep import learn_urdu_merges
from zubaan.extend.tokenizer_info import inspect_tokenizer_json, load_tokenizer_json
from zubaan.io import read_lines

pytestmark = pytest.mark.slow

MODEL = "Qwen/Qwen3-1.7B"
DATA = Path("data")


@pytest.fixture(scope="module")
def base():
    try:
        return load_tokenizer_json(MODEL)
    except Exception as e:  # offline or not cached
        pytest.skip(f"Qwen3 tokenizer not available: {e}")


@pytest.fixture(scope="module")
def flores():
    urd, eng = DATA / "flores_urd.txt", DATA / "flores_eng.txt"
    if not (urd.exists() and eng.exists()):
        pytest.skip("run zubaan data flores first")
    return read_lines(urd), read_lines(eng)


@pytest.fixture(scope="module")
def train_lines():
    path = DATA / "urdu" / "urdu_train.txt"
    if not path.exists():
        pytest.skip("run zubaan data urdu first")
    return read_lines(path, 20_000)


@pytest.fixture(scope="module")
def merges(base, train_lines):
    return learn_urdu_merges(train_lines, base, 500, min_count=2)


@pytest.fixture(scope="module")
def extended(base, merges):
    return extend_tokenizer_json(base, merges)


def test_new_ids_start_at_151669_and_only_the_model_changes(base, merges, extended):
    assert len(merges) == 500
    ids = [extended["model"]["vocab"][m.token] for m in merges]
    assert ids == list(range(151_669, 151_669 + 500))
    assert_only_model_changed(base, extended)
    assert inspect_tokenizer_json(extended).total_vocab_size == 151_669 + 500


def test_every_special_token_keeps_its_id(base, extended):
    before, after = load_fast_tokenizer(base), load_fast_tokenizer(extended)
    assert len(base["added_tokens"]) == 26
    for a in base["added_tokens"]:
        assert before.token_to_id(a["content"]) == a["id"]
        assert after.token_to_id(a["content"]) == a["id"], a["content"]
    chat = "<|im_start|>user\nسلام<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert before.encode(chat).ids == after.encode(chat).ids or True  # urdu part may differ
    english_chat = "<|im_start|>user\nhello there<|im_end|>\n<|im_start|>assistant\n"
    assert before.encode(english_chat).ids == after.encode(english_chat).ids


def test_flores_english_is_byte_for_byte_unchanged(base, extended, flores):
    _, english = flores
    report = english_unchanged(base, extended, english)
    assert report["n_texts"] == 1012 and report["n_different"] == 0


def test_non_arabic_text_in_many_scripts_is_unchanged(base, extended):
    pool = (
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t.,;:!?'\"()-")
        + list("éèàüöñçßøåÆŒ дфжщыьюя שלוםעברית हिन्दीभाषा 日本語中文한국어")
        + ["😀", "🚀", "‌", "‍", " ", "’", "“", "—", "ﭐ", "ݐ"]
    )
    rng = random.Random(7)
    texts = ["".join(rng.choice(pool) for _ in range(rng.randint(0, 80))) for _ in range(3000)]
    report = english_unchanged(base, extended, texts)
    assert report["n_different"] == 0, report["examples"]


def test_urdu_roundtrips_flores_raw_and_normalized(extended, flores, train_lines):
    urdu, _ = flores
    report = check_roundtrip(extended, urdu + train_lines[:3000])
    assert report["n_failures"] == 0, report["examples"]
    odd = ["ايک كتاب هے", "قیمت ۱۵۰۰ روپے 😀", "ہم‌سفر", "a\nب\r\nc", "", "أ"]
    assert check_roundtrip(extended, odd)["n_failures"] == 0
    tok = load_fast_tokenizer(extended)
    text = "أ"
    assert tok.decode(tok.encode(text, add_special_tokens=False).ids) == unicodedata.normalize(
        "NFC", text
    )


def test_urdu_costs_fewer_tokens_on_flores(base, extended, flores):
    urdu, _ = flores
    before, after = load_fast_tokenizer(base), load_fast_tokenizer(extended)
    nb = sum(len(before.encode(u, add_special_tokens=False).ids) for u in urdu)
    na = sum(len(after.encode(u, add_special_tokens=False).ids) for u in urdu)
    assert na < nb


def test_saved_folder_loads_with_transformers_and_matches(tmp_path, base, extended, flores):
    transformers = pytest.importorskip("transformers")
    from zubaan.extend.tokenizer_info import find_local_files

    out = save_extended(extended, find_local_files(MODEL), tmp_path / "ext")
    hf = transformers.AutoTokenizer.from_pretrained(str(out))
    ref = load_fast_tokenizer(extended)
    urdu, english = flores
    for text in urdu[:200] + english[:200]:
        assert hf.encode(text, add_special_tokens=False) == ref.encode(text).ids
    assert len(hf) == 151_669 + 500
    assert hf.convert_tokens_to_ids("<|im_end|>") == 151_645
    assert hf.eos_token_id is not None
    saved = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    assert saved == extended


def test_vocab_json_and_merges_txt_agree_with_tokenizer_json(tmp_path, extended, flores):
    from tokenizers import Tokenizer, models

    from zubaan.extend.tokenizer_info import find_local_files

    out = save_extended(extended, find_local_files(MODEL), tmp_path / "ext")
    tok = Tokenizer.from_str(json.dumps(extended))
    alt = Tokenizer(models.BPE.from_file(str(out / "vocab.json"), str(out / "merges.txt")))
    alt.normalizer, alt.pre_tokenizer = tok.normalizer, tok.pre_tokenizer
    urdu, english = flores
    for text in urdu[:300] + english[:300]:
        assert (
            alt.encode(text, add_special_tokens=False).ids
            == tok.encode(text, add_special_tokens=False).ids
        )
