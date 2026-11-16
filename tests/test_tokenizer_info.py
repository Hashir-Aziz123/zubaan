import json

import pytest

from zubaan.extend.bytelevel import (
    bytes_to_token,
    bytes_to_unicode,
    token_to_bytes,
    unicode_to_bytes,
)
from zubaan.extend.tokenizer_info import (
    inspect_tokenizer,
    inspect_tokenizer_json,
    load_tokenizer_json,
)


def qwen_like() -> dict:
    """Shaped like Qwen3: specials live only in added_tokens, after the model vocabulary."""
    return {
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": [["a", "b"]],
            "byte_fallback": False,
        },
        "added_tokens": [
            {"id": 3, "content": "<|endoftext|>", "special": True},
            {"id": 4, "content": "<tool>", "special": False},
        ],
        "pre_tokenizer": {"type": "Sequence"},
        "normalizer": {"type": "NFC"},
    }


def test_inspect_qwen_like_ids():
    info = inspect_tokenizer_json(qwen_like(), {"vocab_size": 8, "tie_word_embeddings": True})
    assert info.model_vocab_size == 3
    assert info.max_model_id == 2
    assert info.n_added_tokens == 2 and info.n_special_tokens == 1
    assert (info.first_added_id, info.last_added_id) == (3, 4)
    # New tokens start after the added tokens, not at the padded embedding size (8).
    assert info.next_free_id == 5
    assert info.total_vocab_size == 5
    assert info.embedding_rows == 8 and info.tie_word_embeddings is True
    assert info.pre_tokenizer_type == "Sequence" and info.normalizer_type == "NFC"


def test_inspect_counts_overlapping_ids_once():
    data = qwen_like()
    data["added_tokens"] = [{"id": 0, "content": "<s>", "special": True}]
    info = inspect_tokenizer_json(data)
    assert info.total_vocab_size == 3  # id 0 is shared, not counted twice
    assert info.next_free_id == 3


def test_inspect_rejects_non_bpe():
    with pytest.raises(ValueError, match="only BPE"):
        inspect_tokenizer_json({"model": {"type": "Unigram"}})


def test_inspect_handles_missing_pieces():
    info = inspect_tokenizer_json({"model": {"type": "BPE", "vocab": {}, "merges": []}})
    assert info.next_free_id == 0 and info.pre_tokenizer_type == "none"
    assert info.normalizer_type is None and info.first_added_id is None


def test_inspect_local_folder_reads_config(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps(qwen_like()), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 16}), encoding="utf-8")
    info = inspect_tokenizer(str(tmp_path))
    assert info.embedding_rows == 16 and info.next_free_id == 5
    assert load_tokenizer_json(str(tmp_path / "tokenizer.json"))["model"]["type"] == "BPE"


def test_load_missing_tokenizer_json(tmp_path):
    with pytest.raises(ValueError, match="no tokenizer.json"):
        load_tokenizer_json(str(tmp_path))


def test_inspect_on_tiny_tokenizer_matches_real_file(tiny_base_json):
    info = inspect_tokenizer_json(tiny_base_json)
    assert info.model_vocab_size == len(tiny_base_json["model"]["vocab"])
    assert info.n_merges == len(tiny_base_json["model"]["merges"])
    assert info.next_free_id == max(tiny_base_json["model"]["vocab"].values()) + 1


@pytest.mark.slow
def test_real_qwen3_tokenizer_facts():
    """The facts the whole extension design relies on, read from the real Qwen3-1.7B files."""
    info = inspect_tokenizer("Qwen/Qwen3-1.7B")
    assert info.model_vocab_size == 151_643 and info.n_merges == 151_387
    assert (info.first_added_id, info.last_added_id) == (151_643, 151_668)
    assert info.n_added_tokens == 26 and info.n_special_tokens == 14
    assert info.total_vocab_size == 151_669 and info.next_free_id == 151_669
    assert info.embedding_rows == 151_936  # padded above the vocabulary
    assert info.tie_word_embeddings is True
    assert info.normalizer_type == "NFC" and info.byte_fallback is False
    data = load_tokenizer_json("Qwen/Qwen3-1.7B")
    assert data["pre_tokenizer"]["type"] == "Sequence"
    assert data["pre_tokenizer"]["pretokenizers"][1]["type"] == "ByteLevel"


def test_byte_table_is_a_bijection_over_256_bytes():
    fwd, inv = bytes_to_unicode(), unicode_to_bytes()
    assert len(fwd) == 256 and len(set(fwd.values())) == 256
    assert all(inv[fwd[b]] == b for b in range(256))
    assert fwd[ord("a")] == "a"
    assert fwd[0x20] == "Ġ"  # space is shown as the familiar G with dot
    assert not any(c.isspace() for c in fwd.values())


def test_token_bytes_roundtrip():
    for data in (b"hello", "ہے".encode(), bytes(range(256))):
        assert token_to_bytes(bytes_to_token(data)) == data
    assert bytes_to_token("ا".encode()) == "Ø§"  # bytes D8 A7


def test_token_to_bytes_rejects_characters_outside_the_table():
    # A literal space is shown as U+0120 in byte-level tokens, so a real space is not in the table.
    with pytest.raises(ValueError, match="not a byte-level token"):
        token_to_bytes("a b")
    # Special tokens made of printable ASCII do look like byte-level tokens; they are ASCII only.
    assert token_to_bytes("<|endoftext|>") == b"<|endoftext|>"


def test_same_tokenizer_detects_every_kind_of_difference(tiny_base_json):
    import copy

    from zubaan.extend.tokenizer_info import same_tokenizer

    assert same_tokenizer(tiny_base_json, copy.deepcopy(tiny_base_json))
    for mutate in (
        lambda d: d["model"]["vocab"].update({"zz": 9999}),
        lambda d: d["model"]["merges"].pop(),
        lambda d: d["added_tokens"].pop(),
        lambda d: d.update(normalizer=None),
        lambda d: d["pre_tokenizer"]["pretokenizers"].pop(),
    ):
        other = copy.deepcopy(tiny_base_json)
        mutate(other)
        assert not same_tokenizer(tiny_base_json, other)


@pytest.mark.slow
def test_qwen3_0_6b_and_1_7b_share_one_tokenizer():
    """One set of merges serves both models, which is what a shared-vocabulary drafter needs."""
    from zubaan.extend.tokenizer_info import same_tokenizer

    big, small = load_tokenizer_json("Qwen/Qwen3-1.7B"), load_tokenizer_json("Qwen/Qwen3-0.6B")
    assert same_tokenizer(big, small)
