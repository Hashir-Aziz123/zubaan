import json

import pytest

torch = pytest.importorskip("torch")

from zubaan.extend.build import (  # noqa: E402
    METADATA_FILE,
    build_extended_model,
    saved_embedding_rows,
    verify_extended_folder,
)
from zubaan.extend.sweep import learn_urdu_merges  # noqa: E402
from zubaan.extend.tokenizer_info import inspect_tokenizer_json  # noqa: E402


@pytest.fixture
def setup(tmp_path, tiny_base_json, urdu_lines, tiny_qwen3):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "tokenizer.json").write_text(json.dumps(tiny_base_json), encoding="utf-8")
    (base_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    merges = learn_urdu_merges(urdu_lines, tiny_base_json, 80, min_count=2)
    rows = inspect_tokenizer_json(tiny_base_json).total_vocab_size + 8  # padded like Qwen3
    return base_dir, merges, lambda: tiny_qwen3(rows, seed=3), tmp_path / "out"


def build(setup, tiny_base_json, method="mean", n=50, **kw):
    base_dir, merges, factory, out = setup
    return build_extended_model(
        factory, tiny_base_json, base_dir, merges, n, method, out, base_model_name="tiny", **kw
    )


def test_build_saves_model_tokenizer_and_metadata(setup, tiny_base_json, urdu_lines, english_lines):
    meta = build(setup, tiny_base_json)
    out = setup[3]
    for name in ("config.json", "model.safetensors", "tokenizer.json", "vocab.json", "merges.txt"):
        assert (out / name).exists(), name
    assert (out / "tokenizer_config.json").exists()
    total = inspect_tokenizer_json(tiny_base_json).total_vocab_size + 50
    assert meta["vocab_size"] == total and meta["n_new_tokens"] == 50
    assert meta["init_method"] == "mean" and meta["trained"] is False
    assert json.loads((out / METADATA_FILE).read_text(encoding="utf-8")) == meta
    assert saved_embedding_rows(out) == total
    checked = verify_extended_folder(
        out, tiny_base_json, urdu_lines, english_lines, expected_rows=total
    )
    assert checked["embedding_rows"] == total and checked["urdu_texts"] == len(urdu_lines)


def test_saved_model_reloads_and_runs(setup, tiny_base_json):
    transformers = pytest.importorskip("transformers")
    build(setup, tiny_base_json)
    model = transformers.AutoModelForCausalLM.from_pretrained(setup[3])
    total = inspect_tokenizer_json(tiny_base_json).total_vocab_size + 50
    out = model(input_ids=torch.tensor([[1, total - 1]])).logits
    assert out.shape == (1, 2, total)  # the last new id is a valid input and output


@pytest.mark.parametrize("method", ["random", "focus"])
def test_build_other_methods(setup, tiny_base_json, urdu_lines, method):
    meta = build(setup, tiny_base_json, method, focus_lines=urdu_lines * 4, focus_dim=8)
    assert meta["init_method"] == method


def test_build_rejects_zero_tokens(setup, tiny_base_json):
    with pytest.raises(ValueError, match="at least 1"):
        build(setup, tiny_base_json, n=0)


def test_verify_catches_a_wrong_config_vocab_size(setup, tiny_base_json, urdu_lines, english_lines):
    build(setup, tiny_base_json)
    out = setup[3]
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    cfg["vocab_size"] += 1
    (out / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_size"):
        verify_extended_folder(out, tiny_base_json, urdu_lines, english_lines)


def test_verify_catches_a_wrong_row_count(setup, tiny_base_json, urdu_lines, english_lines):
    build(setup, tiny_base_json)
    with pytest.raises(ValueError, match="rows"):
        verify_extended_folder(
            setup[3], tiny_base_json, urdu_lines, english_lines, expected_rows=12345
        )


def test_verify_catches_a_tokenizer_that_changes_english(
    setup, tiny_base_json, urdu_lines, english_lines
):
    build(setup, tiny_base_json)
    out = setup[3]
    tok = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    tok["pre_tokenizer"]["pretokenizers"].pop(0)  # a different pre-tokenizer
    (out / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    with pytest.raises(AssertionError, match="pre_tokenizer"):
        verify_extended_folder(out, tiny_base_json, urdu_lines, english_lines)
