import json

import pytest

from zubaan.report.extension import build_extension_markdown, build_extension_report


def bpc(v):
    return {"bits_per_character": v, "tokens_per_char": 0.3}


def bel(acc, lo, hi):
    return {"acc": acc, "acc_char": acc, "acc_char_ci": [lo, hi]}


def variant(n, method, h, f, e, urd, focus=None):
    v = {
        "n_new_tokens": n,
        "method": method,
        "vocab_size": 151669 + n,
        "bpc": {
            "urdu_heldout": bpc(h),
            "urdu_flores": bpc(f),
            "english_flores": bpc(e),
        },
        "belebele": {
            "urd_Arab": bel(urd, urd - 0.03, urd + 0.03),
            "eng_Latn": bel(0.45, 0.42, 0.48),
        },
        "init": {"focus": focus} if focus else {},
    }
    return v


QUALITY = {
    "model": "Org/Model-1B",
    "meta": {
        "platform": "Linux",
        "python": "3.13",
        "packages": {"torch": "2.11"},
        "model_revisions": {"Org/Model-1B": "abc123"},
        "settings": {"n_texts": {"urdu_heldout": 10, "urdu_flores": 10, "english_flores": 10}},
    },
    "variants": {
        "base": variant(0, None, 2.0, 2.0, 1.2, 0.33),
        "n10000-mean": variant(10000, "mean", 3.0, 3.1, 1.21, 0.26),
        "n10000-focus": variant(
            10000, "focus", 3.1, 3.2, 1.21, 0.25,
            focus={"n_new": 10000, "n_with_vector": 9000, "n_fallback_to_mean": 1000},
        ),
        "n10000-random": variant(10000, "random", 5.0, 5.2, 1.22, 0.26),
    },
}  # fmt: skip


def sweep_row(n, cost):
    return {
        "n_new_tokens": n,
        "vocab_size": 151669 + n,
        "urdu_tokens_per_word": 1.24 * cost,
        "english_tokens_per_word": 1.24,
        "urdu_chars_per_token": 4.0 / cost,
        "cost_multiplier": cost,
        "english_different": 0,
        "urdu_roundtrip_failures": 0,
    }


SWEEP = {"rows": [sweep_row(0, 3.0), sweep_row(10000, 1.5)]}

EXPORT = {
    "gguf": {
        "llama_cpp_tag": "b1",
        "pre_tokenizer": "qwen2",
        "fingerprint": "f" * 64,
        "converter_patched": False,
        "quantized": {"Q4_K_M": "m-Q4_K_M.gguf"},
        "imatrix_calibration": "Urdu Wikipedia",
        "parity": {"n_texts": 100, "n_mismatch": 0},
    }
}


def test_markdown_numbers_come_from_the_results():
    md = build_extension_markdown(SWEEP, QUALITY, EXPORT)
    assert "from **3.00x** to **1.50x**" in md
    assert "bits per character on held-out Urdu goes from 2.00 (base) to 3.00" in md
    assert "50% higher" in md
    assert "Org/Model-1B" in md and "`abc123`" in md
    assert "9,000 of 10,000 new tokens" in md and "1,000 fell back" in md
    assert "Random initialization is the worst in every comparison." in md
    assert "differ by at most 0.10 bits per character" in md
    assert "100 of 100 texts" in md and "`m-Q4_K_M.gguf`" in md
    assert "recognized the extended tokenizer without any change" in md
    assert "\u2014" not in md  # no em dashes


def test_markdown_without_export_has_no_gguf_section():
    md = build_extension_markdown(SWEEP, QUALITY, None)
    assert "## GGUF export" not in md and "parity" not in md.lower().split("limitations")[0]


def test_verdict_notices_when_random_is_not_worst():
    q = json.loads(json.dumps(QUALITY))
    q["variants"]["n10000-random"]["bpc"]["urdu_heldout"]["bits_per_character"] = 2.5
    assert "No initialization is consistently the worst." in build_extension_markdown(SWEEP, q)


def test_markdown_requires_a_base_variant():
    q = {"model": "m", "variants": {"n1-mean": QUALITY["variants"]["n10000-mean"]}}
    with pytest.raises(ValueError, match="base"):
        build_extension_markdown(SWEEP, q)


def test_build_report_writes_markdown_and_charts(tmp_path):
    pytest.importorskip("matplotlib")
    sweep, quality, export = tmp_path / "s.json", tmp_path / "q.json", tmp_path / "e.json"
    sweep.write_text(json.dumps(SWEEP), encoding="utf-8")
    quality.write_text(json.dumps(QUALITY), encoding="utf-8")
    export.write_text(json.dumps(EXPORT), encoding="utf-8")
    out = tmp_path / "report"
    path = build_extension_report(sweep, quality, out, export)
    assert path == out / "index.md"
    for name in ("extension_cost.png", "extension_quality.png"):
        assert (out / name).stat().st_size > 5000
    with pytest.raises(ValueError, match="extend sweep"):
        build_extension_report(tmp_path / "none.json", quality, out)
    with pytest.raises(ValueError, match="extend quality"):
        build_extension_report(sweep, tmp_path / "none.json", out)


def test_report_explains_nfc_only_parity_differences():
    export = json.loads(json.dumps(EXPORT))
    export["gguf"]["parity"] = {"n_texts": 100, "n_mismatch": 4, "n_nfc_only": 4}
    md = build_extension_markdown(SWEEP, QUALITY, export)
    assert "96 of 100 texts, and the other 4 differ only because the text is not in NFC form" in md
    assert "The other 4 texts are not in Unicode NFC form" in md
    assert "No text differed." in build_extension_markdown(SWEEP, QUALITY, EXPORT)
