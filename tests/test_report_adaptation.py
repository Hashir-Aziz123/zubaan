"""Stage 3 report section: every number and every direction word comes from the results."""

import pytest

from zubaan.report.adaptation import (
    adaptation_section,
    adaptation_summary,
    check_adapt_results,
)


def _variant(bpc: dict, tpc: float, bel: tuple, vocab: int) -> dict:
    sets = {
        s: {"bits_per_character": v, "tokens_per_char": tpc if s.startswith("urdu") else 0.21}
        for s, v in bpc.items()
    }
    acc, lo, hi = bel
    belebele = {
        "urd_Arab": {"acc_char": acc, "acc_char_ci": [lo, hi]},
        "eng_Latn": {"acc_char": 0.45, "acc_char_ci": [0.42, 0.48]},
    }
    return {"bpc": sets, "belebele": belebele, "vocab_size": vocab}


@pytest.fixture
def adapt():
    return {
        "order": ["base", "extended-untrained", "adapted"],
        "meta": {"settings": {"n_texts": {"urdu_heldout": 1000}, "dtype": "float32"}},
        "variants": {
            "base": _variant(
                {"urdu_heldout": 2.31, "urdu_flores": 2.339, "english_flores": 1.208},
                0.668,
                (0.334, 0.304, 0.366),
                151669,
            ),
            "extended-untrained": _variant(
                {"urdu_heldout": 3.57, "urdu_flores": 3.562, "english_flores": 1.21},
                0.275,
                (0.253, 0.226, 0.283),
                161669,
            ),
            "adapted": _variant(
                {"urdu_heldout": 1.486, "urdu_flores": 1.718, "english_flores": 1.066},
                0.275,
                (0.34, 0.31, 0.372),
                161669,
            ),
        },
    }


def test_summary_states_cost_and_quality_from_the_numbers(adapt):
    s = adaptation_summary(adapt)
    # 0.668 / 0.275 = 2.43x fewer tokens; 2.339 - 1.718 = 0.621 fewer bits per character.
    assert "2.43x fewer tokens" in s and "0.62 fewer bits per character" in s
    assert "2.339 to 1.718" in s and "not yet known" in s


def test_section_table_checks_and_reading(adapt):
    md = adaptation_section(adapt, stage2_base=adapt["variants"]["base"])
    assert "| `adapted` | 161,669 | 0.275 | 1.486 | 1.718 | 1.066 |" in md
    assert "**PASS, within noise**: Belebele" in md  # 0.340 vs 0.334 with overlapping intervals
    assert "**PASS**: Urdu held-out bits per character below the untrained model" in md
    assert "went past the base" in md
    assert "not evidence that English got better" in md  # English went down
    assert "did not measurably change" in md
    assert "reproduced its stage 2 bits per character to three decimals" in md
    assert "not attributed to the tokenizer" in md
    assert "Serving the adapted model" not in md  # no serving results given


def test_section_words_follow_the_numbers_when_they_change(adapt):
    a = adapt["variants"]["adapted"]
    a["bpc"]["urdu_flores"]["bits_per_character"] = 2.5  # worse than the base
    a["bpc"]["english_flores"]["bits_per_character"] = 1.25  # English got worse
    a["belebele"]["urd_Arab"] = {"acc_char": 0.45, "acc_char_ci": [0.42, 0.48]}
    md = adaptation_section(adapt)
    assert "went past the base" not in md
    assert "not evidence that English got better" not in md
    assert "English moved by +0.042" in md
    assert "the intervals separate" in md


def test_serving_part_reports_parity_quant_cost_and_the_thinking_problem(adapt):
    export = {
        "gguf": {
            "llama_cpp_tag": "b12477",
            "pre_tokenizer": "qwen2",
            "quantized": {"Q4_K_M": "m-Q4_K_M.gguf"},
            "imatrix_calibration": "Urdu Wikipedia (training split)",
            "parity": {"n_texts": 10, "n_mismatch": 1, "n_nfc_only": 1, "n_real_mismatch": 0},
        }
    }
    q = {"value": 0.0341}
    check = {
        "threads": 4,
        "decoding": "greedy",
        "kld_settings": {"ctx": 512, "chunks": 20},
        "quant_cost": {
            "urdu": {
                "quants": {
                    "Q4_K_M": {
                        "mean_kld": q,
                        "same_top_p_percent": {"value": 85.9},
                        "ppl_ratio": {"value": 1.0247},
                    }
                }
            }
        },
        "samples": {"Q4_K_M": [{"tokens_per_s": 14.0}, {"tokens_per_s": 16.0}]},
    }
    vllm = {
        "passed": True,
        "vllm": "0.34.0",
        "gpu": "Tesla T4",
        "dtype": "half",
        "tokenization_identical": 5,
        "samples": {"chat_thinking": {"text": "<think>\nreasoning only", "finish_reason": "stop"}},
    }
    md = adaptation_section(adapt, export, check, vllm)
    assert "9 of 10 texts give identical ids" in md and "0 differ for real" in md
    assert "| FLORES Urdu | Q4_K_M | 0.0341 | 85.9% | 1.0247 |" in md
    assert "Q4_K_M 15.0" in md  # median of 14 and 16
    assert "ended the turn inside it" in md and "serve it with thinking off" in md
    vllm["samples"]["chat_thinking"]["text"] = "<think>x</think>جواب"
    assert "ended the turn inside it" not in adaptation_section(adapt, export, check, vllm)


def test_missing_model_is_refused(adapt):
    del adapt["variants"]["extended-untrained"]
    with pytest.raises(ValueError, match="extended-untrained"):
        check_adapt_results(adapt)


def test_chart_is_written(adapt, tmp_path):
    pytest.importorskip("matplotlib")
    from zubaan.report.charts import adaptation_quality_chart

    out = adaptation_quality_chart(adapt["variants"], adapt["order"], tmp_path / "a.png")
    assert out.stat().st_size > 1000
    variants = {**adapt["variants"], "control": adapt["variants"]["base"]}
    with pytest.raises(ValueError, match="no chart color"):
        adaptation_quality_chart(variants, ["base", "control"], tmp_path / "b.png")
