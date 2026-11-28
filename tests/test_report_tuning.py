"""Stage 4 report section: numbers and direction words come from the results."""

import pytest

from zubaan.report.extension import build_extension_markdown
from zubaan.report.tuning import check_tune_results, tuning_section, tuning_summary
from zubaan.tune.evaluate import stage4_checks


def _group(answered, right, hit, loops):
    return {
        "n": 60,
        "answered": answered,
        "answer_rate": answered / 60,
        "hit_limit": hit,
        "loops": loops,
        "right_language": right,
        "language_due": 39,
        "right_language_rate": right / 39,
    }


def _summary(on, off):
    return {
        m: {"all": _group(*g), "urd": _group(*g)} for m, g in (("think", on), ("no_think", off))
    }


def _variant(fl, ho, en, bel):
    return {
        "bpc": {
            "urdu_flores": {"bits_per_character": fl},
            "urdu_heldout": {"bits_per_character": ho},
            "english_flores": {"bits_per_character": en},
        },
        "belebele": {
            "urd_Arab": {"acc_char": bel, "acc_char_ci": [bel - 0.03, bel + 0.03]},
            "eng_Latn": {"acc_char": 0.5, "acc_char_ci": [0.467, 0.533]},
        },
    }


@pytest.fixture
def results():
    chat = {
        "base": _summary((58, 32, 22, 29), (60, 37, 22, 26)),
        "stage3": _summary((13, 8, 8, 11), (60, 37, 12, 12)),
        "stage4": _summary((60, 38, 9, 23), (60, 38, 7, 13)),
    }
    quality = {
        "variants": {
            "stage3": _variant(1.718, 1.486, 1.066, 0.340),
            "stage4": _variant(1.735, 1.534, 1.074, 0.351),
        }
    }
    checks = {
        "checks": stage4_checks(chat, quality["variants"]),
        "chat": chat,
        "vllm": "0.34.0",
        "gpu": "Tesla T4",
    }
    return checks, quality


def test_summary_states_the_thinking_rates_and_the_checks(results):
    s = tuning_summary(*results)
    assert "answered 60 of 60 prompts, against 13 for Stage 3 and 58 for the base" in s
    assert "5 of 5 success checks pass" in s and "1.735" in s and "Claude" in s


def test_section_tables_checks_and_reading(results):
    md = tuning_section(*results)
    assert "| Stage 4 | vLLM 0.34.0, float16, Tesla T4 | on | 60/60 | 38/39 | 9 | 23 |" in md
    assert "| Stage 3 | 1.718 | 1.486 | 1.066 | 34.0% (31.0 to 37.0) |" in md
    # No base row given: the recorded notebook 08 numbers, without held-out Urdu.
    assert "| base | 2.339 | n/a | 1.208 | 33.4% (30.4 to 36.6) | n/a |" in md
    assert md.count("**PASS**") == 5
    assert "barely moved" in md and "1.718 (Stage 3) to 1.735" in md
    assert "Q4_K_M" not in md  # no serving results given, only vLLM


def test_reading_follows_the_numbers(results):
    checks, quality = results
    quality["variants"]["stage4"]["bpc"]["urdu_heldout"]["bits_per_character"] = 1.60
    md = tuning_section(checks, quality)
    assert "got worse by up to 0.114" in md and "barely moved" not in md


def test_serving_part_with_the_q4_chat_check_and_ollama(results):
    checks, quality = results
    q4 = {
        "seconds": 4340.0,
        "server": {"command": ["-m", "x.gguf", "-np", "1", "-t", "4"]},
        "summary": _summary((60, 38, 11, 18), (60, 39, 5, 10)),
    }

    def run(mode, answered, thought, reason="stop"):
        return {
            "mode": mode,
            "answered": answered,
            "thought": thought,
            "done_reason": reason,
            "tokens_per_s": 12.0,
        }

    ollama = {
        "ollama": "0.40.16",
        "decoding": "greedy",
        "runs": [
            run("default_thinking_on", True, True),
            run("default_thinking_on", False, True, "length"),
            run("think_false", True, False),
        ],
    }
    md = tuning_section(checks, quality, q4=q4, ollama=ollama)
    assert "| Stage 4 Q4_K_M | llama-server, CPU | off | 60/60 | 39/39 | 5 | 10 |" in md
    assert "72 minutes on 4 CPU threads" in md and "all pass" in md
    assert "answered 1 of 2 prompts (reasoning shown in 2), 1 ran to the token limit" in md
    assert "with `think: false` it answered 1 of 1 prompts" in md
    q4["summary"]["think"]["all"]["answer_rate"] = 40 / 60  # more than 10 points below the base
    assert "not all pass" in tuning_section(checks, quality, q4=q4)


def test_missing_results_are_refused(results):
    checks, quality = results
    del checks["chat"]["stage3"]
    with pytest.raises(ValueError, match="stage3"):
        check_tune_results(checks, quality)
    with pytest.raises(ValueError, match="stage4"):
        check_tune_results(results[0] | {"chat": {}}, {"variants": {"stage3": {}}})


def test_full_report_gets_the_stage4_summary_section_and_limit(results):
    from test_report_extension import QUALITY, SWEEP

    checks, quality = results
    md = build_extension_markdown(SWEEP, QUALITY, tune={"checks": checks, "quality": quality})
    assert "## Stage 4: restoring chat behavior" in md
    assert "**restored thinking mode**" in md
    assert "Stage 4 is one short chat fine-tune" in md
    assert "## Stage 4" not in build_extension_markdown(SWEEP, QUALITY)
