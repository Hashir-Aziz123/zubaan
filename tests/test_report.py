import pytest

from zubaan.report.tax import build_markdown, words_per_target_pass


def _row(model, cost, status="ok"):
    return {
        "model": model,
        "status": status,
        "vocab_size": 1000,
        "target_tokens_per_word": 3.0,
        "reference_tokens_per_word": 1.25,
        "target_chars_per_token": 1.5,
        "cost_multiplier": cost,
        "target_tokens": 300,
        "reference_tokens": 100,
        "reason": "",
    }


def _lang(alpha, lo, hi):
    return {
        "n_texts": 10,
        "n_positions": 500,
        "alpha": alpha,
        "alpha_ci_low": lo,
        "alpha_ci_high": hi,
        "greedy_match": alpha,
        "greedy_ci_low": lo,
        "greedy_ci_high": hi,
        "draft_cost": 0.35,
        "draft_cost_source": "parameter ratio",
        "speedup": [{"gamma": 4, "tokens_per_pass": 2.0, "speedup": 0.83}],
        "run": {"dtype": "bfloat16", "device": "cpu", "mode": "joint"},
    }


COMPARE = {
    "rows": [
        _row("org/good", 1.4),
        _row("Qwen/Qwen3-1.7B", 3.0),
        {**_row("org/gated", 0, "skipped"), "reason": "OSError: gated"},
    ],
    "provenance": {"python": "3.11", "packages": {"zubaan": "0.0.1"}, "data": {}},
}
ACCEPTANCE = {
    "draft": "Qwen/Qwen3-0.6B",
    "target": "Qwen/Qwen3-1.7B",
    "languages": {"urd": _lang(0.55, 0.52, 0.58), "eng": _lang(0.75, 0.72, 0.78)},
}


def test_words_per_target_pass_worked_example():
    # E = (1 - 0.55^5) / 0.45 = 2.11 tokens per pass; / 2.90 tokens per word = 0.73 words.
    assert words_per_target_pass(0.55, 4, 2.90) == pytest.approx(0.728, abs=1e-3)
    with pytest.raises(ValueError):
        words_per_target_pass(0.5, 4, 0)


def test_report_has_numbers_skips_and_no_em_dash():
    md = build_markdown(COMPARE, ACCEPTANCE, "Qwen/Qwen3-1.7B")
    assert "# The Urdu language tax" in md
    assert "**3.00x**" in md
    assert "`org/gated`: OSError: gated" in md
    assert "intervals do not overlap" in md
    assert "Words per target step" in md
    assert "\u2014" not in md


def test_report_says_no_clear_difference_when_intervals_overlap():
    acc = {**ACCEPTANCE, "languages": {"urd": _lang(0.7, 0.6, 0.8), "eng": _lang(0.72, 0.65, 0.8)}}
    assert "no clear difference" in build_markdown(COMPARE, acc, "Qwen/Qwen3-1.7B")


def test_report_without_acceptance():
    assert "Not run yet" in build_markdown(COMPARE, None, "Qwen/Qwen3-1.7B")


def test_report_needs_a_measured_tokenizer():
    with pytest.raises(ValueError):
        build_markdown({"rows": [_row("x", 0, "skipped")]}, None, "x")


def test_charts_render(tmp_path):
    pytest.importorskip("matplotlib")
    from zubaan.report import charts

    rows = COMPARE["rows"]
    assert charts.tokens_per_word_chart(rows, tmp_path / "a.png").stat().st_size > 0
    assert charts.cost_multiplier_chart(rows, tmp_path / "b.png", "Qwen/Qwen3-1.7B").exists()
    langs = ACCEPTANCE["languages"]
    assert charts.acceptance_chart(langs, {"urd": "Urdu"}, tmp_path / "c.png", "t").exists()
