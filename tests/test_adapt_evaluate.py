import pytest

from zubaan.adapt.evaluate import comparison_rows, run_model_eval, success_checks
from zubaan.config import EvaluateConfig


def _row(vocab, urdu, flores_urdu, english, bel_urdu, ci, bel_eng=0.45):
    def bpc(x):
        return {"bits_per_character": x, "tokens_per_char": 0.3}

    return {
        "vocab_size": vocab,
        "bpc": {
            "urdu_heldout": bpc(urdu),
            "urdu_flores": bpc(flores_urdu),
            "english_flores": bpc(english),
        },
        "belebele": {
            "urd_Arab": {"acc_char": bel_urdu, "acc_char_ci": ci},
            "eng_Latn": {"acc_char": bel_eng, "acc_char_ci": [0.41, 0.48]},
        },
    }


def results(adapted_urdu=2.2, adapted_en=1.22, adapted_bel=0.40, ci=(0.37, 0.43)):
    """Stage 2 numbers for base and untrained; the adapted row is a stand-in."""
    return {
        "order": ["base", "untrained", "adapted"],
        "variants": {
            "base": _row(151669, 2.31, 2.34, 1.208, 0.334, [0.304, 0.366]),
            "untrained": _row(161669, 3.57, 3.56, 1.21, 0.254, [0.227, 0.284]),
            "adapted": _row(161669, adapted_urdu, 2.3, adapted_en, adapted_bel, list(ci)),
        },
    }


def by_name(checks):
    return {c["name"].split(" (")[0]: c for c in checks}


def test_all_checks_pass_with_a_real_belebele_gain():
    # Worked example: English 1.22 - 1.208 = +0.012 <= 0.02; Urdu 2.2 < 3.57 and <= 2.31;
    # Belebele 0.40 > 0.334 and lower end 0.37 > upper end 0.366, so the gain is real.
    checks = success_checks(results(), "base", "untrained", "adapted")
    assert [c["passed"] for c in checks] == [True, True, True, True]
    assert "real gain" in checks[3]["detail"]


def test_overlapping_intervals_pass_but_are_called_noise():
    checks = success_checks(
        results(adapted_bel=0.35, ci=(0.32, 0.38)), "base", "untrained", "adapted"
    )
    assert checks[3]["passed"] is True
    assert "not distinguishable from noise" in checks[3]["detail"]


def test_english_regression_fails_and_tolerance_is_inclusive():
    bad = success_checks(results(adapted_en=1.25), "base", "untrained", "adapted")
    assert bad[0]["passed"] is False
    edge = success_checks(results(adapted_en=1.208 + 0.02), "base", "untrained", "adapted")
    assert edge[0]["passed"] is True


def test_urdu_stretch_can_fail_while_the_must_passes():
    checks = success_checks(results(adapted_urdu=2.8), "base", "untrained", "adapted")
    assert checks[1]["passed"] is True
    assert checks[2]["passed"] is False
    assert checks[3]["passed"] is True


def test_belebele_not_above_the_base_fails():
    checks = success_checks(
        results(adapted_bel=0.30, ci=(0.27, 0.33)), "base", "untrained", "adapted"
    )
    assert checks[3]["passed"] is False
    assert "not above" in checks[3]["detail"]


def test_missing_model_raises():
    with pytest.raises(ValueError, match="no results for 'nope'"):
        success_checks(results(), "base", "untrained", "nope")


def test_comparison_rows_have_deltas_to_the_first_model():
    rows = comparison_rows(results())
    assert [r["model"] for r in rows] == ["base", "untrained", "adapted"]
    assert rows[0]["bpc_urdu_heldout_delta"] == 0
    assert rows[1]["bpc_urdu_heldout_delta"] == pytest.approx(3.57 - 2.31)
    assert rows[2]["belebele_urd_Arab_ci"] == [0.37, 0.43]


def test_config_defaults_follow_the_stage_2_protocol():
    cfg = EvaluateConfig(models={"base": "Qwen/Qwen3-1.7B"})
    assert (cfg.quality_texts, cfg.max_chars, cfg.belebele_questions, cfg.dtype) == (
        1000,
        400,
        None,
        "float32",
    )


torch = pytest.importorskip("torch")


def test_run_model_eval_scores_each_model_and_resumes(
    tmp_path, tiny_qwen3, tiny_base_json, monkeypatch
):
    """Two 'folders' (same tiny model and tokenizer) are scored; a rerun skips both."""
    import zubaan.adapt.evaluate as ev
    from zubaan.extend.tokenizer_info import inspect_tokenizer_json

    model = tiny_qwen3(inspect_tokenizer_json(tiny_base_json).total_vocab_size)
    loads, logs = [], []
    out = tmp_path / "r.json"

    def fake_load(path):
        loads.append(path)
        return model

    monkeypatch.setattr(ev, "load_tokenizer_json", lambda path: tiny_base_json)
    sets = {"urdu_heldout": ["اردو متن ہے", "دوسرا متن"]}
    r = run_model_eval({"a": "pa", "b": "pb"}, fake_load, sets, {}, out, log=logs.append)
    assert list(r["variants"]) == ["a", "b"]
    assert r["order"] == ["a", "b"]
    assert r["variants"]["a"]["bpc"]["urdu_heldout"]["bits_per_character"] > 0
    assert loads == ["pa", "pb"]
    run_model_eval({"a": "pa", "b": "pb"}, fake_load, sets, {}, out, log=logs.append)
    assert loads == ["pa", "pb"]  # nothing reloaded
    assert any("already done" in line for line in logs)
