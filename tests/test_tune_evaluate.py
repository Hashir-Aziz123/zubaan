"""Stage 4 success checks from chat-check summaries and quality results."""

from __future__ import annotations

import pytest

from zubaan.tune.evaluate import checks_markdown, reproduces_stage3, stage4_checks


def group(answered, n=40, hit_limit=0, right=None, due=None):
    right = answered if right is None else right
    due = n if due is None else due
    return {
        "n": n,
        "answered": answered,
        "answer_rate": answered / n,
        "hit_limit": hit_limit,
        "loops": 0,
        "right_language": right,
        "language_due": due,
        "right_language_rate": right / due,
    }


def summary(think_answered, no_think_answered, hit_limit=0, urdu_right=None):
    out = {}
    for mode, answered in (("think", think_answered), ("no_think", no_think_answered)):
        out[mode] = {
            "all": group(answered, 60, hit_limit),
            "urd": group(min(answered, 40), 40, right=urdu_right),
        }
    return out


def quality(urdu_flores, english, belebele=0.34):
    return {
        "bpc": {
            "urdu_flores": {"bits_per_character": urdu_flores},
            "english_flores": {"bits_per_character": english},
            "urdu_heldout": {"bits_per_character": 1.486},
        },
        "belebele": {
            "urd_Arab": {"acc_char": belebele, "acc_char_ci": [belebele - 0.03, belebele + 0.03]}
        },
    }


def test_all_checks_pass_for_a_good_model():
    chat = {
        "base": summary(55, 60, hit_limit=6),
        "stage3": summary(2, 60),
        "stage4": summary(52, 59, hit_limit=4),
    }
    q = {"stage3": quality(1.718, 1.066), "stage4": quality(1.74, 1.10)}
    checks = stage4_checks(chat, q)
    assert [c["passed"] for c in checks] == [True] * 5
    assert "Stage 3 3.3%" in checks[0]["detail"]
    md = checks_markdown(checks, chat)
    assert md.count("PASS") == 5 and "| stage4 | think | 52/60 |" in md


def test_each_check_can_fail():
    chat = {
        "base": summary(55, 60, hit_limit=2),
        "stage4": summary(45, 55, hit_limit=3, urdu_right=30),  # 75% vs 91.7%; 91.7%; 30/40
    }
    q = {"stage3": quality(1.718, 1.066), "stage4": quality(1.80, 1.25, belebele=0.29)}
    checks = stage4_checks(chat, q)
    assert [c["passed"] for c in checks] == [False] * 5
    assert "FLORES Urdu within 0.05 of Stage 3" in checks[4]["detail"]
    assert "Belebele Urdu not below" in checks[4]["detail"]


def test_edges_count_as_pass():
    # Exactly 10 points below the base, exactly 95%, exactly 90%, exactly 0.05 and 0.02 above.
    chat = {"base": summary(60, 60), "stage4": summary(54, 57, urdu_right=36)}
    chat["stage4"]["no_think"]["all"]["answer_rate"] = 0.95
    q = {"stage3": quality(1.718, 1.066), "stage4": quality(1.768, 1.228, belebele=0.304)}
    assert all(c["passed"] for c in stage4_checks(chat, q))


def test_missing_results_are_refused():
    with pytest.raises(ValueError, match="stage4"):
        stage4_checks({"base": summary(1, 1)}, {})


def test_stage3_reproduction():
    assert reproduces_stage3(quality(1.718, 1.066)) == []
    assert reproduces_stage3(quality(1.75, 1.066)) == ["urdu_flores: now 1.750, notebook 08 1.718"]


def test_quantized_checks_use_the_recorded_base_numbers():
    from zubaan.tune.evaluate import BASE_CHAT, quantized_checks

    assert BASE_CHAT == {"think_answer_rate": 58 / 60, "hit_limit": 44}
    good = quantized_checks(summary(52, 60, hit_limit=10))  # 86.7%: exactly 10 points below
    assert [c["passed"] for c in good] == [True] * 4
    assert "vs base 96.7%" in good[0]["detail"] and "20 vs base 44" in good[3]["detail"]
    bad = quantized_checks(summary(51, 56, hit_limit=23, urdu_right=35))
    assert [c["passed"] for c in bad] == [False] * 4
