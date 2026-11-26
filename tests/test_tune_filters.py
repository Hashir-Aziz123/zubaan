import pytest

from zubaan.tune.filters import (
    SplitReply,
    has_loop,
    max_ngram_repeats,
    mentions_source,
    split_reply,
)


def test_reply_without_a_think_block_is_all_answer():
    assert split_reply("  اسلام آباد۔ \n") == SplitReply(None, "اسلام آباد۔", True)


def test_closed_think_block_is_split_from_the_answer():
    r = split_reply("<think>\nThe user asks.\n</think>\n\nاسلام آباد۔")
    assert r == SplitReply("The user asks.", "اسلام آباد۔", True)


def test_empty_think_block_counts_as_a_block():
    assert split_reply("<think>\n\n</think>\n\nجواب") == SplitReply("", "جواب", True)


def test_unclosed_think_block_has_no_answer():
    # The Stage 3 failure: the model reasons, then ends its turn.
    assert split_reply("<think>\nThe capital is Islamabad.") == SplitReply(
        "The capital is Islamabad.", "", False
    )


def test_ngram_repeats_worked_example():
    # "a b c" three times: the 3-word run "a b c" occurs 3 times.
    assert max_ngram_repeats("a b c a b c a b c", n=3) == 3
    assert max_ngram_repeats("a b", n=3) == 0
    with pytest.raises(ValueError):
        max_ngram_repeats("a", n=0)


def test_loop_detection():
    sentence = "اردو پاکستان کی قومی زبان ہے اور بہت سے لوگ اسے بولتے ہیں۔ "
    assert has_loop(sentence * 5)
    assert not has_loop(sentence * 2)
    assert not has_loop("ایک دو تین چار پانچ چھ سات آٹھ نو دس گیارہ بارہ")


@pytest.mark.parametrize(
    "text",
    [
        "According to the passage, Lahore is old.",
        "The background says it was founded in 1947.",
        "متن کے مطابق لاہور پرانا شہر ہے۔",
        "اقتباس میں لکھا ہے کہ",
    ],
)
def test_source_mentions_are_found(text):
    assert mentions_source(text)


@pytest.mark.parametrize(
    "text", ["Lahore is an old city.", "لاہور ایک پرانا شہر ہے۔", "A textbook on history."]
)
def test_plain_answers_do_not_mention_a_source(text):
    assert not mentions_source(text)
