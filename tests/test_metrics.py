import pytest

from zubaan.metrics import bits_per_character, words_per_second


def test_words_per_second():
    assert words_per_second(40, 4.0) == pytest.approx(10.0)
    assert words_per_second(38, 1.8) == pytest.approx(21.11, abs=0.01)


def test_bits_per_character():
    # 2,080 nats over 1,000 characters is about 3.0 bits per character.
    assert bits_per_character(2080, 1000) == pytest.approx(3.0, abs=0.01)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        words_per_second(10, 0)
    with pytest.raises(ValueError):
        bits_per_character(1.0, 0)
