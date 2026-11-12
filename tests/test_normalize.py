from zubaan.text.normalize import normalize_urdu, split_words


def test_maps_arabic_lookalikes_to_urdu():
    # Arabic yeh and kaf become Farsi yeh and keheh.
    assert normalize_urdu("يك") == "یک"


def test_strips_diacritics_and_tatweel():
    assert normalize_urdu("کَتـاب") == "کتاب"


def test_keeps_diacritics_when_asked():
    assert "َ" in normalize_urdu("کَتاب", strip_diacritics=False)


def test_collapses_whitespace():
    assert normalize_urdu("  ایک   دو \n تین ") == "ایک دو تین"


def test_split_words_handles_urdu_punctuation():
    assert split_words("یہ ایک جملہ ہے۔ کیا، واقعی؟") == ["یہ", "ایک", "جملہ", "ہے", "کیا", "واقعی"]


def test_zwnj_does_not_split_a_word():
    assert split_words("خوش‌قسمت") == ["خوش‌قسمت"]
