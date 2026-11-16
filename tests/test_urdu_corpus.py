import json
from pathlib import Path

import pytest

from zubaan.data.urdu import (
    build_corpus,
    clean_line,
    flores_shingles,
    is_arabic_block,
    is_heldout,
    iter_parquet_text,
    overlaps_flores,
    split_paragraphs,
    urdu_fraction,
)
from zubaan.text.normalize import normalize_urdu


def test_urdu_fraction_examples():
    assert urdu_fraction("") == 0.0
    assert urdu_fraction("   ") == 0.0
    assert urdu_fraction("اردو") == 1.0
    assert urdu_fraction("abcd") == 0.0
    # 4 Urdu letters and 4 Latin letters; the space is ignored.
    assert urdu_fraction("اردو abcd") == pytest.approx(0.5)


def test_arabic_block_edges():
    assert is_arabic_block("؀") and is_arabic_block("ۿ") and is_arabic_block("ا")
    assert not is_arabic_block("׿") and not is_arabic_block("܀")
    assert not is_arabic_block("a")


def test_split_paragraphs_drops_blank_lines():
    assert split_paragraphs("a\n\n  b  \n") == ["a", "b"]
    assert split_paragraphs("") == []


def test_clean_line_keeps_urdu_sentence_and_normalizes():
    # Arabic yeh (U+064A) must become Farsi yeh (U+06CC), as in the audit.
    line = "پاکستان ايک خوبصورت ملک ہے اور بہت اچھا ہے"
    out = clean_line(line)
    assert out == normalize_urdu(line) and "ي" not in out


def test_clean_line_drops_short_english_and_mixed():
    assert clean_line("جغرافیہ") is None  # heading, too short
    assert clean_line("This is an English sentence with many words in it") is None
    assert clean_line("اردو a b c d e f g h i j k l m n o p q r s t u v") is None  # mostly Latin
    assert clean_line("") is None


def test_shingles_detect_exact_partial_and_short_overlaps(urdu_lines):
    flores = [urdu_lines[0], urdu_lines[1]]
    sh = flores_shingles(flores)
    assert overlaps_flores(urdu_lines[0], sh)  # exact copy
    # A long line that embeds 8 consecutive words of the evaluation sentence.
    words = urdu_lines[1].split()
    embedded = "کچھ اور الفاظ " + " ".join(words[:8]) + " اور مزید الفاظ"
    assert overlaps_flores(embedded, sh)
    assert not overlaps_flores(urdu_lines[5], sh)  # unrelated sentence
    # Only 7 consecutive words of the evaluation sentence is not enough.
    assert not overlaps_flores(" ".join(words[:7]) + " الگ بات", sh)
    # Short evaluation sentences are matched whole.
    short = flores_shingles(["پاکستان زندہ باد"])
    assert overlaps_flores("پاکستان زندہ باد", short)
    assert not overlaps_flores("پاکستان بہت اچھا", short)


def test_overlap_is_normalization_insensitive():
    sh = flores_shingles(["ايک دو تین چار پانچ چھ سات آٹھ نو"])  # Arabic yeh in the source
    assert overlaps_flores("ایک دو تین چار پانچ چھ سات آٹھ نو", sh)  # Farsi yeh in the text


def test_overlap_empty_line_is_false():
    assert not overlaps_flores("", flores_shingles(["ایک دو تین"]))


def test_heldout_split_is_deterministic_and_close_to_percent(urdu_lines):
    assert all(is_heldout(line, 1.0) == is_heldout(line, 1.0) for line in urdu_lines)
    lines = [f"جملہ نمبر {i} ہے" for i in range(20_000)]
    frac = sum(is_heldout(x, 5.0) for x in lines) / len(lines)
    assert 0.04 < frac < 0.06
    assert not any(is_heldout(x, 0.0) for x in lines)
    with pytest.raises(ValueError):
        is_heldout("x", 100.0)
    with pytest.raises(ValueError):
        is_heldout("x", -1.0)


def test_build_corpus_end_to_end(tmp_path, urdu_lines):
    flores = [urdu_lines[3]]
    articles = [
        "\n".join(urdu_lines[:3] + ["جغرافیہ", urdu_lines[3]]),  # heading + FLORES copy
        "\n".join([urdu_lines[0], *urdu_lines[4:9]]),  # first line is a duplicate
        "English only paragraph that must be dropped from the corpus",
    ]
    meta = build_corpus(articles, tmp_path, flores, heldout_percent=0.0)
    st = meta["stats"]
    assert st["articles"] == 3
    assert st["dropped_flores"] == 1
    assert st["dropped_duplicate"] == 1
    assert st["dropped_filter"] == 2  # "جغرافیہ" heading and the English paragraph
    train = (tmp_path / "urdu_train.txt").read_text(encoding="utf-8").splitlines()
    assert train == [normalize_urdu(x) for x in urdu_lines[:3] + urdu_lines[4:9]]
    assert urdu_lines[3] not in train
    assert st["train_lines"] == 8 and st["heldout_lines"] == 0
    saved = json.loads((tmp_path / "urdu_meta.json").read_text(encoding="utf-8"))
    assert saved["stats"] == st and len(saved["files"]["train"]["sha256"]) == 64


def test_build_corpus_train_and_heldout_are_disjoint(tmp_path, urdu_lines):
    lines = [f"{urdu_lines[i % 30]} نمبر {i} ہے" for i in range(600)]
    build_corpus(["\n".join(lines)], tmp_path, [urdu_lines[0] + " الگ"], heldout_percent=10.0)
    train = set((tmp_path / "urdu_train.txt").read_text(encoding="utf-8").splitlines())
    held = set((tmp_path / "urdu_heldout.txt").read_text(encoding="utf-8").splitlines())
    assert train and held and not (train & held)
    assert 20 < len(held) < 120  # about 10% of 600


def test_build_corpus_respects_size_cap(tmp_path, urdu_lines):
    articles = [f"{urdu_lines[i % 30]} حصہ {i} ہے" for i in range(500)]
    meta = build_corpus(
        articles, tmp_path, ["بالکل الگ جملہ ہے جو نہیں آئے گا"], max_mb=0.01, heldout_percent=0.0
    )
    assert meta["stats"]["stopped_at_cap"] is True
    assert 10_000 <= meta["stats"]["train_bytes"] < 11_000
    assert meta["stats"]["articles"] < 500


def test_build_corpus_refuses_when_guard_or_input_is_empty(tmp_path, urdu_lines):
    with pytest.raises(ValueError, match="contamination guard"):
        build_corpus(urdu_lines, tmp_path, [])
    with pytest.raises(ValueError, match="no training lines"):
        build_corpus(["English only text here"], tmp_path, [urdu_lines[0]])


def test_flores_overlap_is_rejected_even_inside_longer_paragraph(tmp_path, urdu_lines):
    flores_sentence = " ".join(urdu_lines[2].split())
    paragraph = urdu_lines[10] + " " + flores_sentence + " " + urdu_lines[11]
    with pytest.raises(ValueError, match="no training lines"):
        build_corpus([paragraph], tmp_path, [flores_sentence])


def test_iter_parquet_text_streams_rows(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"text": ["پہلا مضمون", "", None, "دوسرا مضمون"], "id": [1, 2, 3, 4]})
    path = tmp_path / "w.parquet"
    pq.write_table(table, path)
    assert list(iter_parquet_text(path, batch_size=2)) == ["پہلا مضمون", "دوسرا مضمون"]


@pytest.mark.slow
def test_real_corpus_has_no_flores_sentences():
    """The built Wikipedia file must not contain any FLORES devtest sentence or 8-word run."""
    train = Path("data/urdu/urdu_train.txt")
    flores = Path("data/flores_urd.txt")
    if not train.exists() or not flores.exists():
        pytest.skip("run zubaan data flores and zubaan data urdu first")
    sentences = [normalize_urdu(s) for s in flores.read_text(encoding="utf-8").splitlines() if s]
    shingles = flores_shingles(sentences)
    bad = 0
    with train.open(encoding="utf-8") as f:
        for line in f:
            if overlaps_flores(line.rstrip("\n"), shingles):
                bad += 1
    assert bad == 0
    held = Path("data/urdu/urdu_heldout.txt").read_text(encoding="utf-8").splitlines()
    assert not any(overlaps_flores(x, shingles) for x in held)
