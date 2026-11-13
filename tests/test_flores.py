import io
import tarfile

import pytest

from zubaan.data import flores
from zubaan.data.flores import (
    align_by_id,
    check_parallel,
    clean_sentence,
    download_flores,
    output_name,
    write_parallel,
)


def test_align_by_id_ignores_row_order():
    out = align_by_id(
        {
            "urd_Arab": [{"id": 2, "text": "دو"}, {"id": 1, "text": "ایک"}],
            "eng_Latn": [{"id": 1, "text": "one"}, {"id": 2, "text": "two"}],
        }
    )
    assert out == {"urd_Arab": ["ایک", "دو"], "eng_Latn": ["one", "two"]}


def test_align_by_id_rejects_missing_ids():
    with pytest.raises(ValueError, match="not parallel"):
        align_by_id({"a": [{"id": 1, "text": "x"}], "b": [{"id": 2, "text": "y"}]})


def test_clean_sentence_removes_line_breaks():
    assert clean_sentence("first\nsecond\r\n third ") == "first second third"


def test_clean_sentence_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        clean_sentence("  \n ")


def test_check_parallel_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        check_parallel({"a": ["x", "y"], "b": ["x"]})


def test_output_name():
    assert output_name("urd_Arab") == "flores_urd.txt"


def test_write_parallel_uses_utf8_and_unix_newlines(tmp_path):
    paths = write_parallel({"urd_Arab": ["ایک", "دو"], "eng_Latn": ["one", "two"]}, tmp_path)
    raw = paths["urd_Arab"].read_bytes()
    assert raw == "ایک\nدو\n".encode()
    assert b"\r" not in paths["eng_Latn"].read_bytes()


def _fake_archive(path):
    with tarfile.open(path, "w:gz") as tar:
        for lang, text in {"urd_Arab": "ایک\nدو\n", "eng_Latn": "one\ntwo\n"}.items():
            data = text.encode()
            info = tarfile.TarInfo(f"./flores200_dataset/devtest/{lang}.devtest")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_auto_falls_back_to_meta_archive(tmp_path, monkeypatch):
    def denied(langs, split):
        raise PermissionError("gated")

    monkeypatch.setattr(flores, "load_hf", denied)
    cache = tmp_path / ".cache"
    cache.mkdir()
    _fake_archive(cache / "flores200_dataset.tar.gz")

    result = download_flores(["urd_Arab", "eng_Latn"], out_dir=tmp_path, source="auto")
    assert result.n_lines == 2
    assert result.meta["source"] == "meta"
    assert "gated" in result.meta["fallback_reason"]
    assert (tmp_path / "flores_eng.txt").read_text(encoding="utf-8") == "one\ntwo\n"
    assert (tmp_path / "flores_meta.json").exists()


def test_invalid_split():
    with pytest.raises(ValueError, match="split"):
        download_flores(["urd_Arab", "eng_Latn"], split="train")


@pytest.mark.slow
def test_real_meta_download(tmp_path):
    result = download_flores(["urd_Arab", "eng_Latn"], out_dir=tmp_path, source="meta")
    assert result.n_lines == 1012
