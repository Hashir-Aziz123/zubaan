import pytest
from conftest import make_llama_dir

from zubaan.export.gguf import find_llama_cpp
from zubaan.export.parity import (
    ParityResult,
    check_parity,
    compare_ids,
    first_difference,
    llama_ids,
    parse_ids,
    stress_texts,
    tokenize_command,
)


class FakeProc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


class CharTok:
    """Hugging Face stand-in: ids are code points."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def fake_llama(map_ids=lambda ids: ids):
    """A ``run`` that tokenizes the text file the command points at, like llama-tokenize."""

    def run(cmd, **kwargs):
        with open(cmd[cmd.index("-f") + 1], "rb") as f:
            text = f.read().decode("utf-8")
        ids = map_ids([ord(c) for c in text])
        return FakeProc("noise line\n" + str(ids) + "\n")

    return run


def test_first_difference():
    assert first_difference([1, 2, 3], [1, 2, 3]) is None
    assert first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert first_difference([1, 2], [1, 2, 3]) == 2  # one is a prefix of the other
    assert first_difference([], []) is None


def test_compare_ids_counts_and_describes_mismatches():
    texts = ["aa", "bb", "cc"]
    res = compare_ids(texts, [[1, 2], [3, 4], [5]], [[1, 2], [3, 9], [5, 6]])
    assert (res.n_texts, res.n_mismatch, res.passed) == (3, 2, False)
    assert res.mismatches[0]["text"] == "bb" and res.mismatches[0]["position"] == 1
    assert res.mismatches[1]["llama_length"] == 2
    ok = compare_ids(["a"], [[1]], [[1]])
    assert ok.passed and ok.mismatches == []


def test_compare_ids_limits_examples_and_checks_lengths():
    n = 30
    res = compare_ids(["t"] * n, [[1]] * n, [[2]] * n, max_examples=5)
    assert res.n_mismatch == n and len(res.mismatches) == 5
    with pytest.raises(ValueError, match="same length"):
        compare_ids(["a"], [[1]], [])


def test_empty_result_does_not_count_as_passed():
    assert ParityResult(0, 0).passed is False


def test_parse_ids():
    assert parse_ids("[1, 22, 333]\n") == [1, 22, 333]
    assert parse_ids("some log\n[]\n") == []
    with pytest.raises(ValueError, match="no id list"):
        parse_ids("error: could not load model")


def test_tokenize_command_preserves_bytes_and_disables_escapes(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    cmd = tokenize_command(tools, tmp_path / "m.gguf", tmp_path / "t.txt")
    for flag in ("--ids", "--no-bos", "--no-escape", "--log-disable"):
        assert flag in cmd
    assert cmd[cmd.index("-f") + 1].endswith("t.txt")


def test_llama_ids_keeps_order_and_exact_text(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    texts = ["", "ab", "a\nb\\n", "اردو"]
    got = llama_ids(tools, tmp_path / "m.gguf", texts, run=fake_llama(), workers=3)
    assert got == [[ord(c) for c in t] for t in texts]  # backslash-n stayed a backslash and n


def test_llama_ids_reports_tool_failure(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    with pytest.raises(RuntimeError, match="llama-tokenize failed"):
        llama_ids(tools, tmp_path / "m.gguf", ["a"], run=lambda *a, **k: FakeProc("", 1, "bad"))


def test_check_parity_passes_and_fails(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    texts = ["hello", "پاکستان", ""]
    ok = check_parity(CharTok(), tools, tmp_path / "m.gguf", texts, run=fake_llama())
    assert ok.passed and ok.n_texts == 3
    # A llama.cpp that merges two ids in one text must be caught.
    bad_run = fake_llama(lambda ids: ids[:1] if len(ids) == 5 else ids)
    bad = check_parity(CharTok(), tools, tmp_path / "m.gguf", texts, run=bad_run)
    assert not bad.passed and bad.n_mismatch == 1 and bad.mismatches[0]["text"] == "hello"
    with pytest.raises(ValueError, match="no texts"):
        check_parity(CharTok(), tools, tmp_path / "m.gguf", [], run=fake_llama())


def test_stress_set_covers_the_hard_cases():
    texts = stress_texts()
    assert "" in texts and any("🚀" in t for t in texts)
    assert any(len(t) > 2000 for t in texts)
    assert any("<|im_start|>" in t for t in texts)
    assert any("\r\n" in t for t in texts)
    assert any(any("a" <= c <= "z" for c in t) and any("؀" <= c <= "ۿ" for c in t) for t in texts)


class NfcHf:
    """Hugging Face stand-in that, like Qwen's tokenizer, applies NFC before tokenizing."""

    def encode(self, text, add_special_tokens=False):
        import unicodedata

        return [ord(c) for c in unicodedata.normalize("NFC", text)]


# shadda (class 33) before damma (class 31): NFC puts the damma first.
NON_NFC = "ت\u0651\u064f"


def test_nfc_only_mismatch_is_explained_and_does_not_fail(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    texts = [NON_NFC, "hello", "پاکستان"]
    res = check_parity(NfcHf(), tools, tmp_path / "m.gguf", texts, run=fake_llama())
    assert res.n_mismatch == 1 and res.n_nfc_only == 1 and res.n_real == 0
    assert res.passed and res.mismatches == []
    assert res.nfc_only[0]["position"] == 1  # the swapped marks


def test_a_real_bug_on_a_non_nfc_text_still_fails(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    # llama.cpp drops the last id for this text even after NFC, so NFC does not explain it.
    run = fake_llama(lambda ids: ids[:-1] if len(ids) == 3 else ids)
    res = check_parity(NfcHf(), tools, tmp_path / "m.gguf", [NON_NFC, "hello"], run=run)
    assert res.n_mismatch == 1 and res.n_nfc_only == 0 and res.n_real == 1
    assert not res.passed and res.mismatches[0]["text"] == NON_NFC


def test_a_mismatch_on_an_nfc_text_is_never_excused(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    run = fake_llama(lambda ids: ids[::-1])  # llama.cpp reverses every text
    res = check_parity(NfcHf(), tools, tmp_path / "m.gguf", ["abc"], run=run)
    assert res.n_real == 1 and res.n_nfc_only == 0 and not res.passed
