from pathlib import Path

import pytest
from conftest import FAKE_SOURCE, H_BASE, H_EXT, H_OTHER, make_llama_dir  # noqa: E402

from zubaan.export.gguf import (
    LLAMA_CPP_TAG,
    apply_patch,
    assert_same_pretokenizer,
    convert_command,
    extract_chktxt,
    find_llama_cpp,
    find_pretokenizer_source,
    imatrix_command,
    known_pretokenizers,
    patch_converter_source,
    plan_pretokenizer,
    pretokenizer_fingerprint,
    quantize_command,
    run_checked,
    write_calibration_file,
)

# locating llama.cpp ------------------------------------------------------------------------------


def test_find_llama_cpp_and_tools(tmp_path):
    root = make_llama_dir(tmp_path)
    tools = find_llama_cpp(root)
    assert tools.convert_script == root / "convert_hf_to_gguf.py"
    assert tools.tool("llama-quantize").name == "llama-quantize.exe"
    with pytest.raises(FileNotFoundError, match="llama-bench"):
        tools.tool("llama-bench")


def test_find_llama_cpp_explains_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=LLAMA_CPP_TAG):
        find_llama_cpp(tmp_path)
    root = tmp_path / "x"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no folder"):
        find_llama_cpp(root)


def test_find_pretokenizer_source_looks_in_the_conversion_package(tmp_path):
    root = tmp_path / "l"
    (root / "conversion").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("print('thin entry point')", encoding="utf-8")
    (root / "conversion" / "base.py").write_text(FAKE_SOURCE, encoding="utf-8")
    assert find_pretokenizer_source(root) == root / "conversion" / "base.py"
    with pytest.raises(FileNotFoundError):
        find_pretokenizer_source(tmp_path)


# fingerprint -------------------------------------------------------------------------------------


def test_extract_chktxt_handles_escapes_and_quotes():
    assert extract_chktxt(FAKE_SOURCE) == "abc\n 'quoted' ق"
    with pytest.raises(ValueError, match="chktxt"):
        extract_chktxt("nothing here")


def test_fingerprint_is_sha256_of_the_id_list():
    import hashlib

    got = pretokenizer_fingerprint(lambda t: [1, 2, 3], "ignored")
    assert got == hashlib.sha256(b"[1, 2, 3]").hexdigest()
    assert pretokenizer_fingerprint(lambda t: [3, 2, 1], "x") != got


def test_known_pretokenizers_reads_value_after_comment_lines():
    assert known_pretokenizers(FAKE_SOURCE) == {H_BASE: "qwen2", H_OTHER: "llama-bpe"}


def test_plan_same_hash_needs_no_patch():
    plan = plan_pretokenizer(H_BASE, H_BASE, {H_BASE: "qwen2"})
    assert (plan.pre_type, plan.needs_patch) == ("qwen2", False)


def test_plan_unknown_hash_is_mapped_to_the_base_type():
    plan = plan_pretokenizer(H_BASE, H_EXT, {H_BASE: "qwen2"})
    assert (plan.pre_type, plan.extended_hash, plan.needs_patch) == ("qwen2", H_EXT, True)


def test_plan_known_extended_hash_with_the_same_value_is_fine():
    plan = plan_pretokenizer(H_BASE, H_EXT, {H_BASE: "qwen2", H_EXT: "qwen2"})
    assert plan.needs_patch is False


def test_plan_refuses_collisions_and_unknown_bases():
    with pytest.raises(ValueError, match="maps to 'llama-bpe'"):
        plan_pretokenizer(H_BASE, H_OTHER, {H_BASE: "qwen2", H_OTHER: "llama-bpe"})
    with pytest.raises(ValueError, match="base model's tokenizer is not recognized"):
        plan_pretokenizer(H_BASE, H_EXT, {})


def test_patch_adds_a_branch_before_the_failure_check():
    patched = patch_converter_source(FAKE_SOURCE, H_EXT, "qwen2")
    assert patched.index(H_EXT) < patched.index("if res is None:")
    assert patched.index(H_EXT) > patched.index(H_OTHER)
    ns: dict = {}
    # The patched function must still be valid Python.
    compile(patched, "<patched>", "exec")
    assert known_pretokenizers(patched)[H_EXT] == "qwen2"
    assert ns == {}


def test_patch_is_idempotent_and_needs_the_marker():
    once = patch_converter_source(FAKE_SOURCE, H_EXT, "qwen2")
    assert patch_converter_source(once, H_EXT, "qwen2") == once
    with pytest.raises(ValueError, match="if res is None"):
        patch_converter_source("def get_vocab_base_pre(): pass", H_EXT, "qwen2")


def test_apply_patch_keeps_a_backup_once(tmp_path):
    f = tmp_path / "convert.py"
    f.write_text(FAKE_SOURCE, encoding="utf-8")
    assert apply_patch(f, H_EXT, "qwen2") is True
    assert apply_patch(f, H_EXT, "qwen2") is False  # already patched
    assert (tmp_path / "convert.py.orig").read_text(encoding="utf-8") == FAKE_SOURCE
    assert H_EXT in f.read_text(encoding="utf-8")


def test_real_converter_source_if_present():
    root = Path(__file__).parent.parent / "tools" / f"llama.cpp-{LLAMA_CPP_TAG}"
    if not root.exists():
        pytest.skip("llama.cpp checkout not present")
    source = find_pretokenizer_source(root).read_text(encoding="utf-8")
    known = known_pretokenizers(source)
    assert len(known) > 50
    assert "qwen2" in known.values()
    assert len(extract_chktxt(source)) > 100
    compile(patch_converter_source(source, H_EXT, "qwen2"), "<patched>", "exec")


def test_assert_same_pretokenizer_rejects_a_changed_pretokenizer(tiny_base_json):
    import copy

    ext = copy.deepcopy(tiny_base_json)
    assert_same_pretokenizer(tiny_base_json, ext)  # identical is fine
    ext["pre_tokenizer"]["pretokenizers"].pop(0)
    with pytest.raises(ValueError):
        assert_same_pretokenizer(tiny_base_json, ext)


# commands ----------------------------------------------------------------------------------------


def test_convert_command(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    cmd = convert_command(tools, Path("m"), Path("o.gguf"), "f16", python="py")
    assert cmd == ["py", str(tools.convert_script), "m", "--outfile", "o.gguf", "--outtype", "f16"]
    with pytest.raises(ValueError, match="outtype"):
        convert_command(tools, Path("m"), Path("o.gguf"), "int3")


def test_quantize_command_with_and_without_imatrix(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    plain = quantize_command(tools, Path("a.gguf"), Path("b.gguf"), "Q4_K_M")
    assert plain[1:] == ["a.gguf", "b.gguf", "Q4_K_M"]
    with_im = quantize_command(tools, Path("a.gguf"), Path("b.gguf"), "Q8_0", Path("i.dat"))
    assert with_im[1:3] == ["--imatrix", "i.dat"] and with_im[-1] == "Q8_0"
    with pytest.raises(ValueError, match="quant must be"):
        quantize_command(tools, Path("a"), Path("b"), "Q4_K_X")


def test_imatrix_command(tmp_path):
    tools = find_llama_cpp(make_llama_dir(tmp_path))
    cmd = imatrix_command(tools, Path("m.gguf"), Path("c.txt"), Path("o.dat"), 5, 256, threads=2)
    assert cmd[1:] == [
        "-m",
        "m.gguf",
        "-f",
        "c.txt",
        "-o",
        "o.dat",
        "--chunks",
        "5",
        "-c",
        "256",
    ] + [
        "-t",
        "2",
    ]
    with pytest.raises(ValueError):
        imatrix_command(tools, Path("m"), Path("c"), Path("o"), chunks=0)


def test_calibration_file_must_be_urdu(tmp_path):
    out = tmp_path / "c.txt"
    n = write_calibration_file(["اردو متن", "", "دوسرا جملہ"], out, max_chars=1000)
    assert out.read_text(encoding="utf-8") == "اردو متن\nدوسرا جملہ\n" and n == len(
        out.read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="Urdu"):
        write_calibration_file(["only english text"], tmp_path / "e.txt")
    with pytest.raises(ValueError, match="no calibration"):
        write_calibration_file(["", "  "], tmp_path / "e.txt")


def test_calibration_file_stops_at_the_character_budget(tmp_path):
    lines = ["اردو متن"] * 100
    write_calibration_file(lines, tmp_path / "c.txt", max_chars=50)
    assert 50 <= len((tmp_path / "c.txt").read_text(encoding="utf-8")) < 70


def test_run_checked_returns_output_and_raises_with_the_tail():
    class Proc:
        def __init__(self, code, out="", err=""):
            self.returncode, self.stdout, self.stderr = code, out, err

    assert run_checked(["x"], lambda *a, **k: Proc(0, "ok"), log=lambda s: None) == "ok"
    with pytest.raises(RuntimeError, match="boom"):
        run_checked(["x"], lambda *a, **k: Proc(2, "", "line\nboom"), log=lambda s: None)
