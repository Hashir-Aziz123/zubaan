"""The export pipeline end to end with a fake llama.cpp: commands, order and failure handling."""

import copy
import json
from pathlib import Path

import pytest
import yaml
from conftest import make_llama_dir
from typer.testing import CliRunner

from zubaan import cli
from zubaan.config import ExportConfig
from zubaan.export.gguf import extract_chktxt, known_pretokenizers, pretokenizer_fingerprint
from zubaan.export.pipeline import run_export
from zubaan.extend.merge import extend_tokenizer_json, save_extended
from zubaan.extend.sweep import learn_urdu_merges
from zubaan.extend.tokenizer_info import inspect_tokenizer_json

runner = CliRunner()


class FakeHf:
    """Hugging Face tokenizer stand-in. ``shift`` makes the extended one hash differently."""

    def __init__(self, shift=0):
        self.shift = shift

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + self.shift for c in text]


class FakeProc:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.returncode, self.stderr = stdout, returncode, ""


class FakeLlama:
    """Plays convert, imatrix, quantize and tokenize; records every command."""

    def __init__(self, shift, tokenizer_bug=False):
        self.shift, self.bug, self.calls = shift, tokenizer_bug, []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        name = Path(cmd[0]).name
        if name.startswith("llama-tokenize"):
            with open(cmd[cmd.index("-f") + 1], "rb") as f:
                text = f.read().decode("utf-8")
            ids = [ord(c) + self.shift for c in text]
            if self.bug and "پاکستان" in text:
                ids = ids[:-1]
            return FakeProc(str(ids) + "\n")
        if name.startswith("llama-imatrix"):
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"imatrix")
        elif name.startswith("llama-quantize"):
            Path(cmd[-2]).write_bytes(b"quantized " + cmd[-1].encode())
        else:  # the converter script
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"gguf")
        return FakeProc("ok")


@pytest.fixture
def world(tmp_path, tiny_base_json, urdu_lines):
    base_json = copy.deepcopy(tiny_base_json)
    next_id = inspect_tokenizer_json(base_json).next_free_id
    base_json["added_tokens"].append(
        {
            "id": next_id,
            "content": "<|im_end|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text(json.dumps(base_json), encoding="utf-8")
    merges = learn_urdu_merges(urdu_lines, base_json, 40, min_count=2)
    model = tmp_path / "model"
    save_extended(extend_tokenizer_json(base_json, merges), base, model)
    (model / "config.json").write_text("{}", encoding="utf-8")
    # A converter source that knows the base fingerprint (computed from the fake base tokenizer).
    from conftest import FAKE_SOURCE

    chktxt = extract_chktxt(FAKE_SOURCE)
    h = pretokenizer_fingerprint(FakeHf(0).encode, chktxt)
    source = FAKE_SOURCE.replace("a" * 64, h)
    llama = make_llama_dir(tmp_path, source)
    data = tmp_path / "data"
    data.mkdir()
    (data / "urd.txt").write_text("\n".join(urdu_lines[:5]) + "\n", encoding="utf-8")
    (data / "eng.txt").write_text("hello world\nsecond line\n", encoding="utf-8")
    (data / "calib.txt").write_text("\n".join(urdu_lines) + "\n", encoding="utf-8")
    cfg = ExportConfig(
        model_dir=model,
        base_model=str(base),
        output_dir=tmp_path / "export",
        name="m",
        llama_cpp_dir=llama,
        parity_urdu=data / "urd.txt",
        parity_english=data / "eng.txt",
        calibration=data / "calib.txt",
        parity_workers=2,
    )
    return cfg, llama


def export(cfg, llama_runner, steps=("gguf", "ollama", "vllm"), shift=1):
    return run_export(
        cfg,
        steps,
        log=lambda s: None,
        run=llama_runner,
        load_tokenizer=lambda p: FakeHf(shift if "model" in p else 0),
    )


def test_full_export_order_files_and_summary(world):
    cfg, _ = world
    fake = FakeLlama(shift=1)
    summary = export(cfg, fake)
    kinds = [Path(c[0]).name.split(".")[0] for c in fake.calls]
    first_quantize = kinds.index("llama-quantize")
    assert kinds.index("llama-imatrix") < first_quantize  # importance matrix before quantizing
    # The converter runs first, and every parity call happens before the importance matrix.
    assert "convert_hf_to_gguf" in fake.calls[0][1]
    assert max(i for i, k in enumerate(kinds) if k == "llama-tokenize") < kinds.index(
        "llama-imatrix"
    )
    out = cfg.output_dir
    for name in ("m-F16.gguf", "m-Q4_K_M.gguf", "m-Q8_0.gguf", "Modelfile", "vllm.yaml"):
        assert (out / name).exists(), name
    assert (out / "serve_vllm.sh").exists() and (out / "calibration_urdu.txt").exists()
    assert summary["gguf"]["parity"]["passed"] and summary["gguf"]["parity"]["n_mismatch"] == 0
    assert summary["gguf"]["pre_tokenizer"] == "qwen2"
    assert summary["gguf"]["converter_patched"] is True  # the shifted tokenizer hashes differently
    assert summary["ollama"]["gguf"] == "m-Q4_K_M.gguf"
    assert json.loads((out / "export_summary.json").read_text(encoding="utf-8")) == summary
    # imatrix was passed to every quantize command
    for c in fake.calls:
        if Path(c[0]).name.startswith("llama-quantize"):
            assert "--imatrix" in c


def test_converter_is_not_patched_when_the_fingerprint_is_unchanged(world):
    cfg, llama = world
    summary = export(cfg, FakeLlama(shift=0), steps=("gguf",), shift=0)
    assert summary["gguf"]["converter_patched"] is False
    assert not (llama / "convert_hf_to_gguf.py.orig").exists()


def test_patch_is_applied_and_recognizes_the_extended_hash(world):
    cfg, llama = world
    export(cfg, FakeLlama(shift=1), steps=("gguf",))
    patched = (llama / "convert_hf_to_gguf.py").read_text(encoding="utf-8")
    assert (llama / "convert_hf_to_gguf.py.orig").exists()
    chktxt = extract_chktxt(patched)
    ext_hash = pretokenizer_fingerprint(FakeHf(1).encode, chktxt)
    assert known_pretokenizers(patched)[ext_hash] == "qwen2"


def test_parity_failure_stops_before_quantizing(world):
    cfg, _ = world
    fake = FakeLlama(shift=1, tokenizer_bug=True)
    with pytest.raises(RuntimeError, match="parity failed"):
        export(cfg, fake, steps=("gguf",))
    kinds = [Path(c[0]).name for c in fake.calls]
    assert not any(k.startswith(("llama-imatrix", "llama-quantize")) for k in kinds)
    saved = json.loads((cfg.output_dir / "export_summary.json").read_text(encoding="utf-8"))
    assert saved["parity"]["passed"] is False and saved["parity"]["mismatches"]


def test_existing_files_are_skipped_unless_forced(world):
    cfg, _ = world
    export(cfg, FakeLlama(shift=1))
    again = FakeLlama(shift=1)
    export(cfg, again)
    names = [Path(c[0]).name for c in again.calls]
    assert not any(n.startswith(("llama-imatrix", "llama-quantize")) for n in names)
    forced = FakeLlama(shift=1)
    export(cfg.model_copy(update={"force": True}), forced)
    assert any(Path(c[0]).name.startswith("llama-quantize") for c in forced.calls)


def test_steps_are_validated_and_ollama_needs_a_gguf_first(world):
    cfg, _ = world
    with pytest.raises(ValueError, match="unknown steps"):
        export(cfg, FakeLlama(1), steps=("gguf", "docker"))
    with pytest.raises(ValueError, match="no GGUF"):
        export(cfg, FakeLlama(1), steps=("ollama",))


def test_missing_model_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="extend run"):
        run_export(ExportConfig(model_dir=tmp_path / "none", output_dir=tmp_path / "o"))


# CLI -----------------------------------------------------------------------------------------


def write_cfg(tmp_path, cfg: ExportConfig) -> Path:
    path = tmp_path / "export.yaml"
    data = json.loads(cfg.model_dump_json())
    path.write_text(yaml.safe_dump({"export": data}), encoding="utf-8")
    return path


def test_export_help_lists_the_subcommands():
    result = runner.invoke(cli.app, ["export", "--help"])
    assert result.exit_code == 0
    for name in ("gguf", "ollama", "vllm", "all"):
        assert name in result.output


def test_export_vllm_cli_writes_files_and_json(world, tmp_path):
    cfg, _ = world
    result = runner.invoke(
        cli.app, ["export", "vllm", "--config", str(write_cfg(tmp_path, cfg)), "--json"]
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["vllm"] == {"config": "vllm.yaml", "script": "serve_vllm.sh"}
    assert "dtype: half" in (cfg.output_dir / "vllm.yaml").read_text(encoding="utf-8")


def test_export_gguf_cli_explains_a_missing_llama_cpp(world, tmp_path):
    cfg, _ = world
    cfg = cfg.model_copy(update={"llama_cpp_dir": tmp_path / "nowhere"})
    result = runner.invoke(cli.app, ["export", "gguf", "--config", str(write_cfg(tmp_path, cfg))])
    assert result.exit_code != 0 and "convert_hf_to_gguf.py" in result.output


def test_export_needs_an_export_section(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("audit:\n  tokenizer: x\n  corpus: y\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["export", "all", "--config", str(path)])
    assert result.exit_code != 0 and "no 'export' section" in result.output
