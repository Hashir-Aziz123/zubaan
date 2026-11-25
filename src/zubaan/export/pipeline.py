"""Export an extended model: GGUF with proven tokenizer parity, Ollama Modelfile, vLLM config.

Order matters. The f16 GGUF is converted first and its tokenizer is compared with Hugging
Face on FLORES Urdu, FLORES English and a stress set. If a single id differs the export stops
before quantizing, because every later file would carry the same wrong tokenizer.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable

from zubaan.config import ExportConfig
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
    plan_pretokenizer,
    pretokenizer_fingerprint,
    quantize_command,
    run_checked,
    write_calibration_file,
)
from zubaan.export.ollama import check_stop_tokens, modelfile_text, write_modelfile
from zubaan.export.parity import ParityResult, check_parity, stress_texts
from zubaan.export.vllm import write_vllm_files
from zubaan.extend.tokenizer_info import load_tokenizer_json
from zubaan.io import read_lines
from zubaan.provenance import write_json

STEPS = ("gguf", "ollama", "vllm")
STOP_TOKENS = ("<|im_end|>", "<|im_start|>")


def _load_hf_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def parity_texts(cfg: ExportConfig) -> list[str]:
    """FLORES Urdu, FLORES English and the stress set, in a stable order."""
    urdu = read_lines(cfg.parity_urdu, cfg.max_parity_texts) if cfg.parity_urdu.exists() else []
    english = (
        read_lines(cfg.parity_english, cfg.max_parity_texts) if cfg.parity_english.exists() else []
    )
    return [*urdu, *english, *stress_texts()]


def _gguf_step(cfg: ExportConfig, log: Callable[[str], None], run, load_tokenizer) -> dict:
    tools = find_llama_cpp(cfg.llama_cpp_dir, cfg.llama_cpp_bin)
    base_json = load_tokenizer_json(cfg.base_model)
    ext_json = load_tokenizer_json(str(cfg.model_dir))
    assert_same_pretokenizer(base_json, ext_json)

    # 1. Fingerprint: will the converter recognize the extended tokenizer?
    source_file = find_pretokenizer_source(cfg.llama_cpp_dir)
    source = source_file.read_text(encoding="utf-8")
    chktxt = extract_chktxt(source)
    base_tok = load_tokenizer(cfg.base_model)
    ext_tok = load_tokenizer(str(cfg.model_dir))
    plan = plan_pretokenizer(
        pretokenizer_fingerprint(base_tok.encode, chktxt),
        pretokenizer_fingerprint(ext_tok.encode, chktxt),
        known_pretokenizers(source),
    )
    patched = False
    if plan.needs_patch:
        patched = apply_patch(source_file, plan.extended_hash, plan.pre_type)
    log(
        f"pre-tokenizer type {plan.pre_type!r}; fingerprint "
        f"{'was unknown, patched the converter' if plan.needs_patch else 'already recognized'}"
    )

    # 2. Convert.
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    f16 = out / f"{cfg.name}-{cfg.outtype.upper()}.gguf"
    if cfg.force or not f16.exists():
        run_checked(convert_command(tools, cfg.model_dir, f16, cfg.outtype, cfg.python), run, log)
    else:
        log(f"{f16.name} exists, skipping conversion")

    # 3. Parity on the converted file. Stop here if a single id differs.
    result: ParityResult = check_parity(
        ext_tok,
        tools,
        f16,
        parity_texts(cfg),
        run,
        cfg.parity_workers,
        lambda done, total: log(f"  parity {done}/{total}"),
    )
    parity = {
        "gguf": f16.name,
        "n_texts": result.n_texts,
        "n_mismatch": result.n_mismatch,
        "n_nfc_only": result.n_nfc_only,
        "n_real_mismatch": result.n_real,
        "mismatches": result.mismatches,
        "nfc_only_examples": result.nfc_only,
        "passed": result.passed,
    }
    log(
        f"parity: {result.n_texts - result.n_mismatch}/{result.n_texts} texts identical; "
        f"{result.n_nfc_only} differ only because the text is not NFC; "
        f"{result.n_real} real mismatches"
    )
    if not result.passed:
        write_json({"parity": parity}, out / "export_summary.json")
        raise RuntimeError(
            f"tokenizer parity failed on {result.n_real} of {result.n_texts} texts; "
            f"see {out / 'export_summary.json'}"
        )

    # 4. Importance matrix on Urdu text, then quantize.
    quantized: dict[str, str] = {}
    if cfg.quants:
        calib = out / "calibration_urdu.txt"
        n_chars = write_calibration_file(read_lines(cfg.calibration), calib, cfg.calibration_chars)
        log(f"calibration text: {n_chars:,} characters of Urdu")
        imatrix = out / f"{cfg.name}.imatrix.dat"
        if cfg.force or not imatrix.exists():
            cmd = imatrix_command(
                tools, f16, calib, imatrix, cfg.imatrix_chunks, cfg.imatrix_ctx, cfg.threads
            )
            run_checked(cmd, run, log)
        for q in cfg.quants:
            dst = out / f"{cfg.name}-{q}.gguf"
            if cfg.force or not dst.exists():
                run_checked(quantize_command(tools, f16, dst, q, imatrix), run, log)
            quantized[q] = dst.name
    return {
        "llama_cpp_tag": LLAMA_CPP_TAG,
        "pre_tokenizer": plan.pre_type,
        "fingerprint": plan.extended_hash,
        "converter_patched": patched,
        "f16": f16.name,
        "quantized": quantized,
        "parity": parity,
        "imatrix_calibration": "Urdu Wikipedia (training split)",
    }


def run_export(
    cfg: ExportConfig,
    steps: tuple[str, ...] = STEPS,
    log: Callable[[str], None] = print,
    run=subprocess.run,
    load_tokenizer: Callable[[str], object] = _load_hf_tokenizer,
) -> dict:
    """Run the chosen export steps and write ``export_summary.json``; return the summary."""
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    if not (cfg.model_dir / "tokenizer.json").exists():
        raise FileNotFoundError(
            f"{cfg.model_dir}/tokenizer.json not found; run 'zubaan extend run'"
        )
    summary_path = cfg.output_dir / "export_summary.json"
    summary: dict = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["model_dir"] = str(cfg.model_dir)
    summary["base_model"] = cfg.base_model
    if "gguf" in steps:
        summary["gguf"] = _gguf_step(cfg, log, run, load_tokenizer)
    if "ollama" in steps:
        base_json = load_tokenizer_json(cfg.base_model)
        ext_json = load_tokenizer_json(str(cfg.model_dir))
        stops = check_stop_tokens(ext_json, base_json, STOP_TOKENS)
        gguf = summary.get("gguf", {})
        quants = gguf.get("quantized", {})
        file = quants.get("Q4_K_M") or next(iter(quants.values()), None) or gguf.get("f16")
        if not file:
            raise ValueError("no GGUF file to point the Modelfile at; run the gguf step first")
        text = modelfile_text(file, tuple(stops), num_ctx=cfg.max_model_len)
        path = write_modelfile(cfg.output_dir / "Modelfile", text)
        summary["ollama"] = {"modelfile": path.name, "gguf": file, "stop_token_ids": stops}
        log(f"wrote {path}")
    if "vllm" in steps:
        files = write_vllm_files(
            cfg.output_dir, cfg.model_dir.as_posix(), cfg.name, cfg.max_model_len
        )
        summary["vllm"] = {k: v.name for k, v in files.items()}
        log(f"wrote {files['config']} and {files['script']}")
    write_json(summary, summary_path)
    return summary
