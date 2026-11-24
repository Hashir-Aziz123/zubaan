"""Check exported GGUF files with llama.cpp: does the model still write Urdu, and what did
quantizing cost?

Two checks, both on the files that ``zubaan export gguf`` wrote:

1. **Generation samples.** Fixed Urdu prompts (plain completion and chat) and one English control
   prompt, greedy decoding, on every GGUF file. Bits per character cannot say whether the text is
   readable, so the samples are kept for a person to read. ``urdu_letter_share`` is only a
   coarse flag for drift into other scripts, not a quality measure.
2. **Quantization cost.** llama.cpp's KL-divergence mode compares each quantized file with the
   f16 file on the same text: the mean KL divergence of the next-token distributions, how often
   both pick the same top token, and the perplexity ratio. All files share one tokenizer, so
   per-token numbers are comparable here. They must never be compared with a model that uses a
   different tokenizer (use bits per character for that).

Speeds reported here are tokens per second of one tokenizer on one machine; they compare the
quantizations with each other, not the extended model with the base model.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from zubaan.config import ExportConfig
from zubaan.export.gguf import LLAMA_CPP_TAG, LlamaCpp, Runner, find_llama_cpp
from zubaan.io import read_lines
from zubaan.provenance import run_metadata, write_json
from zubaan.text import normalize_urdu

# Qwen3's chat template with ``enable_thinking=False`` puts an empty think block at the start
# of the answer; with thinking allowed the model writes the block itself.
_NO_THINK = "<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class Prompt:
    """One fixed prompt. ``chat`` wraps it in Qwen's ChatML; ``thinking`` only matters then."""

    name: str
    text: str
    chat: bool = False
    thinking: bool = False


PROMPTS: tuple[Prompt, ...] = (
    # "The capital of Pakistan" (the Stage 2 smoke prompt).
    Prompt("urdu_completion_capital", "پاکستان کا دارالحکومت"),
    # "Lahore is a historic city of Pakistan. Here"
    Prompt("urdu_completion_lahore", "لاہور پاکستان کا ایک تاریخی شہر ہے۔ یہاں"),
    # "What is the capital of Pakistan?"
    Prompt("urdu_chat_capital", "پاکستان کا دارالحکومت کیا ہے؟", chat=True),
    # "Write three sentences about the Urdu language."
    Prompt("urdu_chat_sentences", "اردو زبان کے بارے میں تین جملے لکھیں۔", chat=True),
    # The same question with thinking allowed, to see what happens to the <think> block.
    Prompt("urdu_chat_thinking", "پاکستان کا دارالحکومت کیا ہے؟", chat=True, thinking=True),
    # English control.
    Prompt("english_completion", "The capital of Pakistan is"),
)


# Generation ---------------------------------------------------------------------------------------


def chatml_prompt(user: str, thinking: bool = False) -> str:
    """A single user turn in Qwen's ChatML, ending where the assistant's answer starts."""
    if not user.strip():
        raise ValueError("empty user message")
    prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return prompt if thinking else prompt + _NO_THINK


def prompt_text(p: Prompt) -> str:
    """The exact string given to llama.cpp for a prompt."""
    return chatml_prompt(p.text, p.thinking) if p.chat else p.text


def completion_command(
    tools: LlamaCpp,
    gguf: Path,
    prompt: str,
    n_predict: int = 80,
    threads: int | None = None,
    ctx: int = 2048,
) -> list[str]:
    """``llama-completion``, greedy, special tokens shown, prompt not echoed.

    ``-no-cnv`` keeps it a plain completion (the chat formatting is in the prompt itself), and
    ``-sp`` prints special tokens so a ``<think>`` block or an early ``<|im_end|>`` is visible.
    """
    if n_predict < 1:
        raise ValueError("n_predict must be at least 1")
    cmd = [
        str(tools.tool("llama-completion")),
        "-m",
        str(gguf),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "--temp",
        "0",
        "--seed",
        "0",
        "-c",
        str(ctx),
        "-no-cnv",
        "-sp",
        "--no-display-prompt",
    ]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


_EVAL_SPEED = re.compile(r"(?<!prompt )eval time =.*?([\d.]+) tokens per second")


def parse_eval_speed(log: str) -> float | None:
    """Generation speed (tokens per second) from llama.cpp's timing lines, if printed."""
    found = _EVAL_SPEED.findall(log)
    return float(found[-1]) if found else None


def strip_think(text: str) -> str:
    """The answer without a ``<think>...</think>`` block (an unclosed block is removed too)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"<think>.*", "", text, flags=re.S).strip()


def urdu_letter_share(text: str) -> float | None:
    """Share of letters that are in the Arabic Unicode block (U+0600 to U+06FF).

    share = arabic_block_letters / all_letters. Digits, spaces and punctuation are not letters.
    Returns None when the text has no letters at all. A low share on an Urdu prompt means the
    model drifted into another script; a high share says nothing about whether the text is good.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    return sum("؀" <= c <= "ۿ" for c in letters) / len(letters)


def generate_samples(
    tools: LlamaCpp,
    gguf: Path,
    prompts: Sequence[Prompt] = PROMPTS,
    run: Runner = subprocess.run,
    n_predict: int = 80,
    threads: int | None = None,
) -> list[dict]:
    """Run every prompt on one GGUF file and return the text and simple flags per prompt."""
    out = []
    for p in prompts:
        cmd = completion_command(tools, gguf, prompt_text(p), n_predict, threads)
        proc = run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-10:])
            raise RuntimeError(f"llama-completion failed on {gguf.name} ({p.name}):\n{tail}")
        text = (proc.stdout or "").replace(" [end of text]", "").rstrip()
        answer = strip_think(text)
        out.append(
            {
                "prompt": p.name,
                "input": p.text,
                "chat": p.chat,
                "thinking_allowed": p.thinking,
                "text": text,
                "has_think_block": "<think>" in text and not text.startswith("\n\n</think>"),
                "answer_urdu_letter_share": urdu_letter_share(answer),
                "tokens_per_s": parse_eval_speed(proc.stderr or ""),
            }
        )
    return out


# Quantization cost --------------------------------------------------------------------------------


def write_eval_text(lines: Sequence[str], out: Path, urdu: bool) -> int:
    """Write evaluation text, one line per input line; Urdu is normalized first. Returns chars."""
    chosen = [normalize_urdu(x) if urdu else " ".join(x.split()) for x in lines]
    chosen = [x for x in chosen if x]
    if not chosen:
        raise ValueError("no evaluation text")
    text = "\n".join(chosen) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return len(text)


def kld_base_command(
    tools: LlamaCpp,
    f16: Path,
    text: Path,
    logits: Path,
    ctx: int = 512,
    chunks: int = 20,
    threads: int | None = None,
) -> list[str]:
    """``llama-perplexity`` on the reference file, saving its logits for the comparison."""
    if chunks < 1 or ctx < 8:
        raise ValueError("chunks must be at least 1 and ctx at least 8")
    cmd = [
        str(tools.tool("llama-perplexity")),
        "-m",
        str(f16),
        "-f",
        str(text),
        "-c",
        str(ctx),
        "--chunks",
        str(chunks),
        "--kl-divergence-base",
        str(logits),
    ]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def kld_command(
    tools: LlamaCpp,
    quant: Path,
    logits: Path,
    ctx: int = 512,
    chunks: int = 20,
    threads: int | None = None,
) -> list[str]:
    """``llama-perplexity --kl-divergence``: the quantized file against the saved logits.

    The text and tokens come from the logits file, so the comparison is on identical input.
    """
    if chunks < 1 or ctx < 8:
        raise ValueError("chunks must be at least 1 and ctx at least 8")
    cmd = [
        str(tools.tool("llama-perplexity")),
        "-m",
        str(quant),
        "-c",
        str(ctx),
        "--chunks",
        str(chunks),
        "--kl-divergence-base",
        str(logits),
        "--kl-divergence",
    ]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


_NUM = r"(-?[\d.]+(?:e[-+]?\d+)?)"
_KLD_FIELDS = {
    "ppl_ratio": rf"Mean PPL\(Q\)/PPL\(base\)\s*:\s*{_NUM}\s*±\s*{_NUM}",
    "ln_ppl_ratio": rf"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*{_NUM}\s*±\s*{_NUM}",
    "mean_kld": rf"Mean\s+KLD:\s*{_NUM}\s*±\s*{_NUM}",
    "rms_delta_p_percent": rf"RMS Δp\s*:\s*{_NUM}\s*±\s*{_NUM}\s*%",
    "same_top_p_percent": rf"Same top p:\s*{_NUM}\s*±\s*{_NUM}\s*%",
}
_PPL = re.compile(rf"Final estimate: PPL = {_NUM} \+/- {_NUM}")


def parse_kld(output: str) -> dict[str, dict[str, float]]:
    """Read the summary of ``llama-perplexity --kl-divergence``: value and uncertainty per field.

    ``mean_kld`` is the average over tokens of KL(P_f16 || P_quant) in nats: 0 means the
    quantized model predicts exactly like f16. ``same_top_p_percent`` is how often both pick the
    same most likely token. ``ppl_ratio`` is PPL(quant) / PPL(f16) on the same tokens.
    """
    out = {}
    for name, pattern in _KLD_FIELDS.items():
        m = re.search(pattern, output)
        if not m:
            raise ValueError(f"{name} not found in llama-perplexity output")
        out[name] = {"value": float(m.group(1)), "uncertainty": float(m.group(2))}
    return out


def parse_ppl(output: str) -> dict[str, float]:
    """``Final estimate: PPL = x +/- y`` from a plain llama-perplexity run."""
    m = _PPL.search(output)
    if not m:
        raise ValueError("no final perplexity estimate in llama-perplexity output")
    return {"value": float(m.group(1)), "uncertainty": float(m.group(2))}


def quant_cost(
    tools: LlamaCpp,
    f16: Path,
    quants: dict[str, Path],
    text: Path,
    scratch: Path,
    run: Runner = subprocess.run,
    log: Callable[[str], None] = print,
    ctx: int = 512,
    chunks: int = 20,
    threads: int | None = None,
) -> dict:
    """KL divergence of every quantized file against f16 on one text file.

    The f16 logits file is large (about 2 bytes per vocabulary entry per scored token), so it
    lives in ``scratch`` and is deleted at the end.
    """
    from zubaan.export.gguf import run_checked

    scratch.mkdir(parents=True, exist_ok=True)
    logits = scratch / f"{text.stem}.f16.kld"
    try:
        base_out = run_checked(
            kld_base_command(tools, f16, text, logits, ctx, chunks, threads), run, log
        )
        result: dict = {"f16_ppl": parse_ppl(base_out), "quants": {}}
        for name, path in quants.items():
            out = run_checked(kld_command(tools, path, logits, ctx, chunks, threads), run, log)
            result["quants"][name] = parse_kld(out)
            kld = result["quants"][name]
            log(
                f"  {name}: mean KLD {kld['mean_kld']['value']:.4f}, "
                f"same top token {kld['same_top_p_percent']['value']:.1f}%, "
                f"PPL ratio {kld['ppl_ratio']['value']:.4f}"
            )
    finally:
        logits.unlink(missing_ok=True)
    return result


# Pipeline -----------------------------------------------------------------------------------------


def _gguf_files(cfg: ExportConfig) -> tuple[Path, dict[str, Path]]:
    """The f16 file and the quantized files named in export_summary.json."""
    summary_path = cfg.output_dir / "export_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found; run 'zubaan export gguf' first")
    gguf = json.loads(summary_path.read_text(encoding="utf-8")).get("gguf", {})
    if not gguf.get("parity", {}).get("passed"):
        raise ValueError("export_summary.json has no passed parity check; run 'export gguf'")
    f16 = cfg.output_dir / gguf["f16"]
    quants = {q: cfg.output_dir / f for q, f in gguf.get("quantized", {}).items()}
    for p in [f16, *quants.values()]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (the f16 file must be kept for this check)")
    return f16, quants


def run_check(
    cfg: ExportConfig,
    log: Callable[[str], None] = print,
    run: Runner = subprocess.run,
    prompts: Sequence[Prompt] = PROMPTS,
) -> dict:
    """Generation samples on every GGUF file, then quantization cost on Urdu and English.

    Writes and returns ``export_check.json`` in the export folder.
    """
    tools = find_llama_cpp(cfg.llama_cpp_dir, cfg.llama_cpp_bin)
    f16, quants = _gguf_files(cfg)
    result: dict = {
        "llama_cpp_tag": LLAMA_CPP_TAG,
        "model_dir": str(cfg.model_dir),
        "threads": cfg.threads,
        "n_predict": cfg.check_n_predict,
        "decoding": "greedy (temperature 0)",
        "samples": {},
        "quant_cost": {},
        "kld_settings": {"ctx": cfg.check_ctx, "chunks": cfg.check_chunks},
        "metadata": run_metadata(),
    }
    for name, path in [(cfg.outtype.upper(), f16), *quants.items()]:
        log(f"generating with {path.name}")
        result["samples"][name] = generate_samples(
            tools, path, prompts, run, cfg.check_n_predict, cfg.threads
        )
        for s in result["samples"][name]:
            log(f"  [{s['prompt']}] {s['text'][:160]!r}")
    scratch = cfg.check_scratch or cfg.output_dir / "check_tmp"
    for lang, src, urdu in [("urdu", cfg.check_urdu, True), ("english", cfg.check_english, False)]:
        text = scratch / f"check_{lang}.txt"
        n_chars = write_eval_text(read_lines(src), text, urdu)
        log(f"quantization cost on {lang} ({src}, {n_chars:,} characters)")
        res = quant_cost(
            tools,
            f16,
            quants,
            text,
            scratch,
            run,
            log,
            cfg.check_ctx,
            cfg.check_chunks,
            cfg.threads,
        )
        res["source"] = str(src)
        result["quant_cost"][lang] = res
    if cfg.check_scratch is None:
        shutil.rmtree(scratch, ignore_errors=True)
    write_json(result, cfg.output_dir / "export_check.json")
    return result
