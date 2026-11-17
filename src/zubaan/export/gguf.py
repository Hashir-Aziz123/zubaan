"""Export an extended model to GGUF with llama.cpp: convert, importance matrix, quantize.

llama.cpp is not a Python dependency. It is a checkout (for ``convert_hf_to_gguf.py``) plus
the prebuilt tools ``llama-quantize``, ``llama-imatrix`` and ``llama-tokenize`` at one pinned
release, ``LLAMA_CPP_TAG``. Every command is built here as a list of strings and run through a
``run`` callable, so the logic is testable without llama.cpp.

**Pre-tokenizer fingerprint.** The converter writes ``tokenizer.ggml.pre`` into the GGUF so that
llama.cpp can split text the way the model's pre-tokenizer does. It picks the value by hashing
the token ids of a fixed test string (``chktxt``) and looking the hash up in a table. A
tokenizer whose hash is not in the table makes the converter stop with "BPE pre-tokenizer was
not recognized". Extended tokenizers can hit that, because new merges change the ids. The fix
is to add the extended hash to the table with the base model's value. That is correct only
because the extension leaves the pre-tokenizer untouched, so ``assert_same_pretokenizer``
checks it first. On the pinned release the test string has no Arabic-script characters, so
Urdu-only merges do not change its ids: the hashes are equal and no patch is needed. The code
handles both cases, and the parity test (``export/parity.py``) is the real proof either way.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from zubaan.extend.merge import assert_only_model_changed

LLAMA_CPP_TAG = "b12477"
QUANT_TYPES = (
    "Q2_K",
    "Q3_K_M",
    "Q4_0",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
)
OUTTYPES = ("f32", "f16", "bf16", "q8_0", "auto")

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


# llama.cpp location ------------------------------------------------------------------------------


@dataclass(frozen=True)
class LlamaCpp:
    """A llama.cpp checkout and the folder holding its built tools."""

    root: Path
    bin_dir: Path

    @property
    def convert_script(self) -> Path:
        return self.root / "convert_hf_to_gguf.py"

    def tool(self, name: str) -> Path:
        """Path of a built tool such as ``llama-quantize`` (``.exe`` is added on Windows)."""
        for candidate in (self.bin_dir / name, self.bin_dir / f"{name}.exe"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"{name} not found in {self.bin_dir}")


def find_llama_cpp(root: Path, bin_dir: Path | None = None) -> LlamaCpp:
    """Locate the converter script and the built tools, or raise with what is missing."""
    if not (root / "convert_hf_to_gguf.py").exists():
        raise FileNotFoundError(
            f"{root}/convert_hf_to_gguf.py not found. Get llama.cpp release {LLAMA_CPP_TAG}: "
            "see docs/stage2-usage.md"
        )
    candidates = [bin_dir] if bin_dir else [root / "bin", root / "build" / "bin"]
    for c in candidates:
        if c and c.is_dir():
            return LlamaCpp(root, c)
    raise FileNotFoundError(f"no folder with llama.cpp tools among {[str(c) for c in candidates]}")


# Pre-tokenizer fingerprint -----------------------------------------------------------------------


def find_pretokenizer_source(root: Path) -> Path:
    """The converter file that defines ``get_vocab_base_pre`` (its location varies by release)."""
    candidates = [root / "convert_hf_to_gguf.py", *sorted((root / "conversion").glob("*.py"))]
    for path in candidates:
        if path.exists() and "def get_vocab_base_pre" in path.read_text(encoding="utf-8"):
            return path
    raise FileNotFoundError(f"no get_vocab_base_pre found under {root}")


def extract_chktxt(source: str) -> str:
    """The fixed test string the converter tokenizes to fingerprint a pre-tokenizer."""
    m = re.search(r"^\s*chktxt = ('(?:[^'\\\n]|\\.)*')\s*$", source, re.MULTILINE)
    if not m:
        raise ValueError("could not find chktxt in the converter source")
    return ast.literal_eval(m.group(1))


def pretokenizer_fingerprint(encode: Callable[[str], list[int]], chktxt: str) -> str:
    """The converter's hash: sha256 of ``str(ids)`` for the ids of ``chktxt``.

    ``encode`` must behave like the converter's ``tokenizer.encode`` (special tokens added as
    the tokenizer's defaults dictate).
    """
    return sha256(str(encode(chktxt)).encode()).hexdigest()


_KNOWN = re.compile(
    r'if chkhsh == "([0-9a-f]{64})":\s*(?:#[^\n]*\n\s*)*res = "([^"]+)"',
)


def known_pretokenizers(source: str) -> dict[str, str]:
    """Fingerprint -> ``tokenizer.ggml.pre`` value, as listed in the converter source."""
    return {h: pre for h, pre in _KNOWN.findall(source)}


@dataclass(frozen=True)
class PretokenizerPlan:
    pre_type: str  # value written to tokenizer.ggml.pre, e.g. "qwen2"
    extended_hash: str
    needs_patch: bool  # True when the converter does not know the extended hash yet


def plan_pretokenizer(
    base_hash: str, extended_hash: str, known: dict[str, str]
) -> PretokenizerPlan:
    """Decide how the converter will recognize the extended tokenizer.

    The base tokenizer must be known to the converter. If the extended hash equals the base
    hash, or is already known with the same value, nothing needs patching. If it is unknown,
    it is mapped to the base model's value. If it is known with a different value, the
    extended tokenizer collides with another pre-tokenizer and we stop.
    """
    if base_hash not in known:
        raise ValueError(
            "the base model's tokenizer is not recognized by this llama.cpp release, so there "
            "is no pre-tokenizer type to map the extended tokenizer to"
        )
    pre = known[base_hash]
    if extended_hash == base_hash:
        return PretokenizerPlan(pre, extended_hash, False)
    if extended_hash in known:
        if known[extended_hash] != pre:
            raise ValueError(
                f"the extended fingerprint maps to {known[extended_hash]!r} but the base "
                f"tokenizer is {pre!r}"
            )
        return PretokenizerPlan(pre, extended_hash, False)
    return PretokenizerPlan(pre, extended_hash, True)


def assert_same_pretokenizer(base: dict, extended: dict) -> None:
    """Raise unless the extension changed nothing but ``model.vocab`` and ``model.merges``.

    Mapping the extended fingerprint to the base pre-tokenizer type is valid only then.
    """
    try:
        assert_only_model_changed(base, extended)
    except AssertionError as e:
        raise ValueError(f"the extended tokenizer changed more than its vocabulary: {e}") from e


def patch_converter_source(source: str, extended_hash: str, pre_type: str) -> str:
    """Add ``extended_hash -> pre_type`` to ``get_vocab_base_pre`` (idempotent).

    The new branch goes right before the ``if res is None:`` check that raises the
    "not recognized" error.
    """
    if f'chkhsh == "{extended_hash}"' in source:
        return source
    marker = re.search(r"^(\s*)if res is None:", source, re.MULTILINE)
    if not marker or "def get_vocab_base_pre" not in source[: marker.start()]:
        raise ValueError("could not find the 'if res is None:' check in get_vocab_base_pre")
    indent = marker.group(1)
    block = (
        f'{indent}if chkhsh == "{extended_hash}":\n'
        f"{indent}    # zubaan: extended tokenizer, same pre-tokenizer as the base model\n"
        f'{indent}    res = "{pre_type}"\n\n'
    )
    return source[: marker.start()] + block + source[marker.start() :]


def apply_patch(path: Path, extended_hash: str, pre_type: str) -> bool:
    """Patch the converter file in place, keeping a ``.orig`` backup. True if it changed."""
    source = path.read_text(encoding="utf-8")
    patched = patch_converter_source(source, extended_hash, pre_type)
    if patched == source:
        return False
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8", newline="\n")
    path.write_text(patched, encoding="utf-8", newline="\n")
    return True


# Commands ----------------------------------------------------------------------------------------


def convert_command(
    tools: LlamaCpp, model_dir: Path, outfile: Path, outtype: str = "f16", python: str | None = None
) -> list[str]:
    """``python convert_hf_to_gguf.py <model_dir> --outfile <file> --outtype <type>``."""
    if outtype not in OUTTYPES:
        raise ValueError(f"outtype must be one of {OUTTYPES}")
    return [
        python or sys.executable,
        str(tools.convert_script),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]


def imatrix_command(
    tools: LlamaCpp,
    gguf: Path,
    calibration: Path,
    out: Path,
    chunks: int = 40,
    ctx: int = 512,
    threads: int | None = None,
) -> list[str]:
    """``llama-imatrix`` on a calibration text file."""
    if chunks < 1 or ctx < 8:
        raise ValueError("chunks must be at least 1 and ctx at least 8")
    cmd = [
        str(tools.tool("llama-imatrix")),
        "-m",
        str(gguf),
        "-f",
        str(calibration),
        "-o",
        str(out),
        "--chunks",
        str(chunks),
        "-c",
        str(ctx),
    ]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def quantize_command(
    tools: LlamaCpp, src: Path, dst: Path, quant: str, imatrix: Path | None = None
) -> list[str]:
    """``llama-quantize [--imatrix file] src dst TYPE``."""
    if quant not in QUANT_TYPES:
        raise ValueError(f"quant must be one of {QUANT_TYPES}, got {quant!r}")
    cmd = [str(tools.tool("llama-quantize"))]
    if imatrix:
        cmd += ["--imatrix", str(imatrix)]
    return [*cmd, str(src), str(dst), quant]


def write_calibration_file(lines: Sequence[str], out: Path, max_chars: int = 200_000) -> int:
    """Write target-language calibration text for the importance matrix; returns characters.

    Lines are joined with newlines until ``max_chars`` is reached. The project rule is to
    calibrate on target-language text, not generic English, so an empty or English-only input
    is an error.
    """
    chosen: list[str] = []
    total = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        chosen.append(line)
        total += len(line) + 1
        if total >= max_chars:
            break
    if not chosen:
        raise ValueError("no calibration text")
    text = "\n".join(chosen) + "\n"
    if not any("؀" <= c <= "ۿ" for c in text):
        raise ValueError("calibration text has no Urdu/Arabic-script characters")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return len(text)


def run_checked(cmd: Sequence[str], run: Runner = subprocess.run, log=print) -> str:
    """Run a command, return its output, and raise with the tail of the output on failure."""
    log("$ " + " ".join(cmd))
    proc = run(list(cmd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-15:])
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd[0]}\n{tail}")
    return output
