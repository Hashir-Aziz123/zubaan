# ruff: noqa: E501
"""Writes notebooks/03_extend_export_qwen3.ipynb.

Run: python notebooks/make_03_notebook.py [COMMIT]

COMMIT defaults to the current git HEAD. It must be pushed to GitHub before the notebook runs,
because the notebook installs zubaan from that exact commit.
"""

import json
import subprocess
import sys
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


LLAMA_CPP_TAG = "b12477"

INSTALL = """
# Install zubaan at one exact commit. Do not edit the next line except to change the commit.
COMMIT = "{commit}"
REPO = f"git+https://github.com/Hashir-Aziz123/zubaan.git@{{COMMIT}}"
!pip install -q "zubaan[audit,export] @ {{REPO}}"
# Kaggle may already hold an older copy with the same version number, which pip keeps.
# Force our commit (dependencies are already in place).
!pip install -q --force-reinstall --no-deps "zubaan @ {{REPO}}"
!zubaan version
"""

CHECK_CODE = """
# CHECK 1: the installed code is the commit above, not a cached older copy.
# If this fails, use Run > Restart session and run the cells again from the top.
import json
from importlib import metadata

direct = json.loads(metadata.distribution("zubaan").read_text("direct_url.json"))
installed = direct["vcs_info"]["commit_id"]
print("installed commit:", installed)
assert installed.startswith(COMMIT), f"installed {installed}, wanted {COMMIT}"

import zubaan.export.parity  # noqa: F401
import zubaan.extend.pipeline  # noqa: F401

print("OK: code version verified")
"""

CHECK_MACHINE = """
import os
import shutil

# Quiet progress bars: they can overwhelm the notebook front end.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# CHECK 2: the machine has what the run needs. No GPU is needed.
with open("/proc/meminfo") as f:
    mem = {line.split(":")[0]: int(line.split()[1]) for line in f}
ram_gb = mem["MemAvailable"] / 1e6
disk_gb = shutil.disk_usage("/kaggle/working").free / 1e9
n_cpu = os.cpu_count()
print(f"RAM available {ram_gb:.1f} GB, disk free {disk_gb:.1f} GB, CPUs {n_cpu}")
assert ram_gb >= 12, "need about 12 GB free RAM (1.7B in bfloat16 plus the GGUF conversion)"
assert disk_gb >= 20, "need about 20 GB free disk"
print("OK: machine is big enough")
"""

LLAMA_CPP = f'''
import subprocess
from pathlib import Path

TAG = "{LLAMA_CPP_TAG}"
TOOLS = Path("/tmp/tools")  # outside /kaggle/working so it is not saved as output
LLAMA_DIR = TOOLS / f"llama.cpp-{{TAG}}"
LLAMA_DIR.mkdir(parents=True, exist_ok=True)
base = "https://github.com/ggml-org/llama.cpp"
# Source tree: the converter script and gguf-py.
!curl -sL {{base}}/archive/refs/tags/{{TAG}}.tar.gz | tar xz --strip-components=1 -C {{LLAMA_DIR}}
# Prebuilt CPU tools (llama-quantize, llama-imatrix, llama-tokenize, llama-completion).
BIN_ROOT = TOOLS / "bin"
BIN_ROOT.mkdir(exist_ok=True)
!curl -sL {{base}}/releases/download/{{TAG}}/llama-{{TAG}}-bin-ubuntu-x64.tar.gz | tar xz -C {{BIN_ROOT}}
found = sorted(BIN_ROOT.rglob("llama-quantize"))
assert found, "llama-quantize not found in the downloaded archive"
BIN_DIR = found[0].parent
print("llama.cpp tools in", BIN_DIR)
'''

LLAMA_SELFCHECK = """
# CHECK 3: the prebuilt tools run on this machine. If one does not (a library is missing),
# build llama.cpp from the source tree instead: about 10 minutes on 4 CPUs.
def tools_work(bin_dir):
    for tool in ("llama-quantize", "llama-imatrix", "llama-tokenize", "llama-completion"):
        r = subprocess.run([str(bin_dir / tool), "--help"], capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        # llama-quantize prints its usage and exits with 1, so exit codes 0 and 1 both count as "ran".
        if r.returncode not in (0, 1) or not out.strip() or "shared libraries" in out:
            print(tool, "failed:", r.returncode, out[-300:])
            return False
    return True


if not tools_work(BIN_DIR):
    print("prebuilt tools do not run here; building from source")
    !cmake -S {LLAMA_DIR} -B {LLAMA_DIR}/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF > /dev/null
    !cmake --build {LLAMA_DIR}/build --config Release -j {n_cpu} --target llama-quantize llama-imatrix llama-tokenize llama-completion
    BIN_DIR = LLAMA_DIR / "build" / "bin"
    assert tools_work(BIN_DIR), "built tools do not run either"
print("OK: llama.cpp tools run, bin dir:", BIN_DIR)
"""

CONFIG = """
import os
from pathlib import Path

import yaml

WORK = Path("/kaggle/working")
config = {
    "urdu_corpus": {
        "raw_dir": "data/raw",
        "out_dir": "data/urdu",
        "flores": "data/flores_urd.txt",
        "max_mb": 100,
        "heldout_percent": 1.0,
    },
    "extend": {
        "base_model": BASE_MODEL,
        "corpus": "data/urdu/urdu_train.txt",
        "heldout": "data/urdu/urdu_heldout.txt",
        "eval_urdu": "data/flores_urd.txt",
        "eval_english": "data/flores_eng.txt",
        "n_new_tokens": N_NEW,
        "sizes": [N_NEW],
        "init_method": METHOD,
        "min_count": 2,
        "merges_file": str(WORK / "merges.json"),
        "output_dir": str(WORK / "extended"),
        "model_dtype": "bfloat16",
        "device": "cpu",
    },
    "export": {
        "model_dir": str(WORK / "extended" / "model"),
        "base_model": BASE_MODEL,
        "output_dir": str(WORK / "export"),
        "name": NAME,
        "llama_cpp_dir": str(LLAMA_DIR),
        "llama_cpp_bin": str(BIN_DIR),
        "quants": ["Q4_K_M", "Q8_0"],
        "calibration": "data/urdu/urdu_train.txt",
        "calibration_chars": 200000,
        "imatrix_chunks": 40,
        "threads": os.cpu_count(),
        "parity_urdu": "data/flores_urd.txt",
        "parity_english": "data/flores_eng.txt",
        "parity_workers": os.cpu_count(),
        "max_model_len": 4096,
    },
}
Path("export.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(Path("export.yaml").read_text())
"""

SETTINGS = """
# What to build. Defaults are the project's recommendation (see docs/decisions.md, 2026-11-17).
BASE_MODEL = "Qwen/Qwen3-1.7B"
N_NEW = 10000        # new Urdu tokens
METHOD = "mean"      # mean | focus | random
NAME = "qwen3-1.7b-urdu"
"""

SUMMARY = """
import json

summary = json.loads(Path("/kaggle/working/export/export_summary.json").read_text())
g = summary["gguf"]
p = g["parity"]
print("llama.cpp", g["llama_cpp_tag"], "| pre-tokenizer", g["pre_tokenizer"], "| converter patched:", g["converter_patched"])
print(f"parity: {p['n_texts'] - p['n_mismatch']}/{p['n_texts']} identical, {p['n_nfc_only']} differ only because the text is not NFC, {p['n_real_mismatch']} real mismatches")
assert p["passed"] and p["n_real_mismatch"] == 0, "tokenizer parity failed: see export_summary.json"
assert {"Q4_K_M", "Q8_0"} <= set(g["quantized"]), "a quantized file is missing"
print("OK: GGUF export passed tokenizer parity")
"""

GENERATE = """
# CHECK 4: the Q4_K_M file loads and generates on the CPU. The new rows are untrained, so the
# text will be poor (it may drift into other scripts). It only has to load and produce tokens.
# The prompt, 'The capital of Pakistan', takes 3 tokens here against 13 with the base tokenizer.
prompt = "پاکستان کا دارالحکومت"
q4 = WORK / "export" / f"{NAME}-Q4_K_M.gguf"
!{BIN_DIR}/llama-completion -m {q4} -p "{prompt}" -n 40 -t {n_cpu} --temp 0 2>&1 | tail -15
"""

CLEANUP = """
# Kaggle keeps everything in /kaggle/working as output (limit 20 GB). The f16 GGUF is only an
# intermediate for quantizing, so remove it. Keep the extended model folder: stage 3 starts from it.
for f in (WORK / "export").glob("*-F16.gguf"):
    print("removing", f.name, f"{f.stat().st_size / 1e9:.1f} GB")
    f.unlink()

total = 0
for f in sorted(WORK.rglob("*")):
    if f.is_file() and f.stat().st_size > 5e6:
        total += f.stat().st_size
        print(f"{f.stat().st_size / 1e9:6.2f} GB  {f.relative_to(WORK)}")
print(f"total of files over 5 MB: {total / 1e9:.1f} GB")
assert total < 19e9, "output is close to Kaggle's 20 GB limit"
"""

VLLM = """
# OPTIONAL, GPU session only, and NOT tested by the author: starts vLLM on the extended model with
# the generated config (dtype half) and asks for one completion. Set RUN_VLLM_CHECK = True, switch the
# accelerator to GPU T4, and run this cell last. It installs vllm, which can take several minutes.
RUN_VLLM_CHECK = False

if RUN_VLLM_CHECK:
    import time

    import torch

    assert torch.cuda.is_available(), "turn on a GPU accelerator first"
    !pip install -q vllm
    export_dir = WORK / "export"
    log_file = Path("/kaggle/working/vllm.log")
    server = subprocess.Popen(f"cd {export_dir} && sh serve_vllm.sh > {log_file} 2>&1", shell=True)
    ok = False
    for _ in range(90):
        time.sleep(10)
        r = subprocess.run(["curl", "-s", "localhost:8000/v1/models"], capture_output=True, text=True)
        if NAME in r.stdout:
            ok = True
            break
    print("server up:", ok)
    if ok:
        body = json.dumps({"model": NAME, "prompt": prompt, "max_tokens": 30, "temperature": 0})
        r = subprocess.run(
            ["curl", "-s", "localhost:8000/v1/completions", "-H", "Content-Type: application/json", "-d", body],
            capture_output=True,
            text=True,
        )
        print(r.stdout)
    server.terminate()
    print(log_file.read_text()[-2000:])
"""


def build(commit: str) -> dict:
    cells = [
        md(
            f"""
# Stage 2: extend and export Qwen3-1.7B

Builds the real deliverable for Qwen3-1.7B: the model with 10,000 new Urdu tokens, saved and verified, then converted to GGUF with a tokenizer parity check, quantized (Q4_K_M and Q8_0) with an Urdu importance matrix, plus an Ollama `Modelfile` and a vLLM config.

**Nothing is trained.** The new embedding rows are only initialized (mean of pieces), so Urdu quality is worse than the base model until stage 3. The GGUF files are for checking the pipeline, not for users.

All logic lives in the `zubaan` package; this notebook only calls the CLI.

Settings before running:

- Accelerator: **None** (CPU). The GPU is not used, so do not spend GPU hours on this.
- Internet: **on**.
- No secret is needed: Qwen3, FLORES-200 (Meta archive), Urdu Wikipedia and llama.cpp `{LLAMA_CPP_TAG}` are not gated.

Needs about 12 GB of free RAM and 20 GB of disk. Expected run time: about 1 hour on 4 CPUs (merge learning 5 minutes, extension and verification 5, conversion 5, tokenizer parity 10 to 15, importance matrix 15 to 25, quantizing 5).

**Before the first run:** the commit named in the install cell must be pushed to GitHub (`git push`), because the notebook installs zubaan from it.

**Safest way to run it:** *Save Version > Save & Run All (Commit)*. Finished files are skipped if you run it again.

**Download from the Output tab afterwards:** `export/export_summary.json` (parity result, always), the `.gguf` files you want, `export/Modelfile`, `export/vllm.yaml`, and `extended/model/` (the extended Hugging Face model that stage 3 trains).
"""
        ),
        code(INSTALL.format(commit=commit)),
        code(CHECK_CODE),
        code(CHECK_MACHINE),
        code(SETTINGS),
        md("## 1. llama.cpp (conversion, importance matrix, quantizing, tokenizer parity)"),
        code(LLAMA_CPP),
        code(LLAMA_SELFCHECK),
        md(
            "## 2. Data and config\n\nFLORES-200 (evaluation and parity texts) and the Urdu Wikipedia training text."
        ),
        code(CONFIG),
        code("!zubaan data flores --out data\n!zubaan data urdu --config export.yaml"),
        md(
            "## 3. Learn merges and extend the model\n\nMerge learning is tokenizer-only (about 5 minutes). `extend run` then extends the weights, saves the model and tokenizer, and verifies the saved folder: only the vocabulary and merges changed, English tokenizes exactly as before, Urdu round-trips, and the embedding rows match the vocabulary."
        ),
        code("!zubaan extend learn --config export.yaml"),
        code("!zubaan extend run --config export.yaml"),
        md(
            "## 4. Export\n\nConverts to f16 GGUF, then **compares token ids of Hugging Face and llama.cpp on every FLORES Urdu and English sentence plus a stress set**. If any id differs for a reason other than NFC normalization, the export stops here. Then the Urdu importance matrix and the two quantizations."
        ),
        code("!zubaan export gguf --config export.yaml"),
        code(
            "!zubaan export ollama --config export.yaml\n!zubaan export vllm --config export.yaml"
        ),
        code(SUMMARY),
        code(GENERATE),
        md("## 5. Clean up and list the output"),
        code(CLEANUP),
        md(
            "## 6. Optional: vLLM check on a GPU\n\nSkipped by default. See the comment in the cell."
        ),
        code(VLLM),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    commit = (
        sys.argv[1]
        if len(sys.argv) > 1
        else subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    )
    out = Path(__file__).parent / "03_extend_export_qwen3.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
