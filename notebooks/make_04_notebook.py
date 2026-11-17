# ruff: noqa: E501
"""Writes notebooks/04_vllm_check_qwen3.ipynb.

Run: python notebooks/make_04_notebook.py [COMMIT]

COMMIT defaults to the current git HEAD and must be pushed to GitHub, because the notebook
installs zubaan from that exact commit.
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


INSTALL = """
# Install zubaan at one exact commit (only its small core is needed here: the vLLM config generator).
COMMIT = "{commit}"
REPO = f"git+https://github.com/Hashir-Aziz123/zubaan.git@{{COMMIT}}"
!pip install -q --force-reinstall "zubaan @ {{REPO}}" tokenizers
!zubaan version
"""

CHECK_CODE = """
# CHECK 1: the installed code is the commit above, not a cached older copy.
import json
from importlib import metadata

direct = json.loads(metadata.distribution("zubaan").read_text("direct_url.json"))
installed = direct["vcs_info"]["commit_id"]
print("installed commit:", installed)
assert installed.startswith(COMMIT), f"installed {installed}, wanted {COMMIT}"
print("OK: code version verified")
"""

CHECK_GPU = """
import subprocess

# CHECK 2: a GPU that vLLM supports. vLLM needs compute capability 7.0 or higher: a Kaggle T4 is 7.5,
# a P100 is 6.0 and does NOT work. Pick "GPU T4 x2" in the notebook settings.
r = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
assert r.returncode == 0, "no GPU found: turn on a GPU accelerator (T4) in the notebook settings"
print(r.stdout)
name, cap, mem = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
assert float(cap) >= 7.0, f"{name} has compute capability {cap}; vLLM needs 7.0 or higher. Use a T4."
print("OK: GPU", name, "compute capability", cap, mem)
"""

FIND_MODEL = """
from pathlib import Path

# The extended model comes from the output of the CPU notebook (03), added as an input to this one
# with "Add Input > Notebooks". The folder holds zubaan_extension.json, which marks it.
INPUT_ROOT = Path("/kaggle/input")
markers = sorted(INPUT_ROOT.rglob("zubaan_extension.json"))
assert markers, (
    "no extended model found under /kaggle/input. In the sidebar choose Add Input > Notebooks and add "
    "the finished output of 03_extend_export_qwen3"
)
MODEL_DIR = markers[0].parent
print("model folder:", MODEL_DIR)
print(json.dumps(json.loads(markers[0].read_text()), indent=1))
assert (MODEL_DIR / "model.safetensors").exists() or list(MODEL_DIR.glob("model-*.safetensors")), "no weights in the model folder"
assert (MODEL_DIR / "tokenizer.json").exists(), "no tokenizer.json in the model folder"
"""

INSTALL_VLLM = """
# vLLM goes into its own Python 3.12 environment so it cannot replace Kaggle's torch. Takes 5 to 10 minutes.
!pip install -q uv
!uv venv --python 3.12 /tmp/vllm-env
!uv pip install --python /tmp/vllm-env/bin/python vllm
VLLM_VERSION = subprocess.run(
    ["/tmp/vllm-env/bin/python", "-c", "import vllm; print(vllm.__version__)"], capture_output=True, text=True
).stdout.strip()
print("vLLM", VLLM_VERSION)
assert VLLM_VERSION, "vLLM did not install"
"""

SETTINGS = """
NAME = "qwen3-urdu"
MAX_MODEL_LEN = 4096
"""

WRITE_FILES = """
from zubaan.export.vllm import write_vllm_files

# The same generator that 'zubaan export vllm' uses, pointed at this session's model path.
WORK = Path("/kaggle/working")
files = write_vllm_files(WORK / "vllm", str(MODEL_DIR), NAME, MAX_MODEL_LEN)
config_text = files["config"].read_text()
print(config_text)
print(files["script"].read_text())
assert "dtype: half" in config_text and "bfloat16" not in config_text.lower()
"""

SERVE = """
import os
import signal
import time
import urllib.request

LOG = WORK / "vllm_server.log"
env = dict(os.environ, PATH="/tmp/vllm-env/bin:" + os.environ["PATH"])
server = subprocess.Popen(
    f"cd {WORK / 'vllm'} && sh serve_vllm.sh > {LOG} 2>&1", shell=True, env=env, start_new_session=True
)


def get(path, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:8000{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=300))


up = False
for _ in range(120):  # up to 20 minutes: model load, memory profiling and CUDA graph capture
    time.sleep(10)
    if server.poll() is not None:
        break
    try:
        up = NAME in json.dumps(get("/v1/models"))
    except Exception:
        up = False
    if up:
        break
print("server up:", up)
if not up:
    print(LOG.read_text()[-3000:])
assert up, "vLLM did not start: see the log above and /kaggle/working/vllm_server.log"
"""

TEST = """
import time

# CHECK 3: the server uses the EXTENDED tokenizer. vLLM's own tokenization of Urdu text must equal the
# ids of the extended tokenizer.json, and a base-model tokenizer would give far more tokens.
from tokenizers import Tokenizer

tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
sentences = [
    "پاکستان کا دارالحکومت",
    "پاکستان ایک خوبصورت ملک ہے۔",
    "کیا آپ ٹھیک ہیں؟ جی ہاں!",
    "اردو میں 123 اور English کا مرکب mixed script ٹیسٹ",
    "The quick brown fox jumps over the lazy dog.",
]
mismatches = []
for s in sentences:
    want = tok.encode(s, add_special_tokens=False).ids
    got = get("/tokenize", {"model": NAME, "prompt": s, "add_special_tokens": False})["tokens"]
    if got != want:
        mismatches.append((s, want, got))
print(f"tokenization: {len(sentences) - len(mismatches)}/{len(sentences)} identical to tokenizer.json")
assert not mismatches, mismatches

# CHECK 4: it generates. The new rows are untrained, so the text is poor; it only has to answer.
prompt = sentences[0]
t0 = time.perf_counter()
out = get("/v1/completions", {"model": NAME, "prompt": prompt, "max_tokens": 40, "temperature": 0})
seconds = time.perf_counter() - t0
usage = out["usage"]
text = out["choices"][0]["text"]
print("prompt tokens:", usage["prompt_tokens"], "| completion tokens:", usage["completion_tokens"])
print("text:", repr(text))
assert usage["completion_tokens"] > 0

chat = get(
    "/v1/chat/completions",
    {"model": NAME, "messages": [{"role": "user", "content": "پاکستان کا دارالحکومت کیا ہے؟"}], "max_tokens": 40, "temperature": 0},
)
print("chat:", repr(chat["choices"][0]["message"]["content"][:200]))
assert chat["choices"][0]["message"]["content"]

result = {
    "passed": True,
    "vllm": VLLM_VERSION,
    "gpu": name,
    "compute_capability": cap,
    "dtype": "half",
    "model_dir": str(MODEL_DIR),
    "tokenization_identical": len(sentences),
    "prompt_tokens": usage["prompt_tokens"],
    "completion_tokens": usage["completion_tokens"],
    "completion_seconds_including_first_request": round(seconds, 2),
    "completion_text": text,
}
(WORK / "vllm_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print("OK: vLLM served the extended model")
"""

STOP = """
os.killpg(os.getpgid(server.pid), signal.SIGTERM)
time.sleep(5)
print(LOG.read_text()[-1500:])
"""


def build(commit: str) -> dict:
    cells = [
        md(
            """
# Stage 2: start vLLM on the extended Qwen3 model

Checks that **vLLM can serve the extended model** with the config that `zubaan export vllm` generates: the extra embedding rows match the extended tokenizer, `dtype half` loads, and generation works. It also compares vLLM's tokenization of Urdu with the extended `tokenizer.json`, a third independent tokenizer check after Hugging Face and llama.cpp.

Nothing is trained, so the generated text is poor. The check is that the model loads, tokenizes with the extended vocabulary, and answers.

Settings before running:

- Accelerator: **GPU T4 x2** (one GPU is used). **Not P100**: vLLM needs compute capability 7.0 or higher and a P100 is 6.0.
- Internet: **on**.
- Input: **Add Input > Notebooks** and add the finished output of `03_extend_export_qwen3` (it provides the extended model folder).
- No secret is needed.

Takes about 15 to 25 minutes, most of it installing vLLM and loading the model. vLLM is installed in its own Python 3.12 environment so it cannot break Kaggle's torch.

Results: `/kaggle/working/vllm_check.json` (pass or fail, versions, token counts, a sample completion) and `vllm_server.log`. Send back `vllm_check.json`.
"""
        ),
        code(INSTALL.format(commit=commit)),
        code(CHECK_CODE),
        code(CHECK_GPU),
        code(SETTINGS),
        md("## 1. Find the extended model"),
        code(FIND_MODEL),
        md("## 2. Install vLLM (isolated environment)"),
        code(INSTALL_VLLM),
        md(
            "## 3. Generate the vLLM config and start the server\n\nThe config comes from the same generator as `zubaan export vllm`, so this also tests that generator."
        ),
        code(WRITE_FILES),
        code(SERVE),
        md(
            "## 4. Check tokenization and generation\n\nFails if vLLM tokenizes Urdu differently from the extended `tokenizer.json`."
        ),
        code(TEST),
        md("## 5. Stop the server"),
        code(STOP),
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
    out = Path(__file__).parent / "04_vllm_check_qwen3.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
