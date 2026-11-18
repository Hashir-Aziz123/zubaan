# ruff: noqa: E501
"""Cell code shared by the stage 3 notebooks 06 (speed trial) and 07 (main run).

The install, check, model-finding, data and merge cells are the ones notebook 05 already proved on
Kaggle; they are imported from ``make_05_notebook.py`` unchanged. Only what is new lives here.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_05_notebook", Path(__file__).parent / "make_05_notebook.py"
)
_n5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_n5)

md, code = _n5.md, _n5.code
INSTALL = _n5.INSTALL
CHECK_CODE = _n5.CHECK_CODE
CHECK_GPU = _n5.CHECK_GPU
SECRET = _n5.SECRET
FIND_MODEL = _n5.FIND_MODEL
DATA = _n5.DATA
DATA_CHECK = _n5.DATA_CHECK
CHECK_TRAIN = _n5.CHECK_TRAIN
MERGE = _n5.MERGE
CLEANUP = _n5.CLEANUP

# Hand GPU memory back to the driver after the in-process check, before other processes need it.
FREE_GPU = """
import gc

import torch

gc.collect()
torch.cuda.empty_cache()
print("GPU memory in use now (GB):", round(torch.cuda.memory_allocated() / 1e9, 2))
"""

# The adapt config shared by 06 and 07. ``OUT_NAME``, ``HUB`` and the settings are defined by the
# cell before it. Always 32 blocks (33k tokens) per optimizer step, whatever the micro batch.
CONFIG = """
from pathlib import Path

import yaml

WORK = Path("/kaggle/working")
adapt = {
    "model_path": str(MODEL_DIR),
    "urdu_corpus": "data/urdu/urdu_train.txt",
    "heldout": "data/urdu/urdu_heldout.txt",
    "english_corpus": "data/english/english_replay.txt",
    "english_mb": 12,
    "guard_urdu": ["data/flores_urd.txt", "data/flores_dev/flores_urd.txt"],
    "guard_english": ["data/flores_eng.txt", "data/flores_dev/flores_eng.txt"],
    "guard_belebele": True,
    "drop_overlaps": True,
    "data_dir": "data/adapt",
    "block_size": 1024,
    "english_replay_ratio": 0.1,
    "output_dir": str(WORK / OUT_NAME),
    "micro_batch_size": MICRO_BATCH,
    "grad_accum": 32 // MICRO_BATCH,
    "lr_new_rows": 1e-3,
    "lr_lora": 2e-4,
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "precision": "fp16",
    "gradient_checkpointing": GRAD_CHECKPOINT,
    "device": "cuda",
    "seed": 0,
    "hub_repo": HUB,
    "hub_private": True,
    "merged_dtype": "bfloat16",
    "phase_a_steps": PHASE_A_STEPS,
    "phase_b_steps": PHASE_B_STEPS,
    "warmup_steps": 50,
    "log_every": 10,
    "eval_every": 100,
    "eval_texts": 200,
    "checkpoint_every": 50,
    "keep_checkpoints": 2,
    "milestone_every": 200,
}
config = {
    "urdu_corpus": {"raw_dir": "data/raw", "out_dir": "data/urdu", "flores": "data/flores_urd.txt", "max_mb": 100, "heldout_percent": 1.0},
    "adapt": adapt,
}
Path("adapt.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(Path("adapt.yaml").read_text())
"""

# Speed trial: four settings, each in its own process.
SPEED = """
# The same few real steps under four settings, each in its own process so that an out-of-memory
# error in one cannot hurt the others. Phase B (LoRA plus new rows) is what the main run mostly does,
# so the speed is taken from phase B steps.
import copy
import shutil
import statistics
import subprocess

VARIANTS = [("mb2_ckpt_on", 2, True), ("mb4_ckpt_on", 4, True), ("mb1_ckpt_off", 1, False), ("mb2_ckpt_off", 2, False)]
base = yaml.safe_load(Path("adapt.yaml").read_text())
results = []
for name, mb, gc_on in VARIANTS:
    conf = copy.deepcopy(base)
    out = WORK / f"speed_{name}"
    shutil.rmtree(out, ignore_errors=True)
    conf["adapt"].update(
        output_dir=str(out), hub_repo=None, resume=False, micro_batch_size=mb,
        gradient_checkpointing=gc_on, grad_accum=8 // mb, phase_a_steps=2, phase_b_steps=10,
        warmup_steps=1, log_every=1, eval_every=9999, eval_texts=5, checkpoint_every=9999, keep_checkpoints=1,
    )
    cfg_file = Path(f"speed_{name}.yaml")
    cfg_file.write_text(yaml.safe_dump(conf, sort_keys=False))
    r = subprocess.run(["zubaan", "adapt", "run", "--config", str(cfg_file), "--steps", "8"], capture_output=True, text=True)
    rows = []
    if (out / "train_log.jsonl").exists():
        rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    steps = [x for x in rows if "loss" in x]
    phase_b = [x["tokens_per_s"] for x in steps if x["phase"] == "B"][1:]  # the first B step pays for the switch
    if r.returncode != 0 or len(phase_b) < 3:
        text = r.stdout + r.stderr
        why = "out of GPU memory" if "out of memory" in text.lower() else text.strip().splitlines()[-1:]
        res = {"name": name, "micro_batch": mb, "gradient_checkpointing": gc_on, "ok": False, "why": why}
    else:
        tps = statistics.median(phase_b)
        res = {
            "name": name, "micro_batch": mb, "gradient_checkpointing": gc_on, "ok": True,
            "median_tokens_per_s": round(tps), "peak_gpu_gb": round(max(x["gpu_mem_gb"] for x in steps), 2),
            "hours_per_100_steps": round(100 * 32 * 1024 / tps / 3600, 2),
        }
    print(res)
    results.append(res)
    shutil.rmtree(out, ignore_errors=True)
ok = [x for x in results if x["ok"] and x["peak_gpu_gb"] < 14.0]
assert ok, "no setting worked: send the lines above"
best = max(ok, key=lambda x: x["median_tokens_per_s"])
(WORK / "speed_report.json").write_text(json.dumps({"gpu": gpu_name, "results": results, "best": best["name"]}, indent=2))
print("fastest that fits:", best["name"], "->", best["hours_per_100_steps"], "hours per 100 steps")
print("In notebook 07 set MICRO_BATCH =", best["micro_batch"], "and GRAD_CHECKPOINT =", best["gradient_checkpointing"])
print("Send back speed_report.json.")
"""

# Main run: prove the Hub works before spending hours, and say where the run will start.
HUB_PREFLIGHT = """
# Before hours of training: the repo exists, is private, accepts a write, and tells us where we start.
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(HUB_REPO, private=True, exist_ok=True)
assert api.repo_info(HUB_REPO).private, f"{HUB_REPO} is public: make it private on the Hub first"
api.upload_file(path_or_fileobj=b"ok", path_in_repo="_preflight/ok.txt", repo_id=HUB_REPO)
api.delete_folder("_preflight", repo_id=HUB_REPO)
files = set(api.list_repo_files(HUB_REPO))
if "last/meta.json" in files:
    from huggingface_hub import hf_hub_download

    step = json.loads(Path(hf_hub_download(HUB_REPO, "last/meta.json")).read_text())["step"]
    print(f"OK: Hub writable. A checkpoint at step {step} exists: this run RESUMES from it.")
else:
    print("OK: Hub writable. No checkpoint yet: this run starts from step 0.")
print("milestones already saved:", sorted({f.split("/")[1] for f in files if f.startswith("milestones/")}))
"""

MAIN = """
from zubaan.adapt.pipeline import run_adapt
from zubaan.config import load_config

cfg = load_config("adapt.yaml").adapt
summary = run_adapt(cfg, log=print)
print({k: summary[k] for k in ("resumed_from_step", "final_step", "total_steps", "finished")})
assert summary["finished"], "the schedule did not finish: run this notebook again with the same settings to resume"
for e in summary["evals"]:
    print(e["step"], e["eval"])
"""
