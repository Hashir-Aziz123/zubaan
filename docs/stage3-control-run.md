# Stage 3 control run: how to run notebook 11

This page explains how to run `notebooks/11_adapt_control_qwen3.ipynb` on Kaggle: what to set up before you start, what each section does, what a good run prints, and what to do when something goes wrong. The design and its reasons are in `docs/decisions.md` ("compute-matched control run, design", 2026-11-25).

## What the run answers

The adapted model (notebook 07) scores 1.718 bits per character on FLORES Urdu, against 2.339 for the base model. It also trained on 16M tokens of Urdu that the base model never saw. So the gain could come from the extended tokenizer, from the extra Urdu training, or from both.

The control run trains the **base model with its own tokenizer**, using the same recipe for the same number of tokens:

| | Main run (notebook 07) | Control (notebook 11) |
| --- | --- | --- |
| Model and tokenizer | Qwen3-1.7B + 10,000 Urdu tokens | Qwen3-1.7B, base tokenizer |
| Steps | 100 (new rows only) + 700 (LoRA + new rows) | 800 (LoRA only) |
| Tokens per step | 32 blocks x 1,024 = 32,768 | same |
| Total tokens | 26.2M | same |
| Urdu text covered | 1.47 epochs | about 0.64 epochs |
| LoRA, learning rate, fp16, seed, 10% English | as listed in decisions | identical |

The base tokenizer needs about 2.29x more tokens for the same Urdu, so at equal cost the control sees less of the text. That is what matching on compute means. If the adapted model still beats the control, the extended tokenizer made better use of the same training budget. If the control matches or beats it, the gain came from the training, not the tokenizer. Both outcomes are worth reporting.

## Before you start

1. **Push the code.** The notebook installs zubaan from one pinned commit (`ad7692d` or later). Kaggle downloads it from GitHub, so it must be pushed. The first code cell fails with "installed ..., wanted ..." if the wrong code was installed.
2. **Pick a new private Hub repo name**, for example `your-name/zubaan-qwen3-control`. Never reuse the main run's repo (`zubaan-qwen3-ckpt`). The notebook creates the repo as private if it does not exist, and stops if the repo already holds a main-run checkpoint.
3. **Check the `HF_TOKEN` Kaggle secret.** It must have write access. It is the same secret notebook 07 used.
4. **Check the GPU quota.** A full run needs about 10 GPU hours: about 15 minutes of setup and data, about 8.5 to 9 hours of training, about 10 minutes for the merge, and about 45 minutes for the optional evaluation. Kaggle gives about 30 GPU hours per week and at most 12 hours per session.

## Kaggle settings

In the notebook editor:

- **Accelerator:** GPU T4 x2. One GPU is used. Do not use the P100.
- **Internet:** on. It downloads the base model, Urdu and English Wikipedia, FLORES and Belebele.
- **Add-ons > Secrets:** attach `HF_TOKEN`.
- **Input:** none. The base model is downloaded to `/tmp`, so it is not saved as output.

Upload the notebook with *File > Import Notebook*, or paste the cells from the repo copy.

## The settings cell

The first code cell is the only one you edit:

```python
HUB_REPO = ""            # set this: a NEW private repo
MICRO_BATCH = 1          # do not change
GRAD_CHECKPOINT = False  # do not change
PHASE_A_STEPS = 0        # do not change
PHASE_B_STEPS = 800      # do not change
STOP_AT_STEP = None      # None = all 800 steps; 400 = a shorter run
RUN_EVAL = True          # score the control after the merge, if there is time
```

- **Do not change** the four schedule lines, either now or between sessions of the same run. They decide which data and which learning rate each step gets, and a resumed run refuses a checkpoint made with different values.
- **`STOP_AT_STEP`** can change between sessions. `400` stops at the step of the main run's step-400 milestone, which had seen the same number of tokens. A later session with `None` continues the same run to 800. The 400-step comparison is weaker than the full one because the two learning rate schedules are at different points by then (the main run's schedule restarted after its 100 steps of phase A).
- **`RUN_EVAL`** turns the evaluation at the end on or off. Even when it is on, the evaluation is skipped if more than 10.75 hours of the session have passed, so it cannot push the session past 12 hours.

## Running it

Use **Save Version > Save & Run All (Commit)**, not the interactive editor. A committed run keeps going with the browser tab closed, and its files appear in the version's **Output** tab when it ends.

For a first look, you can run the cells up to and including section 4 (the five-step check) interactively, then stop and start the committed run. The five-step check takes a few minutes and catches most setup problems.

## What each section does and what to look for

**Install and checks (first four code cells).** The install cell installs the pinned commit and removes Kaggle's `torchao`, which breaks `peft`. The checks print the installed commit, the GPU, and the torch, transformers and peft versions. Expect `OK: code version verified` and `OK: GPU Tesla T4 compute capability 7.5`.

**Secret.** Prints `Hub checkpoints go to the PRIVATE repo <your repo>`.

**1. Base model.** Downloads `Qwen/Qwen3-1.7B` at a fixed revision and prints that revision. Write it down. The cell checks that the tokenizer has no Urdu extension (vocabulary under 160,000; the base has 151,669). Expect `OK: base model and base tokenizer`.

**2. Configuration.** Writes `adapt.yaml` and prints it. Check `control: true`, `phase_a_steps: 0`, `phase_b_steps: 800`, `english_mb: 20` and your `hub_repo`.

**3. Data** (about 15 minutes). Downloads FLORES (devtest and dev), builds the Urdu Wikipedia corpus and the English replay text, and tokenizes both with the base tokenizer into `data/adapt_control/`. Any training line that overlaps FLORES or Belebele is dropped and counted. The check cell then prints:

- the number of Urdu and English blocks and Urdu tokens;
- `base / extended tokens on the same Urdu text`, which should be about **2.29x** (the cell stops outside 2.0 to 2.6, which would mean a different corpus or tokenizer);
- the epochs that 800 steps cover, about **0.64** (main run: 1.47).

**4. Five-step check.** Five real fp16 LoRA steps on the base model in a throwaway folder, without the Hub. All steps must be phase B with finite losses. A step or two skipped by the fp16 loss scaler (gradient norm `inf`) at the start is normal. Expect `OK: 5 fp16 LoRA steps, finite losses [...]`. The next cell frees GPU memory.

**5. Hub check.** Writes and deletes a test file to prove the repo is private and writable, then says where the run starts:

- `No checkpoint yet: this run starts from step 0.` for a new run;
- `A checkpoint at step N exists: this run RESUMES from it.` followed by `OK: the existing checkpoint is from this control run` when you continue a run.

If it says the repo holds a main-run checkpoint, you set the wrong `HUB_REPO`.

**6. Train** (about 8.5 to 9 hours for 800 steps). Logs a line every 10 steps:

```
step 120/800 B loss 2.1345 grad 0.41 860 tok/s
```

and the held-out Urdu and English bits per character every 100 steps. Things to watch:

- **Speed** should be close to the main run's 817 to 850 tokens/s. The base model's vocabulary is smaller, so it may be slightly faster. A rate far below 800 means about 10 hours or more for the full run.
- **Held-out Urdu bits per character** should fall steadily. The main run's curve went from 2.41 at step 100 to 1.52 at step 800 (log in `reports/notebook7_logs.txt`). Bits per character is comparable across the two tokenizers; loss and tokens/s are not.
- **English bits per character** should stay near or below the base's 1.21.

A checkpoint goes to `<repo>/last` every 50 steps, and every 200th step is also kept in `<repo>/milestones/step-NNNNNNN`. The cell ends by asserting that the run reached its target step.

**7. Merge, save and verify** (about 10 minutes). Folds LoRA into the weights in float32 and saves a plain model to `control/model/`. It refuses to save if merging changed the logits, if any embedding row differs from the base, or if the tokenizer files differ from the base. It writes `control/model/zubaan_adapt.json` (not `zubaan_extension.json`) and prints it. Check `"control": true` and the step.

**8. Evaluate (optional, about 45 minutes).** Scores the merged control model with the notebook 08 protocol: 1,000 texts per set, all 900 Belebele questions per language, float32. It first checks that the held-out Urdu file has the same hash notebook 08 used, so the numbers sit in the same table. It prints the control's bits per character next to the adapted model's numbers from notebook 08 and writes `control_quality.json`.

**9. Output size.** Deletes the local checkpoints (they are on the Hub) and lists the large files left. The merged model is about 3.5 GB; the total must stay under Kaggle's 20 GB output limit.

## When something goes wrong

**The session ended before training finished** (12-hour limit, crash, disconnect). Start a new committed run with the same settings. The Hub check should say it resumes from the last step, and training continues with exactly the data and learning rate it would have used. At most 50 steps (about 30 minutes) are repeated.

**The install check fails** ("installed ..., wanted ..."). Kaggle kept an old copy. Use *Run > Restart session* and run from the top. If the commit cannot be found at all, it was not pushed.

**Out of GPU memory.** Unlikely: the main run peaked at 13.3 GB with a larger vocabulary. Do not change `MICRO_BATCH` in the middle of a run. If it happens at step 0, ask before changing settings: a different micro batch changes the speed, not the data per step, but it must then be used for the whole run.

**Loss becomes NaN** (`TrainingDiverged ... set precision: fp32`). The main run never hit this in fp16. Switching to fp32 is a different run: it needs a new `HUB_REPO`, and it may not fit in memory or time on a T4. Stop and decide before spending the quota.

**The token ratio check fails.** The Wikipedia dump or a tokenizer differs from what was measured. Do not train. Send the printed numbers.

**The evaluation was skipped** (not enough time, or `RUN_EVAL = False`). The merged model is in the version's output under `control/model/`. Scoring it later needs a GPU session with that output as an input and `zubaan adapt eval` on it. That cell does not exist yet as its own notebook; it is a short addition when needed.

**You want to stop early.** Set `STOP_AT_STEP` and run again; it stops at that global step. Any milestone can also be merged later with `zubaan adapt merge --config adapt.yaml --milestone STEP`.

## What to send back

From the version's Output tab, save into the local `reports/` folder:

- `control/adapt_summary.json`: steps, resume point, data size, in-run evaluations;
- `control/train_log.jsonl`: every logged step and evaluation;
- `control_quality.json`, if section 8 ran;
- the base model revision printed in section 1.

Keep the notebook version saved: its `control/model/` is the input for any later evaluation.

## After the run

1. Compare the control with the adapted model on bits per character (held-out Urdu, FLORES Urdu, FLORES English) and Belebele `acc_char`, with intervals. Never compare loss, perplexity or tokens/s across the two tokenizers.
2. For a 400-step run, compare against the main run's milestone 400 (merge it with `--milestone 400` from the main run's repo and score both the same way), and state that the learning rate schedules differ at that point.
3. Record the result in `docs/decisions.md` and in the Stage 3 report section, including the case where the control wins.
