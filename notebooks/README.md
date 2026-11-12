# Kaggle notebooks

GPU-heavy stages (adapt, draft training, larger acceptance audits, vLLM benchmarks) run on Kaggle.

Rules:

- Notebooks install this package and call library functions. No logic lives only in a notebook.
  ```python
  !pip install -q "git+https://github.com/<user>/zubaan.git#egg=zubaan[train]"
  ```
- Set the accelerator to GPU T4 x2 or P100 and turn internet on.
- Store `HF_TOKEN` as a Kaggle secret, never in code.
- T4 and P100: use fp16 (no BF16, no FP8).
- Push checkpoints to the Hugging Face Hub during training and resume with `resume_from_checkpoint=True`.
- Run long jobs as background commits (Save Version, Save and Run All).

Naming: `NN_stage_purpose.ipynb`, for example `03_adapt_qwen3_1.7b.ipynb`.
