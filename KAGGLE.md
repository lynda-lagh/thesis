# Running on Kaggle

Repo: https://github.com/lynda-lagh/thesis

In the Kaggle notebook settings: **Accelerator = GPU T4 x2** and **Internet = On**.

> Do NOT use the P100. Its compute capability (sm_60) is too old for the current
> PyTorch/bitsandbytes build, so 4-bit QLoRA crashes with
> `Error named symbol not found ... ops.cu`. The T4 (sm_75) works.

### Cell 1 — get the code
```python
!git clone https://github.com/lynda-lagh/thesis.git
%cd thesis
```
(To pull later changes instead: `!git -C /kaggle/working/thesis pull`.)

### Cell 2 — build the benchmark (Step 1)
```python
!pip install -q pykeen
!python -m src.data.fetch_yago_pykeen --dst data/raw/YAGO3-10
!python -m src.data.build_benchmark --raw-dir data/raw/YAGO3-10 --out-dir data/processed/YAGO3-10
```

### Cell 3 — QLoRA 3-way fine-tune (Step 2)
```python
!pip install -q -U transformers peft bitsandbytes accelerate datasets
!CUDA_VISIBLE_DEVICES=0 python -m src.train.qlora_finetune \
    --data-dir data/processed/YAGO3-10 \
    --base Qwen/Qwen2.5-7B-Instruct \
    --out /kaggle/working/models/qwen-yago-3way \
    --epochs 2 --batch-size 4 --grad-accum 4 --lr 2e-4 --max-seq-len 512
```
`CUDA_VISIBLE_DEVICES=0` uses a single T4 (the 7B fits in 4-bit on one card),
avoiding multi-GPU sharding issues.
Output (adapter + `label_info.json`) lands in `/kaggle/working/models/qwen-yago-3way`
so it is saved as notebook output. If you hit OOM on a single GPU: `--batch-size 2
--grad-accum 8`.

### Cell 4 — P(True) scoring (Step 3)
```python
!cd thesis && pip install -q -U transformers peft bitsandbytes accelerate
!cd thesis && python -m src.scoring.ptrue \
    --adapter /kaggle/working/models/qwen-yago-3way \
    --data data/processed/YAGO3-10/test.jsonl \
    --out results/test_scores.jsonl \
    --batch-size 8
```
Single forward pass per example (no generation), so it is fast even at 7B.
Output: `results/test_scores.jsonl` with `p_true/p_false/p_unknown`, `predicted_label`,
`confidence`, `correct`, and the Step-1 metadata (`gen_strategy`, `unknown_type`,
`rare_head`) passed through for Step 4's breakdown analysis.

### Cell 5 — calibration / selective-risk evaluation (Step 4)
```python
!cd thesis && python -m src.eval.calibration --scores results/test_scores.jsonl --out results/calibration.json
```
No GPU or heavy deps needed (pure stdlib). You can instead copy `test_scores.jsonl`
to your own PC and run this step there:
```powershell
cd abstain-kgc
python -m src.eval.calibration --scores results/test_scores.jsonl --out results/calibration.json
```
Reports: accuracy/macro-F1, confusion matrix, ECE/Brier/NLL, risk-coverage curve +
AURC, abstention-AUROC (does `p_unknown` separate gold-Unknown from gold-True/False?),
a breakdown by `unknown_type`/`rare_head`, and a KG-LLM-style forced-choice binary
baseline simulation (how confidently wrong a closed-world model is on Unknown triples).

### Cell 6 — uncertainty frameworks comparison (Step 5a)
```python
!cd thesis && python -m src.uncertainty.frameworks --scores results/test_scores.jsonl --out results/framework_comparison.json
```
Pure Python, no GPU needed. Re-expresses the model's (p_true,p_false,p_unknown)
as probabilistic (baseline) / evidential (DST mass, Bel, Pl, BetP) /
possibilistic (Dubois-Prade transform, necessity) / fuzzy (5-point linguistic
scale), and reports the same Step-4 metrics for each side by side.

### Cell 7 — Dempster-Shafer multi-source fusion (Step 5b, flagship)
```python
!cd thesis && python -m src.uncertainty.dempster_fusion \
    --scores results/test_scores.jsonl \
    --train data/raw/YAGO3-10/train.txt \
    --out results/fused_scores.jsonl \
    --discount2 0.85
```
Fuses the LLM's own judgment with an independent structural/KG-derived source
(relation functionality + known tails) via Dempster's combination rule.
Prints before/after accuracy, confusion matrix, and average conflict K.
See the module docstring for an important, verified finding: fusing with a
mostly-agnostic second source reinforces the more decisive source rather than
pulling toward Unknown - a real DST property, not a bug, and a pointer to the
natural refinement (a 3-element frame with Unknown as its own atomic
hypothesis).

### Cell 8 — prompt-engineering phase (frozen model, no adapter)
```python
!cd thesis && pip install -q -U transformers bitsandbytes accelerate
!cd thesis && python -m src.prompting.run_frozen_eval \
    --base Qwen/Qwen2.5-7B-Instruct \
    --data data/processed/YAGO3-10/test.jsonl \
    --train-data data/processed/YAGO3-10/train.jsonl \
    --raw-train data/raw/YAGO3-10/train.txt \
    --variant zero_shot \
    --out results/frozen_zero_shot_scores.jsonl \
    --batch-size 8 --limit 300
```
Change `--variant` to `few_shot` / `evidence` / `cot` to run the other three
(or use `bash scripts/run_prompting_kaggle.sh` to run all four in sequence).
`zero_shot`/`few_shot`/`evidence` are single forward passes (fast); `cot`
additionally generates a short reasoning trace, so give it more time and keep
`--limit` modest until you've confirmed it works. Drop `--limit` for final
numbers. Feed any of these `_scores.jsonl` files into Step 4's
`src.eval.calibration` for the same metrics as the fine-tuned model, to
compare "no training" vs "fine-tuned" head to head.

### Faster data option
Instead of regenerating each run, upload `data/processed/YAGO3-10` once as a
Kaggle Dataset, add it as an input, and skip Cell 2 — point `--data-dir` at
`/kaggle/input/<your-dataset>/YAGO3-10`.
