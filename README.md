# abstain-kgc

**Calibrated, abstention-aware Knowledge Graph Completion with LLMs.**

Thesis project extending *KG-LLM* (Yao et al., ICASSP 2025) along the axes the
original paper leaves empty: **evidence context (Stage 2)** and **calibration /
abstention (Stage 6)**. Instead of forcing a confident `True/False` on every
triple, we let the model answer **True / False / Unknown** and evaluate whether
it *abstains and calibrates* correctly — using the generation–discrimination gap
(P(True)) and test-time discriminative distillation (SECL, arXiv 2604.09624).

## The four novel hooks

| # | Hook | Skeleton location |
|---|------|-------------------|
| 1 | Open-world **"unknown"** target class | new Stage 0.5 axis: *target epistemic status* |
| 2 | **Selective** predict-or-abstain output | new Stage 5 formulation |
| 3 | **Test-time adapted** regime (SECL port) | new Stage 4(a) regime |
| 4 | **Calibration** metrics (ECE, Brier, AURC) | new Stage 7 metric family |

## Environment

- Base model: **Qwen2.5-7B-Instruct** (4-bit QLoRA)
- Target hardware: **Kaggle P100 16GB** or **T4 x2** (all steps fit in 4-bit)
- Dataset: **YAGO3-10**

## Pipeline (built one step at a time)

```
Step 1  data/build_benchmark.py   ->  true/false/unknown JSONL + data card   [DONE]
Step 2  train/qlora_finetune.py   ->  QLoRA 3-way instruction tuning         [stub]
Step 3  scoring/ptrue.py          ->  P(True) discriminative scoring         [stub]
Step 4  eval/calibration.py       ->  ECE / Brier / risk-coverage / AURC     [stub]
Step 5  secl/test_time.py         ->  test-time discriminative distillation  [stub]
```

## Quick start (Step 1)

```bash
# 1) put YAGO3-10 train.txt / valid.txt / test.txt under data/raw/YAGO3-10/
#    (or run a smoke test on a synthetic sample first)
python -m src.data.build_benchmark --smoke-test

# 2) real build
python -m src.data.build_benchmark \
    --raw-dir data/raw/YAGO3-10 \
    --out-dir data/processed/YAGO3-10 \
    --n-per-class 6000 \
    --seed 42
```

Output: `train.jsonl`, `valid.jsonl`, `test.jsonl` (fields: `head, relation,
tail, label, prompt, response, gen_strategy, rel_functionality`) plus
`data_card.json` with class balance and leakage checks.

## Design note — how we separate *False* from *Unknown*

This is the core methodological contribution of Step 1. Under the open-world
assumption a missing triple is **not** automatically false. We only label a
corrupted triple **False** when we have positive grounds:

- **False**: relation `r` is (near-)**functional** on the tail side *and* the
  head already has a known, different tail. Asserting a different type-consistent
  value then contradicts the KG → confidently false.
- **Unknown**: relation `r` is **non-functional** (many-to-many) and the
  corrupted triple is type-consistent, absent from every split, and not derivable
  via a symmetric/inverse edge → genuinely unverifiable → unknown.
- **True**: the observed triple.

Functionality is measured from the data (`fun(r) = |distinct heads| /
|triples|`), not hardcoded, so the protocol transfers to any KG.
