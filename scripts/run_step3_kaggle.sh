#!/usr/bin/env bash
# Step 3 - P(True) discriminative scoring on the test set (Kaggle GPU).
set -e

pip install -q -U "transformers>=4.45" "peft>=0.12" "bitsandbytes>=0.43" "accelerate>=0.33"

python -m src.scoring.ptrue \
    --adapter /kaggle/working/models/qwen-yago-3way \
    --data data/processed/YAGO3-10/test.jsonl \
    --out results/test_scores.jsonl \
    --batch-size 8

# Also score valid.jsonl if you want threshold tuning for Step 4/5:
# python -m src.scoring.ptrue --adapter /kaggle/working/models/qwen-yago-3way \
#     --data data/processed/YAGO3-10/valid.jsonl --out results/valid_scores.jsonl --batch-size 8
