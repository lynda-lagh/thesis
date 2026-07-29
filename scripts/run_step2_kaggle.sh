#!/usr/bin/env bash
# Step 2 - QLoRA 3-way fine-tuning of Qwen2.5-7B-Instruct on Kaggle (P100 / T4).
# Run this in a Kaggle notebook cell (GPU accelerator ON) after building the
# benchmark and adding the repo + data as inputs.
set -e

pip install -q -U "transformers>=4.45" "peft>=0.12" "bitsandbytes>=0.43" "accelerate>=0.33" datasets

python -m src.train.qlora_finetune \
    --data-dir data/processed/YAGO3-10 \
    --base Qwen/Qwen2.5-7B-Instruct \
    --out models/qwen-yago-3way \
    --epochs 2 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 2e-4 \
    --max-seq-len 512

# If you hit OOM on a single T4/P100: drop --batch-size to 2 and raise
# --grad-accum to 8 (same effective batch, less memory).
