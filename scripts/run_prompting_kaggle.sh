#!/usr/bin/env bash
# Prompt-engineering phase - frozen base model, no adapter, across variants.
set -e

pip install -q -U "transformers>=4.45" "bitsandbytes>=0.43" "accelerate>=0.33"

for VARIANT in zero_shot few_shot evidence cot; do
    echo "=== variant: $VARIANT ==="
    python -m src.prompting.run_frozen_eval \
        --base Qwen/Qwen2.5-7B-Instruct \
        --data data/processed/YAGO3-10/test.jsonl \
        --train-data data/processed/YAGO3-10/train.jsonl \
        --raw-train data/raw/YAGO3-10/train.txt \
        --variant $VARIANT \
        --out results/frozen_${VARIANT}_scores.jsonl \
        --batch-size 8 --limit 300
done

# --limit 300 keeps this affordable (esp. the CoT variant, which generates
# text). Drop --limit to score the full 1800-example test set once you know
# the pipeline works and want the final numbers.
