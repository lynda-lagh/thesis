#!/usr/bin/env bash
# Step 1 - build the True/False/Unknown benchmark from YAGO3-10.
set -e
cd "$(dirname "$0")/.."

# Smoke test first (no data needed):
python -m src.data.build_benchmark --smoke-test

# Real build (uncomment once data/raw/YAGO3-10/{train,valid,test}.txt are in place):
# python -m src.data.get_yago --src /kaggle/input/yago3-10 --dst data/raw/YAGO3-10
# python -m src.data.build_benchmark \
#     --raw-dir data/raw/YAGO3-10 \
#     --out-dir data/processed/YAGO3-10 \
#     --n-per-class 6000 --seed 42
