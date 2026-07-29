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

### Faster data option
Instead of regenerating each run, upload `data/processed/YAGO3-10` once as a
Kaggle Dataset, add it as an input, and skip Cell 2 — point `--data-dir` at
`/kaggle/input/<your-dataset>/YAGO3-10`.
