"""Step 2 - QLoRA 3-way instruction tuning of Qwen2.5-7B-Instruct.

Fine-tunes the base model to answer True / False / Unknown on the Step 1
benchmark. 4-bit (nf4) so it fits a single 16 GB Kaggle P100 / T4. The loss is
masked to the assistant's answer tokens only (completion-only), so the model
learns to *produce the label*, not to reproduce the prompt.

Run (Kaggle)
------------
    pip install -U transformers peft bitsandbytes accelerate datasets
    python -m src.train.qlora_finetune \
        --data-dir data/processed/YAGO3-10 \
        --base Qwen/Qwen2.5-7B-Instruct \
        --out models/qwen-yago-3way \
        --epochs 2 --batch-size 4 --grad-accum 4 --lr 2e-4 --max-seq-len 512

Outputs
-------
    <out>/                LoRA adapter + tokenizer (load with PeftModel)
    <out>/label_info.json label strings + first-token ids (used by Step 3 P(True))

Notes
-----
* T4 and P100 do not support bfloat16 -> we train in fp16.
* gradient_checkpointing + paged_adamw_8bit keep memory well under 16 GB.
* This script must run on a GPU machine (Kaggle). It will refuse to run on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def load_yaml_cfg(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------- #
# Dataset: tokenize chat messages, mask everything before the assistant answer
# --------------------------------------------------------------------------- #

def build_dataset(jsonl_path: Path, tokenizer, max_seq_len: int):
    import torch
    from torch.utils.data import Dataset

    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]

    class KGCDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            row = rows[i]
            messages = row["messages"]                 # system + user
            answer = row["response"]                    # "True"/"False"/"Unknown"

            # prompt = messages + generation prompt (assistant header, no content)
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
            )
            # full = messages + assistant answer
            full_ids = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": answer}],
                add_generation_prompt=False, tokenize=True,
            )
            # truncate from the left of the prompt if needed (keep the answer)
            if len(full_ids) > max_seq_len:
                overflow = len(full_ids) - max_seq_len
                prompt_ids = prompt_ids[overflow:]
                full_ids = full_ids[overflow:]

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            labels = labels[: len(full_ids)]
            return {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    return KGCDataset()


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids, lab = b["input_ids"], b["labels"]
            pad = maxlen - len(ids)
            input_ids.append(torch.cat([ids, torch.full((pad,), self.pad_id, dtype=torch.long)]))
            labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
        }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="QLoRA 3-way fine-tuning (Step 2).")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--data-dir", default="data/processed/YAGO3-10")
    ap.add_argument("--base", default=None, help="override base model")
    ap.add_argument("--out", default="models/qwen-yago-3way")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU found. Step 2 must run on a GPU (Kaggle P100 / T4). "
            "Build the dataset locally, then run this on Kaggle."
        )

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        TrainingArguments, Trainer, set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(args.seed)
    cfg = load_yaml_cfg(args.config)
    base = args.base or cfg.get("model", {}).get("base", "Qwen/Qwen2.5-7B-Instruct")
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f">> base model: {base}")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,   # T4/P100: fp16 (no bf16)
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = build_dataset(data_dir / "train.jsonl", tokenizer, args.max_seq_len)
    eval_path = data_dir / "valid.jsonl"
    eval_ds = build_dataset(eval_path, tokenizer, args.max_seq_len) if eval_path.exists() else None
    collator = PadCollator(tokenizer.pad_token_id)
    print(f">> train examples: {len(train_ds)}"
          + (f" | eval: {len(eval_ds)}" if eval_ds else ""))

    # Trainer checkpoints go to a scratch dir; the final --out folder ends up
    # holding ONLY the trained adapter (+ tokenizer + label_info). save_only_model
    # keeps even the scratch checkpoints adapter-only (no optimizer state), and we
    # delete the scratch dir at the end.
    ckpt_dir = str(out) + "_ckpt"
    ta_kwargs = dict(
        output_dir=ckpt_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        save_steps=args.save_steps,
        save_total_limit=1,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps if eval_ds else None,
        report_to="none",
        seed=args.seed,
    )
    try:
        targs = TrainingArguments(save_only_model=True, **ta_kwargs)  # transformers >= 4.42
    except TypeError:
        targs = TrainingArguments(**ta_kwargs)
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()

    # save ONLY the trained adapter (PEFT save_model writes adapter weights only,
    # ~tens of MB — not the 15 GB base model), plus tokenizer + label info.
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    # remove the scratch checkpoint dir so --out holds only the final adapter
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    # save label token info for Step 3 (P(True) scoring)
    labels = cfg.get("labels", ["True", "False", "Unknown"])
    label_info = {
        "labels": labels,
        "first_token_id": {lab: tokenizer.encode(lab, add_special_tokens=False)[0] for lab in labels},
        "space_first_token_id": {lab: tokenizer.encode(" " + lab, add_special_tokens=False)[0] for lab in labels},
        "base_model": base,
    }
    with open(out / "label_info.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=2)

    total = sum(p.stat().st_size for p in Path(out).rglob("*") if p.is_file())
    print(f">> saved adapter + tokenizer + label_info.json to {out}")
    print(f">> output size: {total/1e6:.1f} MB (adapter only; base model NOT saved)")
    print(">> to reload: PeftModel.from_pretrained(base_model, '%s')" % out)


if __name__ == "__main__":
    main()
