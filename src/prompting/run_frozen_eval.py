"""Prompt-engineering phase: score the FROZEN base model (no QLoRA adapter)
across prompt variants - the no-training baseline for comparison against the
fine-tuned model (Step 2/3), and the source of the "before fine-tuning"
number in the thesis's training-baseline comparison.

zero_shot / few_shot / evidence use the same single-forward-pass P(True)-style
scoring as Step 3 (fast, no generation). cot additionally generates a short
reasoning trace, then does one more forward pass with "Final answer:" forced
as the next tokens, so it still yields a comparable discriminative score.

Run (Kaggle, GPU)
------------------
    python -m src.prompting.run_frozen_eval \
        --base Qwen/Qwen2.5-7B-Instruct \
        --data data/processed/YAGO3-10/test.jsonl \
        --train-data data/processed/YAGO3-10/train.jsonl \
        --raw-train data/raw/YAGO3-10/train.txt \
        --variant zero_shot \
        --out results/frozen_zero_shot_scores.jsonl \
        --batch-size 8 --limit 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.prompting.variants import (  # noqa: E402
    build_zero_shot, build_few_shot, build_evidence, build_cot,
    pick_fewshot_exemplars, neighbor_evidence,
)

LABELS = ["True", "False", "Unknown"]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser(description="Frozen-model prompt-variant evaluation.")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True, help="test.jsonl to score")
    ap.add_argument("--train-data", default=None, help="train.jsonl (needed for --variant few_shot)")
    ap.add_argument("--raw-train", default=None, help="raw train.txt (needed for --variant evidence)")
    ap.add_argument("--variant", choices=["zero_shot", "few_shot", "evidence", "cot"], default="zero_shot")
    ap.add_argument("--k-shot", type=int, default=1, help="few-shot: exemplars per class")
    ap.add_argument("--max-facts", type=int, default=3, help="evidence: max neighbor facts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--cot-max-new-tokens", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None, help="debug: score only first N rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. This must run on a GPU (Kaggle T4).")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f">> base model (frozen, no adapter): {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label_ids = [tokenizer.encode(lab, add_special_tokens=False)[0] for lab in LABELS]
    print(f">> label token ids: {dict(zip(LABELS, label_ids))}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    rows = load_rows(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]
    print(f">> scoring {len(rows)} examples with variant={args.variant}")

    # variant-specific side data
    exemplars = None
    kg = None
    if args.variant == "few_shot":
        if not args.train_data:
            raise SystemExit("--variant few_shot requires --train-data")
        train_rows = load_rows(Path(args.train_data))
        exemplars = pick_fewshot_exemplars(train_rows, k_per_class=args.k_shot, seed=args.seed)
        print(f">> few-shot exemplars: {exemplars}")
    if args.variant == "evidence":
        if not args.raw_train:
            raise SystemExit("--variant evidence requires --raw-train")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.data.build_benchmark import KGStats, load_triples
        kg = KGStats(load_triples(Path(args.raw_train)))

    def build_prompt(row):
        h, r, t = row["head"], row["relation"], row["tail"]
        if args.variant == "zero_shot":
            return build_zero_shot(h, r, t)
        if args.variant == "few_shot":
            return build_few_shot(h, r, t, exemplars)
        if args.variant == "evidence":
            nb = neighbor_evidence(h, r, kg, max_facts=args.max_facts)
            return build_evidence(h, r, t, nb)
        if args.variant == "cot":
            return build_cot(h, r, t)

    id_tensor = torch.tensor(label_ids, device=model.device)
    out_rows, n_correct = [], 0

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [build_prompt(r) for r in batch]

            if args.variant == "cot":
                # stage 1: generate a short reasoning trace
                gen_texts = [
                    tokenizer.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False)
                    for p in prompts
                ]
                enc = tokenizer(gen_texts, return_tensors="pt", padding=True, truncation=True,
                                 max_length=args.max_seq_len, add_special_tokens=False).to(model.device)
                gen_out = model.generate(
                    **enc, max_new_tokens=args.cot_max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                reasoning = tokenizer.batch_decode(gen_out[:, enc["input_ids"].shape[1]:],
                                                    skip_special_tokens=True)
                # stage 2: force "Final answer:" as the next tokens, read logits there
                final_texts = [gt + r.rsplit("Final answer:", 1)[0] + "Final answer:"
                               for gt, r in zip(gen_texts, reasoning)]
                enc2 = tokenizer(final_texts, return_tensors="pt", padding=True, truncation=True,
                                  max_length=args.max_seq_len + args.cot_max_new_tokens,
                                  add_special_tokens=False).to(model.device)
                logits = model(**enc2).logits[:, -1, :]
            else:
                texts = [
                    tokenizer.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False)
                    for p in prompts
                ]
                enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                 max_length=args.max_seq_len, add_special_tokens=False).to(model.device)
                logits = model(**enc).logits[:, -1, :]

            label_logits = logits[:, id_tensor]
            probs = F.softmax(label_logits.float(), dim=-1)

            for i, r in enumerate(batch):
                p = probs[i].tolist()
                pred_idx = int(probs[i].argmax())
                pred_label = LABELS[pred_idx]
                correct = pred_label == r["label"]
                n_correct += int(correct)
                out_rows.append({
                    "head": r["head"], "relation": r["relation"], "tail": r["tail"],
                    "label": r["label"], "predicted_label": pred_label,
                    **{f"p_{lab.lower()}": round(p[j], 6) for j, lab in enumerate(LABELS)},
                    "confidence": round(p[pred_idx], 6), "correct": correct,
                    "variant": args.variant,
                    "gen_strategy": r.get("gen_strategy"), "unknown_type": r.get("unknown_type"),
                    "rare_head": r.get("rare_head"),
                })

            done = start + len(batch)
            if done % (args.batch_size * 5) == 0 or done == len(rows):
                print(f"   {done}/{len(rows)}  running acc={n_correct/done:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n>> [{args.variant}] overall accuracy: {n_correct/len(rows):.4f}")
    print(f">> wrote {out_path}")


if __name__ == "__main__":
    main()
