"""Step 3 - P(True) discriminative scoring.

Implements the generation-discrimination gap that SECL (arXiv 2604.09624) and
Kadavath et al. exploit: instead of trusting a free-text generation, read the
model's raw logits at the answer position and take the probability mass it
places on the True / False / Unknown tokens. Renormalizing over just those
three tokens (NormPTrue in SECL's terms) gives a discriminative distribution
that later becomes the input to every uncertainty layer (Steps 4-5), including
the Dempster-Shafer fusion (mass on {True}, {False}, Omega).

This is a single forward pass per example (no sampling), so it is fast even at
7B: batch the prompts, left-pad, and read logits[:, -1, :] restricted to the
three label token ids from label_info.json (written by Step 2).

Run (Kaggle, GPU)
------------------
    python -m src.scoring.ptrue \
        --adapter /kaggle/working/models/qwen-yago-3way \
        --data data/processed/YAGO3-10/test.jsonl \
        --out results/test_scores.jsonl \
        --batch-size 8

Output (JSONL, one row per triple)
-----------------------------------
    head, relation, tail, label (gold), predicted_label,
    p_true, p_false, p_unknown (renormalized over the label set, sum to 1),
    confidence (= prob of predicted_label), correct,
    in_label_space (whether the model's own top-1 token over the FULL
        vocabulary was one of the three label tokens - a sanity/OOD check),
    gen_strategy, unknown_type, rare_head  (passed through for Step 4 breakdowns)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_label_info(adapter_dir: Path, variant: str) -> dict:
    with open(adapter_dir / "label_info.json", encoding="utf-8") as f:
        info = json.load(f)
    key = "first_token_id" if variant == "first" else "space_first_token_id"
    ids = info[key]
    labels = info["labels"]
    return {"labels": labels, "ids": [ids[lab] for lab in labels], "id_map": ids}


def main():
    ap = argparse.ArgumentParser(description="P(True) discriminative scoring (Step 3).")
    ap.add_argument("--adapter", required=True, help="path to the QLoRA adapter dir from Step 2")
    ap.add_argument("--base", default=None, help="override base model (default: read from label_info.json)")
    ap.add_argument("--data", required=True, help="jsonl file to score (e.g. test.jsonl)")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--token-variant", choices=["first", "space"], default="first",
                    help="which label token ids to read (Qwen's chat template puts the "
                         "answer right after a newline, so 'first' -no leading space- "
                         "is almost always correct; 'space' is a fallback).")
    ap.add_argument("--limit", type=int, default=None, help="debug: score only first N rows")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Step 3 must run on a GPU (Kaggle T4).")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_dir = Path(args.adapter)
    label_info = load_label_info(adapter_dir, args.token_variant)
    labels, label_ids = label_info["labels"], label_info["ids"]
    print(f">> labels: {labels} | token ids ({args.token_variant}): {label_ids}")

    base = args.base
    if base is None:
        with open(adapter_dir / "label_info.json", encoding="utf-8") as f:
            base = json.load(f)["base_model"]
    print(f">> base model: {base}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.padding_side = "left"          # so the answer position is always index -1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    rows = load_rows(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]
    print(f">> scoring {len(rows)} examples")

    id_tensor = torch.tensor(label_ids, device=model.device)
    out_rows = []
    n_correct = 0
    per_class = {lab: {"n": 0, "correct": 0} for lab in labels}

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    r["messages"], add_generation_prompt=True, tokenize=False,
                )
                for r in batch
            ]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_seq_len, add_special_tokens=False,
            ).to(model.device)

            logits = model(**enc).logits[:, -1, :]              # (B, vocab)
            label_logits = logits[:, id_tensor]                  # (B, 3)
            probs = F.softmax(label_logits.float(), dim=-1)       # renormalized over the 3 labels
            top_full = logits.argmax(dim=-1)                     # top-1 over the WHOLE vocab

            for i, r in enumerate(batch):
                p = probs[i].tolist()
                pred_idx = int(probs[i].argmax())
                pred_label = labels[pred_idx]
                gold = r["label"]
                correct = pred_label == gold
                n_correct += int(correct)
                per_class[gold]["n"] += 1
                per_class[gold]["correct"] += int(correct)

                out_rows.append({
                    "head": r["head"], "relation": r["relation"], "tail": r["tail"],
                    "label": gold, "predicted_label": pred_label,
                    **{f"p_{lab.lower()}": round(p[j], 6) for j, lab in enumerate(labels)},
                    "confidence": round(p[pred_idx], 6),
                    "correct": correct,
                    "in_label_space": bool(int(top_full[i]) in label_ids),
                    "gen_strategy": r.get("gen_strategy"),
                    "unknown_type": r.get("unknown_type"),
                    "rare_head": r.get("rare_head"),
                })

            done = start + len(batch)
            if done % (args.batch_size * 10) == 0 or done == len(rows):
                print(f"   {done}/{len(rows)}  running acc={n_correct/done:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n>> overall accuracy: {n_correct/len(rows):.4f}")
    for lab in labels:
        c = per_class[lab]
        acc = c["correct"] / c["n"] if c["n"] else float("nan")
        print(f"   {lab:8s} n={c['n']:5d}  acc={acc:.4f}")
    oos = sum(1 for r in out_rows if not r["in_label_space"])
    print(f">> top-1 token outside {{True,False,Unknown}} for {oos}/{len(rows)} examples "
          f"({oos/len(rows):.2%}) - should be near 0 if fine-tuning worked.")
    print(f">> wrote {out_path}")


if __name__ == "__main__":
    main()
