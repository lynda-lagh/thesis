"""Step 3 (STUB) - P(True) discriminative scoring.

Implements the generation-discrimination gap (Kadavath et al.; SECL): instead of
trusting the model's verbalized answer, read the token probability mass on the
True / False / Unknown label tokens and derive a calibrated confidence.

Planned interface
-----------------
    python -m src.scoring.ptrue --model models/qwen-yago-3way \
        --data data/processed/YAGO3-10/test.jsonl --out preds/test_scores.jsonl

Outputs per example: p_true, p_false, p_unknown, predicted_label, confidence.
"""

def main():
    raise NotImplementedError("Step 3 - implement after Step 2.")


if __name__ == "__main__":
    main()
