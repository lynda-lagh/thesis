"""Step 4 (STUB) - Calibration + selective-prediction evaluation.

The metric family the KGC field almost never reports:
  * ECE / adaptive ECE, Brier score, negative log-likelihood
  * risk-coverage curve + AURC (area under risk-coverage)
  * abstention quality: correct-abstain rate on the Unknown class, AUROC of
    confidence vs correctness
  * standard accuracy / macro-F1 for reference (comparable to KG-LLM)

Planned interface
-----------------
    python -m src.eval.calibration --scores preds/test_scores.jsonl \
        --out results/calibration.json --plots results/plots/
"""

def main():
    raise NotImplementedError("Step 4 - implement after Step 3.")


if __name__ == "__main__":
    main()
