"""Step 4 - Calibration and selective-prediction evaluation.

Consumes the P(True) scores written by Step 3 (results/test_scores.jsonl) and
reports the metric family the KGC field almost never uses: not "is the answer
right" but "does the model's confidence deserve to be trusted, and does it
abstain correctly on genuinely-unknown triples."

Pure numpy/stdlib - runs on CPU, no GPU or torch needed, so it can be verified
locally and run anywhere.

Metrics
-------
  * accuracy, macro-F1, confusion matrix               (reference / sanity)
  * ECE (Expected Calibration Error, equal-width bins)
  * multiclass Brier score, NLL
  * risk-coverage curve + AURC (area under risk-coverage)
  * abstention-AUROC: does p_unknown (or 1 - confidence) separate
    gold-Unknown triples from gold-True/False triples?
  * breakdown by unknown_type / rare_head (Step 1 metadata)
  * KG-LLM forced-choice baseline simulation: renormalize p_true/(p_true+p_false),
    ignoring p_unknown entirely, forcing an answer even on gold=Unknown triples -
    quantifies exactly how "confidently wrong" a closed-world binary model is.

Run
---
    python -m src.eval.calibration --scores results/test_scores.jsonl \
        --out results/calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

LABELS = ["True", "False", "Unknown"]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def probs_of(row: dict) -> list[float]:
    return [row["p_true"], row["p_false"], row["p_unknown"]]


# --------------------------------------------------------------------------- #
# Reference metrics
# --------------------------------------------------------------------------- #

def confusion_matrix(rows: list[dict]) -> dict:
    cm = {g: {p: 0 for p in LABELS} for g in LABELS}
    for r in rows:
        cm[r["label"]][r["predicted_label"]] += 1
    return cm


def accuracy_macro_f1(rows: list[dict]) -> dict:
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else float("nan")
    f1s = []
    for lab in LABELS:
        tp = sum(1 for r in rows if r["label"] == lab and r["predicted_label"] == lab)
        fp = sum(1 for r in rows if r["label"] != lab and r["predicted_label"] == lab)
        fn = sum(1 for r in rows if r["label"] == lab and r["predicted_label"] != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return {"accuracy": acc, "macro_f1": sum(f1s) / len(f1s), "per_class_f1": dict(zip(LABELS, f1s))}


# --------------------------------------------------------------------------- #
# Calibration: ECE, Brier, NLL
# --------------------------------------------------------------------------- #

def ece(rows: list[dict], n_bins: int = 10) -> dict:
    """Expected Calibration Error over the model's own top confidence."""
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        conf = r["confidence"]
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append(r)
    n = len(rows)
    total = 0.0
    bin_report = []
    for i, b in enumerate(bins):
        if not b:
            bin_report.append({"bin": i, "n": 0, "acc": None, "avg_conf": None})
            continue
        acc = sum(r["correct"] for r in b) / len(b)
        avg_conf = sum(r["confidence"] for r in b) / len(b)
        total += (len(b) / n) * abs(acc - avg_conf)
        bin_report.append({"bin": i, "n": len(b), "acc": round(acc, 4), "avg_conf": round(avg_conf, 4)})
    return {"ece": round(total, 6), "bins": bin_report}


def brier_score(rows: list[dict]) -> float:
    """Multiclass Brier: mean squared distance between predicted probs and one-hot gold."""
    total = 0.0
    for r in rows:
        p = probs_of(r)
        y = [1.0 if lab == r["label"] else 0.0 for lab in LABELS]
        total += sum((pi - yi) ** 2 for pi, yi in zip(p, y))
    return total / len(rows) if rows else float("nan")


def nll(rows: list[dict], eps: float = 1e-12) -> float:
    total = 0.0
    for r in rows:
        idx = LABELS.index(r["label"])
        p_gold = max(probs_of(r)[idx], eps)
        total -= math.log(p_gold)
    return total / len(rows) if rows else float("nan")


# --------------------------------------------------------------------------- #
# Selective prediction: risk-coverage curve + AURC
# --------------------------------------------------------------------------- #

def risk_coverage(rows: list[dict]) -> dict:
    """Sort by confidence descending; at each coverage level c, risk(c) = error
    rate among the c*n most-confident predictions (the rest are 'abstained').
    AURC = area under risk(coverage), trapezoidal, over coverage in (0,1]."""
    ordered = sorted(rows, key=lambda r: -r["confidence"])
    n = len(ordered)
    if n == 0:
        return {"aurc": float("nan"), "curve": []}
    curve = []
    errors = 0
    for i, r in enumerate(ordered, start=1):
        if not r["correct"]:
            errors += 1
        coverage = i / n
        risk = errors / i
        curve.append({"coverage": round(coverage, 4), "risk": round(risk, 6)})
    # trapezoidal AURC over the curve (coverage from 1/n to 1)
    aurc = 0.0
    for a, b in zip(curve, curve[1:]):
        dx = b["coverage"] - a["coverage"]
        aurc += dx * (a["risk"] + b["risk"]) / 2
    # selective accuracy at a few standard coverage checkpoints
    checkpoints = {}
    for target in (0.5, 0.8, 1.0):
        closest = min(curve, key=lambda c: abs(c["coverage"] - target))
        checkpoints[f"risk@coverage={target}"] = closest["risk"]
    return {"aurc": round(aurc, 6), "checkpoints": checkpoints, "curve": curve}


# --------------------------------------------------------------------------- #
# Abstention quality: AUROC of "gold is Unknown" detection
# --------------------------------------------------------------------------- #

def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC (Mann-Whitney U), no sklearn dependency.
    labels: 1 = positive class, 0 = negative. scores: higher = more positive."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0   # 1-indexed average rank for ties
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    n_pos = sum(1 for _, l in pairs if l == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = sum(r for (_, l), r in zip(pairs, ranks) if l == 1)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def abstention_quality(rows: list[dict]) -> dict:
    """Can p_unknown separate gold-Unknown triples from gold-True/False triples?"""
    scores = [r["p_unknown"] for r in rows]
    gold_unknown = [1 if r["label"] == "Unknown" else 0 for r in rows]
    auc = auroc(scores, gold_unknown)

    # correct-abstention rate: of the truly-Unknown triples, how many did the
    # model actually predict Unknown for?
    unk_rows = [r for r in rows if r["label"] == "Unknown"]
    correct_abstain = sum(1 for r in unk_rows if r["predicted_label"] == "Unknown")
    car = correct_abstain / len(unk_rows) if unk_rows else float("nan")

    # false-abstention rate: of the True/False (answerable) triples, how many
    # did the model wrongly abstain on?
    ans_rows = [r for r in rows if r["label"] != "Unknown"]
    false_abstain = sum(1 for r in ans_rows if r["predicted_label"] == "Unknown")
    far = false_abstain / len(ans_rows) if ans_rows else float("nan")

    return {
        "abstention_auroc": round(auc, 6),
        "correct_abstention_rate": round(car, 6),
        "false_abstention_rate": round(far, 6),
    }


def breakdown_by_subtype(rows: list[dict]) -> dict:
    """Abstention quality split by WHY a triple is Unknown (Step 1 metadata)."""
    unk_rows = [r for r in rows if r["label"] == "Unknown"]
    out = {}
    for key in ("unknown_type", "rare_head"):
        groups = defaultdict(list)
        for r in unk_rows:
            groups[r.get(key)].append(r)
        out[key] = {}
        for g, grows in groups.items():
            car = sum(1 for r in grows if r["predicted_label"] == "Unknown") / len(grows)
            avg_conf = sum(r["confidence"] for r in grows) / len(grows)
            out[key][str(g)] = {
                "n": len(grows),
                "correct_abstention_rate": round(car, 4),
                "avg_confidence": round(avg_conf, 4),
            }
    return out


# --------------------------------------------------------------------------- #
# KG-LLM forced-choice baseline simulation
# --------------------------------------------------------------------------- #

def kgllm_forced_choice_baseline(rows: list[dict]) -> dict:
    """Simulate a closed-world binary model (KG-LLM's paradigm): renormalize
    p_true/(p_true+p_false), ignore p_unknown, and force True/False even on
    gold=Unknown triples. Reports how confidently wrong it is on those -
    the concrete number behind "KG-LLM guesses confidently on what it can't
    know."""
    forced = []
    for r in rows:
        pt, pf = r["p_true"], r["p_false"]
        denom = pt + pf if (pt + pf) > 0 else 1e-9
        p_true_forced = pt / denom
        pred = "True" if p_true_forced >= 0.5 else "False"
        conf = max(p_true_forced, 1 - p_true_forced)
        forced.append({**r, "forced_pred": pred, "forced_conf": conf})

    unk = [r for r in forced if r["label"] == "Unknown"]
    ansable = [r for r in forced if r["label"] != "Unknown"]

    avg_conf_on_unknown = sum(r["forced_conf"] for r in unk) / len(unk) if unk else float("nan")
    acc_on_answerable = (
        sum(1 for r in ansable if r["forced_pred"] == r["label"]) / len(ansable) if ansable else float("nan")
    )
    return {
        "note": "KG-LLM-style binary model forced to answer True/False on every triple, including Unknown ones.",
        "avg_confidence_on_unknowable_triples": round(avg_conf_on_unknown, 4),
        "accuracy_on_answerable_triples": round(acc_on_answerable, 4),
        "n_unknown_forced": len(unk),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Calibration + selective-risk evaluation (Step 4).")
    ap.add_argument("--scores", required=True, help="jsonl written by Step 3 (src.scoring.ptrue)")
    ap.add_argument("--out", default="results/calibration.json")
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    rows = load_rows(Path(args.scores))
    print(f">> loaded {len(rows)} scored examples")

    report = {
        "n": len(rows),
        "reference": accuracy_macro_f1(rows),
        "confusion_matrix": confusion_matrix(rows),
        "calibration": {
            "ece": ece(rows, args.n_bins),
            "brier": round(brier_score(rows), 6),
            "nll": round(nll(rows), 6),
        },
        "selective_prediction": risk_coverage(rows),
        "abstention": abstention_quality(rows),
        "abstention_breakdown": breakdown_by_subtype(rows),
        "kgllm_forced_choice_baseline": kgllm_forced_choice_baseline(rows),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    ref = report["reference"]
    cal = report["calibration"]
    sel = report["selective_prediction"]
    abst = report["abstention"]
    base = report["kgllm_forced_choice_baseline"]

    print(f"\n== reference ==\naccuracy={ref['accuracy']:.4f}  macro_f1={ref['macro_f1']:.4f}")
    print(f"per-class F1: {ref['per_class_f1']}")
    print(f"\n== calibration ==\nECE={cal['ece']['ece']:.4f}  Brier={cal['brier']:.4f}  NLL={cal['nll']:.4f}")
    print(f"\n== selective prediction ==\nAURC={sel['aurc']:.4f}  checkpoints={sel['checkpoints']}")
    print(f"\n== abstention quality ==\n{abst}")
    print(f"\n== KG-LLM forced-choice baseline (closed-world) ==\n{base}")
    print(f"\n>> full report written to {out_path}")


if __name__ == "__main__":
    main()
