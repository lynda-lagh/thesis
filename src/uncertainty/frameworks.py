"""Step 5a - Four uncertainty representations over one model's output.

Takes the model's own discriminative distribution p = (p_true, p_false,
p_unknown) - written by Step 3 (src.scoring.ptrue) - and re-expresses it in
four formalisms from the Reasoning-under-Uncertainty course:

  * probabilistic   : p itself (the reference / baseline).
  * evidential (DST): a basic probability assignment (mass function) on the
                       frame Omega = {True, False}, with focal elements
                       {True}, {False}, and Omega itself. m(Omega) is the
                       explicit ignorance mass - the thing plain probability
                       cannot represent (Section 1 of the thesis motivation).
  * possibilistic   : the Dubois-Prade probability-to-possibility transform,
                       which provably gives possibility >= probability for
                       every event - a principled "upper envelope" of belief.
  * fuzzy           : a 5-point linguistic scale (False/Unlikely/Unknown/
                       Likely/True) via a simple truth-score membership.

HONEST NOTE on the evidential layer with a SINGLE source: the simple BPA
m({True})=p_true, m({False})=p_false, m(Omega)=p_unknown satisfies
Bel({True})+Bel({False})+m(Omega) = p_true+p_false+p_unknown = 1, so its
*decision* (argmax) is numerically identical to the plain probabilistic
decision - the extra value here is in the Pl/Bel interval width as a reported
uncertainty measure, and in enabling genuine Dempster's-rule FUSION with a
second, independent source (see src.uncertainty.dempster_fusion), where the
combined mass really does diverge from any single source. Possibilistic and
fuzzy, by contrast, differ from probability even with a single source: the
Dubois-Prade transform is a different function of p, not a relabeling of it.

Usage
-----
    python -m src.uncertainty.frameworks --scores results/test_scores.jsonl \
        --out results/framework_comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["True", "False", "Unknown"]


# --------------------------------------------------------------------------- #
# 1. Evidential / Dempster-Shafer  (single source; see docstring caveat)
# --------------------------------------------------------------------------- #

def to_evidential(p_true: float, p_false: float, p_unknown: float) -> dict:
    m_true, m_false, m_omega = p_true, p_false, p_unknown
    bel_true, bel_false = m_true, m_false
    pl_true, pl_false = m_true + m_omega, m_false + m_omega
    # pignistic transform: split ignorance mass evenly (Smets' BetP)
    betp_true = m_true + m_omega / 2
    betp_false = m_false + m_omega / 2
    return {
        "m_true": m_true, "m_false": m_false, "m_omega": m_omega,
        "bel_true": bel_true, "bel_false": bel_false,
        "pl_true": pl_true, "pl_false": pl_false,
        "betp_true": betp_true, "betp_false": betp_false,
        "uncertainty_interval": m_omega,        # Pl(True)-Bel(True) == m_omega here
        # decision scores over the 3-way label set: belief in the frame ITSELF
        # (m_omega) stands for "Unknown" as a first-class outcome.
        "score_true": bel_true, "score_false": bel_false, "score_unknown": m_omega,
    }


# --------------------------------------------------------------------------- #
# 2. Possibilistic (Zadeh 1978; Dubois & Prade 1987)
# --------------------------------------------------------------------------- #

def to_possibilistic(p_true: float, p_false: float, p_unknown: float) -> dict:
    """Dubois-Prade probability-to-possibility transform: sort probabilities
    descending p_(1) >= p_(2) >= ... ; then pi_(i) = sum_{j>=i} p_(j) - the
    SUFFIX sum, i.e. the mass of all events at most as likely as rank i. This
    gives pi_(1) = 1 (the most probable event is fully possible - required by
    possibility-measure normalization) and pi decreasing for less-likely
    ranks, down to pi_(n) = p_(n) for the least likely one. Guarantees
    pi(A) >= P(A) for every event A (a genuine, provably different, upper
    envelope of belief - not merely a relabeling of the probabilities)."""
    items = list(zip(LABELS, [p_true, p_false, p_unknown]))
    items.sort(key=lambda kv: -kv[1])
    pi = {}
    running = 0.0
    for lab, val in reversed(items):     # accumulate from the LEAST likely upward
        running += val
        pi[lab] = running
    pi_true, pi_false, pi_unknown = pi["True"], pi["False"], pi["Unknown"]

    # necessity N(A) = 1 - pi(not A)
    def necessity(lab):
        pi_not = max(v for l, v in pi.items() if l != lab)
        return 1 - pi_not

    n_true, n_false, n_unknown = necessity("True"), necessity("False"), necessity("Unknown")
    # HONEST NOTE: for a consonant (nested) transform derived from a SINGLE
    # source, necessity is 0 for every rank except the top one, where it
    # equals that label's own probability (N_(1) = p_(1)). So argmax(N) is
    # numerically the same decision as argmax(p) here - same single-source
    # degeneracy as the evidential layer. The reported (N, pi) INTERVAL width
    # (pi - N) is the genuinely new quantity: it is the possibilistic
    # counterpart of the DST Pl-Bel gap, and it too only diverges from a
    # single source's raw uncertainty once fused with a second source.
    return {
        "pi_true": pi_true, "pi_false": pi_false, "pi_unknown": pi_unknown,
        "n_true": n_true, "n_false": n_false, "n_unknown": n_unknown,
        "score_true": n_true, "score_false": n_false, "score_unknown": n_unknown,
    }


# --------------------------------------------------------------------------- #
# 3. Fuzzy (Zadeh 1965) - graded linguistic scale
# --------------------------------------------------------------------------- #

FUZZY_SCALE = ["False", "Unlikely", "Unknown", "Likely", "True"]


def to_fuzzy(p_true: float, p_false: float, p_unknown: float) -> dict:
    """Truth score s = p_true - p_false in [-1,1]; p_unknown is the ignorance
    membership. Triangular membership functions over 5 linguistic anchors
    centered at s in {-1,-0.5,0,0.5,1}, width scaled by (1-p_unknown) so high
    ignorance flattens the truth-scale memberships toward 'Unknown'."""
    s = p_true - p_false
    centers = {"False": -1.0, "Unlikely": -0.5, "Unknown": 0.0, "Likely": 0.5, "True": 1.0}
    width = max(0.25, 1.0 - p_unknown)   # more ignorance -> broader/flatter membership
    mu = {}
    for lab, c in centers.items():
        mu[lab] = max(0.0, 1.0 - abs(s - c) / width)
    # fold ignorance directly into the Unknown anchor (it IS the fuzzy set of
    # "cannot be verified" statements)
    mu["Unknown"] = max(mu["Unknown"], p_unknown)
    total = sum(mu.values()) or 1.0
    mu_norm = {k: v / total for k, v in mu.items()}
    linguistic_label = max(mu_norm, key=mu_norm.get)
    return {
        "truth_score": s, **{f"mu_{k.lower()}": v for k, v in mu_norm.items()},
        "linguistic_label": linguistic_label,
        # collapse the 5-point scale back to the 3-way task for comparison:
        # True/Likely -> True, False/Unlikely -> False, Unknown -> Unknown
        "score_true": mu_norm["True"] + mu_norm["Likely"],
        "score_false": mu_norm["False"] + mu_norm["Unlikely"],
        "score_unknown": mu_norm["Unknown"],
    }


# --------------------------------------------------------------------------- #
# Reinterpret rows through a framework, then reuse calibration.py's metrics
# --------------------------------------------------------------------------- #

FRAMEWORKS = {
    "probabilistic": lambda pt, pf, pu: {"score_true": pt, "score_false": pf, "score_unknown": pu},
    "evidential": to_evidential,
    "possibilistic": to_possibilistic,
    "fuzzy": to_fuzzy,
}


def apply_framework(rows: list[dict], framework: str) -> list[dict]:
    fn = FRAMEWORKS[framework]
    out = []
    for r in rows:
        rep = fn(r["p_true"], r["p_false"], r["p_unknown"])
        scores = {"True": rep["score_true"], "False": rep["score_false"], "Unknown": rep["score_unknown"]}
        total = sum(scores.values()) or 1.0
        norm = {k: v / total for k, v in scores.items()}   # renormalize for Brier/ECE comparability
        pred = max(norm, key=norm.get)
        out.append({
            **{k: r[k] for k in ("head", "relation", "tail", "label",
                                  "gen_strategy", "unknown_type", "rare_head")
               if k in r},
            "p_true": norm["True"], "p_false": norm["False"], "p_unknown": norm["Unknown"],
            "predicted_label": pred,
            "confidence": norm[pred],
            "correct": pred == r["label"],
            "framework_detail": rep,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare uncertainty frameworks (Step 5a).")
    ap.add_argument("--scores", required=True, help="test_scores.jsonl from Step 3")
    ap.add_argument("--out", default="results/framework_comparison.json")
    args = ap.parse_args()

    # reuse Step 4's metric functions instead of duplicating them
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.eval.calibration import (
        accuracy_macro_f1, confusion_matrix, ece, brier_score, nll,
        risk_coverage, abstention_quality,
    )

    rows = [json.loads(l) for l in open(args.scores, encoding="utf-8")]
    print(f">> loaded {len(rows)} scored examples\n")

    comparison = {}
    for name in FRAMEWORKS:
        fr = apply_framework(rows, name)
        comparison[name] = {
            "reference": accuracy_macro_f1(fr),
            "confusion_matrix": confusion_matrix(fr),
            "calibration": {"ece": ece(fr)["ece"], "brier": round(brier_score(fr), 6), "nll": round(nll(fr), 6)},
            "selective_prediction": {k: v for k, v in risk_coverage(fr).items() if k != "curve"},
            "abstention": abstention_quality(fr),
        }
        ref, cal, abst = comparison[name]["reference"], comparison[name]["calibration"], comparison[name]["abstention"]
        print(f"== {name:14s} == acc={ref['accuracy']:.4f}  ECE={cal['ece']:.4f}  "
              f"Brier={cal['brier']:.4f}  abst-AUROC={abst['abstention_auroc']:.4f}  "
              f"correct-abstain={abst['correct_abstention_rate']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n>> wrote {out_path}")


if __name__ == "__main__":
    main()
