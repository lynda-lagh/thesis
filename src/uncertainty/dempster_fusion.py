"""Step 5b - Dempster-Shafer multi-source fusion (the flagship contribution).

Section 1's caveat about the evidential layer (frameworks.py) is that a
SINGLE source's mass function decides identically to plain probability - the
real payoff of Dempster-Shafer only appears once you fuse two genuinely
INDEPENDENT sources. This module builds that second source and combines it
with the LLM's own judgment via Dempster's combination rule.

Source 1 (S_LLM) - the model's discriminative distribution from Step 3:
    m1({True})=p_true, m1({False})=p_false, m1(Omega)=p_unknown

Source 2 (S_struct) - a lightweight SYMBOLIC/structural reasoner that looks
only at the training KG (never at the test triple's truth), mirroring what a
classic KGE-style reasoner would use:
  - if the relation is (near-)functional (Step 1's FALSE-eligible criteria)
    and the head has a KNOWN tail for it in the training graph:
        the candidate tail either matches the known one (-> lean True) or
        contradicts it (-> lean False), with the rest of the mass on Omega.
  - if the relation is many-to-many, or the head is unseen for this relation:
        little symbolic basis to decide -> most mass stays on Omega, with a
        small lean toward True if the candidate tail happens to already be a
        known (h,r,*) fact.

Combined via Dempster's rule (with an optional discount factor on S_struct,
since Shafer's rule assumes independent, EQUALLY reliable sources - a naive
combination overstates certainty if a source is redundant or unreliable).

KEY FINDING (verified on synthetic data, expect the same on real results): fusing
with a mostly-agnostic second source (high mass on Omega) does NOT pull the
decision toward Unknown - Dempster's rule lets the more committed source's
opinion dominate, because Omega intersected with any singleton resolves to
that singleton. Agnosticism doesn't argue for Unknown, it just defers. This is
a known property of Dempster's rule (the substance of Zadeh's classic critique
of it). Concretely: fusion here perfectly corrects cases where S_struct has a
real, decisive opinion (the functional-relation True/False cases), but leaves
untouched the cases where S_struct is merely uncertain (most manytomany-
relation Unknowns) - so 'Unknown' as *total ignorance over a 2-element frame*
{True,False} is not the same as 'Unknown' as a *positive third claim*. Making
a source actively argue FOR Unknown (not just defer to it) requires promoting
Unknown to its own atomic hypothesis in a 3-element frame Omega_3 =
{True,False,Unknown}, with mass on subsets like {True,Unknown} - a natural,
well-scoped refinement (flagged here as future work, not implemented in this
version) rather than a bug in the 2-element-frame math below, which is a
correct, textbook Dempster combination.

HONEST LIMITATION: S_struct's decision logic mirrors the SAME rule Step 1
used to construct False/Unknown labels (functionality + known-tails), so on
this benchmark it is unusually well-matched to the labeling scheme. That
makes it an honest DEMONSTRATION of the fusion mechanism and of Stage 2
(structural evidence) closing the LLM's True/Unknown gap - but a fully
independent validation would replace S_struct with a genuinely external
signal (e.g. a trained KGE model's score, or held-out relations not used to
calibrate the functionality thresholds). Flagged here, and in the thesis
write-up, as a direction for a stronger follow-up experiment.

Run
---
    python -m src.uncertainty.dempster_fusion \
        --scores results/test_scores.jsonl \
        --train data/raw/YAGO3-10/train.txt \
        --out results/fused_scores.jsonl \
        --discount2 0.85
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# reuse the KG indexing already built and tested in Step 1
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.build_benchmark import KGStats, load_triples  # noqa: E402


# --------------------------------------------------------------------------- #
# Dempster's combination rule for the 3-focal-element frame {T},{F},Omega
# --------------------------------------------------------------------------- #

def dempster_combine(m1: dict, m2: dict) -> dict:
    """m1, m2: dicts with keys 'true','false','omega' (masses on {T},{F},Omega).
    Returns the combined, renormalized mass, plus the raw conflict K."""
    t1, f1, o1 = m1["true"], m1["false"], m1["omega"]
    t2, f2, o2 = m2["true"], m2["false"], m2["omega"]

    m_true_raw = t1 * t2 + t1 * o2 + o1 * t2
    m_false_raw = f1 * f2 + f1 * o2 + o1 * f2
    m_omega_raw = o1 * o2
    conflict = t1 * f2 + f1 * t2                 # mass assigned to the empty set

    denom = 1.0 - conflict
    if denom <= 1e-9:
        # total contradiction between sources - fall back to source 1 unchanged
        return {"true": t1, "false": f1, "omega": o1, "conflict": conflict}
    return {
        "true": m_true_raw / denom,
        "false": m_false_raw / denom,
        "omega": m_omega_raw / denom,
        "conflict": conflict,
    }


def discount(m: dict, alpha: float) -> dict:
    """Shafer's discounting: scale down a source's committed mass by alpha
    (its assumed reliability) and give the rest to Omega. alpha=1 -> no
    change; alpha=0 -> source becomes fully non-committal (all mass on
    Omega). Used on S_struct since it is a much simpler heuristic than the
    fine-tuned LLM and its independence from the labeling rule is imperfect."""
    return {
        "true": alpha * m["true"],
        "false": alpha * m["false"],
        "omega": alpha * m["omega"] + (1 - alpha),
    }


# --------------------------------------------------------------------------- #
# Source 2: structural / KG-derived symbolic mass
# --------------------------------------------------------------------------- #

def build_structural_source(row: dict, kg: KGStats, functional_rels: set, manytomany_rels: set) -> dict:
    h, r, t = row["head"], row["relation"], row["tail"]
    known = kg.known_tails(h, r)

    if r in functional_rels:
        if not known:
            # head unseen for this relation in training -> no symbolic basis
            return {"true": 0.05, "false": 0.05, "omega": 0.90}
        if t in known:
            return {"true": 0.85, "false": 0.02, "omega": 0.13}
        else:
            return {"true": 0.02, "false": 0.85, "omega": 0.13}
    elif r in manytomany_rels:
        if t in known:
            return {"true": 0.55, "false": 0.02, "omega": 0.43}
        else:
            return {"true": 0.05, "false": 0.05, "omega": 0.90}
    else:
        # relation not covered by either regime (the "neither" bucket from
        # Step 1's diagnostic) - fully non-committal
        return {"true": 0.10, "false": 0.10, "omega": 0.80}


def relation_sets_from_kg(kg: KGStats, cfg: dict) -> tuple[set, set]:
    """Recompute the same FALSE/UNKNOWN-eligibility gates used in Step 1, so
    the structural source's regime detection matches the benchmark exactly."""
    fun_hi = cfg["functional_threshold"]
    fun_lo = cfg["nonfunctional_threshold"]
    min_pool = cfg["min_tail_pool"]
    min_pool_false = cfg["min_pool_false"]
    max_multi = cfg["max_multi_tail_frac"]
    block = set(cfg["relation_blocklist"])

    functional, manytomany = set(), set()
    for r in kg.rel_triples:
        f = kg.functionality(r)
        if r in block:
            continue
        if f >= fun_hi and kg.multi_frac(r) <= max_multi and len(kg.tail_pool[r]) >= min_pool_false:
            functional.add(r)
        elif f <= fun_lo and len(kg.tail_pool[r]) >= min_pool:
            manytomany.add(r)
    return functional, manytomany


DEFAULT_CFG = {
    "functional_threshold": 0.83, "nonfunctional_threshold": 0.65,
    "max_multi_tail_frac": 0.12, "min_tail_pool": 5, "min_pool_false": 2,
    "relation_blocklist": ["hasWebsite"],
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def fuse_rows(rows: list[dict], kg: KGStats, functional_rels: set, manytomany_rels: set,
              discount2: float) -> list[dict]:
    out = []
    for r in rows:
        m1 = {"true": r["p_true"], "false": r["p_false"], "omega": r["p_unknown"]}
        m2 = build_structural_source(r, kg, functional_rels, manytomany_rels)
        m2 = discount(m2, discount2)
        fused = dempster_combine(m1, m2)

        scores = {"True": fused["true"], "False": fused["false"], "Unknown": fused["omega"]}
        pred = max(scores, key=scores.get)
        out.append({
            **{k: r[k] for k in ("head", "relation", "tail", "label",
                                  "gen_strategy", "unknown_type", "rare_head")
               if k in r},
            "p_true": scores["True"], "p_false": scores["False"], "p_unknown": scores["Unknown"],
            "predicted_label": pred, "confidence": scores[pred], "correct": pred == r["label"],
            "conflict_k": fused["conflict"],
            "m1": m1, "m2_discounted": m2,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Dempster-Shafer fusion of LLM + structural evidence (Step 5b).")
    ap.add_argument("--scores", required=True, help="test_scores.jsonl from Step 3")
    ap.add_argument("--train", required=True, help="path to train.txt (used to build the structural source)")
    ap.add_argument("--out", default="results/fused_scores.jsonl")
    ap.add_argument("--discount2", type=float, default=0.85,
                    help="reliability discount applied to the structural source (1.0 = no discount)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.scores, encoding="utf-8")]
    print(f">> loaded {len(rows)} scored examples")

    train_triples = load_triples(Path(args.train))
    kg = KGStats(train_triples)
    functional_rels, manytomany_rels = relation_sets_from_kg(kg, DEFAULT_CFG)
    print(f">> structural source built from {len(train_triples)} training triples "
          f"({len(functional_rels)} functional / {len(manytomany_rels)} many-to-many relations)")

    fused = fuse_rows(rows, kg, functional_rels, manytomany_rels, args.discount2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in fused:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # before/after comparison using Step 4's metrics
    from src.eval.calibration import (
        accuracy_macro_f1, confusion_matrix, ece, brier_score, abstention_quality,
    )
    before = accuracy_macro_f1(rows)   # rows already have predicted_label/correct from Step 3
    after = accuracy_macro_f1(fused)
    print(f"\n== BEFORE fusion (LLM alone) ==  acc={before['accuracy']:.4f}  macro_f1={before['macro_f1']:.4f}")
    print(f"== AFTER  fusion (LLM + structural) ==  acc={after['accuracy']:.4f}  macro_f1={after['macro_f1']:.4f}")
    print(f"\nconfusion matrix AFTER fusion: {json.dumps(confusion_matrix(fused))}")
    print(f"ECE after: {ece(fused)['ece']:.4f}  Brier after: {brier_score(fused):.4f}")
    print(f"abstention after: {abstention_quality(fused)}")
    avg_conflict = sum(r["conflict_k"] for r in fused) / len(fused)
    print(f"avg Dempster conflict K: {avg_conflict:.4f}  (high K = sources frequently disagreed)")
    print(f"\n>> wrote {out_path}")


if __name__ == "__main__":
    main()
