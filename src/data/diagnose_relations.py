"""Relation-functionality diagnostic for the Step 1 benchmark.

Surfaces the two label-quality risks before you run experiments:

  Risk A (bad FALSE): a relation is called functional (fun >= threshold) but some
          heads actually have MORE THAN ONE tail. For those heads a corrupted tail
          is not provably false -> `multi_tail_head_frac` flags it.
  Risk B (borderline): a relation sits in the grey zone between the functional and
          many-to-many thresholds, so it feeds NEITHER class (wasted) or is
          fragile to the threshold choice.

For every relation it prints: functionality, inverse functionality, triple count,
tail-pool size, avg/max tails per head, the class it feeds, and warning flags.
Optionally dumps a few generated examples per relation for eyeballing.

Usage
-----
    python -m src.data.diagnose_relations --smoke-test
    python -m src.data.diagnose_relations --raw-dir data/raw/YAGO3-10 \
        --config config/default.yaml --samples 2 --out data/processed/YAGO3-10/relation_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .build_benchmark import KGStats, load_all, load_config, synthetic_kg
from .verbalize import verbalize_triple


def relation_report(splits, cfg):
    all_triples = list(dict.fromkeys(splits["train"] + splits["valid"] + splits["test"]))
    kg = KGStats(all_triples)

    fun_hi = cfg["functional_threshold"]
    fun_lo = cfg["nonfunctional_threshold"]
    min_pool = cfg["min_tail_pool"]
    min_pool_false = cfg.get("min_pool_false", 2)
    max_multi = cfg.get("max_multi_tail_frac", 0.12)
    blocklist = set(cfg.get("relation_blocklist", []))
    margin = 0.05  # grey-zone width for "borderline" flag

    # tails per head, per relation
    tails_per_head = defaultdict(lambda: defaultdict(set))   # r -> h -> {t}
    heads_per_tail = defaultdict(lambda: defaultdict(set))   # r -> t -> {h}
    for h, r, t in all_triples:
        tails_per_head[r][h].add(t)
        heads_per_tail[r][t].add(h)

    rows = []
    for r in sorted(kg.rel_triples):
        n = kg.rel_triples[r]
        fun = kg.functionality(r)                       # |heads| / |triples|  (tail-side)
        tph = tails_per_head[r]
        hpt = heads_per_tail[r]
        inv_fun = len(hpt) / n if n else 0.0            # head-side
        tail_pool = len(kg.tail_pool[r])

        counts = [len(v) for v in tph.values()]
        avg_tph = sum(counts) / len(counts) if counts else 0.0
        max_tph = max(counts) if counts else 0
        multi = sum(1 for c in counts if c > 1)
        multi_frac = multi / len(counts) if counts else 0.0

        # eligibility (mirrors build_benchmark)
        blocked = r in blocklist
        false_ok = (fun >= fun_hi and multi_frac <= max_multi
                    and tail_pool >= min_pool_false and not blocked)
        unknown_ok = fun <= fun_lo and tail_pool >= min_pool and not blocked
        if false_ok:
            feeds = "FALSE"
        elif unknown_ok:
            feeds = "UNKNOWN"
        else:
            feeds = "neither"

        flags = []
        if blocked:
            flags.append("BLOCKLISTED")
        # would-be functional but rejected because it is actually multi-valued
        if fun >= fun_hi and multi_frac > max_multi and not blocked:
            flags.append(f"MULTI_VALUED:{multi_frac:.0%}-heads-have->1-tail")
        if not false_ok and not unknown_ok and not blocked:
            if fun_lo < fun < fun_hi and multi_frac <= max_multi:
                flags.append("BORDERLINE:grey-zone")
            if fun >= fun_hi and tail_pool < min_pool_false:
                flags.append("SMALL_POOL")
        if false_ok and (fun_hi - margin) <= fun < (fun_hi + margin):
            flags.append("NEAR_FUNC_THRESH")
        if unknown_ok and (fun_lo - margin) < fun <= (fun_lo + margin):
            flags.append("NEAR_NONFUNC_THRESH")

        rows.append({
            "relation": r,
            "triples": n,
            "functionality": round(fun, 4),
            "inv_functionality": round(inv_fun, 4),
            "tail_pool": tail_pool,
            "avg_tails_per_head": round(avg_tph, 3),
            "max_tails_per_head": max_tph,
            "multi_tail_head_frac": round(multi_frac, 4),
            "feeds": feeds,
            "flags": flags,
        })
    return rows, kg


def sample_examples(rows, kg, splits, cfg, n_samples, rng):
    """Attach a few generated corruptions per relation for manual inspection."""
    all_set = kg.all_set
    by_rel_pos = defaultdict(list)
    for h, r, t in dict.fromkeys(splits["train"] + splits["valid"] + splits["test"]):
        by_rel_pos[r].append((h, r, t))

    for row in rows:
        r = row["relation"]
        ex = []
        for (h, _, t) in by_rel_pos[r][: n_samples * 4]:
            if len(ex) >= n_samples:
                break
            pool = kg.tail_pool[r]
            if not pool:
                break
            cand = pool[rng.randrange(len(pool))]
            if (h, r, cand) in all_set:
                continue
            label = row["feeds"] if row["feeds"] in ("FALSE", "UNKNOWN") else "n/a"
            ex.append({
                "positive": verbalize_triple(h, r, t),
                "corrupted": verbalize_triple(h, r, cand),
                "would_label": label,
            })
        row["samples"] = ex
    return rows


def print_table(rows):
    hdr = f"{'relation':28s} {'trip':>7s} {'fun':>6s} {'ifun':>6s} {'pool':>6s} {'avgT':>6s} {'maxT':>5s} {'multi%':>7s}  {'feeds':8s} flags"
    print(hdr)
    print("-" * len(hdr))
    order = {"FALSE": 0, "UNKNOWN": 1, "neither": 2}
    for row in sorted(rows, key=lambda x: (order[x["feeds"]], -x["functionality"])):
        print(f"{row['relation'][:28]:28s} {row['triples']:>7d} "
              f"{row['functionality']:>6.3f} {row['inv_functionality']:>6.3f} "
              f"{row['tail_pool']:>6d} {row['avg_tails_per_head']:>6.2f} "
              f"{row['max_tails_per_head']:>5d} {row['multi_tail_head_frac']*100:>6.1f}%  "
              f"{row['feeds']:8s} {','.join(row['flags'])}")


def summarize(rows):
    feeds = defaultdict(int)
    risky = [r["relation"] for r in rows if any(f.startswith("MULTI_VALUED") for f in r["flags"])]
    border = [r["relation"] for r in rows if any("BORDERLINE" in f for f in r["flags"])]
    for r in rows:
        feeds[r["feeds"]] += 1
    print("\n=== summary ===")
    print(f"relations: {len(rows)} | FALSE-eligible: {feeds['FALSE']} "
          f"| UNKNOWN-eligible: {feeds['UNKNOWN']} | neither: {feeds['neither']}")
    if risky:
        print(f"\n[i] MULTI_VALUED (near-functional but excluded from FALSE by the multi gate):")
        for r in risky:
            print(f"    - {r}")
        print("    -> correctly routed away from FALSE (corruptions there are Unknown, not False).")
    if border:
        print(f"\n[!] BORDERLINE (grey-zone, feed nothing / threshold-fragile):")
        for r in border:
            print(f"    - {r}")
    if not risky and not border:
        print("no risky or borderline relations flagged.")


def main():
    ap = argparse.ArgumentParser(description="Diagnose relation functionality / label quality.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--samples", type=int, default=0, help="generated examples per relation to show")
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(cfg["seed"])

    if args.smoke_test:
        splits = synthetic_kg(cfg["seed"])
    else:
        splits = load_all(Path(args.raw_dir or cfg.get("raw_dir", "data/raw/YAGO3-10")))

    rows, kg = relation_report(splits, cfg)
    if args.samples:
        rows = sample_examples(rows, kg, splits, cfg, args.samples, rng)

    print(f"thresholds: functional >= {cfg['functional_threshold']} | "
          f"many-to-many <= {cfg['nonfunctional_threshold']} | min_pool = {cfg['min_tail_pool']}\n")
    print_table(rows)
    summarize(rows)

    if args.samples:
        print("\n=== sample corruptions (eyeball these) ===")
        for row in rows:
            if row.get("samples"):
                print(f"\n[{row['relation']}] -> would label {row['feeds']}")
                for e in row["samples"]:
                    print(f"    + {e['positive']}")
                    print(f"    ? {e['corrupted']}   => {e['would_label']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"thresholds": {
                "functional": cfg["functional_threshold"],
                "nonfunctional": cfg["nonfunctional_threshold"],
                "min_pool": cfg["min_tail_pool"],
            }, "relations": rows}, f, ensure_ascii=False, indent=2)
        print(f"\nwrote report -> {args.out}")


if __name__ == "__main__":
    main()
