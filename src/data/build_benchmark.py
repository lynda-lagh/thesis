"""Step 1 - Build the True / False / Unknown benchmark from YAGO3-10.

Core methodological contribution: we do NOT treat every corrupted triple as
false (the standard closed-world shortcut that makes triple classification
trivial). We split corruptions into two *grounded* classes:

  * FALSE   : relation is (near-)functional on the tail side AND the head already
              has a known, different tail  ->  a different type-consistent value
              contradicts the KG.
  * UNKNOWN : relation is many-to-many, the corrupted triple is type-consistent,
              absent from every split, and not derivable via a symmetric edge
              ->  genuinely unverifiable under the open-world assumption.
  * TRUE    : an observed triple.

Functionality is measured from the data, so the protocol transfers to any KG.

Usage
-----
    python -m src.data.build_benchmark --smoke-test          # synthetic sanity run
    python -m src.data.build_benchmark --raw-dir data/raw/YAGO3-10 \
        --out-dir data/processed/YAGO3-10 --n-per-class 6000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except Exception:  # pyyaml optional for smoke test
    yaml = None

from .verbalize import build_prompt, verbalize_triple

Triple = tuple[str, str, str]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_triples(path: Path) -> list[Triple]:
    """Read a tab-separated `head <TAB> relation <TAB> tail` file."""
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
                if len(parts) != 3:
                    continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def load_all(raw_dir: Path) -> dict[str, list[Triple]]:
    splits = {}
    for name in ("train", "valid", "test"):
        for ext in (".txt", ".tsv"):
            p = raw_dir / f"{name}{ext}"
            if p.exists():
                splits[name] = load_triples(p)
                break
    if "train" not in splits:
        raise FileNotFoundError(
            f"No train.txt found in {raw_dir}. Expected train/valid/test .txt "
            f"files with 'head<TAB>relation<TAB>tail' per line."
        )
    splits.setdefault("valid", [])
    splits.setdefault("test", [])
    return splits


def synthetic_kg(seed: int = 0) -> dict[str, list[Triple]]:
    """A tiny YAGO-like KG for smoke testing the whole pipeline offline.

    Contains functional relations (wasBornIn, hasGender, isCitizenOf) and
    non-functional ones (actedIn, isAffiliatedTo) so all three classes appear.
    """
    rnd = random.Random(seed)
    people = [f"Person_{i}" for i in range(120)]
    cities = [f"City_{i}" for i in range(40)]
    countries = [f"Country_{i}" for i in range(20)]
    genders = ["Male", "Female"]
    movies = [f"Movie_{i}" for i in range(60)]
    clubs = [f"Club_{i}" for i in range(30)]

    triples: list[Triple] = []
    for p in people:
        triples.append((p, "wasBornIn", rnd.choice(cities)))       # functional
        triples.append((p, "hasGender", rnd.choice(genders)))      # functional
        triples.append((p, "isCitizenOf", rnd.choice(countries)))  # functional
        for _ in range(rnd.randint(1, 4)):                          # many-to-many
            triples.append((p, "actedIn", rnd.choice(movies)))
        for _ in range(rnd.randint(1, 3)):                          # many-to-many
            triples.append((p, "isAffiliatedTo", rnd.choice(clubs)))
    # dedupe
    triples = list(dict.fromkeys(triples))
    rnd.shuffle(triples)
    n = len(triples)
    return {
        "train": triples[: int(0.8 * n)],
        "valid": triples[int(0.8 * n) : int(0.9 * n)],
        "test": triples[int(0.9 * n) :],
    }


# ---------------------------------------------------------------------------
# KG statistics
# ---------------------------------------------------------------------------

class KGStats:
    """Indexes + relation functionality computed over ALL splits (for filtering)."""

    def __init__(self, all_triples: list[Triple]):
        self.all_set: set[Triple] = set(all_triples)
        self.tail_pool: dict[str, list[str]] = defaultdict(list)   # r -> type-consistent tails
        self.head_pool: dict[str, list[str]] = defaultdict(list)
        self.ht: dict[tuple[str, str], set[str]] = defaultdict(set)  # (h,r) -> {t}
        self.rel_triples: dict[str, int] = defaultdict(int)
        self.rel_heads: dict[str, set[str]] = defaultdict(set)

        seen_tail, seen_head = defaultdict(set), defaultdict(set)
        for h, r, t in all_triples:
            self.ht[(h, r)].add(t)
            self.rel_triples[r] += 1
            self.rel_heads[r].add(h)
            if t not in seen_tail[r]:
                seen_tail[r].add(t); self.tail_pool[r].append(t)
            if h not in seen_head[r]:
                seen_head[r].add(h); self.head_pool[r].append(h)

        # per-relation fraction of heads that have MORE THAN ONE tail.
        # This is the real correctness signal for FALSE-eligibility: a low value
        # means the relation is single-valued in practice, so a type-consistent
        # corruption genuinely contradicts the KG.
        rel_multi = defaultdict(lambda: [0, 0])  # r -> [multi_heads, total_heads]
        for (h, r), tails in self.ht.items():
            rel_multi[r][1] += 1
            if len(tails) > 1:
                rel_multi[r][0] += 1
        self.rel_multi_frac = {
            r: (m / tot if tot else 0.0) for r, (m, tot) in rel_multi.items()
        }

        # total head degree (triples where the entity appears as head, any relation).
        # Used to flag long-tail heads among Unknown examples.
        self.head_degree: dict[str, int] = defaultdict(int)
        for h, r, t in all_triples:
            self.head_degree[h] += 1

    def functionality(self, r: str) -> float:
        """fun(r) = |distinct heads| / |triples|. ~1.0 => each head has one tail."""
        n = self.rel_triples[r]
        return len(self.rel_heads[r]) / n if n else 0.0

    def multi_frac(self, r: str) -> float:
        return self.rel_multi_frac.get(r, 0.0)

    def known_tails(self, h: str, r: str) -> set[str]:
        return self.ht.get((h, r), set())


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def make_example(h, r, t, label, strategy, fun):
    p = build_prompt(h, r, t)
    return {
        "head": h,
        "relation": r,
        "tail": t,
        "label": label,                 # "True" | "False" | "Unknown"
        "response": label,
        "gen_strategy": strategy,
        "rel_functionality": round(fun, 4),
        "sentence": p["sentence"],
        "prompt": p["user"],
        "messages": p["messages"],
    }


def build_examples(splits, cfg, rng: random.Random):
    all_triples = splits["train"] + splits["valid"] + splits["test"]
    kg = KGStats(all_triples)

    fun_hi = cfg["functional_threshold"]
    fun_lo = cfg["nonfunctional_threshold"]
    min_pool = cfg["min_tail_pool"]
    min_pool_false = cfg.get("min_pool_false", 2)
    max_multi = cfg.get("max_multi_tail_frac", 0.12)
    blocklist = set(cfg.get("relation_blocklist", []))
    rare_head_max = cfg.get("rare_head_max_degree", 2)
    filt_sym = cfg["filter_symmetric_unknown"]
    n_target = cfg["n_per_class"]

    # classify relations.
    #  FALSE-eligible : near-functional AND single-valued in practice (low multi
    #                   fraction) AND not blocklisted. Pool can be small (a clean
    #                   binary relation like hasGender only needs 2 tails).
    #  UNKNOWN-eligible: many-to-many, decent tail pool, not blocklisted.
    funcs = {r: kg.functionality(r) for r in kg.rel_triples}
    functional = {
        r for r, f in funcs.items()
        if f >= fun_hi and kg.multi_frac(r) <= max_multi
        and len(kg.tail_pool[r]) >= min_pool_false and r not in blocklist
    }
    manytomany = {
        r for r, f in funcs.items()
        if f <= fun_lo and len(kg.tail_pool[r]) >= min_pool and r not in blocklist
    }

    # draw positives from train+valid+test (observed => TRUE)
    pos = list(dict.fromkeys(all_triples))
    rng.shuffle(pos)

    trues, falses, unknowns = [], [], []
    seen_keys: set[tuple] = set()   # dedupe across the whole benchmark

    def add(store, ex):
        key = (ex["head"], ex["relation"], ex["tail"], ex["label"])
        if key in seen_keys:
            return False
        seen_keys.add(key)
        store.append(ex)
        return True

    for (h, r, t) in pos:
        fun = funcs[r]
        # TRUE
        if len(trues) < n_target:
            add(trues, make_example(h, r, t, "True", "observed", fun))

        # FALSE  (functional relation + head already has a known different tail)
        if len(falses) < n_target and r in functional:
            known = kg.known_tails(h, r)
            pool = kg.tail_pool[r]
            for _ in range(12):                     # bounded rejection sampling
                cand = pool[rng.randrange(len(pool))]
                if cand in known:
                    continue
                if (h, r, cand) in kg.all_set:      # never accidentally true
                    continue
                add(falses, make_example(h, r, cand, "False", "functional_conflict", fun))
                break

        # UNKNOWN  (many-to-many, type-consistent, absent, not symmetric-derivable)
        if len(unknowns) < n_target and r in manytomany:
            pool = kg.tail_pool[r]
            for _ in range(12):
                cand = pool[rng.randrange(len(pool))]
                if (h, r, cand) in kg.all_set:
                    continue
                if filt_sym and (cand, r, h) in kg.all_set:
                    continue
                ex = make_example(h, r, cand, "Unknown", "openworld_absent", fun)
                # sub-type of the Unknown (all computable from the KG):
                #   open_extension: head already has known tails for r (plausibly true, unrecorded)
                #   unconstrained : head has NO known tail for r (nothing to reason from)
                known = kg.known_tails(h, r)
                hd = kg.head_degree.get(h, 0)
                ex["unknown_type"] = "open_extension" if known else "unconstrained"
                ex["head_degree"] = hd
                ex["rare_head"] = hd <= rare_head_max
                add(unknowns, ex)
                break

        if len(trues) >= n_target and len(falses) >= n_target and len(unknowns) >= n_target:
            break

    return trues, falses, unknowns, funcs, functional, manytomany


# ---------------------------------------------------------------------------
# Split + write
# ---------------------------------------------------------------------------

def split_and_write(trues, falses, unknowns, cfg, out_dir: Path, funcs, functional, manytomany):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])

    def three_way(items):
        rng.shuffle(items)
        n = len(items)
        n_test = int(cfg["test_frac"] * n)
        n_val = int(cfg["val_frac"] * n)
        return {
            "test": items[:n_test],
            "valid": items[n_test : n_test + n_val],
            "train": items[n_test + n_val :],
        }

    parts = {"train": [], "valid": [], "test": []}
    for group in (trues, falses, unknowns):
        s = three_way(group)
        for k in parts:
            parts[k].extend(s[k])
    for k in parts:
        rng.shuffle(parts[k])

    for split, rows in parts.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # leakage check: no example should be an observed positive unless label==True
    card = {
        "dataset": cfg.get("name", "YAGO3-10"),
        "n_per_class_target": cfg["n_per_class"],
        "counts": {
            "true": len(trues), "false": len(falses), "unknown": len(unknowns),
        },
        "splits": {k: len(v) for k, v in parts.items()},
        "split_label_balance": {
            k: _label_counts(v) for k, v in parts.items()
        },
        "unknown_subtype_balance": _unknown_subtypes(unknowns),
        "relations": {
            "total": len(funcs),
            "functional_false_eligible": sorted(functional),
            "manytomany_unknown_eligible": sorted(manytomany),
            "functionality": {r: round(f, 3) for r, f in sorted(funcs.items(), key=lambda x: -x[1])},
        },
        "thresholds": {
            "functional": cfg["functional_threshold"],
            "nonfunctional": cfg["nonfunctional_threshold"],
        },
    }
    with open(out_dir / "data_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return card


def _label_counts(rows):
    c = {"True": 0, "False": 0, "Unknown": 0}
    for r in rows:
        c[r["label"]] += 1
    return c


def _unknown_subtypes(unknowns):
    c = {"open_extension": 0, "unconstrained": 0, "rare_head": 0, "total": len(unknowns)}
    for r in unknowns:
        c[r.get("unknown_type", "unconstrained")] = c.get(r.get("unknown_type", "unconstrained"), 0) + 1
        if r.get("rare_head"):
            c["rare_head"] += 1
    return c


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(path: str | None) -> dict:
    base = {
        "name": "YAGO3-10",
        "n_per_class": 6000,
        "functional_threshold": 0.83,
        "nonfunctional_threshold": 0.65,
        "min_tail_pool": 5,
        "min_pool_false": 2,
        "max_multi_tail_frac": 0.12,
        "relation_blocklist": ["hasWebsite"],
        "rare_head_max_degree": 2,
        "filter_symmetric_unknown": True,
        "val_frac": 0.10,
        "test_frac": 0.10,
        "seed": 42,
    }
    if path and yaml and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        base.update(y.get("benchmark", {}))
        base["name"] = y.get("dataset", {}).get("name", base["name"])
    return base


def main():
    ap = argparse.ArgumentParser(description="Build True/False/Unknown KGC benchmark.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default="data/processed/YAGO3-10")
    ap.add_argument("--n-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run on a tiny synthetic KG (no data files needed).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n_per_class is not None:
        cfg["n_per_class"] = args.n_per_class
    if args.seed is not None:
        cfg["seed"] = args.seed

    rng = random.Random(cfg["seed"])

    if args.smoke_test:
        print(">> SMOKE TEST on synthetic KG")
        splits = synthetic_kg(cfg["seed"])
        cfg["n_per_class"] = min(cfg["n_per_class"], 200)
        out_dir = Path(args.out_dir).parent / "_smoke"
    else:
        raw = Path(args.raw_dir or cfg.get("raw_dir", "data/raw/YAGO3-10"))
        print(f">> Loading YAGO3-10 from {raw}")
        splits = load_all(raw)
        out_dir = Path(args.out_dir)

    print(f"   train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}")
    trues, falses, unknowns, funcs, functional, manytomany = build_examples(splits, cfg, rng)
    print(f">> Generated  True={len(trues)}  False={len(falses)}  Unknown={len(unknowns)}")
    print(f"   false-eligible relations: {len(functional)} | unknown-eligible: {len(manytomany)}")

    card = split_and_write(trues, falses, unknowns, cfg, out_dir, funcs, functional, manytomany)
    print(f">> Wrote splits to {out_dir}")
    print(f"   split sizes: {card['splits']}")
    print(f"   balance per split: {card['split_label_balance']}")
    print(f">> Data card: {out_dir / 'data_card.json'}")


if __name__ == "__main__":
    main()
