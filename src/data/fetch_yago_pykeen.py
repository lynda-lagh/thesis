"""Fetch YAGO3-10 and write train/valid/test.txt for Step 1.

Uses pykeen, which bundles the canonical YAGO3-10 splits (same ones used across
the KGC literature). Run this on a machine with open internet (e.g. your PC),
then Step 1 reads the .txt files.

    pip install pykeen
    python -m src.data.fetch_yago_pykeen --dst data/raw/YAGO3-10

Output: data/raw/YAGO3-10/{train,valid,test}.txt  (head <TAB> relation <TAB> tail)
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="data/raw/YAGO3-10")
    args = ap.parse_args()

    try:
        from pykeen.datasets import YAGO310
    except ImportError:
        raise SystemExit("pykeen not installed. Run:  pip install pykeen")

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(">> downloading / loading YAGO3-10 via pykeen (first run downloads it)...")
    ds = YAGO310()
    splits = {"train": ds.training, "valid": ds.validation, "test": ds.testing}

    for name, tf in splits.items():
        triples = tf.triples  # numpy array of shape (n, 3) with string labels
        out = dst / f"{name}.txt"
        with open(out, "w", encoding="utf-8") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")
        print(f"   wrote {out}  ({len(triples):,} triples)")

    print("\nDone. Now run:")
    print(f"   python -m src.data.diagnose_relations --raw-dir {dst} --samples 3")
    print(f"   python -m src.data.build_benchmark   --raw-dir {dst}")


if __name__ == "__main__":
    main()
