"""Helper: locate / prepare YAGO3-10 raw files for Step 1.

YAGO3-10 ships as three tab-separated files (train.txt, valid.txt, test.txt),
each line: `head <TAB> relation <TAB> tail`. It is distributed with the KG-LLM
repo (data/YAGO3-10/) and with most KGE libraries (e.g. the ConvE release,
pykeen, LibKGE). This script does not hard-download (network is restricted in
some environments); it validates a folder and, on Kaggle, points you to add the
dataset.

On Kaggle
---------
Add a YAGO3-10 dataset to the notebook (Add Input), then:

    python -m src.data.get_yago --src /kaggle/input/yago3-10 --dst data/raw/YAGO3-10
    python -m src.data.build_benchmark --raw-dir data/raw/YAGO3-10
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

EXPECTED = ["train.txt", "valid.txt", "test.txt"]


def validate(folder: Path) -> bool:
    ok = True
    for name in EXPECTED:
        p = folder / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                first = f.readline().strip()
            cols = len(first.split("\t")) if "\t" in first else len(first.split())
            status = "OK" if cols == 3 else f"WARN: {cols} cols (need 3)"
            print(f"  {name:12s} {status}  e.g. {first[:60]!r}")
        else:
            print(f"  {name:12s} MISSING")
            if name != "valid.txt":  # valid optional
                ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="folder that already has the 3 files")
    ap.add_argument("--dst", default="data/raw/YAGO3-10")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    if args.src:
        src = Path(args.src)
        for name in EXPECTED:
            p = src / name
            if p.exists():
                shutil.copy(p, dst / name)
                print(f"copied {p} -> {dst / name}")

    print(f"\nValidating {dst}:")
    if validate(dst):
        print("\nReady. Run: python -m src.data.build_benchmark --raw-dir", dst)
    else:
        print("\nMissing files. Provide --src pointing to a folder with:",
              ", ".join(EXPECTED))


if __name__ == "__main__":
    main()
