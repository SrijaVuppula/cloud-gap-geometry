"""Rebuild and check the chip splits in splits/ (Section 4.1).

1. Evaluation set: 500 chips sampled with random.Random(42) from the sorted
   list of the 3,083 training-split chips.
2. Fine-tuning pool: the remaining 2,583 chips, minus every chip that shares
   an edge or a corner with an evaluation chip (529 removed, 2,054 left).
   Chips are cut from a regular grid; chip_RRR_CCC is row RRR, column CCC.
3. The pool was split at random into 2,004 training and 50 validation chips.
   The seed of that last draw was not recorded, so this script checks that
   the released lists partition the pool exactly instead of redrawing them.
   Use --write to overwrite splits/ with a fresh draw (not the paper's split).

  python scripts/01_make_splits.py           # check (needs data/training_data.txt from 00_download.py)
"""
import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cloudgap.config import load_config  # noqa: E402
from cloudgap.data import SPLITS, read_list  # noqa: E402

PAT = re.compile(r"(chip_(\d+)_(\d+))")


def rowcol(name):
    m = PAT.search(name)
    return int(m.group(2)), int(m.group(3))


def training_chips(cfg):
    txt = cfg["paths"]["chips_dir"].parent / "training_data.txt"
    if not txt.exists():
        raise SystemExit(f"{txt} not found. Run scripts/00_download.py first.")
    ids = {PAT.search(line).group(1) for line in txt.read_text().split() if PAT.search(line)}
    return sorted(f"{i}_merged.tif" for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write a fresh 2004/50 draw to splits/")
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config()
    n_eval, seed = cfg["evaluation"]["n_chips"], cfg["evaluation"]["chip_seed"]

    allc = training_chips(cfg)
    print(f"training split: {len(allc)} chips (paper: 3,083)")
    ev = random.Random(seed).sample(allc, n_eval)
    eset = set(ev)
    pool = [c for c in allc if c not in eset]
    erc = {rowcol(c) for c in ev}
    adjacent = [c for c in pool if any(abs(rowcol(c)[0] - r) <= 1 and abs(rowcol(c)[1] - k) <= 1 for r, k in erc)]
    pool = sorted(set(pool) - set(adjacent))
    print(f"pool: {len(allc) - n_eval} -> removed {len(adjacent)} adjacent (paper: 529) -> {len(pool)} (paper: 2,054)")

    ok = True
    released_eval = read_list("eval_500.txt")
    same = sorted(ev) == sorted(released_eval)
    ok &= same
    print("evaluation set matches splits/eval_500.txt:", same)

    tr, va = read_list("finetune_train_2004.txt"), read_list("finetune_val_50.txt")
    part = sorted(tr + va) == pool and not set(tr) & set(va)
    ok &= part
    print(f"released train ({len(tr)}) + val ({len(va)}) partition the pool exactly:", part)

    if a.write:
        rng = random.Random(a.split_seed)
        p = pool[:]
        rng.shuffle(p)
        (SPLITS / "eval_500.txt").write_text("\n".join(sorted(ev)) + "\n")
        (SPLITS / "finetune_val_50.txt").write_text("\n".join(sorted(p[:50])) + "\n")
        (SPLITS / "finetune_train_2004.txt").write_text("\n".join(sorted(p[50:])) + "\n")
        print("wrote new splits (this is NOT the paper's train/val draw)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
