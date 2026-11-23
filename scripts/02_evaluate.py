"""Paired contiguous vs scattered evaluation (Sections 4.2 and 4.4).

For every evaluation chip, mask ratio and trial, hide the same number of
summer patches as one block and at scattered positions, reconstruct with the
model and score PSNR over the hidden pixels. Masks are replayed from
splits/trial_seeds.csv, so every configuration sees identical masks.

  python scripts/02_evaluate.py --backbone tiny                       # zero-shot
  python scripts/02_evaluate.py --backbone tiny --finetuned --seed 1  # fine-tuned run 1
  python scripts/02_evaluate.py --backbone tiny --limit 5             # quick check on 5 chips

Writes runs/{zeroshot|finetuned/seed{N}}/results_{backbone}.csv (resumable).
When it finishes it compares every PSNR with the released file in results/.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cloudgap.config import REPO_ROOT, finetuned_weights, load_config, zeroshot_weights  # noqa: E402
from cloudgap.data import FRAME, IMG, T, eval_chips, load_eval_chip, to_reflectance, trial_seeds  # noqa: E402
from cloudgap.masking import build_paired_masks, pixel_map  # noqa: E402
from cloudgap.metrics import masked_psnr  # noqa: E402
from cloudgap.model import load_prithvi, run_masked_forward  # noqa: E402

FIELDS = ["backbone", "chip", "mask_ratio", "trial", "trial_seed", "summer_masked",
          "block_psnr", "random_psnr", "delta_rand_minus_block"]


def reconstruct(model, x, noise, global_ratio, mean, std, device):
    rec = run_masked_forward(model, x, global_ratio, noise.unsqueeze(0).to(device))
    return to_reflectance(rec, mean, std, FRAME)


def compare_with_released(out_csv, released_csv):
    if not released_csv.exists():
        return
    k = ["chip", "mask_ratio", "trial"]
    a = pd.read_csv(out_csv, dtype={"mask_ratio": str})
    b = pd.read_csv(released_csv, dtype={"mask_ratio": str})
    m = a.merge(b, on=k, suffixes=("", "_rel"))
    if m.empty:
        return
    d = np.maximum((m.block_psnr - m.block_psnr_rel).abs(), (m.random_psnr - m.random_psnr_rel).abs())
    print(f"compared {len(m)} rows with {released_csv.relative_to(REPO_ROOT)}: "
          f"max |dPSNR| = {d.max():.4f} dB, median = {d.median():.4f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["tiny", "100M", "300M", "600M"])
    ap.add_argument("--finetuned", action="store_true")
    ap.add_argument("--seed", type=int, help="fine-tuning run (1-10), with --finetuned")
    ap.add_argument("--limit", type=int, help="evaluate only the first N chips")
    ap.add_argument("--out", type=Path, help="output CSV (default under runs/)")
    a = ap.parse_args()
    if a.finetuned and a.seed is None:
        ap.error("--finetuned needs --seed")

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub = Path("finetuned") / f"seed{a.seed}" if a.finetuned else Path("zeroshot")
    out = a.out or REPO_ROOT / "runs" / sub / f"results_{a.backbone}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt, cfg_json = finetuned_weights(cfg, a.backbone, a.seed) if a.finetuned else zeroshot_weights(cfg, a.backbone)
    model, mean, std, patch = load_prithvi(ckpt, cfg_json, num_frames=T, device=device)
    assert patch == cfg["backbones"][a.backbone]["patch_size"]
    print(f"{a.backbone} ({'fine-tuned seed ' + str(a.seed) if a.finetuned else 'zero-shot'}), patch {patch}, {device}")

    chips = eval_chips(cfg["paths"]["chips_dir"])[: a.limit]
    seeds = trial_seeds()
    ratios = cfg["evaluation"]["ratios"]
    n_trials = cfg["evaluation"]["trials_per_chip"]

    new_file = not out.exists() or out.stat().st_size == 0
    done = set()
    if not new_file:
        with open(out) as f:
            done = {(r["chip"], round(float(r["mask_ratio"]), 2), int(r["trial"])) for r in csv.DictReader(f)}
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, chip in enumerate(chips):
            x, gt = load_eval_chip(chip, mean, std, device)
            for r in ratios:
                for t in range(n_trials):
                    key = (chip.name, round(r, 2), t)
                    if key in done:
                        continue
                    seed = seeds[key]
                    masks = build_paired_masks(r, patch, seed, IMG, T, FRAME)
                    psnr = {}
                    for geo, (noise, gr, idx) in masks.items():
                        rec = reconstruct(model, x, noise, gr, mean, std, device)
                        psnr[geo] = masked_psnr(rec, gt, pixel_map(idx, patch, IMG, T, FRAME))
                    w.writerow({"backbone": a.backbone, "chip": chip.name, "mask_ratio": r, "trial": t,
                                "trial_seed": seed, "summer_masked": len(masks["contiguous"][2]),
                                "block_psnr": round(psnr["contiguous"], 4), "random_psnr": round(psnr["scattered"], 4),
                                "delta_rand_minus_block": round(psnr["scattered"] - psnr["contiguous"], 4)})
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(chips)} chips")
    print(f"wrote {out}")
    compare_with_released(out, cfg["paths"]["results_dir"] / sub / f"results_{a.backbone}.csv")


if __name__ == "__main__":
    main()
