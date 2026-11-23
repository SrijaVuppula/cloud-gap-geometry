"""Per-patch error against distance to visible context (Section 5.1, Figure 2).

For every hidden patch, d = chessboard distance (in patches) to the nearest
visible summer patch. One trial per chip (trial 0), all 500 chips, all
ratios, both geometries; zero-shot and fine-tuned run 1 for each backbone.
Every masked-region PSNR is checked against the logged evaluation.

  python scripts/05_patch_distance.py --backbone tiny
  python scripts/05_patch_distance.py --backbone tiny --limit 5

Writes runs/distance/patch_mse_{backbone}_{zs,ft}.csv with the summed patch
MSE per (geometry, chip, ratio, d); analysis/compute_stats.py turns these
into per-distance PSNR.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_cdt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cloudgap.config import REPO_ROOT, finetuned_weights, load_config, zeroshot_weights  # noqa: E402
from cloudgap.data import FRAME, IMG, T, eval_chips, load_eval_chip, to_reflectance, trial_seeds  # noqa: E402
from cloudgap.masking import build_paired_masks, hidden_grid, pixel_map  # noqa: E402
from cloudgap.metrics import masked_psnr, patch_mse  # noqa: E402
from cloudgap.model import load_prithvi, run_masked_forward  # noqa: E402

FIELDS = ["backbone", "cond", "geometry", "chip", "mask_ratio", "dist", "n_patches", "mse_sum"]


def logged_psnr(path):
    d = pd.read_csv(path)
    d = d[d.trial == 0]
    return {(c, round(r, 2)): {"contiguous": b, "scattered": s}
            for c, r, b, s in zip(d.chip, d.mask_ratio, d.block_psnr, d.random_psnr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["tiny", "100M", "300M", "600M"])
    ap.add_argument("--seed", type=int, default=1, help="fine-tuning run used for the 'ft' condition")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = cfg["paths"]["results_dir"]
    chips = eval_chips(cfg["paths"]["chips_dir"])[: a.limit]
    seeds = trial_seeds()
    out_dir = REPO_ROOT / "runs" / "distance"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in ("zs", "ft"):
        if cond == "zs":
            ckpt, cj = zeroshot_weights(cfg, a.backbone)
            logged = logged_psnr(res / "zeroshot" / f"results_{a.backbone}.csv")
        else:
            ckpt, cj = finetuned_weights(cfg, a.backbone, a.seed)
            logged = logged_psnr(res / "finetuned" / f"seed{a.seed}" / f"results_{a.backbone}.csv")
        model, mean, std, patch = load_prithvi(ckpt, cj, num_frames=T, device=device)
        maxdiff = 0.0
        out = out_dir / f"patch_mse_{a.backbone}_{cond}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            for i, chip in enumerate(chips):
                x, gt = load_eval_chip(chip, mean, std, device)
                for r in cfg["evaluation"]["ratios"]:
                    masks = build_paired_masks(r, patch, seeds[(chip.name, round(r, 2), 0)], IMG, T, FRAME)
                    for geo, (noise, gr, idx) in masks.items():
                        rec = to_reflectance(run_masked_forward(model, x, gr, noise.unsqueeze(0).to(device)), mean, std, FRAME)
                        ps = masked_psnr(rec, gt, pixel_map(idx, patch, IMG, T, FRAME))
                        maxdiff = max(maxdiff, abs(ps - logged[(chip.name, round(r, 2))][geo]))
                        h = hidden_grid(idx, patch, IMG, FRAME)
                        pm = patch_mse(rec, gt, patch)
                        dist = distance_transform_cdt(h, metric="chessboard")
                        for dv in np.unique(dist[h]):
                            sel = h & (dist == dv)
                            w.writerow([a.backbone, cond, geo, chip.name, r, int(dv), int(sel.sum()), repr(float(pm[sel].sum()))])
                if (i + 1) % 50 == 0:
                    print(f"  [{a.backbone} {cond}] {i + 1}/{len(chips)} chips, max |PSNR - logged| = {maxdiff:.4f} dB")
        print(f"wrote {out}; max |PSNR - logged| = {maxdiff:.4f} dB")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
