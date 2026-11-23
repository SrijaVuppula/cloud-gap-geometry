"""Chip complexity of the summer frame (Section 5.1).

Mean of three scores, each min-max normalised over the 500 evaluation chips:
Canny edge density, spatial standard deviation and Shannon entropy of the
grey-scale true-colour image (2-98% stretch). No GPU needed.

  python scripts/06_chip_complexity.py

Writes runs/chip_complexity.csv (released copy: results/chip_complexity.csv).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.measure import shannon_entropy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cloudgap.config import REPO_ROOT, load_config  # noqa: E402
from cloudgap.data import FRAME, eval_chips, read_chip  # noqa: E402


def summer_rgb(path):
    s = read_chip(path)[FRAME]
    rgb = np.stack([s[2], s[1], s[0]], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def scores(rgb):
    gray = rgb2gray(rgb)
    return float(canny(gray, sigma=1.0).mean()), float(gray.std()), float(shannon_entropy(gray))


def norm(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


def main():
    cfg = load_config()
    chips = sorted(eval_chips(cfg["paths"]["chips_dir"]))
    names = [c.name for c in chips]
    edge, std, ent = map(list, zip(*(scores(summer_rgb(c)) for c in chips)))
    ne, ns, nen = norm(edge), norm(std), norm(ent)
    comp = (ne + ns + nen) / 3.0
    rows = sorted(zip(names, comp, edge, std, ent, ne, ns, nen), key=lambda r: r[1], reverse=True)
    out = REPO_ROOT / "runs" / "chip_complexity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip", "complexity", "rank", "edge_density", "spatial_std", "entropy", "edge_norm", "std_norm", "entropy_norm"])
        for k, (n, c, e, s, en, a, b, d) in enumerate(rows, 1):
            w.writerow([n, f"{c:.4f}", k, f"{e:.4f}", f"{s:.4f}", f"{en:.4f}", f"{a:.3f}", f"{b:.3f}", f"{d:.3f}"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
