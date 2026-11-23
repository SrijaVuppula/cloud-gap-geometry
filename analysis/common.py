"""Shared loaders, rounding and the crossover computation for the analysis scripts."""
import glob
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DERIVED = RESULTS / "derived"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

BB = ["tiny", "100M", "300M", "600M"]
RATIOS = [0.2, 0.4, 0.6, 0.8]
SEEDS = range(1, 11)

CHIP_KM2 = (224 * 30 / 1000) ** 2      # one non-overlapping 224 x 224 chip at 30 m, 45.16 km^2
CONUS_KM2 = 8.08e6                     # area of the contiguous United States
SAVING_FLOOR = 0.05                    # energy differences below 5% are within measurement uncertainty


def q2(x):
    """Round half up to two decimals (as printed in the paper)."""
    return Decimal(repr(round(float(x), 6))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def f2(x):
    return str(q2(x))


def includes_zero(lo, hi):
    return lo <= 0 <= hi


def seed_stats():
    d = pd.read_csv(DERIVED / "seed_stats.csv")
    d["ratio"] = d.ratio.astype(float).round(2)
    return d.set_index(["backbone", "ratio"])


def distance_stats():
    return pd.read_csv(DERIVED / "distance_stats.csv")


def pass_energy(finetuned=False):
    """{(backbone, ratio): J per pass} under contiguous occlusion (files store mJ)."""
    d = pd.read_csv(RESULTS / "energy" / ("inference_finetuned.csv" if finetuned else "inference_zeroshot.csv"))
    d = d[d.mask_type == "block"]
    return {(b, round(float(r), 2)): float(e) / 1000.0 for b, r, e in zip(d.backbone, d.mask_ratio, d.energy_mean)}


def training_cost():
    """Mean over the ten runs: hours, MJ, W, number of runs."""
    out = {}
    for bb in BB:
        files = sorted(glob.glob(str(RESULTS / "training" / "seed*" / f"{bb}_finetune_cost.csv")))
        d = pd.concat([pd.read_csv(f) for f in files])
        out[bb] = dict(h=d.elapsed_s.mean() / 3600, mj=d.energy_kj.mean() / 1000, w=d.avg_power_w.mean(),
                       h_sd=d.elapsed_s.std() / 3600, mj_sd=d.energy_kj.std() / 1000, n=len(files))
    return out


def psnr_medians():
    d = pd.read_csv(DERIVED / "psnr_medians.csv")
    d["ratio"] = d.ratio.round(2)
    return d.set_index(["backbone", "ratio"])


def crossover(M=None, tc=None, ez=None, S=None):
    """Crossover point N* (Eq. 3) for every fine-tuned backbone and ratio.

    z = smallest zero-shot backbone whose median contiguous PSNR >= fine-tuned b
    (unrounded medians). N* = E_train(b) / (E_pass(zs z) - E_pass(zs b)), defined
    only if z is larger than b, the saving per pass is at least 5%, and z exists.
    Returns (defined rows, cells where same-size zero-shot is already as good, undefined cells).
    """
    M = psnr_medians() if M is None else M
    S = seed_stats() if S is None else S
    tc = training_cost() if tc is None else tc
    ez = pass_energy() if ez is None else ez
    rows, no_saving, undefined = [], [], []
    for b in BB:
        for r in RATIOS:
            ft = M.loc[(b, r), "ft_contig"]
            z = next((c for c in BB if M.loc[(c, r), "zs_contig"] >= ft), None)
            if z is None:
                undefined.append((b, r))
                continue
            if BB.index(z) <= BB.index(b):
                no_saving.append(dict(backbone=b, ratio=r, margin=M.loc[(b, r), "zs_contig"] - ft,
                                      G_con=float(S.loc[(b, r), "G_contiguous_med"])))
                continue
            saved = ez[(z, r)] - ez[(b, r)]
            if saved < SAVING_FLOOR * ez[(z, r)]:
                continue
            n_star = tc[b]["mj"] * 1e6 / saved
            rows.append(dict(backbone=b, ratio=r, reference=z, excess=M.loc[(z, r), "zs_contig"] - ft,
                             saving=saved / ez[(z, r)] * 100, n_star=n_star,
                             area_mkm2=n_star * CHIP_KM2 / 1e6))
    return pd.DataFrame(rows), no_saving, undefined


# ---------------------------------------------------------------- figure style (matches the paper)
PLASMA = ["#2a0593", "#9e199d", "#e16462", "#fca636"]   # tiny, 100M, 300M, 600M
INK, BANNER = "#222222", "#EDEDED"
RC = {
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.4,
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
}
