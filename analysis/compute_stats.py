"""Chip-level statistics behind every table, figure and number (Section 4.4).

Protocol: reduce each chip to one value (mean over 5 trials; for fine-tuned
models, also over 10 runs), report the median over the 500 chips, a 95%
bootstrap interval (10,000 resamples of chips) and the share of chips with a
positive difference. Bootstrap streams are seeded, so the output is identical
on every run.

Writes results/derived/:
  seed_stats.csv              PSNR medians, Delta, G, change in Delta, per backbone x ratio
  psnr_medians.csv            unrounded median PSNR per configuration (for the crossover point)
  distance_stats.csv          patch-level PSNR by distance d (Figure 2)
  complexity_correlation.csv  Spearman rho of Delta and G with chip complexity
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import BB, DERIVED, RATIOS, RESULTS, SEEDS

KEYS = ["chip", "mask_ratio", "trial_seed"]
COLS = ["block_psnr", "random_psnr"]


def load(p):
    d = pd.read_csv(p, dtype={k: str for k in KEYS})
    for c in COLS:
        d[c] = d[c].astype(float)
    return d


def chip_means(d):
    return d.groupby(["mask_ratio", "chip"])[COLS].mean()


def load_chip_means():
    """{backbone: (zero-shot chip means, [chip means of each fine-tuning run])}."""
    out = {}
    for bb in BB:
        zs = load(RESULTS / "zeroshot" / f"results_{bb}.csv")
        zc = chip_means(zs)
        runs = []
        for s in SEEDS:
            ft = load(RESULTS / "finetuned" / f"seed{s}" / f"results_{bb}.csv")
            assert len(ft) == len(zs) and len(zs.merge(ft, on=KEYS)) == len(zs), (bb, s, "trials differ")
            fc = chip_means(ft)
            assert fc.index.equals(zc.index), (bb, s)
            runs.append(fc)
        out[bb] = (zc, runs)
    return out


# ---------------------------------------------------------------- seed_stats.csv
def quantities(z, f):
    gc = (f.block_psnr - z.block_psnr).values
    gs = (f.random_psnr - z.random_psnr).values
    return {"G_contiguous": gc, "G_scattered": gs,
            "gap_zeroshot": (z.random_psnr - z.block_psnr).values,
            "gap_finetuned": (f.random_psnr - f.block_psnr).values,
            "dgap": gs - gc}


def seed_stats(cm):
    rng = np.random.default_rng(0)

    def boot(x):
        n = len(x)
        meds = np.median(x[rng.integers(0, n, (10000, n))], axis=1)
        lo, hi = np.percentile(meds, [2.5, 97.5])
        return float(np.median(x)), float(lo), float(hi), float((x > 0).mean())

    rows = []
    for bb in BB:
        zc, runs = cm[bb]
        fmean = sum(runs) / len(runs)
        for ratio in sorted(zc.index.get_level_values(0).unique()):
            z, f = zc.xs(ratio, level=0), fmean.xs(ratio, level=0)
            per_run = [quantities(z, p.xs(ratio, level=0)) for p in runs]
            row = {"backbone": bb, "ratio": float(ratio),
                   "zs_contig_med": round(float(z.block_psnr.median()), 2),
                   "ft_contig_med": round(float(f.block_psnr.median()), 2),
                   "zs_scatt_med": round(float(z.random_psnr.median()), 2),
                   "ft_scatt_med": round(float(f.random_psnr.median()), 2)}
            for name, x in quantities(z, f).items():
                med, lo, hi, pos = boot(x)
                sm = np.array([np.median(q[name]) for q in per_run])
                row.update({f"{name}_med": round(med, 6), f"{name}_lo": round(lo, 6),
                            f"{name}_hi": round(hi, 6), f"{name}_pos": round(pos, 6),
                            f"{name}_seedmean": round(float(sm.mean()), 6),
                            f"{name}_seedsd": round(float(sm.std(ddof=1)), 6),
                            f"{name}_seedmin": round(float(sm.min()), 6),
                            f"{name}_seedmax": round(float(sm.max()), 6)})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["backbone", "ratio"])


# ---------------------------------------------------------------- distance_stats.csv
def distance_stats():
    rng = np.random.default_rng(0)
    min_chips = 50

    def boot(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        m = np.median(x[rng.integers(0, len(x), (10_000, len(x)))], axis=1)
        return np.median(x), np.percentile(m, 2.5), np.percentile(m, 97.5), (x > 0).mean(), len(x)

    rows = []

    def add(**kw):
        med, lo, hi, pos, n = boot(kw.pop("x"))
        rows.append(dict(**kw, med=med, lo=lo, hi=hi, pos=pos, n_chips=n))

    def psnr(mse_sum, n):
        return 10 * np.log10(n / mse_sum)

    for bb in BB:
        d = pd.concat([pd.read_csv(RESULTS / "distance" / f"patch_mse_{bb}_{c}.csv", dtype={"chip": str}) for c in ("zs", "ft")])
        key = ["cond", "geometry", "mask_ratio", "chip"]
        pdist = d.groupby(key + ["dist"])[["mse_sum", "n_patches"]].sum()
        pdist["psnr"] = psnr(pdist.mse_sum, pdist.n_patches)
        pall = d.groupby(key)[["mse_sum", "n_patches"]].sum()
        pall["psnr"] = psnr(pall.mse_sum, pall.n_patches)
        P, A = pdist.psnr.unstack("dist"), pall.psnr
        for cond in ["zs", "ft"]:
            for r in sorted(d.mask_ratio.unique()):
                Pc, Ps = P.loc[(cond, "contiguous", r)], P.loc[(cond, "scattered", r)]
                Ac, As = A.loc[(cond, "contiguous", r)], A.loc[(cond, "scattered", r)]
                add(backbone=bb, cond=cond, ratio=r, stat="psnr_all", geometry="contiguous", dist=0, x=Ac)
                add(backbone=bb, cond=cond, ratio=r, stat="psnr_all", geometry="scattered", dist=0, x=As)
                add(backbone=bb, cond=cond, ratio=r, stat="gap_all", geometry="-", dist=0, x=(As - Ac))
                for k in P.columns:
                    for g, M in [("contiguous", Pc), ("scattered", Ps)]:
                        x = M[k].dropna()
                        if len(x) >= min_chips:
                            add(backbone=bb, cond=cond, ratio=r, stat="psnr_d", geometry=g, dist=k, x=x)
                            if k > 1:   # within-chip change relative to the chip's d = 1 bin
                                add(backbone=bb, cond=cond, ratio=r, stat="drop_vs_d1", geometry=g, dist=k, x=(M[k] - M[1]).dropna())
                add(backbone=bb, cond=cond, ratio=r, stat="gap_d1", geometry="-", dist=1, x=(Ps[1] - Pc[1]))
        for g in ["contiguous", "scattered"]:
            for r in sorted(d.mask_ratio.unique()):
                Z, F = P.loc[("zs", g, r)], P.loc[("ft", g, r)]
                add(backbone=bb, cond="ft-zs", ratio=r, stat="G_all", geometry=g, dist=0, x=A.loc[("ft", g, r)] - A.loc[("zs", g, r)])
                for k in P.columns:
                    x = (F[k] - Z[k]).dropna()
                    if len(x) >= min_chips:
                        add(backbone=bb, cond="ft-zs", ratio=r, stat="G_d", geometry=g, dist=k, x=x)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- complexity_correlation.csv
def complexity_correlation(cm):
    comp = pd.read_csv(RESULTS / "chip_complexity.csv").set_index("chip").complexity
    rows = []
    for bb in BB:
        zc, runs = cm[bb]
        fmean = sum(runs) / len(runs)
        for r in RATIOS:
            z, f = zc.xs(str(r), level=0), fmean.xs(str(r), level=0)
            c = comp[z.index]
            for name, x in [("gap_zeroshot", z.random_psnr - z.block_psnr),
                            ("G_contiguous", f.block_psnr - z.block_psnr),
                            ("G_scattered", f.random_psnr - z.random_psnr)]:
                rows.append(dict(backbone=bb, ratio=r, quantity=name, spearman_rho=spearmanr(x, c).statistic))
    return pd.DataFrame(rows)


def psnr_medians(cm):
    """Unrounded median PSNR per configuration (used for the crossover point, Table 4)."""
    rows = []
    for bb in BB:
        zc, runs = cm[bb]
        f = sum(runs) / len(runs)
        for r in RATIOS:
            z, ff = zc.xs(str(r), level=0), f.xs(str(r), level=0)
            rows.append(dict(backbone=bb, ratio=r, zs_contig=z.block_psnr.median(), ft_contig=ff.block_psnr.median(),
                             zs_scatt=z.random_psnr.median(), ft_scatt=ff.random_psnr.median()))
    return pd.DataFrame(rows)


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    cm = load_chip_means()
    seed_stats(cm).to_csv(DERIVED / "seed_stats.csv", index=False)
    psnr_medians(cm).to_csv(DERIVED / "psnr_medians.csv", index=False, float_format="%.6f")
    distance_stats().to_csv(DERIVED / "distance_stats.csv", index=False, float_format="%.6f")
    complexity_correlation(cm).to_csv(DERIVED / "complexity_correlation.csv", index=False, float_format="%.6f")
    print(f"wrote seed_stats.csv, distance_stats.csv, complexity_correlation.csv to {DERIVED}")


if __name__ == "__main__":
    main()
