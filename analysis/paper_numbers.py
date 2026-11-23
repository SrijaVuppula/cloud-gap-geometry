"""Print every number quoted in the text of the paper, section by section,
next to the sentence it supports. Run after compute_stats.py."""
import numpy as np
import pandas as pd

from common import (BB, CHIP_KM2, CONUS_KM2, DERIVED, RATIOS, crossover, distance_stats, f2, includes_zero,
                    pass_energy, psnr_medians, seed_stats, training_cost)

LARGE = BB[1:]


def rng(xs, fmt=f2):
    return f"{fmt(min(xs))}-{fmt(max(xs))}"


def main():
    S, M, D = seed_stats(), psnr_medians(), distance_stats()
    ez, ef, tc = pass_energy(), pass_energy(finetuned=True), training_cost()
    g = lambda c, b, r: float(S.loc[(b, r), c])
    ci0 = lambda n, b, r: includes_zero(g(f"{n}_lo", b, r), g(f"{n}_hi", b, r))
    cells = [(b, r) for b in BB for r in RATIOS]
    lcells = [(b, r) for b in LARGE for r in RATIOS]

    def dist(**k):
        q = D
        for a, v in k.items():
            q = q[np.isclose(q[a], v)] if isinstance(v, float) else q[q[a] == v]
        return q

    print("== 5.1 Zero-shot reconstruction")
    zc = lambda b, r: g("zs_contig_med", b, r)
    print("  tiny->100M adds", rng([zc("100M", r) - zc("tiny", r) for r in RATIOS]), "dB;",
          "100M->300M", rng([zc("300M", r) - zc("100M", r) for r in RATIOS]), "; 100M->600M",
          rng([zc("600M", r) - zc("100M", r) for r in RATIOS]), "; 600M-300M at r=80%:", f2(zc("600M", 0.8) - zc("300M", 0.8)))
    gz = [g("gap_zeroshot_med", *c) for c in cells]
    print("  geometry gap Delta:", rng(gz), "dB; positive on at least",
          f"{min(g('gap_zeroshot_pos', *c) for c in cells) * 100:.1f}% of chips")
    print("  MSE ratio contiguous/scattered = 10^(Delta/10):", rng([10 ** (x / 10) for x in gz], lambda v: f"{v:.1f}"))
    print("  largest gap at every ratio:", {int(r * 100): max(BB, key=lambda b: g("gap_zeroshot_med", b, r)) for r in RATIOS},
          "| peak ratio:", {b: int(max(RATIOS, key=lambda r: g("gap_zeroshot_med", b, r)) * 100) for b in BB})
    for bb in BB:
        d = pd.read_csv(f"{DERIVED.parent}/distance/patch_mse_{bb}_zs.csv")
        tot = d.groupby(["geometry", "mask_ratio"]).n_patches.sum()
        d1 = d[d.dist == 1].groupby(["geometry", "mask_ratio"]).n_patches.sum()
        sh = (d1 / tot).unstack()
        print(f"  share of hidden patches at d=1, {bb:4s}: contiguous", rng(sh.loc["contiguous"] * 100, lambda v: f"{v:.0f}%"),
              "| scattered", [f"{v * 100:.0f}%" for v in sh.loc["scattered"]])
    drop = [dist(stat="drop_vs_d1", backbone=b, cond="zs", ratio=r, geometry="contiguous", dist=2).med.iloc[0] for b, r in cells]
    print("  zero-shot PSNR drop d=1 -> d=2:", rng([-x for x in drop]), "dB")
    ring = {r: [dist(stat="gap_d1", backbone=b, cond="zs", ratio=r).med.iloc[0] for b in BB] for r in RATIOS}
    print("  edge ring (d=1) vs scattered d=1: r<=40%", rng(ring[0.2] + ring[0.4]), "| r=60%", rng(ring[0.6]), "| r=80%", rng(ring[0.8]))
    cc = pd.read_csv(DERIVED / "complexity_correlation.csv")
    rho = lambda q: cc[cc.quantity == q].spearman_rho
    print("  Spearman rho(Delta, complexity):", rng(rho("gap_zeroshot")), "in all 16 cells")

    print("\n== 5.2 Fine-tuning on contiguous masks")
    tg = [g("G_contiguous_med", "tiny", r) for r in RATIOS]
    print("  tiny G_con:", rng(tg), "dB; chips improving >=", f"{min(g('G_contiguous_pos', 'tiny', r) for r in RATIOS) * 100:.0f}%")
    lg = [g("G_contiguous_med", *c) for c in lcells]
    print("  larger G_con:", rng(lg), "dB; CI includes zero in", sum(ci0("G_contiguous", *c) for c in lcells), "of 12 cells")
    print("  r=80% larger: G_con", [f2(g("G_contiguous_med", b, 0.8)) for b in LARGE],
          "chips improving", [f"{g('G_contiguous_pos', b, 0.8) * 100:.0f}%" for b in LARGE], "| 300M r=20%:", f2(g("G_contiguous_med", "300M", 0.2)))
    print("  run-to-run SD of median G: contiguous <=", f2(max(g("G_contiguous_seedsd", *c) for c in cells)),
          "scattered <=", f2(max(g("G_scattered_seedsd", *c) for c in cells)))
    print("  tiny G_sct:", rng([g("G_scattered_med", "tiny", r) for r in RATIOS]))
    lose = [c for c in lcells if g("G_scattered_hi", *c) < 0]
    print("  larger G_sct: lose", rng([-g("G_scattered_med", *c) for c in lose]), "dB in", len(lose), "of 12 cells; other cell(s):",
          [(b, int(r * 100), round(g("G_scattered_med", b, r), 3)) for b, r in lcells if (b, r) not in lose])
    dg = [(c, g("dgap_med", *c)) for c in cells]
    ex = [(c, v) for c, v in dg if not ci0("dgap", *c)]
    print("  change in Delta: negative in", sum(v < 0 for _, v in dg), "of 16; CI excludes zero in", len(ex),
          "with reductions", rng([-v for _, v in ex]), "; exception:", [c for c, _ in dg if ci0("dgap", *c)])
    gf = [g("gap_finetuned_med", *c) for c in cells]
    print("  fine-tuned Delta:", rng(gf), "dB; positive on at least", f"{min(g('gap_finetuned_pos', *c) for c in cells) * 100:.0f}% of chips")
    print("  larger: fall in scattered > rise in contiguous in",
          sum(-g("G_scattered_med", *c) > g("G_contiguous_med", *c) for c in lcells), "of 12 cells")
    tiny_d = dist(stat="G_d", backbone="tiny", geometry="contiguous")
    print("  tiny G by distance: d=1", rng(tiny_d[tiny_d.dist == 1].med), "| d>=4",
          rng(tiny_d[tiny_d.dist >= 4].groupby("ratio").med.agg(["min", "max"]).values.ravel()))
    l80 = dist(stat="G_d", ratio=0.8, geometry="contiguous")
    l80 = l80[l80.backbone.isin(LARGE)]
    print("  larger, r=80%: G at d=1 <=", f2(l80[l80.dist == 1].med.max()), "| d>=5", rng(l80[l80.dist >= 5].med))
    print("  |rho(G_con, complexity)| <=", f2(rho("G_contiguous").abs().max()),
          "(G_sct: up to", f2(rho("G_scattered").abs().max()), ")")

    print("\n== 5.3 Cost and crossover")
    step = [ez[(BB[i + 1], r)] / ez[(BB[i], r)] for i in range(3) for r in RATIOS]
    print("  energy per pass, each size step x", rng(step, lambda v: f"{v:.1f}"), "| 600M/tiny x",
          rng([ez[("600M", r)] / ez[("tiny", r)] for r in RATIOS], lambda v: f"{v:.1f}"))
    print("  drop r=20% -> 80%:", rng([(1 - ez[(b, 0.8)] / ez[(b, 0.2)]) * 100 for b in BB], lambda v: f"{v:.0f}%"))
    e = pd.read_csv(DERIVED.parent / "energy" / "inference_zeroshot.csv")
    print("  board power, every condition:", rng(e.power_mean, lambda v: f"{v:.0f} W"))
    d = [(ef[(b, r)] - ez[(b, r)]) / ez[(b, r)] * 100 for b in BB for r in RATIOS]
    print("  fine-tuned vs zero-shot energy per pass:", rng([-x for x in d], lambda v: f"{v:.2f}%"), "less")
    rv = pd.read_csv(DERIVED.parent / "energy" / "inference_zeroshot_reversed.csv")
    m = e.merge(rv, on=["backbone", "mask_ratio", "mask_type"], suffixes=("", "_rev"))
    m["ch"] = ((m.energy_mean_rev - m.energy_mean) / m.energy_mean).abs() * 100
    b = m[m.mask_type == "block"].ch
    print(f"  reversed order, contiguous conditions: energy changes by at most {b.max():.1f}% (median {b.median():.1f}%)"
          f" | all conditions incl. scattered: at most {m.ch.max():.1f}% (median {m.ch.median():.1f}%)")
    print("  training: time", [f2(tc[b]["h"]) for b in BB], "h, energy", [f2(tc[b]["mj"]) for b in BB], "MJ;",
          "per step x", rng([tc[BB[i + 1]]["mj"] / tc[BB[i]]["mj"] for i in range(3)], lambda v: f"{v:.1f}"),
          "| SD/mean: energy", f"{max(tc[b]['mj_sd'] / tc[b]['mj'] for b in BB) * 100:.1f}%",
          "time", f"{max(tc[b]['h_sd'] / tc[b]['h'] for b in BB) * 100:.1f}%")
    print(f"  fine-tuning tiny used {tc['tiny']['mj'] / 3.6:.2f} kWh; 600M {tc['600M']['mj'] / 3.6:.2f} kWh")
    closes = [(M.loc[("tiny", r), "ft_contig"] - M.loc[("tiny", r), "zs_contig"])
              / (M.loc[("100M", r), "zs_contig"] - M.loc[("tiny", r), "zs_contig"]) * 100 for r in RATIOS]
    closes_r = [(g("ft_contig_med", "tiny", r) - zc("tiny", r)) / (zc("100M", r) - zc("tiny", r)) * 100 for r in RATIOS]
    print("  fine-tuned tiny closes", rng(closes_r, lambda v: f"{v:.0f}%"), "of the gap to zero-shot 100M",
          f"(unrounded medians: {rng(closes, lambda v: f'{v:.0f}%')})",
          "at", rng([(1 - ez[('tiny', r)] / ez[('100M', r)]) * 100 for r in RATIOS], lambda v: f"{v:.0f}%"), "less energy")
    cx, nos, undef = crossover()
    t = cx[cx.backbone == "tiny"]
    o = cx[cx.backbone != "tiny"]
    print("  tiny: remaining gap", rng(t.excess), "dB; N*", rng(t.n_star / 1e3, lambda v: f"{v:.0f}k"),
          "=", rng(t.area_mkm2, lambda v: f"{v:.1f}"), "million km2 =", rng(t.area_mkm2 * 1e6 / CONUS_KM2, lambda v: f"{v:.2f}"), "x CONUS")
    print("  larger: crossover in", len(o), "cells", [(b, int(r * 100)) for b, r in zip(o.backbone, o.ratio)],
          "| energy ratio x", rng(1 / (1 - o.saving / 100), lambda v: f"{v:.1f}"), "| N*", rng(o.n_star / 1e3, lambda v: f"{v:.0f}k"))
    print("  same-size zero-shot already as good in", len(nos), "cells; undefined:", [(b, int(r * 100)) for b, r in undef])
    print("  reference margin: <0.01 dB in", int((cx.excess < 0.01).sum()), "cell, others", rng(cx.excess[cx.excess >= 0.01]))
    print(f"  one chip = {CHIP_KM2:.1f} km2; one pass over CONUS = {CONUS_KM2 / CHIP_KM2 / 1e3:.0f}k chips")


if __name__ == "__main__":
    main()
