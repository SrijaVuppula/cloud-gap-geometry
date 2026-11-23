"""Tables 1-4 of the paper, as LaTeX (tables/*.tex) and Markdown (tables/*.md)."""
from common import BB, RATIOS, TABLES, crossover, f2, includes_zero, pass_energy, q2, seed_stats, training_cost

DAG = "†"


def num(x, dag=False, latex=True):
    q = q2(x)
    body = ("−" if q < 0 and not latex else "-" if q < 0 else "") + str(abs(q))
    if latex:
        body = f"${body}" + (r"^{\dagger}$" if dag else "$") if (q < 0 or dag) else body
    elif dag:
        body += DAG
    return body


def write(name, header, rows, latex_rows):
    TABLES.mkdir(exist_ok=True)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    (TABLES / f"{name}.md").write_text("\n".join(md) + "\n")
    (TABLES / f"{name}.tex").write_text("\n".join(" & ".join(r) + r" \\" for r in latex_rows) + "\n")
    print(f"\n{name}\n" + "\n".join(md))


def main():
    S = seed_stats()
    g = lambda col, b, r: float(S.loc[(b, r), col])
    ci0 = lambda name, b, r: includes_zero(g(f"{name}_lo", b, r), g(f"{name}_hi", b, r))
    pct = [f"{int(r * 100)}%" for r in RATIOS]

    # Table 1: zero-shot contiguous PSNR, scattered PSNR, geometry gap Delta
    rows = [[b] + [f2(g("zs_contig_med", b, r)) for r in RATIOS] + [f2(g("zs_scatt_med", b, r)) for r in RATIOS]
            + [f2(g("gap_zeroshot_med", b, r)) for r in RATIOS] for b in BB]
    write("table1_zeroshot",
          ["Backbone"] + [f"Contig. {p}" for p in pct] + [f"Scatt. {p}" for p in pct] + [f"Δ {p}" for p in pct],
          rows, rows)

    # Table 2: geometry gap after fine-tuning, and its change from zero-shot
    md, tex = [], []
    for b in BB:
        fd = [f2(g("gap_finetuned_med", b, r)) for r in RATIOS]
        md.append([b] + fd + [num(g("dgap_med", b, r), ci0("dgap", b, r), latex=False) for r in RATIOS])
        tex.append([b] + fd + [num(g("dgap_med", b, r), ci0("dgap", b, r)) for r in RATIOS])
    write("table2_finetuned_gap", ["Backbone"] + [f"FT Δ {p}" for p in pct] + [f"Change in Δ {p}" for p in pct], md, tex)

    # Table 3: energy per pass (released weights, contiguous) and one-time training cost
    ez, tc = pass_energy(), training_cost()
    rows = [[b, f2(ez[(b, 0.2)]), f2(ez[(b, 0.8)]), f2(tc[b]["h"]), f2(tc[b]["mj"])] for b in BB]
    write("table3_cost", ["Model", "J/pass r=20%", "J/pass r=80%", "Training time (h)", "Training energy (MJ)"], rows, rows)

    # Table 4: crossover point N*
    cx, _, _ = crossover()
    rows = []
    for x in cx.itertuples():
        ex = "<0.01" if x.excess < 0.01 else f"{x.excess:.2f}"
        rows.append([x.backbone, str(int(x.ratio * 100)), x.reference, ex, f"{x.saving:.1f}", f"{x.n_star / 1e3:.0f}"])
    tex = [[c.replace("<0.01", "$<$0.01") for c in r] for r in rows]
    write("table4_crossover", ["Fine-tuned", "r (%)", "Zero-shot reference", "Excess (dB)", "Saving (%)", "N* (10^3)"], rows, tex)

    # Figure 1c / Section 5.2: fine-tuning gain G (shown as a heat map in the paper)
    md = [[b] + [num(g("G_contiguous_med", b, r), ci0("G_contiguous", b, r), latex=False) for r in RATIOS]
          + [num(g("G_scattered_med", b, r), ci0("G_scattered", b, r), latex=False) for r in RATIOS] for b in BB]
    tex = [[b] + [num(g("G_contiguous_med", b, r), ci0("G_contiguous", b, r)) for r in RATIOS]
           + [num(g("G_scattered_med", b, r), ci0("G_scattered", b, r)) for r in RATIOS] for b in BB]
    write("fig1c_gain", ["Backbone"] + [f"G con {p}" for p in pct] + [f"G sct {p}" for p in pct], md, tex)
    print(f"\n{DAG} = 95% bootstrap interval includes zero")


if __name__ == "__main__":
    main()
