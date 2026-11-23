"""Figure 3: crossover of fine-tuned tiny against zero-shot 100M at r = 40%.

Cumulative energy against number of inferences:
  fine-tuned tiny = E_train(tiny) + N * E_pass(tiny)   (fine-tuned energy per pass = zero-shot, Section 5.3)
  zero-shot 100M  = N * E_pass(100M)
Shaded band: N* over r = 20-80% (Table 4). Dotted line: one pass over the contiguous US.
Reads results/ (training cost, energy) and results/derived/. Writes figures/fig3_crossover.{pdf,png}.
"""
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CHIP_KM2, CONUS_KM2, FIGURES, INK, PLASMA, RC, crossover, pass_energy, training_cost  # noqa: E402

mpl.rcParams.update(RC)

e_train = training_cost()["tiny"]["mj"] * 1e6          # J, mean of ten runs
ez = pass_energy()
e_tiny, e_100m = ez[("tiny", 0.4)], ez[("100M", 0.4)]
cx, _, _ = crossover()
band = cx[(cx.backbone == "tiny") & (cx.reference == "100M")].n_star
nstar = e_train / (e_100m - e_tiny)
conus_chips = CONUS_KM2 / CHIP_KM2

fig, ax = plt.subplots(figsize=(3.35, 2.05))
fig.subplots_adjust(left=0.135, right=0.97, bottom=0.19, top=0.80)
N = np.linspace(0, 320e3, 400)
ax.axvspan(band.min() / 1e3, band.max() / 1e3, color=PLASMA[0], alpha=0.10, lw=0, zorder=0)
ax.plot(N / 1e3, N * e_100m / 1e6, color=PLASMA[1], lw=1.4, ls=(0, (4, 2)), zorder=2, label="zero-shot 100M")
ax.plot(N / 1e3, (e_train + N * e_tiny) / 1e6, color=PLASMA[0], lw=1.6, zorder=3, label="fine-tuned tiny (incl. training)")
y_star = (e_train + nstar * e_tiny) / 1e6
ax.scatter([nstar / 1e3], [y_star], s=16, color=PLASMA[0], zorder=4)
ax.annotate(f"$N^*$ = {nstar / 1e3:.0f}k", xy=(nstar / 1e3, y_star), xytext=(nstar / 1e3 + 14, 0.47), fontsize=7,
            color=PLASMA[0], fontweight="bold", arrowprops=dict(arrowstyle="-", color=PLASMA[0], lw=0.6))
ax.axvline(conus_chips / 1e3, color="#8A8A8A", lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
ax.text(conus_chips / 1e3 - 4, 0.05, "one pass over\nthe contiguous US", ha="right", va="bottom", fontsize=6.3,
        style="italic", color="#555555", linespacing=1.1)
ax.text((band.min() + band.max()) / 2e3, 0.97, "$N^*$, $r$ = 20–80%", ha="center", va="top", fontsize=6.3, color=PLASMA[0])
ax.text(8, e_train / 1e6 - 0.035, "training energy", ha="left", va="top", fontsize=6.3, style="italic", color="#555555")

ax.set_xlim(0, 320)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Inferences (thousands)", labelpad=2)
ax.set_ylabel("Cumulative energy (MJ)", labelpad=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, linestyle=(0, (1, 2)))
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=2.2, borderaxespad=0.2)
top = ax.secondary_xaxis("top", functions=(lambda k: k * 1e3 * CHIP_KM2 / 1e6, lambda m: m * 1e6 / CHIP_KM2 / 1e3))
top.set_xlabel("Area at one pass per chip (million km$^2$)", labelpad=3, color=INK)
top.set_xticks([0, 4, 8, 12])
top.tick_params(labelsize=7)

FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "fig3_crossover.pdf")
fig.savefig(FIGURES / "fig3_crossover.png", dpi=300)
print(f"N* at r=40%: {nstar:,.0f}; band {band.min():,.0f}-{band.max():,.0f}; one pass over CONUS = {conus_chips:,.0f} chips")
