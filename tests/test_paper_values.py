"""The released results reproduce the numbers printed in the paper (no GPU needed).

  pytest tests/
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import BB, RATIOS, crossover, f2, includes_zero, pass_energy, seed_stats, training_cost  # noqa: E402

S = seed_stats()

# Table 1: zero-shot contiguous PSNR, scattered PSNR, Delta
TABLE1 = {
    "tiny": ([31.85, 31.32, 30.97, 30.59], [34.64, 34.33, 33.68, 32.62], [2.43, 2.69, 2.47, 1.81]),
    "100M": ([33.86, 33.02, 32.54, 32.06], [36.42, 35.81, 35.05, 33.69], [2.19, 2.49, 2.20, 1.60]),
    "300M": ([34.02, 33.22, 32.90, 32.39], [36.82, 36.32, 35.47, 34.15], [2.53, 2.67, 2.33, 1.64]),
    "600M": ([34.52, 33.93, 33.52, 32.53], [37.70, 37.35, 36.37, 34.86], [2.88, 3.00, 2.63, 2.08]),
}
# Table 2: fine-tuned Delta and change in Delta
TABLE2 = {
    "tiny": ([1.88, 1.95, 1.67, 1.20], [-0.55, -0.61, -0.63, -0.54]),
    "100M": ([2.08, 2.24, 1.89, 1.25], [-0.13, -0.22, -0.26, -0.26]),
    "300M": ([2.18, 2.35, 2.03, 1.27], [-0.36, -0.34, -0.31, -0.31]),
    "600M": ([2.82, 2.80, 2.29, 1.66], [-0.01, -0.20, -0.33, -0.43]),
}
# Table 3: J/pass at r = 20%, 80%; training hours; training MJ
TABLE3 = {"tiny": (1.77, 1.69, 0.28, 0.33), "100M": (3.54, 3.06, 0.55, 0.69),
          "300M": (7.54, 6.86, 1.32, 1.64), "600M": (19.76, 16.15, 3.00, 3.74)}
# Table 4: backbone, r, reference, excess (dB), saving (%), N* (10^3)
TABLE4 = [("tiny", 20, "100M", "0.93", "50.0", 188), ("tiny", 40, "100M", "0.77", "45.8", 227),
          ("tiny", 60, "100M", "0.69", "45.5", 231), ("tiny", 80, "100M", "0.56", "44.9", 243),
          ("100M", 80, "300M", "0.17", "55.4", 182), ("300M", 20, "600M", "0.33", "61.9", 134),
          ("300M", 40, "600M", "0.55", "63.2", 132), ("300M", 80, "600M", "<0.01", "57.5", 177)]
# Figure 1c: fine-tuning gain G (contiguous, scattered)
FIG1C = {
    "tiny": ([1.05, 0.86, 0.80, 0.78], [0.50, 0.25, 0.15, 0.20]),
    "100M": ([0.02, 0.03, 0.06, 0.14], [-0.14, -0.20, -0.21, -0.12]),
    "300M": ([0.31, 0.16, 0.11, 0.18], [-0.04, -0.18, -0.25, -0.19]),
    "600M": ([0.06, 0.02, 0.06, 0.17], [0.01, -0.20, -0.32, -0.23]),
}
FIG1C_DAGGER = {("100M", 0.2, "c"), ("100M", 0.4, "c"), ("600M", 0.2, "c"), ("600M", 0.4, "c"), ("600M", 0.2, "s")}


def cell(col, b, r):
    return float(f2(S.loc[(b, r), col]))


@pytest.mark.parametrize("b", BB)
def test_table1(b):
    con, sct, gap = TABLE1[b]
    assert [cell("zs_contig_med", b, r) for r in RATIOS] == con
    assert [cell("zs_scatt_med", b, r) for r in RATIOS] == sct
    assert [cell("gap_zeroshot_med", b, r) for r in RATIOS] == gap


@pytest.mark.parametrize("b", BB)
def test_table2(b):
    ftgap, change = TABLE2[b]
    assert [cell("gap_finetuned_med", b, r) for r in RATIOS] == ftgap
    assert [cell("dgap_med", b, r) for r in RATIOS] == change


def test_table3():
    ez, tc = pass_energy(), training_cost()
    for b, (e20, e80, h, mj) in TABLE3.items():
        assert (float(f2(ez[(b, 0.2)])), float(f2(ez[(b, 0.8)])), float(f2(tc[b]["h"])), float(f2(tc[b]["mj"]))) == (e20, e80, h, mj)
        assert tc[b]["n"] == 10


def test_table4():
    cx, no_saving, undefined = crossover()
    got = [(x.backbone, int(x.ratio * 100), x.reference, "<0.01" if x.excess < 0.01 else f"{x.excess:.2f}",
            f"{x.saving:.1f}", round(x.n_star / 1e3)) for x in cx.itertuples()]
    assert got == TABLE4
    assert len(no_saving) == 6 and undefined == [("600M", 0.2), ("600M", 0.8)]


@pytest.mark.parametrize("b", BB)
def test_fig1c(b):
    con, sct = FIG1C[b]
    assert [cell("G_contiguous_med", b, r) for r in RATIOS] == con
    assert [cell("G_scattered_med", b, r) for r in RATIOS] == sct
    for r in RATIOS:
        for k, name in (("c", "G_contiguous"), ("s", "G_scattered")):
            assert includes_zero(S.loc[(b, r), f"{name}_lo"], S.loc[(b, r), f"{name}_hi"]) == ((b, r, k) in FIG1C_DAGGER)


def test_headline_numbers():
    gz = [S.loc[(b, r), "gap_zeroshot_med"] for b in BB for r in RATIOS]
    assert (f2(min(gz)), f2(max(gz))) == ("1.60", "3.00")                         # 1.6-3.0 dB
    assert min(S.loc[(b, r), "gap_zeroshot_pos"] for b in BB for r in RATIOS) >= 0.986
    closes = [(S.loc[("tiny", r), "ft_contig_med"] - S.loc[("tiny", r), "zs_contig_med"])
              / (S.loc[("100M", r), "zs_contig_med"] - S.loc[("tiny", r), "zs_contig_med"]) for r in RATIOS]
    assert (round(min(closes) * 100), round(max(closes) * 100)) == (54, 62)       # closes 54-62% of the gap
