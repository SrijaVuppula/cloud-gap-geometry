"""Masking protocol checks (CPU only; needs torch)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cloudgap.masking import build_paired_masks, hidden_grid  # noqa: E402

RATIOS = [0.2, 0.4, 0.6, 0.8]


@pytest.mark.parametrize("patch", [16, 14])
@pytest.mark.parametrize("ratio", RATIOS)
def test_only_summer_hidden_and_counts_match(patch, ratio):
    ppf = (224 // patch) ** 2
    for seed in range(20):
        m = build_paired_masks(ratio, patch, seed)
        (nb, grb, ib), (nr, grr, ir) = m["contiguous"], m["scattered"]
        assert len(ib) == len(ir)                                    # same number hidden
        assert grb == grr == len(ib) / (3 * ppf)                     # global ratio passed to the encoder
        for idx, noise in ((ib, nb), (ir, nr)):
            assert ((idx >= ppf) & (idx < 2 * ppf)).all()            # summer only
            assert (noise[idx] == 1.0).all() and (noise[noise != 1.0] < 0.9).all()
        assert abs(len(ib) / ppf - ratio) <= 0.0125 + 1e-9           # within 1.25 points of r


def test_block_is_one_rectangle():
    g = hidden_grid(build_paired_masks(0.4, 16, 7)["contiguous"][2], 16)
    rows, cols = np.where(g)
    assert g[rows.min():rows.max() + 1, cols.min():cols.max() + 1].all()


def test_replays_logged_masks():
    """trial_seed reproduces the logged patch counts and the Figure 1a masks exactly."""
    seeds = pd.read_csv(ROOT / "splits" / "trial_seeds.csv").head(400)
    logged = pd.read_csv(ROOT / "results" / "zeroshot" / "results_tiny.csv").head(400)
    for (_, s), (_, l) in zip(seeds.iterrows(), logged.iterrows()):
        assert len(build_paired_masks(s.mask_ratio, 16, int(s.trial_seed))["contiguous"][2]) == l.summer_masked
    z = np.load(ROOT / "results" / "figure_assets" / "fig1a_masks.npz")
    m = build_paired_masks(float(z["ratio"]), int(z["patch"]), int(z["trial_seed"]))
    assert (hidden_grid(m["contiguous"][2], 16) == z["contiguous"]).all()
    assert (hidden_grid(m["scattered"][2], 16) == z["scattered"]).all()
