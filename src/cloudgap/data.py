"""HLS chip loading and normalisation (Section 4.1).

Each chip is a GeoTIFF with 18 bands: three acquisitions (spring, summer,
fall) x six HLS bands (Blue, Green, Red, NIR, SWIR1, SWIR2), in HLS digital
numbers. Surface reflectance = DN / 10^4, clipped to [0, 1].
"""
import csv
from pathlib import Path

import numpy as np
import rasterio
import torch

from .config import REPO_ROOT

IMG, BANDS, T, FRAME = 224, 6, 3, 1
SCALE = 10000.0
SPLITS = REPO_ROOT / "splits"


def read_chip(path):
    """Raw chip as float32 array (T, BANDS, IMG, IMG), HLS digital numbers."""
    with rasterio.open(path) as src:
        data = src.read()
    return data.reshape(T, BANDS, IMG, IMG).astype(np.float32)


def normalize(raw, mean, std):
    """(..., T, BANDS, H, W) raw tensor -> (..., BANDS, T, H, W) model input.
    The released mean/std are on the raw DN scale."""
    raw = torch.as_tensor(raw, dtype=torch.float32)
    shape = [1] * raw.dim()
    shape[-3] = BANDS
    m = torch.tensor(mean[:BANDS], dtype=torch.float32).reshape(shape)
    s = torch.tensor(std[:BANDS], dtype=torch.float32).reshape(shape)
    return ((raw - m) / s).transpose(-4, -3)


def load_eval_chip(path, mean, std, device="cpu"):
    """Model input (1, BANDS, T, H, W) and summer reflectance target (BANDS, H, W)."""
    raw = read_chip(path)
    gt = np.clip(raw[FRAME] / SCALE, 0.0, 1.0)
    x = normalize(raw, mean, std).unsqueeze(0).to(device)
    return x, gt


def to_reflectance(rec, mean, std, frame=FRAME):
    """Normalised reconstruction (1, BANDS, T, H, W) -> reflectance of one frame (BANDS, H, W)."""
    pred = rec[0, :, frame]
    m = torch.tensor(mean[:BANDS]).reshape(-1, 1, 1)
    s = torch.tensor(std[:BANDS]).reshape(-1, 1, 1)
    return torch.clamp((pred * s + m) / SCALE, 0.0, 1.0).numpy()


def read_list(name):
    return (SPLITS / name).read_text().split()


def eval_chips(chips_dir):
    """The 500 evaluation chips (splits/eval_500.txt), as paths."""
    paths = [Path(chips_dir) / n for n in read_list("eval_500.txt")]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} evaluation chips missing, e.g. {missing[0]}. "
                                "Run scripts/00_download.py first.")
    return paths


def trial_seeds():
    """{(chip, ratio, trial): seed} from splits/trial_seeds.csv.

    The seeds were drawn once (with Python's per-process string hash) and
    logged. Every evaluation, zero-shot and fine-tuned, replays them, so all
    configurations see identical masks."""
    out = {}
    with open(SPLITS / "trial_seeds.csv") as f:
        for row in csv.DictReader(f):
            out[(row["chip"], round(float(row["mask_ratio"]), 2), int(row["trial"]))] = int(row["trial_seed"])
    return out
