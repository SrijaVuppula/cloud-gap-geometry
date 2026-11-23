"""Reconstruction quality (Section 4.4)."""
import numpy as np


def masked_psnr(recon, gt, pixel_mask):
    """PSNR (peak 1) over the six bands and the hidden pixels only.

    recon, gt: (BANDS, H, W) reflectance. pixel_mask: (H, W) bool, True = hidden.
    Visible pixels are excluded: they were given to the model as input.
    """
    m = pixel_mask.numpy() if hasattr(pixel_mask, "numpy") else pixel_mask
    if m.sum() == 0:
        return float("nan")
    mse = float(np.mean((recon[:, m] - gt[:, m]) ** 2))
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def patch_mse(recon, gt, patch_size):
    """Mean squared error per patch, (grid, grid)."""
    g = recon.shape[-1] // patch_size
    e = ((recon.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(0)
    return e.reshape(g, patch_size, g, patch_size).mean((1, 3))
