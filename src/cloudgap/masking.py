"""Contiguous and scattered occlusion of the summer acquisition (Section 4.2).

Both geometries hide only patches of the masked frame (summer), so spring and
fall stay fully visible, and both hide exactly the same number of patches.
The only difference between them is how the hidden patches are arranged.

How this works with TerraTorch: `random_masking` keeps the tokens with the
lowest noise, and its token budget is global over all frames. We therefore
give every token we want to keep a noise value in [0, 0.9) and every token we
want to hide a value of 1.0, and pass the *global* ratio
(n_hidden / total_tokens) to the encoder. TerraTorch then keeps exactly the
tokens we intended. Always pass the returned `global_ratio`, never the nominal
mask ratio, to `cloudgap.model.run_masked_forward`.

The random number streams are unchanged from the code that produced the paper,
so a given `trial_seed` reproduces the logged masks exactly.
"""
import numpy as np
import torch


def _grid(img_size, patch_size):
    g = img_size // patch_size
    return g, g


def _find_block_dims(n_target, grid_h, grid_w):
    """Rectangle (h, w) of whole patches whose area is closest to n_target."""
    best_h, best_w = 1, min(n_target, grid_w)
    best_diff = abs(best_h * best_w - n_target)
    for h in range(1, grid_h + 1):
        w = max(1, min(round(n_target / h), grid_w))
        diff = abs(h * w - n_target)
        if diff < best_diff or (diff == best_diff and abs(h - w) < abs(best_h - best_w)):
            best_h, best_w, best_diff = h, w, diff
    return best_h, best_w


def _base_noise(total_tokens, rng):
    # every token starts as guaranteed-keep: strictly below the 1.0 mask value
    return torch.rand(total_tokens, generator=rng) * 0.9


def _frame_slice(patch_size, img_size, num_frames, frame_idx):
    grid_h, grid_w = _grid(img_size, patch_size)
    ppf = grid_h * grid_w
    total = num_frames * ppf
    offset = frame_idx * ppf
    return grid_h, grid_w, ppf, total, offset


def build_block_noise_mask(mask_ratio, patch_size=16, img_size=224,
                           num_frames=3, frame_idx=1, trial_seed=None):
    """Contiguous geometry: one rectangle of whole patches at a random position.

    Returns (noise, global_ratio, masked_global_idx).
    """
    rng = torch.Generator()
    if trial_seed is not None:
        rng.manual_seed(trial_seed)

    grid_h, grid_w, ppf, total, offset = _frame_slice(patch_size, img_size, num_frames, frame_idx)

    n_target = max(1, min(int(round(mask_ratio * ppf)), ppf - 1))
    bh, bw = _find_block_dims(n_target, grid_h, grid_w)
    r0 = int(torch.randint(0, grid_h - bh + 1, (1,), generator=rng).item())
    c0 = int(torch.randint(0, grid_w - bw + 1, (1,), generator=rng).item())

    grid = torch.zeros(grid_h, grid_w, dtype=torch.bool)
    grid[r0:r0 + bh, c0:c0 + bw] = True
    masked_global = grid.flatten().nonzero(as_tuple=True)[0] + offset

    noise = _base_noise(total, rng)
    noise[masked_global] = 1.0
    return noise, len(masked_global) / total, masked_global


def build_random_noise_mask(mask_ratio, patch_size=16, img_size=224,
                            num_frames=3, frame_idx=1, trial_seed=None,
                            n_summer_masked=None):
    """Scattered geometry: patches at random positions in the masked frame.

    Pass `n_summer_masked=len(block_idx)` to hide exactly as many patches as
    the paired contiguous trial (this is what the paper does).
    Returns (noise, global_ratio, masked_global_idx).
    """
    rng = torch.Generator()
    if trial_seed is not None:
        rng.manual_seed(trial_seed)

    grid_h, grid_w, ppf, total, offset = _frame_slice(patch_size, img_size, num_frames, frame_idx)

    n = n_summer_masked if n_summer_masked is not None \
        else max(1, min(int(round(mask_ratio * ppf)), ppf - 1))

    masked_global = torch.randperm(ppf, generator=rng)[:n] + offset

    noise = _base_noise(total, rng)
    noise[masked_global] = 1.0
    return noise, len(masked_global) / total, masked_global


def build_paired_masks(mask_ratio, patch_size, trial_seed, img_size=224, num_frames=3, frame_idx=1):
    """Contiguous and scattered masks of one trial, with matched patch counts."""
    block = build_block_noise_mask(mask_ratio, patch_size, img_size, num_frames, frame_idx, trial_seed=trial_seed)
    scattered = build_random_noise_mask(mask_ratio, patch_size, img_size, num_frames, frame_idx,
                                        trial_seed=trial_seed, n_summer_masked=len(block[2]))
    return {"contiguous": block, "scattered": scattered}


def pixel_map(masked_global, patch_size=16, img_size=224, num_frames=3, frame_idx=1):
    """Boolean (img_size, img_size) map of the hidden pixels in the masked frame."""
    grid_h, grid_w = _grid(img_size, patch_size)
    ppf = grid_h * grid_w
    local = masked_global - frame_idx * ppf
    local = local[(local >= 0) & (local < ppf)]
    flat = torch.zeros(ppf, dtype=torch.bool)
    flat[local] = True
    grid = flat.reshape(grid_h, grid_w)
    return grid.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)[:img_size, :img_size]


def hidden_grid(masked_global, patch_size=16, img_size=224, frame_idx=1):
    """Boolean (grid, grid) map of hidden patches in the masked frame."""
    g = img_size // patch_size
    local = masked_global.numpy() - frame_idx * g * g
    local = local[(local >= 0) & (local < g * g)]
    h = np.zeros(g * g, bool)
    h[local] = True
    return h.reshape(g, g)
