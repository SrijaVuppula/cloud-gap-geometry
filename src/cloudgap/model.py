"""Load Prithvi-EO-2.0 through TerraTorch and run a forward pass with a chosen mask.

TerraTorch's PrithviViT.forward() does not accept a noise tensor, so the
encoder forward pass is reproduced here with our noise injected into
`random_masking`. Acquisition date and location are never passed (Section 4.3),
so the temporal and location encodings are skipped.
"""
import json
from pathlib import Path

import torch
from terratorch.models.backbones.prithvi_mae import PrithviMAE

# Decoder settings shared by all four released TL checkpoints.
_DECODER = dict(decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0)


def read_backbone_config(config_path):
    """Architecture and per-band normalisation statistics from a checkpoint's config.json."""
    with open(config_path) as f:
        raw = json.load(f)
    cfg = raw.get("pretrained_cfg", raw)
    return {
        "embed_dim": cfg["embed_dim"],
        "depth": cfg["depth"],
        "num_heads": cfg["num_heads"],
        "patch_size": tuple(cfg["patch_size"]),
        "mean": cfg["mean"],
        "std": cfg["std"],
        "bands": cfg["bands"],
    }


def load_prithvi(checkpoint_path, config_path, num_frames=3, device="cpu", img_size=224):
    """Build PrithviMAE and load a checkpoint (released or fine-tuned; same architecture).

    Returns (model, mean, std, spatial_patch_size).
    """
    cfg = read_backbone_config(config_path)
    model = PrithviMAE(
        img_size=img_size,
        patch_size=cfg["patch_size"],
        num_frames=num_frames,
        in_chans=len(cfg["bands"]),
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        coords_encoding=["time", "location"],
        **_DECODER,
    ).to(device)

    try:
        state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    except TypeError:  # older torch
        state = torch.load(checkpoint_path, map_location=device)
    # Positional embeddings are fixed sinusoids; use the ones built for num_frames.
    if "encoder.pos_embed" in state:
        state["encoder.pos_embed"] = model.encoder.pos_embed
    if "decoder.decoder_pos_embed" in state:
        state["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg["mean"], cfg["std"], cfg["patch_size"][1]


def encode_with_noise(model, x, mask_ratio, noise):
    """Encoder forward pass with our noise tensor. No torch.no_grad here, so
    the same function is used for fine-tuning; wrap it for inference."""
    enc = model.encoder
    x_enc = enc.patch_embed(x)
    pos_embed = enc.interpolate_pos_encoding(x.shape[-3:])
    x_enc = x_enc + pos_embed[:, 1:, :]
    x_enc, mask, ids_restore = enc.random_masking(x_enc, mask_ratio, noise=noise)
    cls = (enc.cls_token + pos_embed[:, :1, :]).expand(x_enc.shape[0], -1, -1)
    x_enc = torch.cat((cls, x_enc), dim=1)
    for block in enc.blocks:
        x_enc = block(x_enc)
    return enc.norm(x_enc), mask, ids_restore


def masked_loss(model, x, mask_ratio, noise):
    """Pretraining objective: MSE on hidden patches, in normalised space (used for fine-tuning)."""
    latent, mask, ids_restore = encode_with_noise(model, x, mask_ratio, noise)
    pred = model.decoder(latent, ids_restore, None, None, input_size=x.shape)
    return model.forward_loss(x, pred, mask)


@torch.no_grad()
def run_masked_forward(model, x, mask_ratio, noise):
    """Reconstruct the hidden patches.

    x: (1, C, T, H, W) normalised input. noise: (1, n_tokens).
    Returns the composite reconstruction (visible input + predicted hidden
    patches), normalised, on CPU, shape (1, C, T, H, W).
    """
    h, w = x.shape[-2:]
    latent, mask, ids_restore = encode_with_noise(model, x, mask_ratio, noise)
    pred = model.decoder(latent, ids_restore, None, None, input_size=x.shape)
    pred_img = model.unpatchify(pred.detach().cpu(), image_size=(h, w))
    mask_img = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1]).detach().cpu(), image_size=(h, w))
    rec = x.detach().cpu().clone()
    rec[mask_img == 1] = pred_img[mask_img == 1]
    return rec
