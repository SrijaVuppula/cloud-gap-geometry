"""Fine-tune one Prithvi-EO-2.0 backbone on contiguous masks (Section 4.3).

Same objective as pretraining (MSE on hidden patches, normalised space).
Every training sample gets a new contiguous block on the summer frame with a
ratio drawn uniformly from 20-80%, so one model covers all evaluation ratios.
AdamW (lr 1e-5, weight decay 0.05), 200 warmup steps then cosine decay,
gradient clipping at 1.0, batch size 1, 20 epochs without early stopping.
The checkpoint with the lowest validation loss is kept. All encoder and
decoder weights are updated; nothing is frozen or added.

The seed sets the order of training chips and the sequence of training masks.
Total board energy of the whole job is logged (the training cost E_train).

  python scripts/03_finetune.py --backbone tiny --seed 1
  python scripts/03_finetune.py --backbone tiny --seed 1 --smoke   # 20 chips, 1 epoch

Writes weights/finetuned/seed{N}/{backbone}/best.pt (+ config.json) and
runs/training/seed{N}/{backbone}_{train_log,finetune_cost}.csv.
"""
import argparse
import csv
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cloudgap.config import REPO_ROOT, finetuned_weights, load_config, zeroshot_weights  # noqa: E402
from cloudgap.data import FRAME, IMG, T, normalize, read_chip, read_list  # noqa: E402
from cloudgap.energy import PowerSampler, gpu_handle, pynvml  # noqa: E402
from cloudgap.masking import build_block_noise_mask  # noqa: E402
from cloudgap.model import load_prithvi, masked_loss  # noqa: E402


class ChipDataset(Dataset):
    """Raw (T, BANDS, H, W) chips; normalisation is backbone-specific and done in the loop."""
    def __init__(self, names, chips_dir):
        self.paths = [Path(chips_dir) / n for n in names]
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} chips missing, e.g. {missing[0]}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(read_chip(self.paths[i]))


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validation_loss(model, val_raw, mean, std, patch, device, ratios=(0.2, 0.4, 0.6, 0.8)):
    """Validation masks are fixed: one per ratio, seeded by the ratio, so the loss is comparable across epochs."""
    model.eval()
    losses = []
    for raw in val_raw:
        x = normalize(raw.unsqueeze(0), mean, std).to(device)
        for r in ratios:
            noise, gr, _ = build_block_noise_mask(r, patch, IMG, T, FRAME, trial_seed=hash((r,)) % (2 ** 31))
            losses.append(float(masked_loss(model, x, gr, noise.unsqueeze(0).to(device))))
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["tiny", "100M", "300M", "600M"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="20 train chips, 5 val chips, 1 epoch")
    a = ap.parse_args()

    cfg = load_config()
    hp = cfg["finetune"]
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_names, val_names, epochs = read_list("finetune_train_2004.txt"), read_list("finetune_val_50.txt"), hp["epochs"]
    if a.smoke:
        train_names, val_names, epochs = train_names[:20], val_names[:5], 1
    chips_dir = cfg["paths"]["chips_dir"]
    train_ds = ChipDataset(train_names, chips_dir)
    val_raw = [ChipDataset(val_names, chips_dir)[i] for i in range(len(val_names))]
    print(f"{a.backbone} seed {a.seed}: {len(train_ds)} train chips, {len(val_raw)} val chips, {epochs} epochs, {device}")

    ckpt0, cfg_json = zeroshot_weights(cfg, a.backbone)
    model, mean, std, patch = load_prithvi(ckpt0, cfg_json, num_frames=T, device=device)
    model.train()

    loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,
                        num_workers=hp["num_workers"], drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    total_steps = max(1, epochs * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, hp["warmup_steps"], total_steps))

    best_path, _ = finetuned_weights(cfg, a.backbone, a.seed)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_json, best_path.parent / "config.json")
    log_dir = REPO_ROOT / "runs" / "training" / f"seed{a.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_dir / f"{a.backbone}_train_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "step", "train_loss", "val_loss", "lr", "elapsed_s"])

    sampler = None
    if pynvml is not None and torch.cuda.is_available():
        handle, _ = gpu_handle(0)
        sampler = PowerSampler(handle, cfg["energy"]["power_poll_s"]).start()

    lo, hi = hp["ratio_range"]
    rng = np.random.default_rng(a.seed)
    best_val, step, t0 = float("inf"), 0, time.perf_counter()
    for epoch in range(epochs):
        losses = []
        for raw in loader:
            ratio = float(rng.uniform(lo, hi))
            x = normalize(raw, mean, std).to(device)
            noises, grs = [], []
            for _ in range(x.shape[0]):
                noise, gr, _ = build_block_noise_mask(ratio, patch, IMG, T, FRAME, trial_seed=int(rng.integers(0, 2 ** 31)))
                noises.append(noise)
                grs.append(gr)
            loss = masked_loss(model, x, float(np.mean(grs)), torch.stack(noises).to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 500 == 0:
                print(f"  epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.perf_counter() - t0) / 60:.1f} min")

        val = validation_loss(model, val_raw, mean, std, patch, device)
        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), best_path)
        elapsed = time.perf_counter() - t0
        log.writerow([epoch, step, float(np.mean(losses)), val, sched.get_last_lr()[0], round(elapsed, 1)])
        log_f.flush()
        print(f"  epoch {epoch}: train {np.mean(losses):.4f} val {val:.4f} (best {best_val:.4f})")
    log_f.close()

    if sampler is not None:
        power = sampler.stop()
        elapsed = time.perf_counter() - t0
        with open(log_dir / f"{a.backbone}_finetune_cost.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["backbone", "epochs", "steps", "elapsed_s", "avg_power_w", "energy_kj"])
            w.writerow([a.backbone, epochs, step, round(elapsed, 1), round(power, 2), round(power * elapsed / 1000, 2)])
        print(f"training cost: {elapsed / 3600:.2f} h, {power:.1f} W, {power * elapsed / 1e6:.3f} MJ")
        pynvml.nvmlShutdown()
    print(f"best val loss {best_val:.4f}; checkpoint {best_path}")


if __name__ == "__main__":
    main()
