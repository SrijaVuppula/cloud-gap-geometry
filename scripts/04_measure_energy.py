"""Energy and time per forward pass (Section 4.5).

For each backbone, mask ratio and geometry: 3 s of warmup passes so the GPU
reaches a steady state, then 3 untimed + 20 timed passes at batch size 1
while a background thread polls board power every 20 ms. Energy per pass =
mean board power x mean pass time (encoder + decoder, no data loading).
The work of a pass depends only on the number of visible tokens, so all
passes use the same chip.

  python scripts/04_measure_energy.py                         # released weights
  python scripts/04_measure_energy.py --order reversed        # repeat in reverse order (uncertainty check)
  python scripts/04_measure_energy.py --finetuned --seed 1    # fine-tuned weights

Writes runs/energy/inference_{zeroshot|finetuned}[_reversed].csv (energy in mJ).
Absolute values depend on the GPU; the paper used one RTX 3090 (350 W cap).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cloudgap import BACKBONES  # noqa: E402
from cloudgap.config import REPO_ROOT, finetuned_weights, load_config, zeroshot_weights  # noqa: E402
from cloudgap.data import FRAME, IMG, T, load_eval_chip  # noqa: E402
from cloudgap.energy import PowerSampler, gpu_handle, pynvml  # noqa: E402
from cloudgap.masking import build_block_noise_mask, build_random_noise_mask  # noqa: E402
from cloudgap.model import encode_with_noise, load_prithvi  # noqa: E402

DEVICE = "cuda"


def noise_fn(geometry, ratio, patch, n_match):
    def f(seed):
        if geometry == "block":
            noise, gr, idx = build_block_noise_mask(ratio, patch, IMG, T, FRAME, trial_seed=seed)
        else:
            noise, gr, idx = build_random_noise_mask(ratio, patch, IMG, T, FRAME, trial_seed=seed, n_summer_masked=n_match)
        return noise.unsqueeze(0).to(DEVICE), gr, len(idx)
    return f


@torch.no_grad()
def one_pass(model, x, gr, noise):
    latent, _, ids_restore = encode_with_noise(model, x, gr, noise)
    model.decoder(latent, ids_restore, None, None, input_size=x.shape)
    return latent.shape[1]


@torch.no_grad()
def measure(model, x, fn, handle, e):
    t_end, s = time.perf_counter() + e["gpu_warmup_s"], 0
    while time.perf_counter() < t_end:          # steady clocks before every condition
        noise, gr, _ = fn(s)
        s += 1
        one_pass(model, x, gr, noise)
        torch.cuda.synchronize()

    sampler = PowerSampler(handle, e["power_poll_s"]).start()
    times, enc, dec, tokens = [], [], [], None
    for i in range(e["n_warmup_passes"] + e["n_timed_passes"]):
        noise, gr, n_masked = fn(1000 + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        latent, _, ids_restore = encode_with_noise(model, x, gr, noise)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        model.decoder(latent, ids_restore, None, None, input_size=x.shape)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        tokens = int(latent.shape[1])
        if i >= e["n_warmup_passes"]:
            times.append((t2 - t0) * 1000)
            enc.append((t1 - t0) * 1000)
            dec.append((t2 - t1) * 1000)
    power = sampler.stop()
    energy = [power * t for t in times]          # W x ms = mJ
    return times, enc, dec, power, energy, n_masked, tokens


def stats(v):
    t = torch.tensor(v, dtype=torch.float64)
    return float(t.mean()), float(t.min()), float(t.max()), float(t.std(unbiased=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", action="store_true")
    ap.add_argument("--seed", type=int, default=1, help="fine-tuning run, with --finetuned")
    ap.add_argument("--order", choices=["forward", "reversed"], default="forward")
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    a = ap.parse_args()

    cfg = load_config()
    e = cfg["energy"]
    handle, gpu = gpu_handle(0)
    print(f"GPU: {gpu}")
    chip = cfg["paths"]["chips_dir"] / e["chip"]
    backbones, ratios, geometries = list(a.backbones), list(e["ratios"]), ["random", "block"]
    if a.order == "reversed":
        backbones, ratios, geometries = backbones[::-1], ratios[::-1], geometries[::-1]

    name = "finetuned" if a.finetuned else "zeroshot"
    out = REPO_ROOT / "runs" / "energy" / f"inference_{name}{'_reversed' if a.order == 'reversed' else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for bb in backbones:
        ckpt, cj = finetuned_weights(cfg, bb, a.seed) if a.finetuned else zeroshot_weights(cfg, bb)
        model, mean, std, patch = load_prithvi(ckpt, cj, num_frames=T, device=DEVICE)
        x, _ = load_eval_chip(chip, mean, std, DEVICE)
        n_tokens = T * (IMG // patch) ** 2
        for r in ratios:
            n_match = len(build_block_noise_mask(r, patch, IMG, T, FRAME, trial_seed=0)[2])
            for geo in geometries:
                tm, en, de, pw, ej, n_masked, tok = measure(model, x, noise_fn(geo, r, patch, n_match), handle, e)
                row = {"backbone": bb, "mask_ratio": r, "mask_type": geo, "summer_masked": n_masked,
                       "total_tokens": n_tokens, "tokens_encoded": tok, "global_ratio": round(n_masked / n_tokens, 4)}
                for k, v in (("time", tm), ("enc", en), ("dec", de), ("power", [pw] * len(tm)), ("energy", ej)):
                    for s, val in zip(("mean", "min", "max", "std"), stats(v)):
                        row[f"{k}_{s}"] = round(val, 4)
                rows.append(row)
                print(f"  {bb:4s} r={int(r * 100):2d}% {geo:6s} {row['time_mean']:7.2f} ms {pw:5.1f} W {row['energy_mean']:8.1f} mJ")
        del model
        torch.cuda.empty_cache()
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    pynvml.nvmlShutdown()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
