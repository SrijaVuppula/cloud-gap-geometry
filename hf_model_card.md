---
license: apache-2.0
base_model:
  - ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL
  - ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL
  - ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
  - ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL
datasets:
  - ibm-nasa-geospatial/multi-temporal-crop-classification
tags:
  - earth-observation
  - remote-sensing
  - cloud-gap-filling
  - masked-autoencoder
  - prithvi
  - hls
---

# Prithvi-EO-2.0 fine-tuned on contiguous masks

Checkpoints from **"Can Foundation Models Compensate for Incomplete Satellite Imagery? An Empirical Cost-Benefit Analysis"** (CV4EO workshop, WACV 2027).

Code, results and instructions: **https://github.com/SrijaVuppula/cloud-gap-geometry**

## What these are

All four Prithvi-EO-2.0 TL backbones (tiny, 100M, 300M, 600M), fine-tuned to fill **contiguous** gaps (one rectangular block of hidden patches) in the middle acquisition of a three-date HLS time series. They keep the pretraining objective (MSE on hidden patches) and architecture. Only the weights change: every encoder and decoder parameter is updated, and nothing is added or frozen. A checkpoint is a drop-in replacement for the released `Prithvi_EO_V2_*_TL.pt` of the same size.

| | |
|---|---|
| Data | HLS Multi-Temporal Crop Classification, 2,004 training / 50 validation chips (disjoint from and non-adjacent to the 500 evaluation chips) |
| Masks | one contiguous block on the summer frame, ratio uniform in 20–80% per sample |
| Optimiser | AdamW, lr 1e-5, weight decay 0.05, 200 warmup steps + cosine decay, grad clip 1.0 |
| Schedule | batch size 1, 20 epochs, best validation loss kept |
| Inputs | 3 acquisitions × 6 bands (Blue, Green, Red, NIR, SWIR1, SWIR2), 224 × 224, per-band normalisation from `config.json`; no date or location metadata |
| Hardware | one NVIDIA RTX 3090; 0.28 h (tiny) to 3.00 h (600M) per run |

## Layout

```
seed{N}/{tiny,100M,300M,600M}/best.pt
seed{N}/{tiny,100M,300M,600M}/config.json
```

Each seed is one of the ten fine-tuning runs in the paper. Runs differ in chip order and training masks; they start from the same weights.

## Use

```bash
git clone https://github.com/SrijaVuppula/cloud-gap-geometry && cd cloud-gap-geometry
pip install -r requirements.txt
python scripts/00_download.py --finetuned --seeds 1
python scripts/02_evaluate.py --backbone tiny --finetuned --seed 1
```

or directly:

```python
from huggingface_hub import hf_hub_download
from cloudgap.model import load_prithvi   # from the GitHub repo
ckpt = hf_hub_download("SrijaVuppula/cloud-gap-geometry-checkpoints", "seed1/tiny/best.pt")
cfg = hf_hub_download("SrijaVuppula/cloud-gap-geometry-checkpoints", "seed1/tiny/config.json")
model, mean, std, patch = load_prithvi(ckpt, cfg, num_frames=3)
```

## Results (median masked-region PSNR over 500 chips, contiguous occlusion, mean of 10 runs)

| Backbone | r = 20% | 40% | 60% | 80% | Gain over zero-shot |
|---|---|---|---|---|---|
| tiny | 32.93 | 32.25 | 31.85 | 31.50 | +0.78 to +1.05 dB |
| 100M | 33.79 | 32.98 | 32.50 | 32.21 | +0.02 to +0.14 dB |
| 300M | 34.19 | 33.38 | 32.89 | 32.52 | +0.11 to +0.31 dB |
| 600M | 34.55 | 33.89 | 33.45 | 32.71 | +0.02 to +0.17 dB |

Fine-tuning helps mainly the smallest backbone. Contiguous gaps stay 1.2–2.8 dB harder than the same number of scattered hidden patches after fine-tuning. See the paper for details and limitations: one year (2022), contiguous US, cropland-weighted, synthetic rectangular masks.

## Citation

```bibtex
@inproceedings{vuppula2027cloudgap,
  title     = {Can Foundation Models Compensate for Incomplete Satellite Imagery? An Empirical Cost-Benefit Analysis},
  author    = {Vuppula, Srija Yadav and Ramaswamy, Lakshmish and Mishra, Deepak and Maniyar, Chintan B. and Kumar, Abhishek and Kethavath, Harshith},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year      = {2027}
}
```
