"""Download the HLS chips and the Prithvi-EO-2.0 weights.

  python scripts/00_download.py                 # data + released weights (zero-shot)
  python scripts/00_download.py --finetuned     # also our fine-tuned checkpoints
  python scripts/00_download.py --skip-data     # weights only

Data: IBM-NASA HLS Multi-Temporal Crop Classification dataset (Hugging Face).
Only the training-split archive is needed (10.6 GB); from it we extract the
chips listed in splits/ (500 evaluation + 2,054 fine-tuning chips, merged
GeoTIFFs only), so the extracted folder is about 2,554 files.
"""
import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from huggingface_hub import hf_hub_download  # noqa: E402

from cloudgap import BACKBONES  # noqa: E402
from cloudgap.config import load_config  # noqa: E402
from cloudgap.data import read_list  # noqa: E402


def download_data(cfg, keep_archive):
    chips_dir = cfg["paths"]["chips_dir"]
    chips_dir.mkdir(parents=True, exist_ok=True)
    repo = cfg["data"]["hf_dataset"]
    # training_data.txt lists every chip of the training split; 01_make_splits.py uses it.
    hf_hub_download(repo, "training_data.txt", repo_type="dataset", local_dir=chips_dir.parent)

    wanted = set(read_list("eval_500.txt") + read_list("finetune_train_2004.txt") + read_list("finetune_val_50.txt"))
    have = {p.name for p in chips_dir.glob("chip_*_merged.tif")}
    if wanted <= have:
        print(f"All {len(wanted)} chips already in {chips_dir}")
        return

    print("Downloading training_chips.tgz (10.6 GB) ...")
    archive = Path(hf_hub_download(repo, "training_chips.tgz", repo_type="dataset", local_dir=chips_dir.parent))
    print(f"Extracting {len(wanted)} chips to {chips_dir} ...")
    n = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in wanted:
                member.name = name  # flatten
                tar.extract(member, chips_dir, filter="data") if sys.version_info >= (3, 12) \
                    else tar.extract(member, chips_dir)
                n += 1
    missing = wanted - {p.name for p in chips_dir.glob("chip_*_merged.tif")}
    print(f"Extracted {n} chips; missing: {len(missing)}")
    if missing:
        raise SystemExit(f"Missing chips, e.g. {sorted(missing)[:3]}")
    if not keep_archive:
        archive.unlink()


def download_weights(cfg, finetuned, seeds):
    for bb in BACKBONES:
        spec = cfg["backbones"][bb]
        out = cfg["paths"]["weights_dir"] / bb
        for f in (spec["file"], "config.json"):
            hf_hub_download(spec["hf_repo"], f, local_dir=out)
        print(f"zero-shot {bb}: {out}")
    if finetuned:
        repo = cfg["finetuned_hf_repo"]
        for s in seeds:
            for bb in BACKBONES:
                for f in ("best.pt", "config.json"):
                    hf_hub_download(repo, f"seed{s}/{bb}/{f}", local_dir=cfg["paths"]["finetuned_dir"])
            print(f"fine-tuned seed {s}: {cfg['paths']['finetuned_dir'] / f'seed{s}'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--finetuned", action="store_true", help="also download our fine-tuned checkpoints")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--keep-archive", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if not a.skip_data:
        download_data(cfg, a.keep_archive)
    download_weights(cfg, a.finetuned, a.seeds)


if __name__ == "__main__":
    main()
