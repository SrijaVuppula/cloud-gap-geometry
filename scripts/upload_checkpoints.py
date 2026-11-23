"""Upload fine-tuned checkpoints to the Hugging Face Hub (run once, by the authors).

  huggingface-cli login
  python scripts/upload_checkpoints.py \
      --src ~/Prithvi/Prithvi-Sandbox/multi_tile_generalization/block_masking_study/checkpoints/seeds \
      --seeds 1

--src must contain seed{N}/{backbone}/best.pt and config.json. Seed 1 is the run
used for the patch-level analysis (Figure 2); add more seeds with --seeds 1 2 ... 10.
The model card is hf_model_card.md.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cloudgap import BACKBONES  # noqa: E402
from cloudgap.config import REPO_ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--repo", default=load_config()["finetuned_hf_repo"])
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    src = a.src.expanduser()
    for s in a.seeds:
        for bb in BACKBONES:
            for f in ("best.pt", "config.json"):
                p = src / f"seed{s}" / bb / f
                if not p.exists():
                    raise SystemExit(f"missing {p}")

    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_file(path_or_fileobj=REPO_ROOT / "hf_model_card.md", path_in_repo="README.md", repo_id=a.repo)
    api.upload_file(path_or_fileobj=REPO_ROOT / "LICENSE", path_in_repo="LICENSE", repo_id=a.repo)
    for s in a.seeds:
        for bb in BACKBONES:
            folder = src / f"seed{s}" / bb
            print(f"uploading seed{s}/{bb} ...")
            api.upload_folder(folder_path=folder, path_in_repo=f"seed{s}/{bb}", repo_id=a.repo,
                              allow_patterns=["best.pt", "config.json"],
                              commit_message=f"seed {s}, {bb}")
    print(f"done: https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
