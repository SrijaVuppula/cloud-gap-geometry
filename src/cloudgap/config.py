"""Load configs/default.yaml and resolve paths (environment variables take precedence)."""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

_ENV = {
    "chips_dir": "CLOUDGAP_CHIPS",
    "weights_dir": "CLOUDGAP_WEIGHTS",
    "finetuned_dir": "CLOUDGAP_FINETUNED",
    "results_dir": "CLOUDGAP_RESULTS",
}


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for key, env in _ENV.items():
        value = os.environ.get(env, cfg["paths"][key])
        p = Path(value).expanduser()
        cfg["paths"][key] = p if p.is_absolute() else REPO_ROOT / p
    return cfg


def zeroshot_weights(cfg, backbone):
    """(checkpoint_path, config_json_path) of the released Prithvi-EO-2.0 TL weights."""
    d = cfg["paths"]["weights_dir"] / backbone
    return d / cfg["backbones"][backbone]["file"], d / "config.json"


def finetuned_weights(cfg, backbone, seed):
    """(checkpoint_path, config_json_path) of one of our fine-tuned checkpoints."""
    d = cfg["paths"]["finetuned_dir"] / f"seed{seed}" / backbone
    return d / "best.pt", d / "config.json"
