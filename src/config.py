"""
config.py — one small config loader used everywhere, plus seed/device helpers.

All hyperparameters and paths live in configs/*.yaml; source code reads them
through here. Paths in the YAML are relative to the project root, which is
resolved automatically so scripts work from any working directory.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Project root = parent of this file's parent (src/ -> root).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_yaml(name: str) -> Dict[str, Any]:
    """Load a single YAML file from configs/ (name with or without .yaml)."""
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"
    path = CONFIG_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config() -> Dict[str, Any]:
    """Load and merge data.yaml, model.yaml, train.yaml into one dict."""
    cfg: Dict[str, Any] = {}
    for name in ("data", "model", "train"):
        cfg[name] = load_yaml(name)
    return cfg


def resolve(path_str: str) -> Path:
    """Resolve a config path (relative to project root) to an absolute Path."""
    p = Path(os.path.expanduser(path_str))
    return p if p.is_absolute() else (ROOT / p)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Auto-select CUDA if available, else CPU (laptop-friendly)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
