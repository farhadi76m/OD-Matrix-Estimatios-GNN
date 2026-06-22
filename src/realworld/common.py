"""
common.py — shared helpers for the real-data calibration stages.

Everything here is signal-agnostic plumbing reused across the diagnostic, the
warm-start, the SPSA loop, and validation:
  * the 9-channel feature layout of the enriched packs,
  * a realistic "what does a probe source observe" coverage mask (named edges),
  * OD <-> marginals, Pearson / pooled off-diagonal cell-correlation,
  * gravity (Furness) reconstruction wrappers + a beta sweep.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from src.config import resolve
from src.od_reconstruct import furness, zone_distance

# Channel order of the enriched packs (data/pack_enr_*.pt), see build_enriched.py.
FEAT_NAMES = ["length", "lanes", "speed_limit", "flow", "speed",
              "density", "traveltime", "out_turn", "in_turn"]
FEAT_IDX = {n: i for i, n in enumerate(FEAT_NAMES)}


# ─────────────────────────────────────────────────────────────────────────────
# Network introspection
# ─────────────────────────────────────────────────────────────────────────────
def edge_names(net_path) -> dict:
    """{edge_id: name} for every non-internal edge that carries a street name."""
    names = {}
    for _, e in ET.iterparse(str(resolve(net_path)), events=("end",)):
        if e.tag != "edge":
            continue
        eid = e.get("id", "")
        nm = e.get("name", "")
        if eid and not eid.startswith(":") and nm:
            names[eid] = nm
        e.clear()
    return names


def named_mask(edge_ids: list[str], net_path, min_name_len: int = 1) -> np.ndarray:
    """Boolean[N] over edge_ids: True where the edge has a street name (an
    arterial a probe source like Neshan is likely to cover)."""
    names = edge_names(net_path)
    return np.array([len(names.get(e, "")) >= min_name_len for e in edge_ids], dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# OD <-> marginals and metrics (definitions match src/evaluate.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def marginals_from_od(od: np.ndarray):
    """Return (production, attraction). od is [...,Z,Z]; production = row sums
    (trips originating per zone), attraction = column sums (trips ending)."""
    return od.sum(axis=-1), od.sum(axis=-2)


def pearson(a, b) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pooled_cell_corr(pred: np.ndarray, true: np.ndarray) -> float:
    """Pooled off-diagonal Pearson over all samples (the headline cell-corr)."""
    z = pred.shape[-1]
    off = ~np.eye(z, dtype=bool)
    return pearson(pred[..., off], true[..., off])


def total_flow_err(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean relative error on per-sample total off-diagonal trips."""
    z = pred.shape[-1]
    off = ~np.eye(z, dtype=bool)
    p = pred[..., off].sum(-1)
    t = true[..., off].sum(-1)
    return float((np.abs(p - t) / np.clip(t, 1.0, None)).mean())


def cell_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    z = pred.shape[-1]
    off = ~np.eye(z, dtype=bool)
    return float(np.sqrt(((pred[..., off] - true[..., off]) ** 2).mean()))


# ─────────────────────────────────────────────────────────────────────────────
# Gravity reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_many(prod: np.ndarray, attr: np.ndarray, zone_ids: list[str],
                     beta: float, iters: int = 40) -> np.ndarray:
    """Furness-reconstruct an OD per sample. prod/attr: [n,Z]."""
    deter = np.exp(-beta * zone_distance(zone_ids))
    return np.stack([furness(prod[k], attr[k], deter, iters) for k in range(len(prod))])


def pick_beta(prod_true: np.ndarray, attr_true: np.ndarray, true_od: np.ndarray,
              zone_ids: list[str], betas: list[float], iters: int = 40,
              subset: int = 300) -> tuple[float, float]:
    """Sweep the deterrence beta using the TRUE marginals; return (best_beta,
    reconstruction-ceiling cell-corr). This is the upper bound a perfect
    marginal predictor could reach via gravity."""
    n = min(subset, len(prod_true))
    best_b, best_c = betas[0], -1.0
    for b in betas:
        rec = reconstruct_many(prod_true[:n], attr_true[:n], zone_ids, b, iters)
        c = pooled_cell_corr(rec, true_od[:n])
        if c > best_c:
            best_b, best_c = b, c
    return best_b, best_c


# ─────────────────────────────────────────────────────────────────────────────
# Enriched pack access
# ─────────────────────────────────────────────────────────────────────────────
def load_pack(split: str) -> dict:
    """Load an enriched pack: x [n,N,9], ew [n,L], y [n,Z,Z], meta."""
    return torch.load(resolve(f"data/pack_enr_{split}.pt"), weights_only=False)
