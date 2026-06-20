"""
metrics.py — regression-appropriate metrics for OD estimation.

All OD metrics are computed in trip-count space (predictions are expm1'd back
from log1p) over the off-diagonal cells only, since the diagonal is structurally
zero. GEH is computed on implied link flows (see evaluate.py for the linear
assignment surrogate that stands in for SUMO re-simulation).
"""

from __future__ import annotations

import numpy as np
import torch


def to_counts(pred_log: torch.Tensor) -> torch.Tensor:
    """Map log1p-space predictions back to non-negative trip counts.

    The log input is clamped (expm1(16) ~ 9e6, far above any real OD total of
    ~2000 trips) so an over-confident prediction can never overflow float32.
    """
    return torch.expm1(pred_log.clamp(max=16.0)).clamp(min=0.0)


def _offdiag_mask(z: int, device) -> torch.Tensor:
    return (1.0 - torch.eye(z, device=device)).bool()


@torch.no_grad()
def od_metrics(pred_counts: torch.Tensor, true_counts: torch.Tensor) -> dict:
    """
    pred_counts, true_counts: [B, Z, Z] in trips.
    Returns per-cell MAE/RMSE (off-diagonal) and total-flow error.
    """
    z = pred_counts.shape[-1]
    mask = _offdiag_mask(z, pred_counts.device)  # [Z,Z]
    p = pred_counts[:, mask]   # [B, Z*(Z-1)]
    t = true_counts[:, mask]

    err = p - t
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()

    tot_p = p.sum(dim=1)
    tot_t = t.sum(dim=1)
    tot_err = (tot_p - tot_t).abs() / tot_t.clamp(min=1.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "total_flow_err": tot_err.mean().item(),  # mean relative error on total trips
    }


def geh(model_flow: np.ndarray, obs_flow: np.ndarray, good_threshold: float = 5.0) -> dict:
    """
    GEH statistic on link flows (hourly). model_flow, obs_flow: [..., E].
    Returns mean GEH and fraction of links with GEH < good_threshold.
    """
    m = np.asarray(model_flow, dtype=np.float64)
    c = np.asarray(obs_flow, dtype=np.float64)
    denom = m + c
    g = np.where(denom > 0, np.sqrt(2.0 * (m - c) ** 2 / np.maximum(denom, 1e-9)), 0.0)
    return {
        "geh_mean": float(g.mean()),
        "geh_frac_good": float((g < good_threshold).mean()),
    }
