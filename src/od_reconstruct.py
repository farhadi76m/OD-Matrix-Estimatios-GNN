"""
od_reconstruct.py — turn predicted zone marginals into a full OD matrix.

Link counts cannot identify a full OD matrix, but they DO constrain the zone
marginals (trips produced/attracted per zone). We therefore predict the
marginals and reconstruct the matrix with a doubly-constrained gravity model
(Furness / iterative proportional fitting) using a grid-distance deterrence.

Validation on this dataset: Furness from the TRUE marginals reproduces the OD
at cell-correlation 0.85 (vs 0.06 for the old direct-cell GNN), so marginal
quality is what determines OD quality.
"""

from __future__ import annotations

import numpy as np


def zone_distance(zone_ids: list[str]) -> np.ndarray:
    """Euclidean distance between zone centroids. Zone ids encode a grid 'row_col'."""
    pos = np.array([[int(z.split("_")[0]), int(z.split("_")[1])] for z in zone_ids],
                   dtype=float)
    return np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)  # [Z, Z]


def furness(production: np.ndarray, attraction: np.ndarray, deterrence: np.ndarray,
            iters: int = 40) -> np.ndarray:
    """
    Doubly-constrained gravity via IPF.

    production [Z], attraction [Z] : target row / column sums (trips).
    deterrence [Z, Z]             : distance-decay weights (e.g. exp(-beta*D)).
    Returns OD [Z, Z] with zero diagonal whose marginals match the inputs.
    """
    P = np.clip(production, 0, None).astype(np.float64)
    A = np.clip(attraction, 0, None).astype(np.float64)
    if A.sum() > 0:
        A = A * (P.sum() / A.sum())            # balance totals
    T = deterrence.astype(np.float64).copy()
    np.fill_diagonal(T, 0.0)
    for _ in range(iters):
        rs = T.sum(1); rs[rs < 1e-9] = 1.0
        T *= (P / rs)[:, None]
        cs = T.sum(0); cs[cs < 1e-9] = 1.0
        T *= (A / cs)[None, :]
    return T


def reconstruct_od(production, attraction, zone_ids, beta: float = 1.0,
                   iters: int = 40) -> np.ndarray:
    """Convenience: build deterrence from zone grid distance and run Furness."""
    D = zone_distance(zone_ids)
    return furness(production, attraction, np.exp(-beta * D), iters)
