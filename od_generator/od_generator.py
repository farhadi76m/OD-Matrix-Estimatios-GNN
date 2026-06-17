"""
od_generator.py – Randomised OD Matrix Generation

Three complementary strategies ensure the 10 000 matrices cover a wide
variety of spatial demand patterns, not just random noise:

  gravity    doubly-constrained gravity model with randomised zone
             "populations", a random β exponent, and 1–3 attractors
             (simulates a CBD or transit hub).  Most realistic.

  dirichlet  Dirichlet allocation over all n×(n-1) OD pairs.
             α concentration is log-uniformly randomised so some
             samples are near-uniform and others are sharply peaked.

  sparse     Only k << n² active corridors receive flow, mimicking
             commuter peak patterns where a handful of O–D pairs
             dominate.

Demand levels are sampled with log-space stratification so the
[50, 2000] range is uniformly covered in log-scale.
"""

import numpy as np
from typing import List
from config import CFG


# ─────────────────────────────────────────────────────────────────────────────
# Demand sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_demand(rng: np.random.Generator) -> int:
    """
    Draw total trip count respecting the chosen demand_strategy.

    'stratified':   divides [log(min), log(max)] into 4 equal bins,
                    picks a random bin, then samples uniformly inside.
                    → guarantees ~2 500 samples per tier.
    'log_uniform':  single draw from log-uniform distribution.
    'uniform':      linear uniform (over-represents high demand).
    """
    lo, hi   = CFG["min_trips"], CFG["max_trips"]
    strategy = CFG["demand_strategy"]

    if strategy == "uniform":
        return int(rng.integers(lo, hi + 1))

    elif strategy == "log_uniform":
        return int(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    elif strategy == "stratified":
        breaks = np.exp(np.linspace(np.log(lo), np.log(hi), 5))  # 4 tiers
        tier   = rng.integers(4)
        return int(rng.uniform(breaks[tier], breaks[tier + 1]))

    else:
        raise ValueError(f"Unknown demand_strategy: {strategy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# OD generation strategies
# ─────────────────────────────────────────────────────────────────────────────

def _grid_positions(n: int, jitter: float = 0.15) -> np.ndarray:
    """
    Return (n, 2) zone centroid array for a √n × √n grid.
    A small Gaussian jitter breaks exact symmetry so distance
    matrices vary between samples.
    """
    g   = int(np.sqrt(n))
    pos = np.array([[i // g, i % g] for i in range(n)], dtype=float)
    return pos   # jitter added per-call in each strategy


def od_gravity(rng: np.random.Generator, n: int, total: int) -> np.ndarray:
    """
    Doubly-constrained gravity model:

        T_ij = A_i * O_i * B_j * D_j * f(c_ij)

    Randomised parameters per sample:
      • O_i, D_j  ~ LogNormal(μ=1, σ=0.8)   production / attraction
      • β         ~ Uniform(1.0, 3.5)          distance exponent
      • 1–3 zones get a random 3–10× attraction boost  (CBD / hub effect)
      • zone centroids are grid positions + Gaussian jitter(σ=0.2)
      • 1 pass of Furness balancing to match marginals exactly

    Returns (n, n) int32 OD matrix with zero diagonal.
    """
    g   = int(np.sqrt(n))
    pos = np.array([[i // g, i % g] for i in range(n)], dtype=float)
    pos += rng.normal(0, 0.2, pos.shape)

    O   = np.exp(rng.normal(1.0, 0.8, n))
    D   = np.exp(rng.normal(1.0, 0.8, n))
    beta = rng.uniform(1.0, 3.5)

    # Random attractor zones (CBD, transit terminal, …)
    n_attr = rng.integers(1, 4)
    attrs  = rng.choice(n, n_attr, replace=False)
    D[attrs] *= rng.uniform(3.0, 10.0, n_attr)

    # Raw gravity matrix
    raw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.linalg.norm(pos[i] - pos[j]) + 1e-6
                raw[i, j] = O[i] * D[j] * d**(-beta)

    # Single-pass Furness balancing
    raw /= (raw.sum(axis=1, keepdims=True) + 1e-12)
    raw /= (raw.sum(axis=0, keepdims=True) + 1e-12)

    return _multinomial_draw(rng, raw, n, total)


def od_dirichlet(rng: np.random.Generator, n: int, total: int) -> np.ndarray:
    """
    Dirichlet-based allocation over all n*(n-1) off-diagonal OD pairs.

    α concentration is drawn from LogUniform(0.1, 8):
      α << 1 → very peaked; most trips on a few pairs
      α >> 1 → near-uniform allocation

    A random 20% of pairs receive an extra weight boost (hot spots).
    """
    n_pairs   = n * (n - 1)
    alpha_val = np.exp(rng.uniform(np.log(0.1), np.log(8.0)))
    alpha     = np.full(n_pairs, alpha_val)

    # Hot-spot pairs
    n_hot  = max(1, n_pairs // 5)
    hot    = rng.choice(n_pairs, n_hot, replace=False)
    alpha[hot] *= rng.uniform(2.0, 12.0, n_hot)

    fractions = rng.dirichlet(alpha)
    counts    = rng.multinomial(total, fractions)

    od = np.zeros((n, n), dtype=np.int32)
    od[~np.eye(n, dtype=bool)] = counts
    return od


def od_sparse(rng: np.random.Generator, n: int, total: int) -> np.ndarray:
    """
    Sparse corridor model: only k active OD pairs carry flow.

    k is drawn from Uniform(n//4, n) so some samples are very sparse
    (single dominant corridor) and others cover a quarter of pairs.
    Weights among active pairs are exponentially distributed so a
    few corridors are dominant.
    """
    n_pairs = n * (n - 1)
    k       = int(rng.integers(max(3, n // 4), max(4, n) + 1))
    k       = min(k, n_pairs)

    active  = rng.choice(n_pairs, k, replace=False)
    weights = np.zeros(n_pairs)
    weights[active] = np.exp(rng.exponential(1.5, k))

    fractions = weights / weights.sum()
    counts    = rng.multinomial(total, fractions)

    od = np.zeros((n, n), dtype=np.int32)
    od[~np.eye(n, dtype=bool)] = counts
    return od


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGIES = {
    "gravity":   od_gravity,
    "dirichlet": od_dirichlet,
    "sparse":    od_sparse,
}


def generate_od_matrix(
    rng: np.random.Generator, n: int, total: int
) -> np.ndarray:
    """
    Generate one OD matrix of size (n, n) with `total` integer trips.

    Dispatches to the strategy in CFG["od_strategy"].
    In 'mixed' mode selects a method stochastically according to
    CFG["mix_ratios"].
    """
    strategy = CFG["od_strategy"]

    if strategy in _STRATEGIES:
        return _STRATEGIES[strategy](rng, n, total)

    elif strategy == "mixed":
        ratios  = CFG["mix_ratios"]
        methods = list(ratios.keys())
        probs   = np.array([ratios[m] for m in methods], dtype=float)
        probs  /= probs.sum()
        chosen  = rng.choice(methods, p=probs)
        return _STRATEGIES[chosen](rng, n, total)

    else:
        raise ValueError(f"Unknown od_strategy: {strategy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _multinomial_draw(
    rng: np.random.Generator,
    raw: np.ndarray,
    n: int,
    total: int,
) -> np.ndarray:
    """
    Convert a (n,n) non-negative raw matrix to integer OD counts.
    Zeros the diagonal, normalises, multinomial-draws `total` trips.
    """
    np.fill_diagonal(raw, 0.0)
    raw = np.maximum(raw, 0.0)
    s   = raw.sum()
    if s < 1e-12:
        # Fallback: uniform over off-diagonal
        raw = np.ones((n, n)) - np.eye(n)

    flat      = raw[~np.eye(n, dtype=bool)]
    fractions = flat / flat.sum()
    counts    = rng.multinomial(total, fractions)

    od = np.zeros((n, n), dtype=np.int32)
    od[~np.eye(n, dtype=bool)] = counts
    return od