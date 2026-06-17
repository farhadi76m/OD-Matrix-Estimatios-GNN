"""
config.py – Central configuration for OD Matrix Dataset Generator
All parameters live here so nothing is scattered across files.
"""

from pathlib import Path

CFG = {
    # ── Network files ────────────────────────────────────────────────────────
    "sumo_net":   "/home/mci/mehdi/Traffic/sumo/prune_tab.net.xml",      # SUMO network (from netconvert)
    "taz_file":   "/home/mci/mehdi/Traffic/sumo/tehran_taz.xml",      # 36-zone TAZ definition

    # ── Dataset scale ────────────────────────────────────────────────────────
    "n_zones":    36,                    # Must match TAZ file (6×6 grid)
    "n_samples":  10_000,               # Total OD matrices to generate
    "min_trips":  50,                    # Minimum total demand per sample
    "max_trips":  2000,                  # Maximum total demand per sample

    # ── Output paths ─────────────────────────────────────────────────────────
    "output_hdf5": "od_dataset.h5",      # Final dataset (HDF5)
    "meta_json":   "dataset_meta.json",  # JSON metadata sidecar
    "log_file":    "pipeline.log",
    "work_dir":    "/tmp/od_dataset_gen",  # Temp workspace (auto-cleaned)

    # ── Simulation window ────────────────────────────────────────────────────
    "sim_begin":    0,       # [s]  simulation start
    "sim_end":      3600,    # [s]  simulation end (1 hour)
    "step_length":  1.0,     # [s]  integration time step
    "warm_up":      300,     # [s]  burn-in; data collected from (begin+warm_up → end)

    # ── SUMO binaries ────────────────────────────────────────────────────────
    "sumo_bin":      "sumo",
    "od2trips_bin":  "od2trips",
    "duarouter_bin": "duarouter",

    # ── Demand sampling strategy ─────────────────────────────────────────────
    #
    #   'stratified'  – 4 equal-width log-space tiers, 25% samples each.
    #                   Best coverage of [min, max] range.
    #   'log_uniform' – single log-uniform draw; simpler but may miss extremes.
    #   'uniform'     – linear uniform; over-represents high demand.
    #
    "demand_strategy": "stratified",

    # ── OD generation strategy ───────────────────────────────────────────────
    #
    #   'mixed'     – random mix of the three methods below (recommended)
    #   'gravity'   – doubly-constrained gravity model with random params
    #   'dirichlet' – Dirichlet allocation (uniform → peaky)
    #   'sparse'    – a few dominant corridors (commuter-like)
    #
    "od_strategy": "mixed",
    "mix_ratios": {          # weights for 'mixed'; must sum to 1
        "gravity":   0.40,
        "dirichlet": 0.35,
        "sparse":    0.25,
    },

    # ── Parallelism ──────────────────────────────────────────────────────────
    #
    #   Rule of thumb: n_workers = (CPU cores - 1)
    #   Each SUMO run ≈ 30–120 s real time at this network scale.
    #   10 000 samples / 4 workers × 60 s ≈ 41 h total calendar time.
    #
    "n_workers":  4,
    "batch_size": 64,        # samples per checkpoint write to HDF5

    # ── Reproducibility ──────────────────────────────────────────────────────
    "master_seed": 42,
}