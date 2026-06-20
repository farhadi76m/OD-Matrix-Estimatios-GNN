"""
convert_h5_to_pkl.py — turn the existing HDF5 dataset into the project's
on-disk format: one .pkl per sample + an index.json manifest + norm_stats.json.

Per CLAUDE.md we stop using HDF5 for the ML pipeline (the .h5 stays as a
read-only source). Each sample stores its dynamic edge features, OD target,
RNG seed, demand tier and the regime label derived from that tier.

The 4 balanced log demand tiers (50..2000 trips) are relabelled as regimes
(night -> noon -> morning_peak -> evening_peak) spanning free-flow -> congested,
and the train/val/test split is stratified by tier so every split holds all 4.
Normalisation stats are fit on the TRAIN split only.

Run:
    python -m src.data.convert_h5_to_pkl                # smoke subset (configs/data.yaml)
    python -m src.data.convert_h5_to_pkl --limit 0      # full dataset (0/null = all)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np

from src.config import load_yaml, resolve
from src.data.graph import load_graph


def assign_tiers(total_demand: np.ndarray, lo: int, hi: int, n_tiers: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (tier_index[0..n-1], tier_breaks) using log-spaced demand bins."""
    breaks = np.exp(np.linspace(np.log(lo), np.log(hi), n_tiers + 1))
    tier = np.clip(np.searchsorted(breaks, total_demand, side="right") - 1, 0, n_tiers - 1)
    return tier.astype(np.int64), breaks


def stratified_split(tier: np.ndarray, fractions: dict, seed: int) -> np.ndarray:
    """Assign each sample to 'train'/'val'/'test', stratified within each tier."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(tier), dtype=object)
    for t in np.unique(tier):
        idx = np.where(tier == t)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * fractions["train"]))
        n_va = int(round(n * fractions["val"]))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"
    return split


def main() -> None:
    dcfg = load_yaml("data")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Max samples (stratified). 0 or negative = all. "
                         "Default: configs/data.yaml subset.limit")
    args = ap.parse_args()

    seed = dcfg["seed"]
    lo, hi = dcfg["demand_range"]
    n_tiers = dcfg["n_tiers"]
    tier_labels = dcfg["tier_labels"]
    dyn_fields = dcfg["features"]["dynamic"]

    samples_dir = resolve(dcfg["paths"]["samples_dir"])
    samples_dir.mkdir(parents=True, exist_ok=True)
    h5_path = resolve(dcfg["paths"]["source_hdf5"])

    graph = load_graph()
    n_nodes = graph["n_nodes"]

    with h5py.File(h5_path, "r") as f:
        N = f["od_matrices"].shape[0]
        total_demand = f["total_demand"][:].astype(np.int64)
        tier, breaks = assign_tiers(total_demand, lo, hi, n_tiers)

        # --- choose subset (stratified) -------------------------------------
        limit = args.limit if args.limit is not None else dcfg["subset"]["limit"]
        if limit and limit > 0 and limit < N:
            rng = np.random.default_rng(seed)
            keep = []
            per = limit // n_tiers
            for t in range(n_tiers):
                idx = np.where(tier == t)[0]
                rng.shuffle(idx)
                keep.append(idx[:per])
            sel = np.sort(np.concatenate(keep))
        else:
            sel = np.arange(N)
        print(f"[convert] using {len(sel)}/{N} samples "
              f"(tiers: {np.bincount(tier[sel], minlength=n_tiers).tolist()})")

        split = stratified_split(tier[sel], dcfg["split"]["fractions"], dcfg["split"]["seed"])

        # --- accumulate train-only stats for normalisation ------------------
        sum_, sumsq_, cnt_ = (np.zeros(len(dyn_fields)) for _ in range(3))

        records = []
        for k, i in enumerate(sel):
            i = int(i)
            x_dyn = np.stack([f[fld][i] for fld in dyn_fields], axis=1).astype(np.float32)  # [N,F]
            od = f["od_matrices"][i].astype(np.int32)
            seed_i = int(f["sample_seeds"][i])
            t = int(tier[i])
            sp = str(split[k])

            if sp == "train":
                lx = np.log1p(np.maximum(x_dyn, 0.0))
                sum_ += lx.sum(axis=0)
                sumsq_ += (lx ** 2).sum(axis=0)
                cnt_ += lx.shape[0]

            fname = f"sample_{i:06d}.pkl"
            with open(samples_dir / fname, "wb") as pf:
                pickle.dump({
                    "idx": i,
                    "x_dyn": x_dyn,                 # [n_nodes, F] raw measurements
                    "od": od,                       # [36, 36] int32 trips
                    "total_demand": int(total_demand[i]),
                    "tier": t,
                    "tod": tier_labels[t],
                    "split": sp,
                    "seed": seed_i,
                }, pf)

            records.append({
                "idx": i, "file": f"samples/{fname}",
                "total_demand": int(total_demand[i]),
                "tier": t, "tod": tier_labels[t], "split": sp, "seed": seed_i,
            })

    # --- normalisation stats (dynamic from train; static from the graph) ----
    mean = sum_ / np.maximum(cnt_, 1)
    var = np.maximum(sumsq_ / np.maximum(cnt_, 1) - mean ** 2, 1e-8)
    std = np.sqrt(var)

    ls = np.log1p(np.maximum(graph["static"], 0.0))
    norm = {
        "dynamic": {"names": dyn_fields, "transform": "log1p_standardize",
                    "mean": mean.tolist(), "std": std.tolist()},
        "static": {"names": graph["static_names"], "transform": "log1p_standardize",
                   "mean": ls.mean(axis=0).tolist(),
                   "std": np.maximum(ls.std(axis=0), 1e-8).tolist()},
        "target": {"transform": dcfg["target"]["transform"]},
    }
    with open(resolve(dcfg["paths"]["norm_stats"]), "w") as f:
        json.dump(norm, f, indent=2)

    # --- manifest -----------------------------------------------------------
    counts = {s: int(sum(1 for r in records if r["split"] == s)) for s in ("train", "val", "test")}
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_hdf5": str(h5_path.name),
        "n_samples": len(records),
        "n_zones": dcfg["n_zones"],
        "n_edges": n_nodes,
        "dynamic_features": dyn_fields,
        "tier_labels": tier_labels,
        "tier_breaks": [round(float(b), 1) for b in breaks],
        "splits": counts,
        "samples": records,
    }
    with open(resolve(dcfg["paths"]["manifest"]), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[convert] wrote {len(records)} .pkl  | splits={counts}")
    print(f"[convert] manifest -> {resolve(dcfg['paths']['manifest'])}")
    print(f"[convert] norm_stats -> {resolve(dcfg['paths']['norm_stats'])}")
    # per-split tier coverage sanity (every split must hold all 4 regimes)
    for s in ("train", "val", "test"):
        tc = np.bincount([r["tier"] for r in records if r["split"] == s], minlength=n_tiers)
        print(f"           {s:5s} tier coverage: {tc.tolist()}")


if __name__ == "__main__":
    main()
