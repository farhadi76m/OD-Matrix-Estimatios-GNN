"""
build_enriched.py — turn the re-simulated samples (src/data/resimulate.py) into
an enriched dataset that carries per-link TURN COUNTS, the measurement that makes
the OD identifiable (validated: turn->marginal corr ~0.89 -> Furness cell-corr ~0.78).

Per sample:
  node x   [1897, 9] : static[length,lanes,speed_limit] + aggregate
                       [flow,speed,density,traveltime] + node [out_turn,in_turn]
  edge_w   [n_links]  : turn count on each line-graph link (message-passing weight)
  y        [36,36]    : OD target (unchanged)

Splits are stratified by demand tier. Normalisation (log1p+standardise) is fit on
train only. Outputs packs (data/pack_enr_*.pt) + data/norm_enr.json + manifest.

Run:
    python -m src.data.build_enriched
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.config import load_yaml, resolve
from src.data.graph import load_graph

RESIM_DIR = "data/resim"


def main() -> None:
    dcfg = load_yaml("data")
    graph = load_graph()
    ei = graph["edge_index"]
    src, dst = ei[0], ei[1]
    static = graph["static"].astype(np.float32)           # [N,3]
    lo, hi, n_tiers = *dcfg["demand_range"], dcfg["n_tiers"]
    tier_labels = dcfg["tier_labels"]
    breaks = np.exp(np.linspace(np.log(lo), np.log(hi), n_tiers + 1))

    files = sorted(glob.glob(str(resolve(RESIM_DIR) / "*.pkl")))
    print(f"[enrich] {len(files)} re-simulated samples")
    X, EW, Y, tiers = [], [], [], []
    for fp in files:
        s = pickle.load(open(fp, "rb"))
        ts = s["x_ts"]                                     # [T,N,4]
        flow = ts[:, :, 0].sum(0)                          # aggregate entered
        spd, den, tt = ts[:, :, 1].mean(0), ts[:, :, 2].mean(0), ts[:, :, 3].mean(0)
        turn = s["turn"].astype(np.float32)               # [n_links]
        out_turn = np.zeros(len(static), np.float32); np.add.at(out_turn, src, turn)
        in_turn = np.zeros(len(static), np.float32); np.add.at(in_turn, dst, turn)
        x = np.concatenate([static,
                            np.stack([flow, spd, den, tt, out_turn, in_turn], 1)], 1)  # [N,9]
        X.append(x.astype(np.float32)); EW.append(turn); Y.append(s["od"].astype(np.float32))
        tot = int(s["od"].sum())
        tiers.append(int(np.clip(np.searchsorted(breaks, tot, "right") - 1, 0, n_tiers - 1)))
    X = np.stack(X); EW = np.stack(EW); Y = np.stack(Y); tiers = np.array(tiers)

    # stratified split
    rng = np.random.default_rng(dcfg["split"]["seed"])
    split = np.empty(len(X), dtype=object)
    fr = dcfg["split"]["fractions"]
    for t in np.unique(tiers):
        idx = np.where(tiers == t)[0]; rng.shuffle(idx)
        ntr, nva = int(round(len(idx) * fr["train"])), int(round(len(idx) * fr["val"]))
        split[idx[:ntr]] = "train"; split[idx[ntr:ntr + nva]] = "val"; split[idx[ntr + nva:]] = "test"

    # train-only normalisation (log1p then standardise) for node feats + edge weights
    tr = split == "train"
    lx = np.log1p(np.clip(X[tr], 0, None))
    x_mean, x_std = lx.mean((0, 1)), np.maximum(lx.reshape(-1, X.shape[2]).std(0), 1e-6)
    lew = np.log1p(np.clip(EW[tr], 0, None))
    ew_mean, ew_std = float(lew.mean()), float(max(lew.std(), 1e-6))
    norm = {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
            "ew_mean": ew_mean, "ew_std": ew_std,
            "feat_names": ["length", "lanes", "speed_limit", "flow", "speed",
                           "density", "traveltime", "out_turn", "in_turn"]}
    resolve("data/norm_enr.json").write_text(json.dumps(norm, indent=2))

    out = resolve("data"); counts = {}
    for sp in ("train", "val", "test"):
        m = split == sp
        torch.save({"x": torch.tensor(X[m]), "ew": torch.tensor(EW[m]),
                    "y": torch.tensor(Y[m]),
                    "meta": [{"tier": int(tiers[i]), "tod": tier_labels[int(tiers[i])],
                              "idx": int(i)} for i in np.where(m)[0]]},
                   out / f"pack_enr_{sp}.pt")
        counts[sp] = int(m.sum())
    print(f"[enrich] splits={counts}  feature_dim={X.shape[2]}  n_links={EW.shape[1]}")
    for sp in ("train", "val", "test"):
        tc = np.bincount(tiers[split == sp], minlength=n_tiers)
        print(f"         {sp:5s} tier coverage {tc.tolist()}")


if __name__ == "__main__":
    main()
