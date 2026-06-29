"""
build_packs.py — turn the per-sample pkls into train/val/test packs for training.

Per sample node features x[1897,9] = static[length,lanes,speed_limit] +
[flow,speed,density,traveltime] + node[out_turn,in_turn]; edge weight ew[L] =
per-link turn count; target y[9,9]. Splits come from data9/index.json (stratified
by tier). Normalisation (log1p + standardise) fit on TRAIN only.

Run:
    python -m src.zones9.build_packs
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.config import load_yaml, resolve
from src.zones9.graph import load_graph


def _features(s, static, src, dst, n):
    x_dyn = s["x_dyn"]                                   # [N,4] flow,speed,density,traveltime
    turn = s["turn"].astype(np.float32)                 # [L]
    out_turn = np.zeros(n, np.float32); np.add.at(out_turn, src, turn)
    in_turn = np.zeros(n, np.float32); np.add.at(in_turn, dst, turn)
    x = np.concatenate([static, x_dyn, np.stack([out_turn, in_turn], 1)], 1)  # [N,9]
    return x.astype(np.float32), turn


def main():
    cfg = load_yaml("zones9")
    g = load_graph(); ei = g["edge_index"]; src, dst = ei[0], ei[1]
    static = g["static"].astype(np.float32); n = g["n_nodes"]
    root = resolve(cfg["paths"]["data_dir"])
    manifest = json.loads(resolve(cfg["paths"]["manifest"]).read_text())

    by_split = {"train": [], "val": [], "test": []}
    for r in manifest["samples"]:
        by_split[r["split"]].append(r)

    packs = {}
    for split, rows in by_split.items():
        X, EW, Y, meta = [], [], [], []
        for r in rows:
            s = pickle.load(open(root / r["file"], "rb"))
            x, turn = _features(s, static, src, dst, n)
            X.append(x); EW.append(turn); Y.append(s["od"].astype(np.float32))
            meta.append({"idx": s["idx"], "tier": s["tier"], "tod": s["tod"],
                         "total_demand": s["total_demand"]})
        packs[split] = {"x": np.stack(X), "ew": np.stack(EW), "y": np.stack(Y), "meta": meta}
        print(f"[packs] {split}: {len(rows)} samples")

    # train-only normalisation
    tx = packs["train"]["x"]
    lx = np.log1p(np.clip(tx, 0, None))
    x_mean = lx.mean((0, 1)); x_std = np.maximum(lx.reshape(-1, tx.shape[2]).std(0), 1e-6)
    norm = {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
            "feat_names": ["length", "lanes", "speed_limit", "flow", "speed",
                           "density", "traveltime", "out_turn", "in_turn"]}
    resolve(f"{cfg['paths']['data_dir']}/norm.json").write_text(json.dumps(norm, indent=2))

    for split, p in packs.items():
        torch.save({"x": torch.tensor(p["x"]), "ew": torch.tensor(p["ew"]),
                    "y": torch.tensor(p["y"]), "meta": p["meta"]},
                   root / f"pack_{split}.pt")
    print(f"[packs] saved -> {root}/pack_*.pt + norm.json  feature_dim={tx.shape[2]}")


if __name__ == "__main__":
    main()
