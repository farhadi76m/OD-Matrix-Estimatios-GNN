"""
dataset.py — PyG dataset + dataloaders over the per-sample .pkl files.

The static line-graph (edge_index, zone membership, normalised static node
features) is shared across all samples; only the dynamic edge measurements and
the OD target change per sample. Each item is a torch_geometric Data:

    x          [N, 3+F]  normalised [static | dynamic] node features
    edge_index [2, L]    shared downstream connectivity
    zone       [N]       TAZ index per node (for zone pooling)
    y          [36, 36]  OD target in raw trips (log1p applied in the loss)

Normalisation uses train-only stats from data/norm_stats.json.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.config import load_yaml, resolve
from src.data.graph import load_graph


class ODDataset(torch.utils.data.Dataset):
    def __init__(self, split: str):
        dcfg = load_yaml("data")
        self.split = split

        manifest = json.loads(resolve(dcfg["paths"]["manifest"]).read_text())
        norm = json.loads(resolve(dcfg["paths"]["norm_stats"]).read_text())
        graph = load_graph()

        self.n_zones = graph["n_zones"]
        self.n_nodes = graph["n_nodes"]

        # shared static tensors
        self.edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        self.zone = torch.as_tensor(graph["node_zone"], dtype=torch.long)

        d_mean = torch.tensor(norm["dynamic"]["mean"], dtype=torch.float32)
        d_std = torch.tensor(norm["dynamic"]["std"], dtype=torch.float32)
        s_mean = torch.tensor(norm["static"]["mean"], dtype=torch.float32)
        s_std = torch.tensor(norm["static"]["std"], dtype=torch.float32)
        self._d_mean, self._d_std = d_mean, d_std

        static = torch.tensor(graph["static"], dtype=torch.float32)
        self.static_norm = (torch.log1p(static.clamp(min=0)) - s_mean) / s_std  # [N,3]

        # load this split — via a single packed cache so process startup is fast
        # (reading 6998 individual .pkl files per run is ~12s; the pack loads in ~1s).
        data_root = resolve(dcfg["paths"]["samples_dir"]).parent
        pack_path = data_root / f"pack_{split}.pt"
        if not pack_path.exists():
            self._build_pack(manifest, data_root, split, pack_path)
        pack = torch.load(pack_path, weights_only=False)

        dyn_raw = pack["dyn"].clamp(min=0)                 # [n, N, F] raw measurements
        self.dyn = (torch.log1p(dyn_raw) - d_mean) / d_std  # normalised once, vectorised
        self.y = pack["y"]                                  # [n, 36, 36]
        self.meta = pack["meta"]
        self.feature_dim = self.static_norm.shape[1] + self.dyn.shape[2]

    @staticmethod
    def _build_pack(manifest, data_root, split, pack_path):
        rows = [r for r in manifest["samples"] if r["split"] == split]
        dyn, y, meta = [], [], []
        for r in rows:
            with open(data_root / r["file"], "rb") as f:
                s = pickle.load(f)
            dyn.append(torch.tensor(s["x_dyn"], dtype=torch.float32))
            y.append(torch.tensor(s["od"], dtype=torch.float32))
            meta.append({"idx": s["idx"], "tier": s["tier"], "tod": s["tod"],
                         "total_demand": s["total_demand"]})
        torch.save({"dyn": torch.stack(dyn), "y": torch.stack(y), "meta": meta}, pack_path)
        print(f"[dataset] packed {len(rows)} '{split}' samples -> {pack_path}")

    def __len__(self) -> int:
        return len(self.dyn)

    def __getitem__(self, i: int) -> Data:
        x = torch.cat([self.static_norm, self.dyn[i]], dim=1)  # [N, 3+F]
        return Data(x=x, edge_index=self.edge_index, zone=self.zone,
                    y=self.y[i], num_nodes=self.n_nodes)


def make_loaders(batch_size: int, num_workers: int = 0):
    """Return (train, val, test) loaders plus a small info dict."""
    train = ODDataset("train")
    val = ODDataset("val")
    test = ODDataset("test")
    info = {"feature_dim": train.feature_dim, "n_zones": train.n_zones,
            "n_nodes": train.n_nodes}
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers)
    return mk(train, True), mk(val, False), mk(test, False), info


class ODEnrichedDataset(torch.utils.data.Dataset):
    """Enriched samples (data/pack_enr_*.pt): node features + per-link turn
    counts as message-passing edge weights. Built by src/data/build_enriched.py."""

    def __init__(self, split: str):
        dcfg = load_yaml("data")
        graph = load_graph()
        self.n_zones, self.n_nodes = graph["n_zones"], graph["n_nodes"]
        self.edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        self.zone = torch.as_tensor(graph["node_zone"], dtype=torch.long)

        norm = json.loads(resolve("data/norm_enr.json").read_text())
        xm = torch.tensor(norm["x_mean"]); xs = torch.tensor(norm["x_std"])
        pack = torch.load(resolve(f"data/pack_enr_{split}.pt"), weights_only=False)
        self.x = (torch.log1p(pack["x"].clamp(min=0)) - xm) / xs       # [n,N,9]
        self.ew = torch.log1p(pack["ew"].clamp(min=0))                 # [n,n_links] >=0 weights
        self.y = pack["y"]; self.meta = pack["meta"]
        self.feature_dim = self.x.shape[2]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return Data(x=self.x[i], edge_index=self.edge_index, edge_weight=self.ew[i],
                    zone=self.zone, y=self.y[i], num_nodes=self.n_nodes)


def make_enriched_loaders(batch_size: int, num_workers: int = 0):
    train, val, test = (ODEnrichedDataset(s) for s in ("train", "val", "test"))
    info = {"feature_dim": train.feature_dim, "n_zones": train.n_zones, "n_nodes": train.n_nodes}
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers)
    return mk(train, True), mk(val, False), mk(test, False), info
