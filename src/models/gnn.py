"""
gnn.py — ODLineGraphGNN

Edge-as-node line-graph GNN that maps SUMO edge measurements to a 36x36 OD
matrix:

  1. per-node encoder       : [static|dynamic feats] (+) zone embedding -> hidden
  2. message passing        : K residual SAGE/Graph/GAT layers on the road graph
  3. zone pooling           : mean|max|sum of node states into 36 TAZ vectors
                              (uses the clean edge->zone partition)
  4. pairwise OD readout     : MLP over (z_i, z_j, z_i*z_j, |z_i-z_j|, global)
                              -> log1p(trips), diagonal forced to zero.

Predictions are in log1p space; expm1 recovers trip counts for metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphConv, SAGEConv
from torch_geometric.utils import scatter


def _make_conv(kind: str, dim: int) -> nn.Module:
    if kind == "sage":
        return SAGEConv(dim, dim)
    if kind == "graph":
        return GraphConv(dim, dim)
    if kind == "gat":
        return GATv2Conv(dim, dim, heads=1)
    raise ValueError(f"Unknown conv kind: {kind!r}")


class MLP(nn.Module):
    def __init__(self, sizes, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ODLineGraphGNN(nn.Module):
    def __init__(self, feature_dim: int, n_zones: int, mcfg: dict):
        super().__init__()
        m = mcfg["model"]
        self.n_zones = n_zones
        self.add_reverse = m["gnn"]["add_reverse_edges"]
        self.pool_methods = m["pooling"]["methods"]

        zemb = m["zone_embed_dim"]
        h = m["gnn"]["hidden"]
        self.dropout = m["gnn"]["dropout"]
        self.residual = m["gnn"]["residual"]

        self.zone_embed = nn.Embedding(n_zones, zemb)
        self.encoder = MLP([feature_dim + zemb, m["encoder"]["hidden"], h], self.dropout)

        self.convs = nn.ModuleList(_make_conv(m["gnn"]["conv"], h) for _ in range(m["gnn"]["layers"]))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(m["gnn"]["layers"]))

        zh = m["pooling"]["zone_hidden"]
        self.zone_proj = MLP([h * len(self.pool_methods), zh, zh], self.dropout)
        # LayerNorm keeps zone vectors O(1) so the z_i*z_j products in the
        # readout stay bounded regardless of how many edges a zone contains
        # (sum pooling magnitude scales with node count otherwise).
        self.zone_ln = nn.LayerNorm(zh)

        # pairwise readout: zi, zj, zi*zj, |zi-zj|, global  -> 5 * zh
        rh = m["readout"]["hidden"]
        self.pair_mlp = MLP([5 * zh, rh, rh, 1], m["readout"]["dropout"])

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        zone = data.zone
        B = int(batch.max().item()) + 1
        Z = self.n_zones

        if self.add_reverse:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        h = self.encoder(torch.cat([x, self.zone_embed(zone)], dim=1))
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index)
            out = F.relu(norm(out))
            out = F.dropout(out, p=self.dropout, training=self.training)
            h = h + out if self.residual else out

        # zone pooling: unique group per (graph, zone)
        pool_idx = batch * Z + zone
        pooled = [scatter(h, pool_idx, dim=0, dim_size=B * Z, reduce=r) for r in self.pool_methods]
        zvec = self.zone_proj(torch.cat(pooled, dim=1)).view(B, Z, -1)  # [B, Z, zh]
        zvec = self.zone_ln(zvec)
        g = zvec.mean(dim=1)  # [B, zh] global context

        zi = zvec.unsqueeze(2).expand(B, Z, Z, zvec.size(-1))
        zj = zvec.unsqueeze(1).expand(B, Z, Z, zvec.size(-1))
        gg = g.unsqueeze(1).unsqueeze(1).expand(B, Z, Z, g.size(-1))
        feats = torch.cat([zi, zj, zi * zj, (zi - zj).abs(), gg], dim=-1)

        od = self.pair_mlp(feats).squeeze(-1)  # [B, Z, Z]  (log1p space)
        off = (1.0 - torch.eye(Z, device=od.device)).unsqueeze(0)
        return od * off


def build_model(mcfg: dict, feature_dim: int, n_zones: int) -> ODLineGraphGNN:
    return ODLineGraphGNN(feature_dim, n_zones, mcfg)
