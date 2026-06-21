"""
plot_graph.py — visualise the INPUT graph the GNN is fed, before inference.

The model receives a line-graph: nodes = the 1897 road edges (each with 7
features), connectivity = data.edge_index (downstream links). This renders that
graph at the real road geometry:
  * left  : nodes colored by TAZ zone (the static structure + pooling groups),
  * right : nodes colored by a traffic feature for one sample (the dynamic input).

Run:
    python -m src.plot_graph --random
    python -m src.plot_graph --index 839 --feature flow
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sumolib

from src.config import load_yaml, resolve
from src.data.graph import load_graph

FEATURES = ["flow", "speed", "density", "traveltime"]  # order of x_dyn channels


def node_positions(edge_ids, net_path):
    """Geographic centroid of each road edge, in edge_ids order."""
    net = sumolib.net.readNet(str(net_path))
    pos = np.full((len(edge_ids), 2), np.nan)
    idx = {e: i for i, e in enumerate(edge_ids)}
    for e in net.getEdges():
        i = idx.get(e.getID())
        if i is not None:
            pos[i] = np.mean(e.getShape(), axis=0)
    return pos


def pick_sample_file(dcfg, args):
    data_root = resolve(dcfg["paths"]["samples_dir"]).parent
    if args.sample:
        return Path(args.sample)
    manifest = json.loads(resolve(dcfg["paths"]["manifest"]).read_text())
    rows = [r for r in manifest["samples"] if r["split"] == "test"]
    if args.index is not None and any(r["idx"] == args.index for r in rows):
        row = next(r for r in rows if r["idx"] == args.index)
    else:
        row = rows[np.random.default_rng(args.seed).integers(len(rows))]
    return data_root / row["file"]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    dcfg = load_yaml("data")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--index", type=int)
    g.add_argument("--sample", type=str)
    g.add_argument("--random", action="store_true")
    ap.add_argument("--feature", choices=FEATURES, default="flow")
    ap.add_argument("--out", type=str, default="data/infer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    graph = load_graph()
    pos = node_positions(graph["edge_ids"], resolve(dcfg["paths"]["net_file"]))
    ei = graph["edge_index"]
    segs = np.stack([pos[ei[0]], pos[ei[1]]], axis=1)  # [L, 2, 2]

    pkl = pick_sample_file(dcfg, args)
    with open(pkl, "rb") as f:
        s = pickle.load(f)
    feat = s["x_dyn"][:, FEATURES.index(args.feature)]
    print(f"[plot] sample idx={s['idx']} regime={s['tod']} total={s['total_demand']} "
          f"| nodes={graph['n_nodes']} links={ei.shape[1]} | active edges (flow>0)={int((s['x_dyn'][:,0]>0).sum())}")

    fig, ax = plt.subplots(1, 2, figsize=(21, 10))
    for a in ax:
        a.add_collection(LineCollection(segs, colors="#cfcfcf", linewidths=0.3, alpha=0.5))
        a.set_aspect("equal"); a.axis("off")

    ax[0].scatter(pos[:, 0], pos[:, 1], c=graph["node_zone"], cmap="tab20", s=7, linewidths=0)
    ax[0].set_title(f"input graph: {graph['n_nodes']} road-edge nodes, {ei.shape[1]} links\n"
                    "colored by TAZ zone (the 36 pooling groups)")

    # dim the (many) zero-feature edges; highlight the active ones sized by value
    active = feat > 0
    n_act = int(active.sum())
    ax[1].scatter(pos[~active, 0], pos[~active, 1], c="#e3e3e3", s=4, linewidths=0)
    sc = ax[1].scatter(pos[active, 0], pos[active, 1], c=np.log1p(feat[active]), cmap="plasma",
                       s=12 + 60 * feat[active] / max(feat.max(), 1e-6), linewidths=0)
    ax[1].set_title(f"same graph, nodes colored/sized by {args.feature}\n"
                    f"sample idx={s['idx']} ({s['tod']}, total={s['total_demand']} trips) — "
                    f"only {n_act}/{graph['n_nodes']} edges carry signal")
    fig.colorbar(sc, ax=ax[1], fraction=0.04, label=f"log1p({args.feature})")

    fig.tight_layout()
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    png = out / f"input_graph_{s['idx']}.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {png}")


if __name__ == "__main__":
    main()
