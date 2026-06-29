"""
visualize.py — Step 4: all visualisations for the 9-zone study.

Produces (in data9/viz/):
  zones_map.png        the 9 TAZ zones drawn on the road network
  training_curve.png   train loss + val marginal correlations over epochs
  marginals_scatter.png predicted vs true production/attraction
  cell_scatter.png     pooled off-diagonal OD cell pred vs true
  od_examples.png      pred vs true OD heatmaps for one sample per regime
  metrics_summary.png  cell-corr per regime vs the gravity ceiling + key numbers

Run (after train_eval):
    python -m src.zones9.visualize
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import load_yaml, resolve
from src.zones9.graph import load_graph


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def zones_map(cfg, graph, out):
    plt = _setup()
    from matplotlib.collections import LineCollection
    from src.plot_graph import node_positions
    pos = node_positions(graph["edge_ids"], resolve(cfg["paths"]["net_file"]))
    ei = graph["edge_index"]; segs = np.stack([pos[ei[0]], pos[ei[1]]], axis=1)
    zone = graph["node_zone"]; zone_ids = graph["zone_ids"]
    fig, ax = plt.subplots(figsize=(12, 9)); ax.set_aspect("equal"); ax.axis("off")
    ax.add_collection(LineCollection(segs, colors="#dddddd", linewidths=0.3, alpha=0.6))
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=zone, cmap="tab10", s=10, linewidths=0)
    for zi, zid in enumerate(zone_ids):
        c = pos[zone == zi].mean(0)
        ax.text(c[0], c[1], zid, fontsize=13, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.8))
    ax.set_title(f"9 TAZ zones on the District-2 network (gridDistricts -w 3000)\n"
                 f"{graph['n_nodes']} road edges colored by zone")
    fig.tight_layout(); fig.savefig(out / "zones_map.png", dpi=120); plt.close(fig)


def training_curve(cfg, out):
    plt = _setup()
    hp = resolve(f"{cfg['paths']['data_dir']}/history.json")
    if not hp.exists():
        return
    h = json.loads(hp.read_text())
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(ep, [r["train_loss"] for r in h], color="#d1495b")
    ax[0].set_title("training loss (standardised-MSE marginals)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [r["prod_corr"] for r in h], label="production")
    ax[1].plot(ep, [r["attr_corr"] for r in h], label="attraction")
    ax[1].set_title("validation marginal correlation"); ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("Pearson r"); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "training_curve.png", dpi=120); plt.close(fig)


def marginals_scatter(P, out):
    plt = _setup()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    for a, p, t, ttl in [(ax[0], P["prod_p"], P["prod_t"], "production"),
                         (ax[1], P["attr_p"], P["attr_t"], "attraction")]:
        a.scatter(t.ravel(), p.ravel(), s=10, alpha=0.4, color="#2e7d8c")
        lim = max(t.max(), p.max(), 1); a.plot([0, lim], [0, lim], "k--", lw=1)
        r = np.corrcoef(p.ravel(), t.ravel())[0, 1]
        a.set_title(f"{ttl} (per zone)  r={r:.3f}"); a.set_xlabel("true trips")
        a.set_ylabel("predicted trips"); a.grid(alpha=0.3)
    fig.suptitle("Zone marginals: predicted vs true (test)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out / "marginals_scatter.png", dpi=120); plt.close(fig)


def cell_scatter(P, out):
    plt = _setup()
    z = P["pred"].shape[-1]; off = ~np.eye(z, dtype=bool)
    pv, tv = P["pred"][:, off].ravel(), P["true"][:, off].ravel()
    r = np.corrcoef(pv, tv)[0, 1]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(tv, pv, s=6, alpha=0.25, color="#444")
    lim = max(tv.max(), pv.max(), 1); ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_title(f"OD cells (off-diagonal, all test): Pearson r={r:.3f}")
    ax.set_xlabel("true trips"); ax.set_ylabel("predicted trips"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "cell_scatter.png", dpi=120); plt.close(fig)


def od_examples(cfg, P, out):
    plt = _setup()
    zone_ids = load_graph()["zone_ids"]; tiers = P["tier"]
    picks = []
    for t, name in enumerate(cfg["tier_labels"]):
        idx = np.where(tiers == t)[0]
        if len(idx):
            picks.append((name, idx[np.argmax(P["true"][idx].sum((1, 2)))]))  # busiest in regime
    n = len(picks)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 8))
    ax = np.atleast_2d(ax)
    vmax = max(np.log1p(P["true"]).max(), np.log1p(P["pred"]).max(), 1e-3)
    for c, (name, k) in enumerate(picks):
        for r, (M, ttl) in enumerate([(P["true"][k], "TRUE"), (P["pred"][k], "PRED")]):
            im = ax[r, c].imshow(np.log1p(M), cmap="magma", vmin=0, vmax=vmax)
            ax[r, c].set_title(f"{name}\n{ttl} (tot={M.sum():.0f})", fontsize=10)
            ax[r, c].set_xticks(range(len(zone_ids))); ax[r, c].set_yticks(range(len(zone_ids)))
            ax[r, c].set_xticklabels(zone_ids, fontsize=6, rotation=90)
            ax[r, c].set_yticklabels(zone_ids, fontsize=6)
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.025, label="log1p(trips)")
    fig.suptitle("Predicted vs true 9x9 OD — busiest test sample per regime", fontsize=13)
    fig.savefig(out / "od_examples.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def metrics_summary(cfg, rep, out):
    plt = _setup()
    regimes = list(rep["per_regime"]); cc = [rep["per_regime"][r]["cell_corr"] for r in regimes]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(regimes, cc, color="#2e7d8c", alpha=0.85)
    ax.axhline(rep["reconstruction_ceiling"], color="#d1495b", ls="--",
               label=f"gravity ceiling = {rep['reconstruction_ceiling']:.2f}")
    ax.axhline(rep["cell_corr"], color="#444", ls=":", label=f"overall = {rep['cell_corr']:.2f}")
    ax.set_ylim(0, 1); ax.set_ylabel("OD cell correlation"); ax.legend()
    g = rep.get("geh", {})
    geh_s = f"GEH mean={g['geh_mean']:.1f}" if g.get("geh_mean") is not None else "GEH n/a"
    ax.set_title(f"9-zone OD recovery by regime\n"
                 f"marginals: prod r={rep['marginals']['production_corr']:.2f}, "
                 f"attr r={rep['marginals']['attraction_corr']:.2f} | "
                 f"totErr={rep['total_flow_err']:.2f} | {geh_s}")
    fig.tight_layout(); fig.savefig(out / "metrics_summary.png", dpi=120); plt.close(fig)


def main():
    cfg = load_yaml("zones9")
    out = resolve(f"{cfg['paths']['data_dir']}/viz"); out.mkdir(parents=True, exist_ok=True)
    graph = load_graph()
    zones_map(cfg, graph, out)
    training_curve(cfg, out)
    P = dict(np.load(resolve(cfg["paths"]["predictions"])))
    rep = json.loads(resolve(cfg["paths"]["metrics"]).read_text())
    marginals_scatter(P, out)
    cell_scatter(P, out)
    od_examples(cfg, P, out)
    metrics_summary(cfg, rep, out)
    print(f"[viz] wrote zones_map, training_curve, marginals_scatter, cell_scatter, "
          f"od_examples, metrics_summary -> {out}")


if __name__ == "__main__":
    main()
