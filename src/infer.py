"""
infer.py — run a trained checkpoint on ONE test sample, produce the estimated
OD matrix, visualise it, and (optionally) launch sumo-gui on the predicted OD.

Examples
--------
    python -m src.infer --random
    python -m src.infer --index 3124
    python -m src.infer --regime evening_peak --sumo-gui
    python -m src.infer --sample data/samples/sample_000123.pkl --no-viz

Outputs (in data/infer/):
    pred_od_<idx>.npy / .json   estimated OD (+ zone labels, totals)
    sample_<idx>.png            predicted vs true heatmaps + per-zone + scatter
    sumo_<idx>/                 trips/routes/sumocfg when --sumo-gui is used
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.config import get_device, load_config, resolve, set_seed
from src.data.graph import load_graph
from src.metrics import to_counts
from src.models.gnn import build_model


def pick_sample(dcfg, args) -> Path:
    """Resolve which .pkl to load from --sample / --index / --regime / --random."""
    data_root = resolve(dcfg["paths"]["samples_dir"]).parent
    if args.sample:
        return Path(args.sample)
    manifest = json.loads(resolve(dcfg["paths"]["manifest"]).read_text())
    rows = [r for r in manifest["samples"] if r["split"] == "test"]
    if args.index is not None:
        rows = [r for r in rows if r["idx"] == args.index] or rows
        if not any(r["idx"] == args.index for r in rows):
            print(f"[infer] idx {args.index} not in test split; using a random test sample")
    if args.regime:
        rows = [r for r in rows if r["tod"] == args.regime] or rows
    if args.index is not None and any(r["idx"] == args.index for r in rows):
        row = next(r for r in rows if r["idx"] == args.index)
    else:
        row = rows[np.random.default_rng(args.seed).integers(len(rows))]
    return data_root / row["file"]


def build_single(graph, norm, x_dyn_raw: np.ndarray) -> Data:
    d_mean = torch.tensor(norm["dynamic"]["mean"], dtype=torch.float32)
    d_std = torch.tensor(norm["dynamic"]["std"], dtype=torch.float32)
    s_mean = torch.tensor(norm["static"]["mean"], dtype=torch.float32)
    s_std = torch.tensor(norm["static"]["std"], dtype=torch.float32)
    static = torch.tensor(graph["static"], dtype=torch.float32).clamp(min=0)
    static_n = (torch.log1p(static) - s_mean) / s_std
    dyn = torch.tensor(x_dyn_raw, dtype=torch.float32).clamp(min=0)
    dyn_n = (torch.log1p(dyn) - d_mean) / d_std
    return Data(x=torch.cat([static_n, dyn_n], dim=1),
                edge_index=torch.as_tensor(graph["edge_index"], dtype=torch.long),
                zone=torch.as_tensor(graph["node_zone"], dtype=torch.long),
                num_nodes=graph["n_nodes"])


def build_enriched_single(graph, norm, resim) -> Data:
    """Build the enriched Data (9 node feats + per-link turn weights) for one
    re-simulated sample (mirrors src/data/build_enriched.py)."""
    ei = graph["edge_index"]; src, dst = ei[0], ei[1]
    static = graph["static"].astype(np.float32)
    ts = resim["x_ts"]
    aggr = np.stack([ts[:, :, 0].sum(0), ts[:, :, 1].mean(0),
                     ts[:, :, 2].mean(0), ts[:, :, 3].mean(0)], 1)
    turn = resim["turn"].astype(np.float32)
    out_t = np.zeros(len(static), np.float32); np.add.at(out_t, src, turn)
    in_t = np.zeros(len(static), np.float32); np.add.at(in_t, dst, turn)
    x = np.concatenate([static, aggr, np.stack([out_t, in_t], 1)], 1)
    xn = (np.log1p(np.clip(x, 0, None)) - np.array(norm["x_mean"])) / np.array(norm["x_std"])
    return Data(x=torch.tensor(xn, dtype=torch.float32),
                edge_index=torch.as_tensor(ei, dtype=torch.long),
                edge_weight=torch.tensor(np.log1p(np.clip(turn, 0, None)), dtype=torch.float32),
                zone=torch.as_tensor(graph["node_zone"], dtype=torch.long),
                num_nodes=graph["n_nodes"])


def plot_traffic(graph, dcfg, pred_od, zone_ids, true_turn, idx, out_dir, seed):
    """Simulate the PREDICTED OD and plot its traffic (turn counts on the network)
    next to the true traffic — a reliable static alternative to sumo-gui."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from src import sumo_export as sx
    from src.data.resimulate import _parse_turncounts
    from src.plot_graph import node_positions

    sdir = out_dir / f"sumo_{idx}"; sdir.mkdir(parents=True, exist_ok=True)
    net = resolve(dcfg["paths"]["net_file"]); taz = resolve(dcfg["paths"]["taz_file"])
    trips, routes = sdir / "trips.xml", sdir / "routes.xml"
    sx.write_trips_xml(pred_od, zone_ids, trips, seed=seed)
    ok, err = sx.run_duarouter(net, taz, trips, routes, seed=seed)
    ei = graph["edge_index"]
    eid_idx = {e: i for i, e in enumerate(graph["edge_ids"])}
    link_idx = {(int(ei[0, k]), int(ei[1, k])): k for k in range(ei.shape[1])}
    pred_turn = _parse_turncounts(routes, eid_idx, link_idx) if ok else np.zeros(ei.shape[1])

    pos = node_positions(graph["edge_ids"], net)
    segs = np.stack([pos[ei[0]], pos[ei[1]]], axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(21, 10))
    for a, turn, ttl in [(ax[0], pred_turn, "PREDICTED OD → traffic"),
                         (ax[1], true_turn, "TRUE traffic")]:
        a.set_aspect("equal"); a.axis("off")
        o = np.argsort(turn)
        lc = LineCollection(segs[o], cmap="inferno", linewidths=0.3 + 2.5 * turn[o] / max(turn.max(), 1))
        lc.set_array(np.log1p(turn[o])); a.add_collection(lc)
        a.scatter(pos[:, 0], pos[:, 1], c="#888", s=2, linewidths=0, alpha=0.4)
        a.set_title(f"{ttl}\n{int((turn > 0).sum())} active links, {int(turn.sum())} turns")
    fig.suptitle(f"sample {idx}: predicted vs true traffic (junction turn counts)")
    png = out_dir / f"traffic_{idx}.png"
    fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return png


def visualize(pred, true, zone_ids, meta, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z = pred.shape[0]
    off = ~np.eye(Z, dtype=bool)
    vmax = max(np.log1p(true).max(), np.log1p(pred).max(), 1e-3)
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))

    for a, M, ttl in [(ax[0, 0], pred, "PREDICTED OD"), (ax[0, 1], true, "TRUE OD")]:
        im = a.imshow(np.log1p(M), cmap="magma", vmin=0, vmax=vmax)
        a.set_title(f"{ttl}  (total={M.sum():.0f} trips)")
        a.set_xlabel("destination zone"); a.set_ylabel("origin zone")
        fig.colorbar(im, ax=a, fraction=0.046, label="log1p(trips)")

    # per-zone origin demand (sum over destinations)
    po, to = pred.sum(1), true.sum(1)
    x = np.arange(Z)
    ax[1, 0].bar(x - 0.2, to, 0.4, label="true", color="#444")
    ax[1, 0].bar(x + 0.2, po, 0.4, label="pred", color="#d1495b")
    ax[1, 0].set_title("per-zone outgoing demand"); ax[1, 0].set_xlabel("zone index")
    ax[1, 0].set_ylabel("trips"); ax[1, 0].legend()

    # cell scatter (the honest view of structural accuracy)
    pv, tv = pred[off], true[off]
    corr = np.corrcoef(pv, tv)[0, 1] if pv.std() > 0 and tv.std() > 0 else float("nan")
    ax[1, 1].scatter(tv, pv, s=8, alpha=0.4, color="#2e7d8c")
    lim = max(tv.max(), pv.max(), 1)
    ax[1, 1].plot([0, lim], [0, lim], "k--", lw=1)
    ax[1, 1].set_title(f"per-cell pred vs true (Pearson r={corr:.2f})")
    ax[1, 1].set_xlabel("true trips"); ax[1, 1].set_ylabel("predicted trips")

    mae = np.abs(pv - tv).mean()
    fig.suptitle(f"sample idx={meta['idx']}  regime={meta['tod']}  "
                 f"MAE={mae:.3f}  totErr={abs(po.sum()-to.sum())/max(to.sum(),1):.2f}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    dcfg, tcfg = cfg["data"], cfg["train"]

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--index", type=int, help="test sample by original idx")
    g.add_argument("--sample", type=str, help="explicit .pkl path")
    g.add_argument("--random", action="store_true", help="random test sample")
    ap.add_argument("--regime", type=str, help="night|noon|morning_peak|evening_peak")
    ap.add_argument("--checkpoint", type=str,
                    default=str(resolve(tcfg["checkpoints"]["dir"]) / tcfg["checkpoints"]["best_name"]))
    ap.add_argument("--out", type=str, default="data/infer")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--sumo-gui", action="store_true", help="launch sumo-gui on the predicted OD")
    ap.add_argument("--plot-traffic", action="store_true",
                    help="static plot of predicted vs true traffic (turn counts)")
    ap.add_argument("--seed", type=int, default=tcfg["seed"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph()
    zone_ids = graph["zone_ids"]
    enriched = dcfg.get("enriched")

    # --- load one sample + build the model input graph ----------------------
    if enriched:
        norm = json.loads(resolve("data/norm_enr.json").read_text())
        rdir = resolve("data/resim")
        if args.sample:
            fp = Path(args.sample)
        elif args.index is not None:
            fp = rdir / f"resim_{args.index:06d}.pkl"
        else:
            files = sorted(rdir.glob("*.pkl"))
            fp = files[np.random.default_rng(args.seed).integers(len(files))]
        s = pickle.load(open(fp, "rb"))
        tot = int(s["od"].sum())
        breaks = np.exp(np.linspace(np.log(dcfg["demand_range"][0]),
                                    np.log(dcfg["demand_range"][1]), dcfg["n_tiers"] + 1))
        tier = int(np.clip(np.searchsorted(breaks, tot, "right") - 1, 0, dcfg["n_tiers"] - 1))
        meta = {"idx": s["idx"], "tod": dcfg["tier_labels"][tier], "tier": tier, "total_demand": tot}
        build_data = lambda: build_enriched_single(graph, norm, s)
    else:
        norm = json.loads(resolve(dcfg["paths"]["norm_stats"]).read_text())
        s = pickle.load(open(pick_sample(dcfg, args), "rb"))
        meta = {"idx": s["idx"], "tod": s["tod"], "tier": s["tier"], "total_demand": s["total_demand"]}
        build_data = lambda: build_single(graph, norm, s["x_dyn"])
    true_od = s["od"].astype(np.float64)
    print(f"[infer] idx={meta['idx']}  regime={meta['tod']}  true_total={int(true_od.sum())}")

    # --- inference (brief GPU use; well under the watchdog) -----------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head = ckpt["model_cfg"]["model"].get("head", "cell")
    fcfg = ckpt["model_cfg"]["model"].get("furness", {"beta": 1.0, "iters": 40})
    model = build_model(ckpt["model_cfg"], ckpt["feature_dim"], ckpt["n_zones"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    data = build_data().to(device)
    with torch.no_grad():
        out = model(Batch.from_data_list([data]))
    if head == "marginal":
        from src.od_reconstruct import furness, zone_distance
        ms = ckpt["marg_stats"]
        prod = np.clip(out["production"][0].cpu().numpy() * np.array(ms["ps"]) + np.array(ms["pm"]), 0, None)
        attr = np.clip(out["attraction"][0].cpu().numpy() * np.array(ms["as"]) + np.array(ms["am"]), 0, None)
        pred_od = furness(prod, attr, np.exp(-fcfg["beta"] * zone_distance(zone_ids)), fcfg["iters"])
    else:
        pred_od = to_counts(out)[0].cpu().numpy().astype(np.float64)
    
    # --- report -------------------------------------------------------------
    off = ~np.eye(len(zone_ids), dtype=bool)
    mae = np.abs(pred_od[off] - true_od[off]).mean()
    print(f"[infer] predicted_total={pred_od.sum():.0f}  MAE={mae:.3f}")
    def top(M, k=5):
        idx = np.dstack(np.unravel_index(np.argsort(M, axis=None)[::-1], M.shape))[0][:k]
        return [(zone_ids[i], zone_ids[j], round(float(M[i, j]), 1)) for i, j in idx]
    print(f"[infer] top true OD pairs: {top(true_od)}")
    print(f"[infer] top pred OD pairs: {top(pred_od)}")

    np.save(out_dir / f"pred_od_{meta['idx']}.npy", pred_od)
    (out_dir / f"pred_od_{meta['idx']}.json").write_text(json.dumps({
        "idx": meta["idx"], "regime": meta["tod"], "zone_ids": zone_ids,
        "predicted_total": float(pred_od.sum()), "true_total": float(true_od.sum()),
        "od_pred": np.rint(pred_od).astype(int).tolist()}, indent=2))

    if not args.no_viz:
        png = out_dir / f"sample_{meta['idx']}.png"
        visualize(pred_od, true_od, zone_ids, meta, png)
        print(f"[infer] visualization -> {png}")

    # --- static traffic plot: predicted OD's traffic vs true traffic --------
    if args.plot_traffic:
        true_turn = s["turn"].astype(float) if enriched and "turn" in s else None
        if true_turn is None:
            print("[infer] --plot-traffic needs an enriched (re-simulated) sample with turn counts")
        else:
            tp = plot_traffic(graph, dcfg, pred_od, zone_ids, true_turn, meta["idx"], out_dir, args.seed)
            print(f"[infer] traffic plot -> {tp}")

    # --- optional sumo-gui on the PREDICTED OD ------------------------------
    if args.sumo_gui:
        from src import sumo_export as sx
        sdir = out_dir / f"sumo_{meta['idx']}"
        sdir.mkdir(exist_ok=True)
        net = resolve(dcfg["paths"]["net_file"]); taz = resolve(dcfg["paths"]["taz_file"])
        trips, routes, scfg = sdir / "trips.xml", sdir / "routes.xml", sdir / "sim.sumocfg"
        n_trips = sx.write_trips_xml(pred_od, zone_ids, trips, seed=args.seed)
        print(f"[infer] wrote {n_trips} trips from predicted OD -> {trips}")
        ok, err = sx.run_duarouter(net, taz, trips, routes, seed=args.seed)
        if not ok:
            print(f"[infer] duarouter failed: {err}\n        files in {sdir}; "
                  f"check SUMO_HOME / net-TAZ ids.")
            return
        sx.write_sumocfg(scfg, net, routes)
        launched, info = sx.launch_sumo_gui(scfg)
        if launched:
            print(f"[infer] sumo-gui launched (press Play ▶): {info}")
        else:
            print(f"[infer] {info}. Run it yourself:\n        SUMO_HOME={sx.SUMO_HOME_DEFAULT} "
                  f"sumo-gui -c {scfg} --start --delay 80")


if __name__ == "__main__":
    main()
