"""
infer.py — Step 4: run the trained model on ONE sample and (optionally) animate it.

Predicts the sample's zone marginals, rebuilds the OD with the Furness gravity
prior, prints a short report + saves a predicted-vs-true figure. With --sumo-gui
it launches SUMO on the BASE (true) OD and/or the PREDICTED OD so you can compare
the two traffic states (use --meso for the fast mesoscopic model at high demand).

Run:
    python -m src.odpipe.infer --max-trips
    python -m src.odpipe.infer --index 42 --sumo-gui --which both --meso
    python -m src.odpipe.infer --regime gridlock --sumo-gui --which pred --scale 1
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.config import load_yaml, resolve, set_seed
from src.models.gnn import build_model
from src.od_reconstruct import furness, zone_distance
from src.odpipe import CFG
from src.odpipe.dataset import features
from src.odpipe.graph import load_graph
from src.train import destd_marginals


def pick(cfg, args):
    rows = [r for r in json.loads(resolve(cfg["paths"]["manifest"]).read_text())["samples"]
            if r["split"] == "test"]
    if args.regime:
        rows = [r for r in rows if r["tod"] == args.regime] or rows
    if args.max_trips:
        return max(rows, key=lambda r: r["total_demand"])
    if args.index is not None and any(r["idx"] == args.index for r in rows):
        return next(r for r in rows if r["idx"] == args.index)
    return rows[np.random.default_rng(args.seed).integers(len(rows))]


def launch(cfg, od, zone_ids, tag, idx, out, seed, meso):
    from src import sumo_export as sx
    sdir = out / f"sumo_{idx}_{tag}"; sdir.mkdir(parents=True, exist_ok=True)
    net = resolve(cfg["paths"]["net_file"]); taz = resolve(cfg["paths"]["taz_file"])
    trips, routes, scfg = sdir / "trips.xml", sdir / "routes.xml", sdir / "sim.sumocfg"
    n = sx.write_trips_xml(od, zone_ids, trips, seed=seed, prefix=f"{tag}_")
    ok, err = sx.run_duarouter(net, taz, trips, routes, seed=seed)
    if not ok:
        print(f"[infer] {tag}: duarouter failed: {err}"); return
    sx.write_sumocfg(scfg, net, routes)
    launched, _ = sx.launch_sumo_gui(scfg, meso=meso)
    print(f"[infer] {tag} OD ({n} trips): sumo-gui {'launched' if launched else 'NOT found'} "
          f"(meso={meso})\n        open: sumo-gui -c {scfg} --start --delay 80"
          + (" --mesosim" if meso else ""))


def main():
    cfg = load_yaml(CFG)
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--index", type=int)
    g.add_argument("--random", action="store_true")
    g.add_argument("--max-trips", action="store_true", help="busiest test sample")
    ap.add_argument("--regime", type=str, help="one of the tier_labels")
    ap.add_argument("--sumo-gui", action="store_true")
    ap.add_argument("--which", choices=["base", "pred", "both"], default="both",
                    help="which OD to animate: base=true, pred=predicted")
    ap.add_argument("--meso", action="store_true", help="mesoscopic model in the GUI")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    graph = load_graph(); zone_ids = graph["zone_ids"]; z = len(zone_ids)
    ei = graph["edge_index"]; src, dst = ei[0], ei[1]
    norm = json.loads(resolve(f"{cfg['paths']['data_dir']}/norm.json").read_text())
    xm, xs = np.array(norm["x_mean"]), np.array(norm["x_std"])

    row = pick(cfg, args)
    s = pickle.load(open(resolve(cfg["paths"]["data_dir"]) / row["file"], "rb"))
    x, turn = features(s, graph["static"].astype(np.float32), src, dst, graph["n_nodes"])
    data = Data(x=torch.tensor((np.log1p(np.clip(x, 0, None)) - xm) / xs, dtype=torch.float32),
                edge_index=torch.as_tensor(ei, dtype=torch.long),
                edge_weight=torch.tensor(np.log1p(np.clip(turn, 0, None)), dtype=torch.float32),
                zone=torch.as_tensor(graph["node_zone"], dtype=torch.long),
                num_nodes=graph["n_nodes"]).to(device)

    ckpt = torch.load(resolve(cfg["paths"]["checkpoint"]), map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"], ckpt["feature_dim"], z).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    ms = {k: torch.tensor(v, device=device) for k, v in ckpt["marg_stats"].items()}
    beta = cfg["furness"]["beta"]
    if resolve(cfg["paths"]["metrics"]).exists():
        beta = json.loads(resolve(cfg["paths"]["metrics"]).read_text()).get("gravity_beta", beta)

    with torch.no_grad():
        prod, attr = destd_marginals(model(Batch.from_data_list([data])), ms)
    prod, attr = prod[0].cpu().numpy(), attr[0].cpu().numpy()
    pred_od = furness(prod, attr, np.exp(-beta * zone_distance(zone_ids)), cfg["furness"]["iters"])
    true_od = s["od"].astype(np.float64)

    off = ~np.eye(z, dtype=bool)
    mae = np.abs(pred_od[off] - true_od[off]).mean()
    cc = np.corrcoef(pred_od[off], true_od[off])[0, 1]
    print(f"[infer] idx={s['idx']} regime={s['tod']} true_total={int(true_od.sum())} "
          f"pred_total={pred_od.sum():.0f} MAE={mae:.2f} cell_corr={cc:.3f}")

    out = resolve(f"{cfg['paths']['data_dir']}/infer"); out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"pred_{s['idx']}.npy", pred_od)
    _plot(pred_od, true_od, zone_ids, s, mae, out / f"sample_{s['idx']}.png")
    print(f"[infer] figure -> {out}/sample_{s['idx']}.png")

    if args.sumo_gui:
        if args.which in ("base", "both"):
            launch(cfg, true_od * args.scale, zone_ids, "base", s["idx"], out, args.seed, args.meso)
        if args.which in ("pred", "both"):
            launch(cfg, pred_od * args.scale, zone_ids, "pred", s["idx"], out, args.seed, args.meso)


def _plot(pred, true, zone_ids, s, mae, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = len(zone_ids); off = ~np.eye(z, dtype=bool); lab = z <= 16
    vmax = max(np.log1p(true).max(), np.log1p(pred).max(), 1e-3)
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for a, M, t in [(ax[0, 0], true, "TRUE OD"), (ax[0, 1], pred, "PREDICTED OD")]:
        im = a.imshow(np.log1p(M), cmap="magma", vmin=0, vmax=vmax)
        a.set_title(f"{t} (total={M.sum():.0f})"); a.set_xlabel("destination"); a.set_ylabel("origin")
        if lab:
            a.set_xticks(range(z)); a.set_yticks(range(z))
            a.set_xticklabels(zone_ids, fontsize=7, rotation=90); a.set_yticklabels(zone_ids, fontsize=7)
        fig.colorbar(im, ax=a, fraction=0.046)
    x = np.arange(z)
    ax[1, 0].bar(x - 0.2, true.sum(1), 0.4, label="true", color="#444")
    ax[1, 0].bar(x + 0.2, pred.sum(1), 0.4, label="pred", color="#d1495b")
    ax[1, 0].set_title("per-zone production"); ax[1, 0].legend()
    if lab:
        ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(zone_ids, fontsize=7, rotation=90)
    pv, tv = pred[off], true[off]
    r = np.corrcoef(pv, tv)[0, 1] if pv.std() and tv.std() else float("nan")
    ax[1, 1].scatter(tv, pv, s=14, alpha=0.5, color="#2e7d8c")
    lim = max(tv.max(), pv.max(), 1); ax[1, 1].plot([0, lim], [0, lim], "k--", lw=1)
    ax[1, 1].set_title(f"per-cell pred vs true (r={r:.2f})")
    ax[1, 1].set_xlabel("true trips"); ax[1, 1].set_ylabel("predicted trips")
    fig.suptitle(f"inference — idx={s['idx']} regime={s['tod']} MAE={mae:.2f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(png, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
