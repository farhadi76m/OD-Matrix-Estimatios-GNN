"""
infer.py — Step 5: run the trained 9-zone model on ONE sample.

Loads a test sample, predicts its zone marginals, rebuilds the 9x9 OD with the
Furness gravity prior, prints a short report (top OD pairs, totals, MAE) and
saves a predicted-vs-true figure.

Run:
    python -m src.zones9.infer --random
    python -m src.zones9.infer --index 1234
    python -m src.zones9.infer --regime morning_peak
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
from src.train import destd_marginals
from src.zones9.build_packs import _features
from src.zones9.graph import load_graph


def pick(cfg, args):
    manifest = json.loads(resolve(cfg["paths"]["manifest"]).read_text())
    rows = [r for r in manifest["samples"] if r["split"] == "test"]
    if args.regime:
        rows = [r for r in rows if r["tod"] == args.regime] or rows
    if args.index is not None and any(r["idx"] == args.index for r in rows):
        return next(r for r in rows if r["idx"] == args.index)
    return rows[np.random.default_rng(args.seed).integers(len(rows))]


def main():
    cfg = load_yaml("zones9")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--index", type=int)
    g.add_argument("--random", action="store_true")
    ap.add_argument("--regime", type=str, help="night|noon|morning_peak|evening_peak")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    graph = load_graph(); zone_ids = graph["zone_ids"]; z = len(zone_ids)
    ei = graph["edge_index"]; src, dst = ei[0], ei[1]
    norm = json.loads(resolve(f"{cfg['paths']['data_dir']}/norm.json").read_text())
    xm = np.array(norm["x_mean"]); xs = np.array(norm["x_std"])

    row = pick(cfg, args)
    s = pickle.load(open(resolve(cfg["paths"]["data_dir"]) / row["file"], "rb"))
    x, turn = _features(s, graph["static"].astype(np.float32), src, dst, graph["n_nodes"])
    data = Data(x=torch.tensor((np.log1p(np.clip(x, 0, None)) - xm) / xs, dtype=torch.float32),
                edge_index=torch.as_tensor(ei, dtype=torch.long),
                edge_weight=torch.tensor(np.log1p(np.clip(turn, 0, None)), dtype=torch.float32),
                zone=torch.as_tensor(graph["node_zone"], dtype=torch.long),
                num_nodes=graph["n_nodes"]).to(device)

    ckpt = torch.load(resolve(cfg["paths"]["checkpoint"]), map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"], ckpt["feature_dim"], z).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    ms = {k: torch.tensor(v, device=device) for k, v in ckpt["marg_stats"].items()}
    beta = json.loads(resolve(cfg["paths"]["metrics"]).read_text()).get("gravity_beta",
            cfg["furness"]["beta"]) if resolve(cfg["paths"]["metrics"]).exists() else cfg["furness"]["beta"]

    with torch.no_grad():
        prod, attr = destd_marginals(model(Batch.from_data_list([data])), ms)
    prod, attr = prod[0].cpu().numpy(), attr[0].cpu().numpy()
    pred_od = furness(prod, attr, np.exp(-beta * zone_distance(zone_ids)), cfg["furness"]["iters"])
    true_od = s["od"].astype(np.float64)

    off = ~np.eye(z, dtype=bool)
    mae = np.abs(pred_od[off] - true_od[off]).mean()
    print(f"[infer9] idx={s['idx']} regime={s['tod']} true_total={int(true_od.sum())} "
          f"pred_total={pred_od.sum():.0f} MAE={mae:.2f} cell_corr={np.corrcoef(pred_od[off], true_od[off])[0,1]:.3f}")

    def top(M, k=5):
        ij = np.dstack(np.unravel_index(np.argsort(M, None)[::-1], M.shape))[0][:k]
        return [(zone_ids[i], zone_ids[j], round(float(M[i, j]), 1)) for i, j in ij]
    print(f"[infer9] top TRUE pairs: {top(true_od)}")
    print(f"[infer9] top PRED pairs: {top(pred_od)}")

    out = resolve(f"{cfg['paths']['data_dir']}/infer"); out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"pred_{s['idx']}.npy", pred_od)
    (out / f"pred_{s['idx']}.json").write_text(json.dumps(
        {"idx": s["idx"], "regime": s["tod"], "zone_ids": zone_ids,
         "pred_total": float(pred_od.sum()), "true_total": float(true_od.sum()),
         "od_pred": np.rint(pred_od).astype(int).tolist()}, indent=2))
    _plot(pred_od, true_od, zone_ids, s, mae, out / f"sample_{s['idx']}.png")
    print(f"[infer9] -> {out}/sample_{s['idx']}.png")


def _plot(pred, true, zone_ids, s, mae, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = len(zone_ids); off = ~np.eye(z, dtype=bool)
    vmax = max(np.log1p(true).max(), np.log1p(pred).max(), 1e-3)
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for a, M, t in [(ax[0, 0], true, "TRUE OD"), (ax[0, 1], pred, "PREDICTED OD")]:
        im = a.imshow(np.log1p(M), cmap="magma", vmin=0, vmax=vmax)
        a.set_title(f"{t} (total={M.sum():.0f})"); a.set_xticks(range(z)); a.set_yticks(range(z))
        a.set_xticklabels(zone_ids, fontsize=7, rotation=90); a.set_yticklabels(zone_ids, fontsize=7)
        a.set_xlabel("destination"); a.set_ylabel("origin"); fig.colorbar(im, ax=a, fraction=0.046)
    x = np.arange(z)
    ax[1, 0].bar(x - 0.2, true.sum(1), 0.4, label="true", color="#444")
    ax[1, 0].bar(x + 0.2, pred.sum(1), 0.4, label="pred", color="#d1495b")
    ax[1, 0].set_title("per-zone production"); ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels(zone_ids, fontsize=7, rotation=90); ax[1, 0].legend()
    pv, tv = pred[off], true[off]
    r = np.corrcoef(pv, tv)[0, 1] if pv.std() and tv.std() else float("nan")
    ax[1, 1].scatter(tv, pv, s=14, alpha=0.5, color="#2e7d8c")
    lim = max(tv.max(), pv.max(), 1); ax[1, 1].plot([0, lim], [0, lim], "k--", lw=1)
    ax[1, 1].set_title(f"per-cell pred vs true (r={r:.2f})")
    ax[1, 1].set_xlabel("true trips"); ax[1, 1].set_ylabel("predicted trips")
    fig.suptitle(f"9-zone inference — idx={s['idx']} regime={s['tod']} MAE={mae:.2f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(png, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
