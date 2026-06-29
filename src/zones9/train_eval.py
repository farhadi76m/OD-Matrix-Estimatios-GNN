"""
train_eval.py — Step 3: train the marginal GNN on the 9-zone dataset and evaluate.

Same proven recipe as the 36-zone study: predict per-zone production/attraction
(standardised-MSE) from edge features + turn-count edge weights, then rebuild the
9x9 OD with a Furness gravity prior. Reuses the model factory and the marginal
loss/eval helpers from the main pipeline; only the data and zone count differ.

Trains on CPU by default (tiny model; avoids the GPU watchdog) and is resumable
(--max-seconds). Evaluation reports marginal corr, reconstructed-OD cell metrics
per regime, the gravity ceiling, and GEH on re-simulated link flows.

Run:
    python -m src.zones9.train_eval                 # train then evaluate
    python -m src.zones9.train_eval --mode eval     # evaluate the saved checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.config import load_yaml, resolve, set_seed
from src.metrics import geh, od_metrics
from src.models.gnn import build_model
from src.od_reconstruct import furness, zone_distance
from src.realworld import common as C
from src.train import destd_marginals, evaluate_marginal, marginal_loss, marginal_stats
from src.zones9.graph import load_graph


# ── data ──────────────────────────────────────────────────────────────────────
class Pack9(torch.utils.data.Dataset):
    def __init__(self, split, graph, norm):
        cfg = load_yaml("zones9")
        pack = torch.load(resolve(f"{cfg['paths']['data_dir']}/pack_{split}.pt"), weights_only=False)
        xm = torch.tensor(norm["x_mean"]); xs = torch.tensor(norm["x_std"])
        self.x = (torch.log1p(pack["x"].clamp(min=0)) - xm) / xs
        self.ew = torch.log1p(pack["ew"].clamp(min=0))
        self.y = pack["y"]; self.meta = pack["meta"]
        self.edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        self.zone = torch.as_tensor(graph["node_zone"], dtype=torch.long)
        self.n_zones, self.n_nodes = graph["n_zones"], graph["n_nodes"]
        self.feature_dim = self.x.shape[2]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return Data(x=self.x[i], edge_index=self.edge_index, edge_weight=self.ew[i],
                    zone=self.zone, y=self.y[i], num_nodes=self.n_nodes)


def make_loaders(graph, norm, batch_size):
    ds = {s: Pack9(s, graph, norm) for s in ("train", "val", "test")}
    mk = lambda s, sh: DataLoader(ds[s], batch_size=batch_size, shuffle=sh)
    info = {"feature_dim": ds["train"].feature_dim, "n_zones": graph["n_zones"]}
    return ds, mk("train", True), mk("val", False), mk("test", False), info


# ── train ─────────────────────────────────────────────────────────────────────
def train(cfg, graph, norm, device, max_seconds):
    tc = cfg["train"]
    mcfg = load_yaml("model")
    ds, tr, va, te, info = make_loaders(graph, norm, tc["batch_size"])
    z = info["n_zones"]
    model = build_model(mcfg, info["feature_dim"], z).to(device)
    ms = marginal_stats(ds["train"], device)
    print(f"[train9] feature_dim={info['feature_dim']} n_zones={z} "
          f"params={sum(p.numel() for p in model.parameters()):,} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    warm = tc["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / max(warm, 1) if e < warm
        else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(tc["epochs"] - warm, 1))))

    last_p = resolve(f"{cfg['paths']['data_dir']}/last.pt"); best_p = resolve(cfg["paths"]["checkpoint"])
    start, best_mae, bad = 0, float("inf"), 0
    if last_p.exists():
        prev = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(prev["model_state"]); opt.load_state_dict(prev["opt_state"])
        sched.load_state_dict(prev["sched_state"]); start = prev["epoch"] + 1
        best_mae, bad = prev["best_mae"], prev["bad"]
        print(f"[train9] resumed -> epoch {start} (best_mae={best_mae:.4f})", flush=True)

    def save(p, ep, val):
        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                    "sched_state": sched.state_dict(), "model_cfg": mcfg,
                    "feature_dim": info["feature_dim"], "n_zones": z, "epoch": ep,
                    "val": val, "best_mae": best_mae, "bad": bad,
                    "marg_stats": {k: v.cpu().tolist() for k, v in ms.items()}}, p)

    hist_p = resolve(f"{cfg['paths']['data_dir']}/history.json")
    history = json.loads(hist_p.read_text()) if hist_p.exists() and start else []
    t0 = time.time()
    for ep in range(start, tc["epochs"]):
        model.train(); run = 0.0
        for data in tr:
            data = data.to(device); opt.zero_grad()
            y = data.y.view(data.num_graphs, z, z)
            loss, _, _ = marginal_loss(model(data), y, ms, 1.0, 1.0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step(); run += loss.item()
        sched.step()
        val = evaluate_marginal(model, va, device, ms)
        is_best = val["mae"] < best_mae
        best_mae, bad = (val["mae"], 0) if is_best else (best_mae, bad + 1)
        save(last_p, ep, val)
        if is_best:
            save(best_p, ep, val)
        history.append({"epoch": ep, "train_loss": run / len(tr), "val_mae": val["mae"],
                        "prod_corr": val["prod_corr"], "attr_corr": val["attr_corr"]})
        hist_p.write_text(json.dumps(history))
        print(f"[ep {ep:03d}] loss={run/len(tr):.4f} val_mae={val['mae']:.4f} "
              f"prod_corr={val['prod_corr']:.3f} attr_corr={val['attr_corr']:.3f}"
              f"{'  *best' if is_best else ''}", flush=True)
        if bad >= tc["patience"]:
            print(f"[train9] early stop @ {ep}"); break
        if max_seconds and time.time() - t0 > max_seconds:
            print(f"[train9] PAUSED after epoch {ep} (budget); re-run to continue"); return False
    print(f"[train9] training complete; best val_mae={best_mae:.4f}")
    return True


# ── evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(cfg, graph, norm, device):
    zone_ids = graph["zone_ids"]; z = len(zone_ids)
    ds, _, _, te, info = make_loaders(graph, norm, cfg["train"]["batch_size"])
    ckpt = torch.load(resolve(cfg["paths"]["checkpoint"]), map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"], ckpt["feature_dim"], z).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    ms = {k: torch.tensor(v, device=device) for k, v in ckpt["marg_stats"].items()}

    pp, ap, pt, at = [], [], [], []
    for data in te:
        data = data.to(device)
        prod, attr = destd_marginals(model(data), ms)
        y = data.y.view(data.num_graphs, z, z)
        pp.append(prod.cpu()); ap.append(attr.cpu()); pt.append(y.sum(2).cpu()); at.append(y.sum(1).cpu())
    prod_p, attr_p = torch.cat(pp).numpy(), torch.cat(ap).numpy()
    prod_t, attr_t = torch.cat(pt).numpy(), torch.cat(at).numpy()
    true_od = ds["test"].y.numpy()
    tiers = np.array([m["tier"] for m in ds["test"].meta])

    # gravity ceiling (TRUE marginals) + best beta, then reconstruct predicted
    beta, ceil = C.pick_beta(prod_t, attr_t, true_od, zone_ids, cfg["furness"]["betas"],
                             cfg["furness"]["iters"], subset=len(true_od))
    deter = np.exp(-beta * zone_distance(zone_ids))
    pred_od = np.stack([furness(prod_p[k], attr_p[k], deter, cfg["furness"]["iters"])
                        for k in range(len(prod_p))])

    rep = {"n_test": len(true_od), "gravity_beta": beta, "reconstruction_ceiling": ceil,
           "marginals": {"production_corr": C.pearson(prod_p, prod_t),
                         "attraction_corr": C.pearson(attr_p, attr_t),
                         "production_mae": float(np.abs(prod_p - prod_t).mean()),
                         "attraction_mae": float(np.abs(attr_p - attr_t).mean())},
           "cell_corr": C.pooled_cell_corr(pred_od, true_od),
           "cell_rmse": C.cell_rmse(pred_od, true_od),
           "total_flow_err": C.total_flow_err(pred_od, true_od), "per_regime": {}}
    for t, name in enumerate(cfg["tier_labels"]):
        m = tiers == t
        if m.sum():
            mm = od_metrics(torch.tensor(pred_od[m]), torch.tensor(true_od[m]))
            rep["per_regime"][name] = {**mm, "cell_corr": C.pooled_cell_corr(pred_od[m], true_od[m]),
                                       "n": int(m.sum())}

    rep["geh"] = _geh_resim(cfg, graph, ds["test"], pred_od, n=25)

    resolve(cfg["paths"]["metrics"]).write_text(json.dumps(rep, indent=2))
    np.savez_compressed(resolve(cfg["paths"]["predictions"]), pred=pred_od, true=true_od,
                        prod_p=prod_p, prod_t=prod_t, attr_p=attr_p, attr_t=attr_t, tier=tiers)
    _print_report(rep)
    return rep


def _geh_resim(cfg, graph, test_ds, pred_od, n=25):
    """Re-simulate predicted OD for a test subset; GEH vs the sample's true link flow."""
    from src.realworld.sim import simulate_od
    idx = np.linspace(0, len(pred_od) - 1, min(n, len(pred_od))).astype(int)
    net, taz = resolve(cfg["paths"]["net_file"]), resolve(cfg["paths"]["taz_file"])
    root = resolve(cfg["paths"]["samples_dir"])
    to_h = 3600.0 / (3600 - 300)                          # entered-count window -> veh/h
    mg, cg = [], []
    for k in idx:
        s = __import__("pickle").load(open(root / f"sample_{test_ds.meta[k]['idx']:06d}.pkl", "rb"))
        r = simulate_od(pred_od[k], graph["zone_ids"], graph["edge_ids"], net, taz,
                        "/tmp/zones9_geh", seed=cfg["sim_seed"])
        if not r["ok"]:
            continue
        true_flow = s["x_dyn"][:, 0]
        mask = true_flow > 1.0
        mg.append(r["flow"][mask] * to_h); cg.append(true_flow[mask] * to_h)
    if not mg:
        return {"geh_mean": None}
    return {**geh(np.concatenate(mg), np.concatenate(cg)), "n_samples": len(mg)}


def _print_report(rep):
    m = rep["marginals"]
    print(f"\n=== 9-zone evaluation (test n={rep['n_test']}) ===")
    print(f"  marginals: production_corr={m['production_corr']:.3f} attraction_corr={m['attraction_corr']:.3f}")
    print(f"  reconstructed OD: cell_corr={rep['cell_corr']:.3f} rmse={rep['cell_rmse']:.2f} "
          f"totErr={rep['total_flow_err']:.3f}")
    print(f"  gravity beta={rep['gravity_beta']} ceiling(TRUE marginals)={rep['reconstruction_ceiling']:.3f}")
    for name, mm in rep["per_regime"].items():
        print(f"    {name:13s} cell_corr={mm['cell_corr']:.3f} rmse={mm['rmse']:.2f} "
              f"totErr={mm['total_flow_err']:.3f} (n={mm['n']})")
    if rep["geh"].get("geh_mean") is not None:
        g = rep["geh"]
        print(f"  GEH(re-sim link flow, {g['n_samples']} samples): mean={g['geh_mean']:.2f} "
              f"frac<5={g['geh_frac_good']:.2f}")


def main():
    cfg = load_yaml("zones9")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "train", "eval"], default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args()
    set_seed(cfg["train"]["seed"])
    device = torch.device(args.device)
    graph = load_graph()
    norm = json.loads(resolve(f"{cfg['paths']['data_dir']}/norm.json").read_text())

    done = True
    if args.mode in ("all", "train"):
        done = train(cfg, graph, norm, device, args.max_seconds)
    if args.mode == "eval" or (args.mode == "all" and done):
        evaluate(cfg, graph, norm, device)


if __name__ == "__main__":
    main()
