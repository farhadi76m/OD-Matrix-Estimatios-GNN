"""
train.py — training + validation loop for the OD-estimation GNN.

Two heads (configs/model.yaml -> model.head):
  * "marginal" (default): predict per-zone production & attraction (log1p space);
    loss = Huber on log1p marginals. Early-stop on marginal MAE. The full OD is
    rebuilt at eval/inference with a Furness gravity prior.
  * "cell": old direct 36x36 OD head; Huber on log1p(OD) + total term.

Checkpoints after EVERY epoch and auto-resumes from checkpoints/last.pt, honoring
a per-invocation wall-time budget (--max-seconds) so long trainings run in short
resumable chunks (e.g. under a ~45s GPU watchdog).

Run:
    python -m src.train
    python -m src.train --epochs 30 --max-seconds 18   # one resumable chunk
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_device, load_config, resolve, set_seed
from src.data.dataset import make_enriched_loaders, make_loaders
from src.metrics import od_metrics, to_counts
from src.models.gnn import build_model


def _offdiag(z, device):
    return (1.0 - torch.eye(z, device=device)).bool()


# ── marginal head ────────────────────────────────────────────────────────────

def marginal_stats(dataset, device):
    """Per-zone mean/std of production & attraction from the train split.

    Targets are standardised and fit with MSE so the model matches the marginal
    *variance* (log1p+Huber regressed predictions to the mean -> under-dispersed
    marginals -> Furness collapses to near-uniform OD)."""
    y = dataset.y  # [n, Z, Z]
    prod, attr = y.sum(2), y.sum(1)
    return {"pm": prod.mean(0).to(device), "ps": prod.std(0).clamp(min=1.0).to(device),
            "am": attr.mean(0).to(device), "as": attr.std(0).clamp(min=1.0).to(device)}


def marginal_loss(out, y, ms, wp, wa):
    prod_t = (y.sum(2) - ms["pm"]) / ms["ps"]
    attr_t = (y.sum(1) - ms["am"]) / ms["as"]
    lp = F.mse_loss(out["production"], prod_t)
    la = F.mse_loss(out["attraction"], attr_t)
    return wp * lp + wa * la, lp.item(), la.item()


def destd_marginals(out, ms):
    prod = (out["production"] * ms["ps"] + ms["pm"]).clamp(min=0)
    attr = (out["attraction"] * ms["as"] + ms["am"]).clamp(min=0)
    return prod, attr


@torch.no_grad()
def evaluate_marginal(model, loader, device, ms):
    model.eval()
    pp, pt, ap, at = [], [], [], []
    for data in loader:
        data = data.to(device)
        prod, attr = destd_marginals(model(data), ms)
        y = data.y.view(data.num_graphs, model.n_zones, model.n_zones)
        pp.append(prod.cpu()); pt.append(y.sum(2).cpu())
        ap.append(attr.cpu()); at.append(y.sum(1).cpu())
    pp, pt = torch.cat(pp).numpy(), torch.cat(pt).numpy()
    ap, at = torch.cat(ap).numpy(), torch.cat(at).numpy()
    mae = (np.abs(pp - pt).mean() + np.abs(ap - at).mean()) / 2
    return {"mae": float(mae),
            "prod_corr": float(np.corrcoef(pp.ravel(), pt.ravel())[0, 1]),
            "attr_corr": float(np.corrcoef(ap.ravel(), at.ravel())[0, 1])}


# ── cell head (legacy) ───────────────────────────────────────────────────────

def cell_loss(pred, y, mask, beta, total_w):
    y_log = torch.log1p(y.clamp(min=0))
    m = mask.unsqueeze(0).expand_as(pred)
    main = F.smooth_l1_loss(pred[m], y_log[m], beta=beta)
    if total_w > 0:
        pc = to_counts(pred)
        total = F.l1_loss(torch.log1p((pc * mask).sum((1, 2))),
                          torch.log1p((y * mask).sum((1, 2))))
        return main + total_w * total, main.item(), total.item()
    return main, main.item(), 0.0


@torch.no_grad()
def evaluate_cell(model, loader, z, device):
    model.eval()
    agg = {"mae": 0.0, "rmse": 0.0, "total_flow_err": 0.0}; n = 0
    for data in loader:
        data = data.to(device)
        y = data.y.view(data.num_graphs, z, z)
        m = od_metrics(to_counts(model(data)), y); b = data.num_graphs
        for k in agg:
            agg[k] += m[k] * b
        n += b
    return {k: v / max(n, 1) for k, v in agg.items()}


def _ckpt(model, opt, sched, mcfg, info, epoch, val, best_mae, bad, ms):
    marg = {k: v.cpu().tolist() for k, v in ms.items()} if ms else None
    return {"model_state": model.state_dict(), "opt_state": opt.state_dict(),
            "sched_state": sched.state_dict(), "model_cfg": mcfg,
            "feature_dim": info["feature_dim"], "n_zones": info["n_zones"],
            "epoch": epoch, "val": val, "best_mae": best_mae, "bad": bad,
            "marg_stats": marg}


def main() -> None:
    cfg = load_config()
    tcfg, mcfg = cfg["train"], cfg["model"]
    head = mcfg["model"].get("head", "cell")

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=tcfg["train"]["epochs"])
    ap.add_argument("--batch-size", type=int, default=tcfg["train"]["batch_size"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    set_seed(tcfg["seed"])
    device = torch.device(args.device) if args.device else get_device()
    print(f"[train] head={head} device={device} epochs={args.epochs} batch={args.batch_size}", flush=True)

    loaders = make_enriched_loaders if cfg["data"].get("enriched") else make_loaders
    train_loader, val_loader, _, info = loaders(args.batch_size, tcfg["train"]["num_workers"])
    z = info["n_zones"]
    model = build_model(mcfg, info["feature_dim"], z).to(device)
    print(f"[train] feature_dim={info['feature_dim']} n_nodes={info['n_nodes']} "
          f"train_batches={len(train_loader)} params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    ms = marginal_stats(train_loader.dataset, device) if head == "marginal" else None

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["train"]["lr"],
                            weight_decay=tcfg["train"]["weight_decay"])
    warm = tcfg["train"]["warmup_epochs"]; use_cos = tcfg["train"]["scheduler"] == "cosine"

    def lr_factor(ep):
        if ep < warm:
            return (ep + 1) / max(warm, 1)
        if not use_cos:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(args.epochs - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    ckpt_dir = resolve(tcfg["checkpoints"]["dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / tcfg["checkpoints"]["last_name"]
    best_path = ckpt_dir / tcfg["checkpoints"]["best_name"]

    start_epoch, best_mae, bad = 0, float("inf"), 0
    if last_path.exists() and not args.no_resume:
        prev = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(prev["model_state"]); opt.load_state_dict(prev["opt_state"])
        sched.load_state_dict(prev["sched_state"])
        start_epoch = prev["epoch"] + 1; best_mae = prev.get("best_mae", float("inf"))
        bad = prev.get("bad", 0)
        print(f"[train] resumed @ epoch {prev['epoch']} (best_mae={best_mae:.4f}) "
              f"-> epoch {start_epoch}", flush=True)

    if start_epoch >= args.epochs:
        print(f"[train] COMPLETE (already at {start_epoch}/{args.epochs}); best_mae={best_mae:.4f}", flush=True)
        return

    mask = _offdiag(z, device)
    lc = tcfg["train"]["loss"]
    beta, total_w = lc["beta"], lc["total_demand_weight"]
    wp, wa = lc.get("production_weight", 1.0), lc.get("attraction_weight", 1.0)
    clip = tcfg["train"]["grad_clip"]; log_every = tcfg["train"]["log_every"]
    patience = tcfg["train"]["early_stopping"]["patience"]

    for ep in range(start_epoch, args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for bi, data in enumerate(train_loader):
            data = data.to(device); opt.zero_grad()
            out = model(data); y = data.y.view(data.num_graphs, z, z)
            if head == "marginal":
                loss, a_i, b_i = marginal_loss(out, y, ms, wp, wa)
            else:
                loss, a_i, b_i = cell_loss(out, y, mask, beta, total_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); run += loss.item()
            if log_every and bi % log_every == 0:
                print(f"  ep{ep:03d} b{bi:04d} loss={loss.item():.4f} ({a_i:.4f}/{b_i:.4f})", flush=True)
        sched.step()

        if head == "marginal":
            val = evaluate_marginal(model, val_loader, device, ms)
            extra = f"prod_corr={val['prod_corr']:.3f} attr_corr={val['attr_corr']:.3f}"
        else:
            val = evaluate_cell(model, val_loader, z, device)
            extra = f"rmse={val['rmse']:.3f} totErr={val['total_flow_err']:.3f}"
        print(f"[ep {ep:03d}] train_loss={run/max(len(train_loader),1):.4f} "
              f"val_mae={val['mae']:.4f} {extra} lr={opt.param_groups[0]['lr']:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)

        is_best = val["mae"] < best_mae
        best_mae, bad = (val["mae"], 0) if is_best else (best_mae, bad + 1)
        ck = _ckpt(model, opt, sched, mcfg, info, ep, val, best_mae, bad, ms)
        torch.save(ck, last_path)
        if is_best:
            torch.save(ck, best_path); print(f"          ↳ new best val_mae={best_mae:.4f}", flush=True)

        if bad >= patience:
            print(f"[train] COMPLETE (early stop @ {ep}); best_mae={best_mae:.4f}", flush=True); return
        if args.max_seconds and (time.time() - t_start) > args.max_seconds:
            print(f"[train] PAUSED after epoch {ep} ({time.time()-t_start:.1f}s>{args.max_seconds:.0f}s); "
                  f"best_mae={best_mae:.4f}", flush=True); return

    print(f"[train] COMPLETE ({args.epochs} epochs); best_mae={best_mae:.4f} -> {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
