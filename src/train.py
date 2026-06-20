"""
train.py — training + validation loop for the OD-estimation GNN.

Loss = SmoothL1 on log1p(OD) over off-diagonal cells, plus an optional soft
term matching predicted vs true total trips. Validation metrics are computed in
trip-count space. Best checkpoint is selected on val MAE.

The loop checkpoints after EVERY epoch and auto-resumes from checkpoints/last.pt,
and honours a per-invocation wall-time budget (--max-seconds). This lets long
trainings run in short, resumable chunks (e.g. under a GPU watchdog): a killed
or paused run loses at most the in-progress epoch.

Run:
    python -m src.train                              # full config
    python -m src.train --epochs 30 --max-seconds 18 # one resumable chunk
    python -m src.train --device cpu                 # force CPU
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.config import get_device, load_config, resolve, set_seed
from src.data.dataset import make_loaders
from src.metrics import od_metrics, to_counts
from src.models.gnn import build_model


def _offdiag(z, device):
    return (1.0 - torch.eye(z, device=device)).bool()


def compute_loss(pred, y, mask, beta, total_w):
    """pred, y: [B, Z, Z]; pred in log1p space, y in raw counts."""
    y_log = torch.log1p(y.clamp(min=0))
    m = mask.unsqueeze(0).expand_as(pred)
    main = F.smooth_l1_loss(pred[m], y_log[m], beta=beta)
    if total_w > 0:
        pc = to_counts(pred)
        tot_p = (pc * mask).sum(dim=(1, 2))
        tot_t = (y * mask).sum(dim=(1, 2))
        total = F.l1_loss(torch.log1p(tot_p), torch.log1p(tot_t))
        return main + total_w * total, main.item(), total.item()
    return main, main.item(), 0.0


def evaluate(model, loader, z, device):
    model.eval()
    agg = {"mae": 0.0, "rmse": 0.0, "total_flow_err": 0.0}
    n = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            y = data.y.view(data.num_graphs, z, z)
            m = od_metrics(to_counts(pred), y)
            b = data.num_graphs
            for k in agg:
                agg[k] += m[k] * b
            n += b
    return {k: v / max(n, 1) for k, v in agg.items()}


def _ckpt(model, opt, sched, mcfg, info, epoch, val, best_mae, bad):
    return {
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "sched_state": sched.state_dict(),
        "model_cfg": mcfg,
        "feature_dim": info["feature_dim"],
        "n_zones": info["n_zones"],
        "epoch": epoch,
        "val": val,
        "best_mae": best_mae,
        "bad": bad,
    }


def main() -> None:
    cfg = load_config()
    tcfg, mcfg = cfg["train"], cfg["model"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=tcfg["train"]["epochs"])
    ap.add_argument("--batch-size", type=int, default=tcfg["train"]["batch_size"])
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="Wall-time budget for THIS invocation (0 = unlimited). "
                         "Checkpoints each epoch and pauses when exceeded.")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    set_seed(tcfg["seed"])
    device = torch.device(args.device) if args.device else get_device()
    print(f"[train] device={device}  epochs={args.epochs}  batch={args.batch_size}", flush=True)

    train_loader, val_loader, _, info = make_loaders(
        args.batch_size, num_workers=tcfg["train"]["num_workers"])
    z = info["n_zones"]
    print(f"[train] feature_dim={info['feature_dim']}  n_nodes={info['n_nodes']}  "
          f"train_batches={len(train_loader)}", flush=True)

    model = build_model(mcfg, info["feature_dim"], z).to(device)
    print(f"[train] model params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["train"]["lr"],
                            weight_decay=tcfg["train"]["weight_decay"])
    warm = tcfg["train"]["warmup_epochs"]
    use_cos = tcfg["train"]["scheduler"] == "cosine"

    def lr_factor(ep):
        if ep < warm:
            return (ep + 1) / max(warm, 1)
        if not use_cos:
            return 1.0
        prog = (ep - warm) / max(args.epochs - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    ckpt_dir = resolve(tcfg["checkpoints"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / tcfg["checkpoints"]["last_name"]
    best_path = ckpt_dir / tcfg["checkpoints"]["best_name"]

    start_epoch, best_mae, bad = 0, float("inf"), 0
    if last_path.exists() and not args.no_resume:
        prev = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(prev["model_state"])
        opt.load_state_dict(prev["opt_state"])
        sched.load_state_dict(prev["sched_state"])
        start_epoch = prev["epoch"] + 1
        best_mae = prev.get("best_mae", float("inf"))
        bad = prev.get("bad", 0)
        print(f"[train] resumed from {last_path} @ epoch {prev['epoch']} "
              f"(best_mae={best_mae:.4f}) -> starting epoch {start_epoch}", flush=True)

    if start_epoch >= args.epochs:
        print(f"[train] COMPLETE (already at epoch {start_epoch}/{args.epochs}); "
              f"best val_mae={best_mae:.4f}", flush=True)
        return

    mask = _offdiag(z, device)
    beta = tcfg["train"]["loss"]["beta"]
    total_w = tcfg["train"]["loss"]["total_demand_weight"]
    clip = tcfg["train"]["grad_clip"]
    log_every = tcfg["train"]["log_every"]
    patience = tcfg["train"]["early_stopping"]["patience"]

    for ep in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for bi, data in enumerate(train_loader):
            data = data.to(device)
            opt.zero_grad()
            pred = model(data)
            y = data.y.view(data.num_graphs, z, z)
            loss, main_i, tot_i = compute_loss(pred, y, mask, beta, total_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            run += loss.item()
            if log_every and bi % log_every == 0:
                print(f"  ep{ep:03d} b{bi:04d} loss={loss.item():.4f} "
                      f"(main={main_i:.4f} tot={tot_i:.4f})", flush=True)
        sched.step()

        val = evaluate(model, val_loader, z, device)
        print(f"[ep {ep:03d}] train_loss={run/max(len(train_loader),1):.4f} "
              f"val_mae={val['mae']:.4f} val_rmse={val['rmse']:.4f} "
              f"val_totErr={val['total_flow_err']:.3f} lr={opt.param_groups[0]['lr']:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)

        is_best = val["mae"] < best_mae
        if is_best:
            best_mae, bad = val["mae"], 0
        else:
            bad += 1
        ck = _ckpt(model, opt, sched, mcfg, info, ep, val, best_mae, bad)
        torch.save(ck, last_path)
        if is_best:
            torch.save(ck, best_path)
            print(f"          ↳ new best val_mae={best_mae:.4f} (saved)", flush=True)

        if bad >= patience:
            print(f"[train] COMPLETE (early stop at epoch {ep}, no val_mae gain "
                  f"in {patience}); best val_mae={best_mae:.4f}", flush=True)
            return
        if args.max_seconds and (time.time() - t_start) > args.max_seconds:
            print(f"[train] PAUSED after epoch {ep} ({time.time()-t_start:.1f}s "
                  f">{args.max_seconds:.0f}s budget); resume to continue. "
                  f"best val_mae={best_mae:.4f}", flush=True)
            return

    print(f"[train] COMPLETE (reached {args.epochs} epochs); "
          f"best val_mae={best_mae:.4f}  checkpoints -> {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
