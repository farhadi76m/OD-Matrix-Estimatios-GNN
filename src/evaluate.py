"""
evaluate.py — evaluate a trained checkpoint on the test split.

Reports, in trip-count space and over off-diagonal cells:
  * overall MAE / RMSE / total-flow error
  * the same broken down per time-of-day regime (the 4 demand tiers)
  * GEH on implied link flows.

GEH needs link flows. SUMO is unavailable to re-simulate, so we fit a ridge
linear assignment operator W (OD-pairs -> edge 'entered' counts) on the TRAIN
split (a BPR-style linear-assignment surrogate) and report:
  * GEH(predicted-OD implied flow, observed flow)   — the model's link-flow fit
  * GEH(true-OD implied flow,      observed flow)   — the assignment's own ceiling
both converted from the 3300 s collection window to hourly volumes.

Run:
    python -m src.evaluate
    python -m src.evaluate --checkpoint checkpoints/best.pt --save-preds
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

from src.config import get_device, load_config, resolve, set_seed
from src.data.dataset import ODDataset
from src.metrics import geh, od_metrics, to_counts
from src.models.gnn import build_model
from torch_geometric.loader import DataLoader


def _offdiag_vec_indices(z: int) -> np.ndarray:
    return np.where(~np.eye(z, dtype=bool).reshape(-1))[0]


def load_raw_split(split: str, manifest, samples_root, z):
    """Return (od_offdiag [n,P], flow [n,E]) raw from the .pkl files."""
    off = _offdiag_vec_indices(z)
    ods, flows = [], []
    for r in manifest["samples"]:
        if r["split"] != split:
            continue
        with open(samples_root / r["file"], "rb") as f:
            s = pickle.load(f)
        ods.append(s["od"].reshape(-1)[off].astype(np.float64))
        flows.append(s["x_dyn"][:, 0].astype(np.float64))  # edge_flow is column 0
    return np.stack(ods), np.stack(flows)


def fit_assignment(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge: solve (X^T X + alpha I) W = X^T Y  ->  Y ~ X @ W. Returns W [P,E]."""
    P = X.shape[1]
    G = X.T @ X + alpha * np.eye(P)
    return np.linalg.solve(G, X.T @ Y)


def main() -> None:
    cfg = load_config()
    dcfg, tcfg = cfg["data"], cfg["train"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str,
                    default=str(resolve(tcfg["checkpoints"]["dir"]) / tcfg["checkpoints"]["best_name"]))
    ap.add_argument("--save-preds", action="store_true")
    args = ap.parse_args()

    set_seed(tcfg["seed"])
    device = get_device()

    test_ds = ODDataset("test")
    z = test_ds.n_zones
    loader = DataLoader(test_ds, batch_size=tcfg["train"]["batch_size"], shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"], ckpt["feature_dim"], z).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] loaded {args.checkpoint} (epoch {ckpt['epoch']}, "
          f"val_mae={ckpt['val']['mae']:.4f})")

    # --- predictions over the test split (loader is unshuffled => order matches)
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds.append(to_counts(model(data)).cpu())
            trues.append(data.y.view(data.num_graphs, z, z).cpu())
    preds = torch.cat(preds)  # [Nt, Z, Z]
    trues = torch.cat(trues)
    tiers = np.array([m["tier"] for m in test_ds.meta])
    tier_labels = dcfg["tier_labels"]

    # --- overall + per-regime OD metrics
    overall = od_metrics(preds, trues)
    print("\n=== OD metrics (test, off-diagonal, trips) ===")
    print(f"  overall   MAE={overall['mae']:.4f}  RMSE={overall['rmse']:.4f}  "
          f"totFlowErr={overall['total_flow_err']:.3f}")
    per_regime = {}
    for t, name in enumerate(tier_labels):
        idx = np.where(tiers == t)[0]
        if len(idx) == 0:
            continue
        m = od_metrics(preds[idx], trues[idx])
        per_regime[name] = m
        print(f"  {name:13s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
              f"totFlowErr={m['total_flow_err']:.3f}  (n={len(idx)})")

    # --- GEH via ridge linear-assignment surrogate
    geh_report = {}
    if tcfg["eval"]["geh"]["enabled"]:
        manifest = json.loads(resolve(dcfg["paths"]["manifest"]).read_text())
        samples_root = resolve(dcfg["paths"]["samples_dir"]).parent
        Xtr, Ytr = load_raw_split("train", manifest, samples_root, z)
        W = fit_assignment(Xtr, Ytr, tcfg["eval"]["geh"]["ridge_alpha"])  # [P,E]

        off = _offdiag_vec_indices(z)
        od_pred_vec = preds.reshape(len(preds), -1).numpy()[:, off]
        od_true_vec = trues.reshape(len(trues), -1).numpy()[:, off]
        _, flow_obs = load_raw_split("test", manifest, samples_root, z)

        to_hourly = 3600.0 / tcfg["eval"]["geh"]["collection_window_s"]
        m_pred = np.clip(od_pred_vec @ W, 0, None) * to_hourly
        m_true = np.clip(od_true_vec @ W, 0, None) * to_hourly
        c_obs = flow_obs * to_hourly

        # evaluate only on links that actually carry observed flow
        link_mask = c_obs > 1.0
        thr = tcfg["eval"]["geh"]["geh_good_threshold"]
        g_pred = geh(m_pred[link_mask], c_obs[link_mask], thr)
        g_ceil = geh(m_true[link_mask], c_obs[link_mask], thr)
        geh_report = {"pred_vs_obs": g_pred, "trueOD_vs_obs_ceiling": g_ceil,
                      "links_evaluated": int(link_mask.sum())}
        print("\n=== GEH on implied link flows (hourly; linear-assignment surrogate) ===")
        print(f"  predicted OD : mean GEH={g_pred['geh_mean']:.3f}  "
              f"frac<{thr:.0f}={g_pred['geh_frac_good']:.3f}")
        print(f"  true OD (ceil): mean GEH={g_ceil['geh_mean']:.3f}  "
              f"frac<{thr:.0f}={g_ceil['geh_frac_good']:.3f}")

    # --- save report (+ optional predictions)
    report = {"checkpoint": args.checkpoint, "n_test": int(len(preds)),
              "overall": overall, "per_regime": per_regime, "geh": geh_report}
    out = resolve("data/eval_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[eval] report -> {out}")
    if args.save_preds:
        np.savez_compressed(resolve("data/test_predictions.npz"),
                            pred=preds.numpy(), true=trues.numpy(), tier=tiers)
        print(f"[eval] predictions -> {resolve('data/test_predictions.npz')}")


if __name__ == "__main__":
    main()
