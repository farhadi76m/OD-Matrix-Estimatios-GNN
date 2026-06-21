"""
evaluate.py — evaluate a trained checkpoint on the test split.

Marginal head: predicts per-zone production/attraction, rebuilds the OD with a
Furness gravity prior, and reports
  * marginal recoverability (production/attraction correlation + MAE),
  * reconstructed-OD cell metrics (corr, MAE, total-flow error), overall + per regime,
  * GEH on implied link flows (ridge linear-assignment surrogate).

Run:
    python -m src.evaluate --save-preds
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
from torch_geometric.loader import DataLoader

from src.config import get_device, load_config, resolve, set_seed
from src.data.dataset import ODDataset, ODEnrichedDataset
from src.data.graph import load_graph
from src.metrics import geh, od_metrics, to_counts
from src.models.gnn import build_model
from src.od_reconstruct import furness, zone_distance


def _off_idx(z):
    return np.where(~np.eye(z, dtype=bool).reshape(-1))[0]


def load_raw_split(split, manifest, root, z):
    off = _off_idx(z)
    ods, flows = [], []
    for r in manifest["samples"]:
        if r["split"] != split:
            continue
        with open(root / r["file"], "rb") as f:
            s = pickle.load(f)
        ods.append(s["od"].reshape(-1)[off].astype(np.float64))
        flows.append(s["x_dyn"][:, 0].astype(np.float64))
    return np.stack(ods), np.stack(flows)


def predict(model, loader, head, device, zone_ids, fcfg, ms):
    """Return predicted OD [N,Z,Z] (+ marginal arrays for the marginal head)."""
    z = len(zone_ids)
    if head == "marginal":
        pm = torch.tensor(ms["pm"], device=device); ps = torch.tensor(ms["ps"], device=device)
        am = torch.tensor(ms["am"], device=device); asd = torch.tensor(ms["as"], device=device)
        pp, ap, pt, at = [], [], [], []
        for data in loader:
            data = data.to(device)
            out = model(data)
            y = data.y.view(data.num_graphs, z, z)
            pp.append((out["production"] * ps + pm).clamp(min=0).cpu())
            ap.append((out["attraction"] * asd + am).clamp(min=0).cpu())
            pt.append(y.sum(2).cpu()); at.append(y.sum(1).cpu())
        prod, attr = torch.cat(pp).numpy(), torch.cat(ap).numpy()
        deter = np.exp(-fcfg["beta"] * zone_distance(zone_ids))
        preds = np.stack([furness(prod[k], attr[k], deter, fcfg["iters"]) for k in range(len(prod))])
        marg = {"prod_pred": prod, "attr_pred": attr,
                "prod_true": torch.cat(pt).numpy(), "attr_true": torch.cat(at).numpy()}
        return preds, marg
    else:
        out = []
        for data in loader:
            data = data.to(device)
            out.append(to_counts(model(data)).cpu())
        return torch.cat(out).numpy(), None


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
    graph = load_graph(); zone_ids = graph["zone_ids"]; z = len(zone_ids)

    test_ds = ODEnrichedDataset("test") if dcfg.get("enriched") else ODDataset("test")
    loader = DataLoader(test_ds, batch_size=tcfg["train"]["batch_size"], shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head = ckpt["model_cfg"]["model"].get("head", "cell")
    fcfg = ckpt["model_cfg"]["model"].get("furness", {"beta": 1.0, "iters": 40})
    model = build_model(ckpt["model_cfg"], ckpt["feature_dim"], z).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"[eval] {args.checkpoint} (epoch {ckpt['epoch']}, head={head}, val_mae={ckpt['val']['mae']:.4f})")

    with torch.no_grad():
        preds_np, marg = predict(model, loader, head, device, zone_ids, fcfg, ckpt.get("marg_stats"))
    preds = torch.tensor(preds_np, dtype=torch.float32)
    trues = torch.stack([test_ds.y[i] for i in range(len(test_ds))])
    tiers = np.array([m["tier"] for m in test_ds.meta]); tier_labels = dcfg["tier_labels"]

    report = {"checkpoint": args.checkpoint, "head": head, "n_test": int(len(preds))}

    if marg is not None:
        def corr(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        report["marginals"] = {
            "production_corr": corr(marg["prod_pred"], marg["prod_true"]),
            "attraction_corr": corr(marg["attr_pred"], marg["attr_true"]),
            "production_mae": float(np.abs(marg["prod_pred"] - marg["prod_true"]).mean()),
            "attraction_mae": float(np.abs(marg["attr_pred"] - marg["attr_true"]).mean())}
        print(f"\n=== marginal recoverability (test) ===")
        print(f"  production corr={report['marginals']['production_corr']:.3f} "
              f"MAE={report['marginals']['production_mae']:.2f}")
        print(f"  attraction corr={report['marginals']['attraction_corr']:.3f} "
              f"MAE={report['marginals']['attraction_mae']:.2f}")

    off = ~np.eye(z, dtype=bool)
    cell_corr = float(np.corrcoef(preds.numpy()[:, off].ravel(), trues.numpy()[:, off].ravel())[0, 1])
    overall = od_metrics(preds, trues); overall["cell_corr"] = cell_corr
    report["overall"] = overall
    print(f"\n=== reconstructed-OD metrics (test, off-diagonal, trips) ===")
    print(f"  overall   cell_corr={cell_corr:.3f}  MAE={overall['mae']:.4f}  "
          f"RMSE={overall['rmse']:.4f}  totErr={overall['total_flow_err']:.3f}")
    report["per_regime"] = {}
    for t, name in enumerate(tier_labels):
        idx = np.where(tiers == t)[0]
        if len(idx) == 0:
            continue
        m = od_metrics(preds[idx], trues[idx])
        report["per_regime"][name] = m
        print(f"  {name:13s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
              f"totErr={m['total_flow_err']:.3f}  (n={len(idx)})")

    # GEH via ridge linear-assignment surrogate (link-only dataset only; the
    # surrogate is fit on the old per-sample flow pkls, which don't align with
    # the enriched re-simulated split)
    if tcfg["eval"]["geh"]["enabled"] and not dcfg.get("enriched"):
        manifest = json.loads(resolve(dcfg["paths"]["manifest"]).read_text())
        root = resolve(dcfg["paths"]["samples_dir"]).parent
        Xtr, Ytr = load_raw_split("train", manifest, root, z)
        W = np.linalg.solve(Xtr.T @ Xtr + tcfg["eval"]["geh"]["ridge_alpha"] * np.eye(Xtr.shape[1]),
                            Xtr.T @ Ytr)
        offi = _off_idx(z)
        _, flow_obs = load_raw_split("test", manifest, root, z)
        to_h = 3600.0 / tcfg["eval"]["geh"]["collection_window_s"]
        m_pred = np.clip(preds.numpy().reshape(len(preds), -1)[:, offi] @ W, 0, None) * to_h
        c_obs = flow_obs * to_h
        lm = c_obs > 1.0; thr = tcfg["eval"]["geh"]["geh_good_threshold"]
        report["geh"] = geh(m_pred[lm], c_obs[lm], thr)
        print(f"\n=== GEH (predicted-OD implied flow vs observed) ===")
        print(f"  mean GEH={report['geh']['geh_mean']:.3f}  frac<{thr:.0f}={report['geh']['geh_frac_good']:.3f}")

    out = resolve("data/eval_report.json"); out.write_text(json.dumps(report, indent=2))
    print(f"\n[eval] report -> {out}")
    if args.save_preds:
        np.savez_compressed(resolve("data/test_predictions.npz"),
                            pred=preds.numpy(), true=trues.numpy(), tier=tiers)
        print(f"[eval] predictions -> {resolve('data/test_predictions.npz')}")


if __name__ == "__main__":
    main()
