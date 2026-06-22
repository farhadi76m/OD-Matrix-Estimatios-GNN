"""
diagnose_traveltime.py — Stage A: how much OD signal does TRAVEL TIME carry?

Neshan gives per-segment travel time (~speed), the weakest OD observable.
Before building a calibrator we measure the ceiling on the SYNTHETIC data we
already have: fit a ridge marginal predictor (production/attraction per zone)
from each candidate signal, reconstruct the OD by gravity/Furness, and compare.
This tells us how good a *learned warm-start* can be — and therefore how much
the SUMO-in-the-loop SPSA loop (Stage D) has to carry.

We also restrict to a realistic COVERAGE MASK (named arterials only) so the
diagnostic matches deployment, where Neshan covers major roads, not all 1897.

Run:
    python -m src.realworld.diagnose_traveltime
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.linear_model import Ridge

from src.config import load_yaml, resolve
from src.data.graph import load_graph
from src.realworld import common as C


def _features(x: np.ndarray, channels: list[str], mask: np.ndarray,
              stats: dict | None) -> tuple[np.ndarray, dict]:
    """Build a flat ridge design matrix [n, n_masked*k] from selected channels.

    log1p then per-column z-score (stats fit on TRAIN, reused for val/test).
    Empty channel list -> a single ones column (a mean-predictor floor).
    """
    if not channels:
        return np.ones((len(x), 1), dtype=np.float64), {}
    idx = [C.FEAT_IDX[c] for c in channels]
    f = np.log1p(np.clip(x[:, mask][:, :, idx], 0, None)).astype(np.float64)  # [n,m,k]
    f = f.reshape(len(x), -1)                                                  # [n,m*k]
    if stats is None:
        mu = f.mean(0)
        sd = np.maximum(f.std(0), 1e-6)
        stats = {"mu": mu, "sd": sd}
    f = (f - stats["mu"]) / stats["sd"]
    return f, stats


def _fit_signal(name, channels, mask, packs, marg, zone_ids, beta, alphas, iters):
    """Fit ridge for one signal, pick alpha on val, report test metrics."""
    Xtr, st = _features(packs["train"]["x"], channels, mask, None)
    Xva, _ = _features(packs["val"]["x"], channels, mask, st)
    Xte, _ = _features(packs["test"]["x"], channels, mask, st)
    Ytr = marg["train"]                                   # [n,2Z] = [prod | attr]

    best = None
    for a in alphas:
        r = Ridge(alpha=a).fit(Xtr, Ytr)
        va = np.abs(r.predict(Xva) - marg["val"]).mean()
        if best is None or va < best[0]:
            best = (va, a, r)
    _, alpha, model = best

    z = len(zone_ids)
    pred = np.clip(model.predict(Xte), 0, None)           # [n,2Z]
    prod_p, attr_p = pred[:, :z], pred[:, z:]
    prod_t, attr_t = marg["test"][:, :z], marg["test"][:, z:]
    rec = C.reconstruct_many(prod_p, attr_p, zone_ids, beta, iters)
    true_od = packs["test"]["y"].numpy()

    return {
        "signal": name, "channels": channels, "alpha": alpha,
        "production_corr": C.pearson(prod_p, prod_t),
        "attraction_corr": C.pearson(attr_p, attr_t),
        "production_mae": float(np.abs(prod_p - prod_t).mean()),
        "attraction_mae": float(np.abs(attr_p - attr_t).mean()),
        "cell_corr": C.pooled_cell_corr(rec, true_od),
        "cell_rmse": C.cell_rmse(rec, true_od),
        "total_flow_err": C.total_flow_err(rec, true_od),
    }


def main() -> None:
    rcfg = load_yaml("realworld")
    graph = load_graph()
    zone_ids = graph["zone_ids"]; z = len(zone_ids)
    edge_ids = graph["edge_ids"]

    # coverage mask: named arterials (realistic probe coverage) or all edges
    cmode = rcfg["coverage"]["mode"]
    if cmode == "named":
        mask = C.named_mask(edge_ids, rcfg["paths"]["net_file"],
                            rcfg["coverage"]["min_name_len"])
    else:
        mask = np.ones(len(edge_ids), dtype=bool)
    print(f"[diag] coverage='{cmode}': {int(mask.sum())}/{len(edge_ids)} edges observed "
          f"({100*mask.mean():.1f}%)")

    packs = {sp: C.load_pack(sp) for sp in ("train", "val", "test")}
    for sp in packs:                                       # ridge works in numpy
        packs[sp]["x"] = packs[sp]["x"].numpy()
    marg = {}
    for sp in ("train", "val", "test"):
        od = packs[sp]["y"].numpy()
        prod, attr = C.marginals_from_od(od)
        marg[sp] = np.concatenate([prod, attr], axis=1)    # [n,2Z]
    print(f"[diag] samples train/val/test = "
          f"{len(packs['train']['x'])}/{len(packs['val']['x'])}/{len(packs['test']['x'])}")

    # gravity ceiling: best beta from TRUE marginals
    pt = marg["train"]
    beta, ceil = C.pick_beta(pt[:, :z], pt[:, z:], packs["train"]["y"].numpy(),
                             zone_ids, rcfg["furness"]["betas"], rcfg["furness"]["iters"])
    # reconstruction ceiling on TEST true marginals at that beta
    tt = marg["test"]
    rec_true = C.reconstruct_many(tt[:, :z], tt[:, z:], zone_ids, beta, rcfg["furness"]["iters"])
    ceil_test = C.pooled_cell_corr(rec_true, packs["test"]["y"].numpy())
    print(f"[diag] gravity beta={beta} | reconstruction ceiling (TRUE marginals): "
          f"train cell-corr={ceil:.3f}, test cell-corr={ceil_test:.3f}")

    rows = []
    for name, channels in rcfg["diagnostic"]["signals"].items():
        r = _fit_signal(name, channels, mask, packs, marg, zone_ids, beta,
                        rcfg["diagnostic"]["ridge_alphas"], rcfg["furness"]["iters"])
        rows.append(r)
        print(f"[diag] {name:12s} a={r['alpha']:<6g} "
              f"prod_corr={r['production_corr']:.3f} attr_corr={r['attraction_corr']:.3f} "
              f"-> cell_corr={r['cell_corr']:.3f} rmse={r['cell_rmse']:.2f} "
              f"totErr={r['total_flow_err']:.3f}")

    print("\n=== Stage A summary (ridge, coverage=%s) ===" % cmode)
    print(f"{'signal':12s} {'prod_corr':>9s} {'attr_corr':>9s} {'cell_corr':>9s} "
          f"{'rmse':>7s} {'totErr':>7s}")
    for r in sorted(rows, key=lambda x: -x["cell_corr"]):
        print(f"{r['signal']:12s} {r['production_corr']:9.3f} {r['attraction_corr']:9.3f} "
              f"{r['cell_corr']:9.3f} {r['cell_rmse']:7.2f} {r['total_flow_err']:7.3f}")
    print(f"{'TRUE-marg ceil':12s} {'1.000':>9s} {'1.000':>9s} {ceil_test:9.3f}")

    report = {
        "coverage_mode": cmode, "observed_edges": int(mask.sum()),
        "total_edges": len(edge_ids), "gravity_beta": beta,
        "reconstruction_ceiling_cell_corr": ceil_test, "signals": rows,
    }
    out = resolve(rcfg["diagnostic"]["report"])
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[diag] report -> {out}")

    tt_row = next(r for r in rows if r["signal"] == "traveltime")
    print(f"\n[diag] TAKEAWAY: travel-time warm-start cell-corr={tt_row['cell_corr']:.3f} "
          f"(ceiling {ceil_test:.3f}). The gap is what Stage D (SPSA) must close.")


if __name__ == "__main__":
    main()
