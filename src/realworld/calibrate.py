"""
calibrate.py — Stage C (warm-start) + Stage D (SPSA) OD calibration.

Travel time barely identifies the OD (Stage A: ridge cell-corr ~0.06), so the
learned model is only a seed. The real work is done by matching SIMULATED
travel times to the OBSERVED ones with SUMO in the loop:

  1. warm-start : ridge on synthetic travel time -> zone marginals -> Furness OD0.
  2. SPSA       : optimise 2*Z log-multipliers on the marginals so the simulated
                  per-edge travel time matches Neshan, on the observed edges.
                  Gradient-free (SPSA), 2 SUMO sims / iter, resumable.

The objective uses the congestion RATIO t/t_free (not absolute seconds) to be
robust to the sim-vs-probe offset, plus a small pull of the multipliers toward 1.

Run (resumable; re-run to continue):
    python -m src.realworld.calibrate --max-seconds 300
    python -m src.realworld.calibrate --reset            # start over
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.linear_model import Ridge

from src.config import load_yaml, resolve, set_seed
from src.data.graph import load_graph
from src.od_reconstruct import zone_distance
from src.realworld import common as C
from src.realworld.sim import simulate_od


# ─────────────────────────────────────────────────────────────────────────────
# Stage C — ridge warm-start marginals from travel time
# ─────────────────────────────────────────────────────────────────────────────
def _tt_features(x, mask, stats):
    """log1p + z-score of the travel-time channel on covered edges.

    Accepts a pack tensor [n,N,9] (selects the tt channel) or an already-tt
    array [n,N] (the observation)."""
    x = np.asarray(x)
    tt = x[:, :, C.FEAT_IDX["traveltime"]] if x.ndim == 3 else x
    f = np.log1p(np.clip(tt[:, mask], 0, None)).astype(np.float64)
    if stats is None:
        stats = {"mu": f.mean(0), "sd": np.maximum(f.std(0), 1e-6)}
    return (f - stats["mu"]) / stats["sd"], stats


def warm_start(rcfg, obs, zone_ids, beta):
    """Fit ridge synthetic(tt)->marginals, apply to the observation -> OD0."""
    mask = obs["coverage_mask"]
    alphas = rcfg["diagnostic"]["ridge_alphas"]
    tr, va = C.load_pack("train"), C.load_pack("val")
    Xtr, st = _tt_features(tr["x"], mask, None)
    Xva, _ = _tt_features(va["x"], mask, st)
    z = len(zone_ids)
    Ytr = np.concatenate(C.marginals_from_od(tr["y"].numpy()), axis=1)   # [n,2Z]
    Yva = np.concatenate(C.marginals_from_od(va["y"].numpy()), axis=1)
    best = None
    for a in alphas:
        r = Ridge(alpha=a).fit(Xtr, Ytr)
        v = np.abs(r.predict(Xva) - Yva).mean()
        if best is None or v < best[0]:
            best = (v, a, r)
    model = best[2]

    Xobs, _ = _tt_features(obs["tt_obs"][None, :], mask, st)             # [1, m]
    pred = np.clip(model.predict(Xobs)[0], 0, None)
    prod0, attr0 = pred[:z], pred[z:]
    print(f"[warm] ridge alpha={best[1]} | warm-start marginals: "
          f"prod_total={prod0.sum():.0f} attr_total={attr0.sum():.0f}")
    return prod0, attr0


# ─────────────────────────────────────────────────────────────────────────────
# Forward operator + objective
# ─────────────────────────────────────────────────────────────────────────────
def od_from_theta(theta, prod0, attr0, deter, z, iters, target_total=None):
    """OD from shape-multipliers, renormalised to a fixed total demand.

    Travel time on a subset of edges does NOT constrain total demand (Stage A
    under-determination), so SPSA optimises only the spatial SHAPE; the total is
    pinned (set once by find_total) to avoid the demand blowing up."""
    from src.od_reconstruct import furness
    tp, ta = np.clip(theta[:z], -3, 3), np.clip(theta[z:], -3, 3)
    od = furness(prod0 * np.exp(tp), attr0 * np.exp(ta), deter, iters)
    if target_total and od.sum() > 0:
        od = od * (target_total / od.sum())
    return od


def logtt_loss(tt_sim, ctx) -> float:
    """Robust travel-time mismatch on observed edges, in LOG space.

    log1p compresses congestion outliers and bounds the penalty on edges the
    candidate leaves empty (tt_sim=0 -> -log1p(tt_obs), not -inf). A FIXED scalar
    bias (ctx['bias'], a one-time sim-vs-probe speed offset; 0 for the
    self-consistent demo) is subtracted — note this must NOT be re-estimated per
    evaluation, or the overall congestion level (the demand-magnitude signal)
    cancels out and the total demand becomes unidentifiable.
    """
    m = ctx["mask"]
    d = np.log1p(tt_sim[m]) - np.log1p(ctx["tt_obs"][m]) - ctx["bias"]
    return float(np.mean(d ** 2))


def objective(theta, ctx):
    """Travel-time mismatch of the OD implied by theta (+ shape regulariser)."""
    od = od_from_theta(theta, ctx["prod0"], ctx["attr0"], ctx["deter"], ctx["z"],
                       ctx["iters"], ctx["target_total"])
    r = simulate_od(od, ctx["zone_ids"], ctx["edge_ids"], ctx["net"], ctx["taz"],
                    ctx["work"], seed=ctx["sim_seed"], timeout=ctx["timeout"])
    if not r["ok"]:
        return 1e3, r                       # penalise failed sims
    loss = logtt_loss(r["traveltime"], ctx) + ctx["reg"] * float(np.mean(theta ** 2))
    return loss, r


def find_total(prod0, attr0, ctx, band):
    """1-D search on total demand by matching the CONGESTION INTENSITY (median
    tt/tt_free over edges active in both sim and observation), with a coverage
    guard. This identifies demand magnitude robustly: unlike a log-loss that
    penalises empty edges, it is not biased toward flooding the network to
    activate edges (which would compensate for a weak warm-start shape and badly
    over-estimate the total)."""
    sp, sa = prod0 / prod0.sum(), attr0 / attr0.sum()       # unit-total shape
    warm_total = float(prod0.sum())
    m = ctx["mask"]; tf = np.clip(ctx["tt_free"], 1e-6, None)
    obs_med = float(np.median(ctx["tt_obs"][m] / tf[m]))
    totals = np.geomspace(band["lo"] * warm_total, band["hi"] * warm_total, band["steps"])
    best = None
    for T in totals:
        od = od_from_theta(np.zeros(2 * ctx["z"]), sp * T, sa * T, ctx["deter"],
                           ctx["z"], ctx["iters"], T)
        r = simulate_od(od, ctx["zone_ids"], ctx["edge_ids"], ctx["net"], ctx["taz"],
                        ctx["work"], seed=ctx["sim_seed"], timeout=ctx["timeout"])
        both = m & (r["traveltime"] > 0)
        cov = both.sum() / max(int(m.sum()), 1)
        if both.sum() < 5:
            err, sim_med = 1e3, 0.0
        else:
            sim_med = float(np.median(r["traveltime"][both] / tf[both]))
            err = abs(sim_med - obs_med) + 1.5 * max(0.0, 0.5 - cov)   # coverage guard
        print(f"[scale] total={T:7.0f}  sim_med_ratio={sim_med:.2f} obs={obs_med:.2f} "
              f"cov={cov:.2f}  err={err:.3f}")
        if best is None or err < best[0]:
            best = (err, float(T))
    print(f"[scale] chosen total={best[1]:.0f} (warm-start was {warm_total:.0f})")
    return best[1], sp * best[1], sa * best[1]


# ─────────────────────────────────────────────────────────────────────────────
# SPSA loop (resumable)
# ─────────────────────────────────────────────────────────────────────────────
def run_spsa(ctx, sp, ckpt_path, max_iters, max_seconds):
    state_path = Path(ckpt_path)
    if state_path.exists():
        state = pickle.load(open(state_path, "rb"))
        print(f"[spsa] resume at iter {state['k']} (best_loss={state['best_loss']:.4f})")
    else:
        z2 = 2 * ctx["z"]
        l0, _ = objective(np.zeros(z2), ctx)        # warm-start loss (theta=0)
        state = {"theta": np.zeros(z2), "best_theta": np.zeros(z2), "best_loss": l0,
                 "warmstart_loss": l0, "k": 0, "history": [l0]}
        print(f"[spsa] warm-start loss={l0:.4f}")

    rng = np.random.default_rng(sp["seed"] + state["k"])
    t0 = time.time()
    while state["k"] < max_iters and (time.time() - t0) < max_seconds:
        k = state["k"]
        ak = sp["a"] / (k + 1 + sp["A"]) ** sp["alpha"]
        ck = sp["c"] / (k + 1) ** sp["gamma"]
        delta = rng.choice([-1.0, 1.0], size=len(state["theta"]))
        tp = np.clip(state["theta"] + ck * delta, -3, 3)
        tm = np.clip(state["theta"] - ck * delta, -3, 3)
        lp, _ = objective(tp, ctx)
        lm, _ = objective(tm, ctx)
        ghat = (lp - lm) / (2 * ck) * delta
        state["theta"] = np.clip(state["theta"] - ak * ghat, -3, 3)
        # track best among the two ACTUALLY-evaluated perturbed points (no extra sim)
        for L, th in ((lp, tp), (lm, tm)):
            if L < state["best_loss"]:
                state["best_loss"], state["best_theta"] = L, th.copy()
        state["k"] += 1
        state["history"].append(min(lp, lm))
        print(f"[spsa] iter {state['k']:3d}/{max_iters}  L+={lp:.4f} L-={lm:.4f}  "
              f"best={state['best_loss']:.4f}  (ak={ak:.3f} ck={ck:.3f})")
        pickle.dump(state, open(state_path, "wb"))

    done = state["k"] >= max_iters
    print(f"[spsa] {'DONE' if done else 'paused (budget)'} at iter {state['k']}, "
          f"best_loss={state['best_loss']:.4f} (warm-start was {state['warmstart_loss']:.4f})")
    return state, done


def main() -> None:
    rcfg = load_yaml("realworld")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=int, default=300)
    ap.add_argument("--max-iters", type=int, default=rcfg["calibrate"]["spsa"]["max_iters"])
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    set_seed(rcfg["seed"])

    graph = load_graph(); zone_ids = graph["zone_ids"]; z = len(zone_ids)
    obs = pickle.load(open(resolve(rcfg["paths"]["observation"]), "rb"))
    print(f"[cal] observation: source={obs['source']} observed={int(obs['mask'].sum())} edges "
          f"regime={obs['regime']}")

    beta = rcfg["calibrate"]["beta"]
    deter = np.exp(-beta * zone_distance(zone_ids))
    prod_raw, attr_raw = warm_start(rcfg, obs, zone_ids, beta)

    ckpt = resolve(rcfg["calibrate"]["checkpoint"])
    if args.reset and ckpt.exists():
        ckpt.unlink(); print("[cal] reset SPSA state")

    ctx = {"prod0": None, "attr0": None, "target_total": None, "deter": deter, "z": z,
           "iters": rcfg["furness"]["iters"], "zone_ids": zone_ids,
           "edge_ids": graph["edge_ids"], "net": resolve(rcfg["paths"]["net_file"]),
           "taz": resolve(rcfg["paths"]["taz_file"]), "work": "/tmp/calib_sim",
           "mask": obs["mask"], "tt_obs": obs["tt_obs"], "tt_free": obs["tt_free"],
           "sim_seed": rcfg["calibrate"]["sim_seed"], "timeout": 360, "bias": 0.0,
           "reg": rcfg["calibrate"]["objective"]["reg_warmstart"]}

    # Stage D.1 — pin total demand by matching overall congestion (deterministic).
    target_total, ctx["prod0"], ctx["attr0"] = find_total(
        prod_raw, attr_raw, ctx, rcfg["calibrate"]["total_search"])
    ctx["target_total"] = target_total

    # Optional one-time sim-vs-probe bias for REAL data (demo is self-consistent).
    if rcfg["calibrate"]["objective"]["remove_global_bias"]:
        od0 = od_from_theta(np.zeros(2 * z), ctx["prod0"], ctx["attr0"], deter, z,
                            ctx["iters"], target_total)
        r0 = simulate_od(od0, zone_ids, graph["edge_ids"], ctx["net"], ctx["taz"],
                         ctx["work"], seed=ctx["sim_seed"], timeout=ctx["timeout"])
        m = obs["mask"]
        ctx["bias"] = float(np.median(np.log1p(r0["traveltime"][m]) - np.log1p(obs["tt_obs"][m])))
        print(f"[cal] one-time sim-vs-probe log-bias = {ctx['bias']:.3f}")

    # Stage D.2 — SPSA on the spatial shape (total stays pinned).
    state, done = run_spsa(ctx, rcfg["calibrate"]["spsa"], ckpt, args.max_iters, args.max_seconds)

    # always (re)write the current best calibrated OD so downstream stages can run
    od0 = od_from_theta(np.zeros(2 * z), ctx["prod0"], ctx["attr0"], deter, z, ctx["iters"], target_total)
    od_best = od_from_theta(state["best_theta"], ctx["prod0"], ctx["attr0"], deter, z, ctx["iters"], target_total)
    out = {"od_calibrated": od_best, "od_warmstart": od0, "zone_ids": zone_ids,
           "beta": beta, "target_total": target_total, "best_loss": state["best_loss"],
           "warmstart_loss": state["warmstart_loss"], "history": state["history"],
           "iters_done": state["k"], "done": done, "source": obs["source"],
           "planted_od": obs.get("planted_od")}
    with open(resolve(rcfg["calibrate"]["out_od"]), "wb") as f:
        pickle.dump(out, f)
    print(f"[cal] calibrated OD (total={od_best.sum():.0f}) -> {resolve(rcfg['calibrate']['out_od'])}")
    if not done:
        print("[cal] budget reached — re-run `python -m src.realworld.calibrate --max-seconds N` to continue.")


if __name__ == "__main__":
    main()
