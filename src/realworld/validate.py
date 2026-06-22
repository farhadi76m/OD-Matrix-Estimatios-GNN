"""
validate.py — Stage E: does the calibrated OD reproduce the observed traffic?

Re-simulates the warm-start OD and the calibrated OD, and scores both against
the observation on the observed edges:
  * travel-time correlation + RMSE (linear and log space),
  * GEH on BPR-implied link flows (travel time has no direct flow, so flow is
    inferred from the volume-delay curve — approximate, flagged as such),
  * for the demo (planted OD known): OD cell-corr / marginal corr / total error,
    which honestly exposes the travel-time under-determination from Stage A.
Saves a convergence plot and an observed-vs-simulated travel-time map.

Run:
    python -m src.realworld.validate
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import load_yaml, resolve
from src.data.graph import load_graph
from src.metrics import geh
from src.realworld import common as C
from src.realworld.sim import simulate_od


def bpr_implied_flow(tt, tt_free, lanes, bpr) -> np.ndarray:
    """Invert the BPR volume-delay curve t = t0 (1 + a (v/c)^b) for flow v.
    Only edges with t > t0 carry inferable flow; others -> 0. Approximate."""
    t0 = np.clip(tt_free, 1e-6, None)
    ratio = np.clip(tt / t0 - 1.0, 0, None)
    cap = np.clip(lanes, 1, None) * bpr["capacity_per_lane"]
    voc = (ratio / bpr["alpha"]) ** (1.0 / bpr["beta"])
    return voc * cap                                        # veh/h


def tt_scores(tt_sim, obs):
    m = obs["mask"]
    s, o = tt_sim[m], obs["tt_obs"][m]
    return {
        "corr": C.pearson(s, o),
        "log_corr": C.pearson(np.log1p(s), np.log1p(o)),
        "rmse_s": float(np.sqrt(np.mean((s - o) ** 2))),
        "log_rmse": float(np.sqrt(np.mean((np.log1p(s) - np.log1p(o)) ** 2))),
        "active_match": float(np.mean((s > 0) == (o > 0))),
    }


def main() -> None:
    rcfg = load_yaml("realworld")
    vcfg = rcfg["validate"]
    graph = load_graph(); zone_ids = graph["zone_ids"]
    obs = pickle.load(open(resolve(rcfg["paths"]["observation"]), "rb"))
    cal = pickle.load(open(resolve(rcfg["calibrate"]["out_od"]), "rb"))
    net, taz = resolve(rcfg["paths"]["net_file"]), resolve(rcfg["paths"]["taz_file"])
    sseed = rcfg["calibrate"]["sim_seed"]

    def sim(od, tag):
        r = simulate_od(od, zone_ids, graph["edge_ids"], net, taz,
                        f"/tmp/val_{tag}", seed=sseed)
        if not r["ok"]:
            raise RuntimeError(f"{tag} sim failed: {r.get('err')}")
        return r

    r_warm = sim(cal["od_warmstart"], "warm")
    r_cal = sim(cal["od_calibrated"], "cal")

    report = {"source": cal["source"], "regime": obs["regime"],
              "n_observed_edges": int(obs["mask"].sum()),
              "target_total": cal["target_total"],
              "warmstart_total": float(cal["od_warmstart"].sum()),
              "calibrated_total": float(cal["od_calibrated"].sum()),
              "spsa_iters": cal["iters_done"],
              "loss_warmstart": cal["warmstart_loss"], "loss_calibrated": cal["best_loss"],
              "travel_time": {"warmstart": tt_scores(r_warm["traveltime"], obs),
                              "calibrated": tt_scores(r_cal["traveltime"], obs)}}

    # GEH on BPR-implied flows (approximate; travel time carries no direct count)
    lanes = graph["static"][:, 1]
    f_obs = bpr_implied_flow(obs["tt_obs"], obs["tt_free"], lanes, vcfg["bpr"])
    f_cal = bpr_implied_flow(r_cal["traveltime"], obs["tt_free"], lanes, vcfg["bpr"])
    m = obs["mask"] & (f_obs > 1.0)
    report["geh_bpr_implied"] = geh(f_cal[m], f_obs[m]) if m.sum() else {"geh_mean": None}

    # demo only: honest OD recovery vs the planted truth
    if cal.get("planted_od") is not None:
        true_od = np.asarray(cal["planted_od"], dtype=float)
        cod = cal["od_calibrated"]
        pt, pp = C.marginals_from_od(true_od)[0], C.marginals_from_od(cod)[0]
        at, ap = C.marginals_from_od(true_od)[1], C.marginals_from_od(cod)[1]
        z = len(zone_ids); off = ~np.eye(z, dtype=bool)
        report["od_recovery_vs_planted"] = {
            "cell_corr": C.pearson(cod[off], true_od[off]),
            "production_corr": C.pearson(pp, pt), "attraction_corr": C.pearson(ap, at),
            "total_true": float(true_od.sum()), "total_calibrated": float(cod.sum()),
            "total_rel_err": float(abs(cod.sum() - true_od.sum()) / max(true_od.sum(), 1))}

    resolve(vcfg["report"]).write_text(json.dumps(report, indent=2))

    # ── console summary ──────────────────────────────────────────────────────
    tw, tc = report["travel_time"]["warmstart"], report["travel_time"]["calibrated"]
    print(f"\n=== travel-time match on {report['n_observed_edges']} observed edges ===")
    print(f"  {'':12s} {'corr':>7s} {'log_corr':>9s} {'rmse_s':>8s} {'log_rmse':>9s}")
    print(f"  {'warm-start':12s} {tw['corr']:7.3f} {tw['log_corr']:9.3f} {tw['rmse_s']:8.2f} {tw['log_rmse']:9.3f}")
    print(f"  {'calibrated':12s} {tc['corr']:7.3f} {tc['log_corr']:9.3f} {tc['rmse_s']:8.2f} {tc['log_rmse']:9.3f}")
    print(f"  SPSA loss {report['loss_warmstart']:.4f} -> {report['loss_calibrated']:.4f} "
          f"({report['spsa_iters']} iters)")
    if report["geh_bpr_implied"].get("geh_mean") is not None:
        g = report["geh_bpr_implied"]
        print(f"  GEH(BPR-implied flow) mean={g['geh_mean']:.2f} frac<5={g['geh_frac_good']:.2f} (approx.)")
    if "od_recovery_vs_planted" in report:
        rr = report["od_recovery_vs_planted"]
        print(f"\n=== OD recovery vs planted (demo) ===")
        print(f"  cell_corr={rr['cell_corr']:.3f}  prod_corr={rr['production_corr']:.3f}  "
              f"attr_corr={rr['attraction_corr']:.3f}")
        print(f"  total true={rr['total_true']:.0f} calibrated={rr['total_calibrated']:.0f} "
              f"(rel.err {rr['total_rel_err']:.2f})")
        print("  (modest cell_corr is expected — Stage A: travel time barely identifies the OD)")

    _plots(graph, obs, cal, r_warm, r_cal, resolve(vcfg["out_dir"]))
    print(f"\n[validate] report -> {resolve(vcfg['report'])}")


def _plots(graph, obs, cal, r_warm, r_cal, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from src.plot_graph import node_positions
    out_dir.mkdir(parents=True, exist_ok=True)

    # convergence
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cal["history"], lw=1.5)
    ax.set_xlabel("SPSA evaluation"); ax.set_ylabel("log travel-time loss")
    ax.set_title(f"Calibration convergence ({cal['source']})"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "convergence.png", dpi=120); plt.close(fig)

    # observed vs calibrated-sim travel times on the network + scatter
    pos = node_positions(graph["edge_ids"], resolve(load_yaml("realworld")["paths"]["net_file"]))
    ei = graph["edge_index"]; segs = np.stack([pos[ei[0]], pos[ei[1]]], axis=1)
    m = obs["mask"]
    fig, ax = plt.subplots(1, 3, figsize=(26, 9))
    for a, tt, ttl in [(ax[0], obs["tt_obs"], "OBSERVED travel time"),
                       (ax[1], r_cal["traveltime"], "CALIBRATED OD -> simulated")]:
        a.set_aspect("equal"); a.axis("off")
        a.add_collection(LineCollection(segs, colors="#eee", linewidths=0.3))
        sc = a.scatter(pos[m, 0], pos[m, 1], c=np.log1p(tt[m]), cmap="inferno",
                       s=18, linewidths=0)
        fig.colorbar(sc, ax=a, fraction=0.04, label="log1p(travel time s)")
        a.set_title(ttl)
    s, o = r_cal["traveltime"][m], obs["tt_obs"][m]
    ax[2].scatter(o, s, s=14, alpha=0.5, color="#2e7d8c")
    lim = max(o.max(), s.max(), 1); ax[2].plot([0, lim], [0, lim], "k--", lw=1)
    ax[2].set_xlabel("observed travel time (s)"); ax[2].set_ylabel("calibrated-sim travel time (s)")
    ax[2].set_title(f"observed vs simulated (r={C.pearson(s, o):.2f})"); ax[2].grid(alpha=0.3)
    fig.suptitle("Stage E: calibrated OD reproduces observed travel times", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "traveltime_match.png", dpi=110); plt.close(fig)
    print(f"[validate] plots -> {out_dir}/convergence.png, traveltime_match.png")


if __name__ == "__main__":
    main()
