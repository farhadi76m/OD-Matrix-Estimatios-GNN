"""
scenario.py — build a HIGH-DEMAND OD at a target total and watch it in sumo-gui.

The training dataset spans only 50-2000 trips/hour (chosen to cover demand
regimes for the estimator), so it looks light in the GUI. This makes a realistic
gravity OD at any total you like — trips proportional to zone size (edge count),
distance-decayed — and launches the simulation.

    python -m src.zones9.scenario --trips 5000  --sumo-gui
    python -m src.zones9.scenario --trips 40000 --sumo-gui                 # heavy
    python -m src.zones9.scenario --trips 5000  --depart-window 900 --sumo-gui  # denser (10-min injection)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import load_yaml, resolve, set_seed
from src.od_reconstruct import furness, zone_distance
from src.zones9.graph import load_graph


def gravity_od(zone_ids, node_zone, trips, beta, iters):
    """Doubly-constrained gravity OD: production/attraction ~ zone size (edge
    count), distance-decayed, zero diagonal (no intra-zone trips)."""
    z = len(zone_ids)
    sizes = np.clip(np.bincount(node_zone, minlength=z).astype(float), 1, None)
    marg = sizes / sizes.sum() * trips                     # balanced prod = attr
    deter = np.exp(-beta * zone_distance(zone_ids))
    np.fill_diagonal(deter, 0.0)                           # keep all trips inter-zone
    return furness(marg, marg, deter, iters)


def main():
    cfg = load_yaml("zones9")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", type=float, default=5000, help="target total demand")
    ap.add_argument("--beta", type=float, default=cfg["furness"]["beta"],
                    help="gravity distance decay (higher = more local trips)")
    ap.add_argument("--depart-window", type=int, default=3600,
                    help="seconds over which to inject trips (smaller = denser on screen)")
    ap.add_argument("--sumo-gui", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data9/scenario")
    args = ap.parse_args()
    set_seed(args.seed)

    graph = load_graph(); zone_ids = graph["zone_ids"]
    od = gravity_od(zone_ids, graph["node_zone"], args.trips, args.beta, cfg["furness"]["iters"])
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"od_{int(args.trips)}.npy", od)

    def top(M, k=6):
        ij = np.dstack(np.unravel_index(np.argsort(M, None)[::-1], M.shape))[0][:k]
        return [(zone_ids[i], zone_ids[j], round(float(M[i, j]), 1)) for i, j in ij]
    print(f"[scenario] gravity OD: total={od.sum():.0f} trips over {len(zone_ids)} zones "
          f"(beta={args.beta}, inject over {args.depart_window}s)")
    print(f"[scenario] top pairs: {top(od)}")

    if args.sumo_gui:
        from src import sumo_export as sx
        sdir = out / f"sumo_{int(args.trips)}"; sdir.mkdir(parents=True, exist_ok=True)
        net = resolve(cfg["paths"]["net_file"]); taz = resolve(cfg["paths"]["taz_file"])
        trips_f, routes, scfg = sdir / "trips.xml", sdir / "routes.xml", sdir / "sim.sumocfg"
        n = sx.write_trips_xml(od, zone_ids, trips_f, seed=args.seed, begin=0, end=args.depart_window)
        print(f"[scenario] wrote {n} trips -> {trips_f}")
        ok, err = sx.run_duarouter(net, taz, trips_f, routes, seed=args.seed)
        if not ok:
            print(f"[scenario] duarouter failed: {err}\n        files in {sdir}")
            return
        sx.write_sumocfg(scfg, net, routes, end=max(3600, args.depart_window + 1800))
        launched, info = sx.launch_sumo_gui(scfg)
        print(f"[scenario] sumo-gui {'launched (press Play ▶)' if launched else 'NOT found'}.")
        print(f"[scenario] open it yourself: sumo-gui -c {scfg} --start --delay 80")
    else:
        print(f"[scenario] OD saved -> {out}/od_{int(args.trips)}.npy  (add --sumo-gui to watch it)")


if __name__ == "__main__":
    main()
