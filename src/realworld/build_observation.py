"""
build_observation.py — assemble the per-edge observed travel time the
calibrator matches against (data/neshan_obs.pkl).

Two sources:
  * real : read data/neshan_matched.pkl (from match_edges.py) — per-edge
           observed travel time + the coverage mask of edges Neshan covers.
  * demo : plant a known synthetic OD, simulate it once, and treat the result
           as the "observed" travel times. Self-consistent (no sim-to-real gap),
           so it isolates whether the SPSA loop can recover demand that
           reproduces an observation. The planted OD is stored for validation.

Run:
    python -m src.realworld.build_observation --demo-total 800
    python -m src.realworld.build_observation                 # real (needs match_edges)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import load_yaml, resolve
from src.data.graph import load_graph
from src.realworld import common as C


def free_flow_tt(graph) -> np.ndarray:
    """Free-flow travel time per edge = length / speed_limit (from the net)."""
    length = graph["static"][:, 0]
    speed = np.clip(graph["static"][:, 2], 0.1, None)
    return length / speed


def _save(obs: dict, rcfg) -> None:
    pkl = resolve(rcfg["paths"]["observation"])
    pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl, "wb") as f:
        pickle.dump(obs, f)
    meta = {k: obs[k] for k in ("regime", "source", "congestion_ratio")}
    meta.update({"n_observed": int(obs["mask"].sum()),
                 "n_coverage": int(obs["coverage_mask"].sum()),
                 "n_edges": len(obs["edge_ids"]),
                 "has_planted_od": obs["planted_od"] is not None})
    resolve(rcfg["paths"]["observation_meta"]).write_text(json.dumps(meta, indent=2))
    print(f"[obs] source={obs['source']} regime={obs['regime']} "
          f"observed={meta['n_observed']}/{meta['n_edges']} edges "
          f"congestion_ratio={obs['congestion_ratio']:.2f}")
    print(f"[obs] -> {pkl}")


def build_demo(rcfg, graph, total: int, seed: int) -> dict:
    from src.realworld.sim import simulate_od
    pack = C.load_pack("test")
    tots = pack["y"].numpy().sum((1, 2))
    k = int(np.argmin(np.abs(tots - total)))
    planted = pack["y"][k].numpy()
    tier = int(pack["meta"][k]["tier"]); regime = load_yaml("data")["tier_labels"][tier]
    print(f"[obs] demo: planted synthetic test sample k={k} total={planted.sum():.0f} "
          f"regime={regime}")

    r = simulate_od(planted, graph["zone_ids"], graph["edge_ids"],
                    resolve(rcfg["paths"]["net_file"]), resolve(rcfg["paths"]["taz_file"]),
                    "/tmp/obs_demo", seed=rcfg["calibrate"]["sim_seed"])
    if not r["ok"]:
        raise RuntimeError(f"demo simulation failed: {r.get('err')}")
    tt_obs = r["traveltime"]
    coverage = C.named_mask(graph["edge_ids"], rcfg["paths"]["net_file"],
                            rcfg["coverage"]["min_name_len"])
    mask = coverage & (tt_obs > 0)
    tt_free = free_flow_tt(graph)
    ratio = float(np.median(tt_obs[mask] / np.clip(tt_free[mask], 1e-6, None)))
    return {"edge_ids": graph["edge_ids"], "tt_obs": tt_obs, "tt_free": tt_free,
            "mask": mask, "coverage_mask": coverage, "regime": regime,
            "source": f"demo:k={k}", "planted_od": planted, "congestion_ratio": ratio}


def build_real(rcfg, graph) -> dict:
    matched_path = resolve("data/neshan_matched.pkl")
    if not matched_path.exists():
        raise FileNotFoundError(
            f"{matched_path} not found — run neshan_ingest + match_edges first "
            f"(or use --demo-total for a self-consistent demo).")
    m = pickle.load(open(matched_path, "rb"))
    tt_obs = np.asarray(m["tt_obs"], dtype=float)
    coverage = np.asarray(m["coverage_mask"], dtype=bool)
    mask = coverage & (tt_obs > 0)
    tt_free = free_flow_tt(graph)
    ratio = float(np.median(tt_obs[mask] / np.clip(tt_free[mask], 1e-6, None)))
    return {"edge_ids": graph["edge_ids"], "tt_obs": tt_obs, "tt_free": tt_free,
            "mask": mask, "coverage_mask": coverage,
            "regime": m.get("regime", "observed"), "source": "neshan",
            "planted_od": None, "congestion_ratio": ratio}


def main() -> None:
    rcfg = load_yaml("realworld")
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-total", type=int, default=None,
                    help="build a self-consistent demo observation near this total demand")
    ap.add_argument("--seed", type=int, default=rcfg["seed"])
    args = ap.parse_args()

    graph = load_graph()
    if args.demo_total is not None:
        obs = build_demo(rcfg, graph, args.demo_total, args.seed)
    else:
        obs = build_real(rcfg, graph)
    _save(obs, rcfg)


if __name__ == "__main__":
    main()
