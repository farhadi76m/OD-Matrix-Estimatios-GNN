"""
generate.py — Step 2: build the 9-zone OD dataset.

For each sample: draw a total demand (stratified log tiers), sample a 9x9 OD with
the gravity/dirichlet/sparse mix (gravity uses the REAL zone grid positions),
simulate it through SUMO (eclipse-sumo 1.27), and store per-edge measurements
[flow, speed, density, traveltime] plus junction TURN COUNTS — the measurement
that makes the OD identifiable (turn->marginal corr ~0.9 in the 36-zone study).

Per sample -> data9/samples/sample_NNNNNN.pkl
  {idx, od[9,9], seed, x_dyn[1897,4], turn[L], tier, tod, total_demand}
Resumable (skips existing). Parallel (ProcessPoolExecutor). Writes index.json.

Run:
    python -m src.zones9.generate                       # all configured samples
    python -m src.zones9.generate --start 0 --end 500   # a chunk
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import load_yaml, resolve
from src.zones9.graph import load_graph


# ── OD samplers (gravity uses real zone positions; dirichlet/sparse geometry-free)
def grid_positions(zone_ids):
    return np.array([[float(p) for p in z.split("_")] for z in zone_ids])


def _draw(rng, raw, n, total):
    np.fill_diagonal(raw, 0.0); raw = np.maximum(raw, 0.0)
    if raw.sum() < 1e-12:
        raw = np.ones((n, n)) - np.eye(n)
    flat = raw[~np.eye(n, dtype=bool)]
    od = np.zeros((n, n), dtype=np.int32)
    od[~np.eye(n, dtype=bool)] = rng.multinomial(total, flat / flat.sum())
    return od


def od_gravity(rng, n, total, pos):
    O = np.exp(rng.normal(1.0, 0.8, n)); D = np.exp(rng.normal(1.0, 0.8, n))
    beta = rng.uniform(1.0, 3.5)
    attrs = rng.choice(n, rng.integers(1, 4), replace=False)
    D[attrs] *= rng.uniform(3.0, 10.0, len(attrs))
    p = pos + rng.normal(0, 0.2, pos.shape)
    raw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                raw[i, j] = O[i] * D[j] * (np.linalg.norm(p[i] - p[j]) + 1e-6) ** (-beta)
    raw /= (raw.sum(1, keepdims=True) + 1e-12); raw /= (raw.sum(0, keepdims=True) + 1e-12)
    return _draw(rng, raw, n, total)


def od_dirichlet(rng, n, total, pos=None):
    npairs = n * (n - 1)
    alpha = np.full(npairs, np.exp(rng.uniform(np.log(0.1), np.log(8.0))))
    hot = rng.choice(npairs, max(1, npairs // 5), replace=False)
    alpha[hot] *= rng.uniform(2.0, 12.0, len(hot))
    od = np.zeros((n, n), dtype=np.int32)
    od[~np.eye(n, dtype=bool)] = rng.multinomial(total, rng.dirichlet(alpha))
    return od


def od_sparse(rng, n, total, pos=None):
    npairs = n * (n - 1)
    k = min(int(rng.integers(max(3, n // 4), max(4, n) + 1)), npairs)
    w = np.zeros(npairs); act = rng.choice(npairs, k, replace=False)
    w[act] = np.exp(rng.exponential(1.5, k))
    od = np.zeros((n, n), dtype=np.int32)
    od[~np.eye(n, dtype=bool)] = rng.multinomial(total, w / w.sum())
    return od


_SAMPLERS = {"gravity": od_gravity, "dirichlet": od_dirichlet, "sparse": od_sparse}


# ── per-worker simulation ─────────────────────────────────────────────────────
_G = {}


def _init(cfg):
    g = load_graph()
    ei = g["edge_index"]
    _G.update(cfg=cfg, graph=g,
              eid_idx={e: i for i, e in enumerate(g["edge_ids"])},
              link_idx={(int(ei[0, k]), int(ei[1, k])): k for k in range(ei.shape[1])},
              pos=grid_positions(g["zone_ids"]),
              net=str(resolve(cfg["paths"]["net_file"])),
              taz=str(resolve(cfg["paths"]["taz_file"])))


def _simulate(od, work, sim_seed):
    from src.realworld.sim import _write_additional, _write_cfg, _parse_edgedata
    from src.sumo_export import write_trips_xml, run_duarouter, _bin, ensure_sumo_home
    from src.data.resimulate import _parse_turncounts
    ensure_sumo_home()
    g = _G["graph"]; eid_idx = _G["eid_idx"]; n_edges = len(g["edge_ids"])
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    trips, routes = work / "trips.xml", work / "routes.xml"
    addl, edge_out, cfg = work / "addl.xml", work / "edge.xml", work / "sim.sumocfg"

    n_trips = write_trips_xml(od, g["zone_ids"], trips, seed=sim_seed, prefix="z9_")
    if n_trips == 0:
        return None
    ok, err = run_duarouter(_G["net"], _G["taz"], trips, routes, seed=sim_seed)
    if not ok:
        return {"err": f"duarouter:{err[:80]}"}
    turn = _parse_turncounts(routes, eid_idx, _G["link_idx"])
    _write_additional(addl, edge_out); _write_cfg(cfg, _G["net"], routes, [addl], sim_seed)
    try:
        r = subprocess.run([_bin("sumo"), "-c", str(cfg)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=360)
    except subprocess.TimeoutExpired:
        return {"err": "sumo_timeout"}
    if r.returncode != 0:
        return {"err": f"sumo_rc{r.returncode}"}
    tt, spd, flow, den = _parse_edgedata(edge_out, eid_idx, n_edges)
    return {"x_dyn": np.stack([flow, spd, den, tt], 1).astype(np.float32), "turn": turn}


def _work(task):
    idx, total, tier, method = task
    try:
        cfg = _G["cfg"]; g = _G["graph"]; n = g["n_zones"]
        seed = cfg["master_seed"] + idx
        rng = np.random.default_rng(seed)
        od = _SAMPLERS[method](rng, n, int(total), _G["pos"])
        res = _simulate(od, f"/tmp/zones9_gen/w{idx % cfg['n_workers']}", cfg["sim_seed"])
        if res is None:
            return (idx, False, "zero_trips")
        if "err" in res:
            return (idx, False, res["err"])
        out = resolve(cfg["paths"]["samples_dir"]) / f"sample_{idx:06d}.pkl"
        pickle.dump({"idx": idx, "od": od, "seed": seed, "x_dyn": res["x_dyn"],
                     "turn": res["turn"], "tier": int(tier),
                     "tod": cfg["tier_labels"][int(tier)], "total_demand": int(od.sum())},
                    open(out, "wb"))
        return (idx, True, None)
    except Exception as e:                          # never let one bad sample kill the pool
        return (idx, False, f"exc:{type(e).__name__}")


def plan_tasks(cfg):
    """Round-robin tiers for exact balance; total log-uniform within tier; method by mix."""
    lo, hi, nt = *cfg["demand_range"], cfg["n_tiers"]
    breaks = np.exp(np.linspace(np.log(lo), np.log(hi), nt + 1))
    methods = list(cfg["od_mix"]); probs = np.array([cfg["od_mix"][m] for m in methods])
    probs = probs / probs.sum()
    rng = np.random.default_rng(cfg["master_seed"])
    tasks = []
    for idx in range(cfg["n_samples"]):
        tier = idx % nt
        total = float(rng.uniform(breaks[tier], breaks[tier + 1]))
        method = rng.choice(methods, p=probs)
        tasks.append((idx, total, tier, method))
    return tasks


def write_manifest(cfg):
    sdir = resolve(cfg["paths"]["samples_dir"])
    files = sorted(sdir.glob("sample_*.pkl"))
    rows = []
    for fp in files:
        s = pickle.load(open(fp, "rb"))
        rows.append({"idx": s["idx"], "file": f"samples/{fp.name}", "tier": s["tier"],
                     "tod": s["tod"], "total_demand": s["total_demand"]})
    # stratified split by tier
    fr = cfg["split"]["fractions"]; rng = np.random.default_rng(cfg["split"]["seed"])
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    for t, rs in by_tier.items():
        rng.shuffle(rs)
        ntr = int(round(len(rs) * fr["train"])); nva = int(round(len(rs) * fr["val"]))
        for r in rs[:ntr]: r["split"] = "train"
        for r in rs[ntr:ntr + nva]: r["split"] = "val"
        for r in rs[ntr + nva:]: r["split"] = "test"
    rows.sort(key=lambda r: r["idx"])
    counts = {sp: sum(r["split"] == sp for r in rows) for sp in ("train", "val", "test")}
    manifest = {"n_samples": len(rows), "n_zones": 9, "splits": counts, "samples": rows}
    resolve(cfg["paths"]["manifest"]).write_text(json.dumps(manifest, indent=2))
    print(f"[gen] manifest: {len(rows)} samples, splits={counts}")


def main():
    cfg = load_yaml("zones9")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--workers", type=int, default=cfg["n_workers"])
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    resolve(cfg["paths"]["samples_dir"]).mkdir(parents=True, exist_ok=True)
    load_graph()  # build/cache graph before forking workers
    if args.manifest_only:
        write_manifest(cfg); return

    end = args.end if args.end is not None else cfg["n_samples"]
    sdir = resolve(cfg["paths"]["samples_dir"])
    tasks = [t for t in plan_tasks(cfg) if args.start <= t[0] < end
             and not (sdir / f"sample_{t[0]:06d}.pkl").exists()]
    print(f"[gen] {len(tasks)} samples to simulate (range [{args.start},{end}), "
          f"{args.workers} workers)")
    done = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(cfg,)) as ex:
        for fut in as_completed([ex.submit(_work, t) for t in tasks]):
            idx, ok, err = fut.result()
            done += ok; fail += (not ok)
            if (done + fail) % 50 == 0 or not ok:
                tag = "ok" if ok else f"FAIL({err})"
                print(f"[gen] {done+fail}/{len(tasks)} done={done} fail={fail}  idx={idx} {tag}",
                      flush=True)
    print(f"[gen] finished: done={done} fail={fail}")
    write_manifest(cfg)


if __name__ == "__main__":
    main()
