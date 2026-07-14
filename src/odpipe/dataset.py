"""
dataset.py — ONE command to build the whole OD dataset.

  TAZ (gridDistricts)  ->  road line-graph  ->  simulate OD samples in SUMO
  ->  per-sample .pkl  ->  train/val/test packs

Each sample: draw a total demand (stratified log tiers over demand_range), sample
an OD (gravity / dirichlet / sparse mix), simulate it (mesoscopic if configured),
and store per-edge [flow, speed, density, traveltime] + junction TURN COUNTS —
the measurement that makes the OD identifiable.

  data_dir/samples/sample_NNNNNN.pkl  {idx, od[Z,Z], seed, x_dyn[N,4], turn[L],
                                       tier, tod, total_demand}
  data_dir/index.json                 manifest + stratified split
  data_dir/pack_{train,val,test}.pt   tensors for training
  data_dir/norm.json                  train-only normalisation

Resumable (skips existing samples) and parallel. All knobs: configs/odpipe.yaml.

Run:
    python -m src.odpipe.dataset                        # everything
    python -m src.odpipe.dataset --samples 300          # override n_samples
    python -m src.odpipe.dataset --start 0 --end 500    # a chunk
    python -m src.odpipe.dataset --packs-only           # just rebuild the packs
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.config import load_yaml, resolve
from src.odpipe import CFG
from src.odpipe.graph import build_graph, load_graph
from src.sumo_export import _wheel_home, ensure_sumo_home


# ── 1. TAZ ────────────────────────────────────────────────────────────────────
def make_taz(cfg, width) -> Path:
    ensure_sumo_home()
    import os
    tool = None
    for base in (_wheel_home(), os.environ.get("SUMO_HOME")):
        if base and (Path(base) / "tools/district/gridDistricts.py").exists():
            tool = Path(base) / "tools/district/gridDistricts.py"; break
    if tool is None:
        raise FileNotFoundError("gridDistricts.py not found under SUMO_HOME/tools/district")

    net, out = resolve(cfg["paths"]["net_file"]), resolve(cfg["paths"]["taz_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(tool), "-n", str(net), "-o", str(out),
                        "-w", str(width)], capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"gridDistricts failed:\n{r.stderr[-600:]}")

    sizes = sorted(len((t.get("edges") or "").split())
                   for t in ET.parse(out).getroot().findall("taz"))
    print(f"[taz] width={width}m -> {len(sizes)} zones  edge-counts={sizes}")
    if any(s < 5 for s in sizes):
        print(f"[taz] NOTE: {sum(s < 5 for s in sizes)} tiny zone(s) (<5 edges) — their "
              f"marginals will be noisy. Use a larger width for balanced zones.")
    return out


# ── 2. OD samplers ────────────────────────────────────────────────────────────
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
    D[rng.choice(n, rng.integers(1, max(2, n // 3)), replace=False)] *= rng.uniform(3.0, 10.0)
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
    w = np.zeros(npairs); w[rng.choice(npairs, k, replace=False)] = np.exp(rng.exponential(1.5, k))
    od = np.zeros((n, n), dtype=np.int32)
    od[~np.eye(n, dtype=bool)] = rng.multinomial(total, w / w.sum())
    return od


_SAMPLERS = {"gravity": od_gravity, "dirichlet": od_dirichlet, "sparse": od_sparse}

# ── 3. simulate (parallel workers) ────────────────────────────────────────────
_G = {}


def _init(cfg):
    g = load_graph(); ei = g["edge_index"]
    _G.update(cfg=cfg, graph=g,
              eid_idx={e: i for i, e in enumerate(g["edge_ids"])},
              link_idx={(int(ei[0, k]), int(ei[1, k])): k for k in range(ei.shape[1])},
              pos=grid_positions(g["zone_ids"]),
              net=str(resolve(cfg["paths"]["net_file"])),
              taz=str(resolve(cfg["paths"]["taz_file"])))


def _simulate(od, work, sim_seed, meso):
    from src.data.resimulate import _parse_turncounts
    from src.realworld.sim import _parse_edgedata, _write_additional, _write_cfg
    from src.sumo_export import _bin, run_duarouter, write_trips_xml
    ensure_sumo_home()
    g = _G["graph"]; eid_idx = _G["eid_idx"]
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    trips, routes = work / "trips.xml", work / "routes.xml"
    addl, edge_out, cfgf = work / "addl.xml", work / "edge.xml", work / "sim.sumocfg"

    if write_trips_xml(od, g["zone_ids"], trips, seed=sim_seed, prefix="od_") == 0:
        return None
    ok, err = run_duarouter(_G["net"], _G["taz"], trips, routes, seed=sim_seed)
    if not ok:
        return {"err": f"duarouter:{err[:60]}"}
    turn = _parse_turncounts(routes, eid_idx, _G["link_idx"])
    _write_additional(addl, edge_out); _write_cfg(cfgf, _G["net"], routes, [addl], sim_seed)
    cmd = [_bin("sumo"), "-c", str(cfgf)] + (["--mesosim", "--meso-junction-control", "true"] if meso else [])
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    except subprocess.TimeoutExpired:
        return {"err": "sumo_timeout"}
    if r.returncode != 0:
        return {"err": f"sumo_rc{r.returncode}"}
    tt, spd, flow, den = _parse_edgedata(edge_out, eid_idx, len(g["edge_ids"]))
    return {"x_dyn": np.stack([flow, spd, den, tt], 1).astype(np.float32), "turn": turn}


def _work(task):
    idx, total, tier, method = task
    try:
        cfg = _G["cfg"]; n = _G["graph"]["n_zones"]
        seed = cfg["master_seed"] + idx
        od = _SAMPLERS[method](np.random.default_rng(seed), n, int(total), _G["pos"])
        # unique work dir per sample: a shared dir races across workers
        res = _simulate(od, f"/tmp/odpipe_gen/{idx:06d}", cfg["sim_seed"],
                        cfg.get("mesoscopic", False))
        if res is None:
            return (idx, False, "zero_trips")
        if "err" in res:
            return (idx, False, res["err"])
        pickle.dump({"idx": idx, "od": od, "seed": seed, "x_dyn": res["x_dyn"],
                     "turn": res["turn"], "tier": int(tier),
                     "tod": cfg["tier_labels"][int(tier)], "total_demand": int(od.sum())},
                    open(resolve(cfg["paths"]["samples_dir"]) / f"sample_{idx:06d}.pkl", "wb"))
        return (idx, True, None)
    except Exception as e:
        return (idx, False, f"exc:{type(e).__name__}")


def plan_tasks(cfg, n_samples):
    lo, hi, nt = *cfg["demand_range"], cfg["n_tiers"]
    breaks = np.exp(np.linspace(np.log(lo), np.log(hi), nt + 1))
    methods = list(cfg["od_mix"])
    probs = np.array([cfg["od_mix"][m] for m in methods]); probs /= probs.sum()
    rng = np.random.default_rng(cfg["master_seed"])
    return [(i, float(rng.uniform(breaks[i % nt], breaks[i % nt + 1])), i % nt,
             rng.choice(methods, p=probs)) for i in range(n_samples)]


# ── 4. manifest + packs ───────────────────────────────────────────────────────
def write_manifest(cfg, n_zones):
    rows = []
    for fp in sorted(resolve(cfg["paths"]["samples_dir"]).glob("sample_*.pkl")):
        s = pickle.load(open(fp, "rb"))
        rows.append({"idx": s["idx"], "file": f"samples/{fp.name}", "tier": s["tier"],
                     "tod": s["tod"], "total_demand": s["total_demand"]})
    fr = cfg["split"]["fractions"]; rng = np.random.default_rng(cfg["split"]["seed"])
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    for rs in by_tier.values():                      # stratified by demand tier
        rng.shuffle(rs)
        ntr = int(round(len(rs) * fr["train"])); nva = int(round(len(rs) * fr["val"]))
        for r in rs[:ntr]: r["split"] = "train"
        for r in rs[ntr:ntr + nva]: r["split"] = "val"
        for r in rs[ntr + nva:]: r["split"] = "test"
    rows.sort(key=lambda r: r["idx"])
    counts = {sp: sum(r["split"] == sp for r in rows) for sp in ("train", "val", "test")}
    resolve(cfg["paths"]["manifest"]).write_text(json.dumps(
        {"n_samples": len(rows), "n_zones": n_zones, "splits": counts, "samples": rows}, indent=2))
    print(f"[data] manifest: {len(rows)} samples, splits={counts}")
    return rows


def features(s, static, src, dst, n):
    """Node features [N,9] + per-link turn counts [L] for one sample (also used at inference)."""
    turn = s["turn"].astype(np.float32)
    out_turn = np.zeros(n, np.float32); np.add.at(out_turn, src, turn)
    in_turn = np.zeros(n, np.float32); np.add.at(in_turn, dst, turn)
    x = np.concatenate([static, s["x_dyn"], np.stack([out_turn, in_turn], 1)], 1)
    return x.astype(np.float32), turn


def build_packs(cfg, g, rows):
    ei = g["edge_index"]; src, dst = ei[0], ei[1]
    static = g["static"].astype(np.float32); n = g["n_nodes"]
    root = resolve(cfg["paths"]["data_dir"])

    packs = {}
    for split in ("train", "val", "test"):
        X, EW, Y, meta = [], [], [], []
        for r in [r for r in rows if r["split"] == split]:
            s = pickle.load(open(root / r["file"], "rb"))
            x, turn = features(s, static, src, dst, n)
            X.append(x); EW.append(turn); Y.append(s["od"].astype(np.float32))
            meta.append({"idx": s["idx"], "tier": s["tier"], "tod": s["tod"],
                         "total_demand": s["total_demand"]})
        packs[split] = {"x": np.stack(X), "ew": np.stack(EW), "y": np.stack(Y), "meta": meta}

    lx = np.log1p(np.clip(packs["train"]["x"], 0, None))     # train-only normalisation
    resolve(f"{cfg['paths']['data_dir']}/norm.json").write_text(json.dumps({
        "x_mean": lx.mean((0, 1)).tolist(),
        "x_std": np.maximum(lx.reshape(-1, lx.shape[2]).std(0), 1e-6).tolist(),
        "feat_names": ["length", "lanes", "speed_limit", "flow", "speed", "density",
                       "traveltime", "out_turn", "in_turn"]}, indent=2))
    for split, p in packs.items():
        torch.save({"x": torch.tensor(p["x"]), "ew": torch.tensor(p["ew"]),
                    "y": torch.tensor(p["y"]), "meta": p["meta"]}, root / f"pack_{split}.pt")
    print(f"[data] packs -> {root}/pack_*.pt  feature_dim={packs['train']['x'].shape[2]}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = load_yaml(CFG)
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=cfg["taz"]["width"], help="TAZ grid width (m)")
    ap.add_argument("--samples", type=int, default=cfg["n_samples"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--workers", type=int, default=cfg["n_workers"])
    ap.add_argument("--skip-taz", action="store_true", help="reuse the existing TAZ file")
    ap.add_argument("--packs-only", action="store_true", help="rebuild manifest + packs only")
    args = ap.parse_args()

    resolve(cfg["paths"]["samples_dir"]).mkdir(parents=True, exist_ok=True)

    # 1. TAZ + 2. graph (rebuild the cached graph whenever the TAZ changes)
    if not args.packs_only and not args.skip_taz:
        make_taz(cfg, args.width)
        cache = resolve(cfg["paths"]["graph_cache"])
        if cache.exists():
            cache.unlink()
    g = load_graph()

    # 3. simulate
    if not args.packs_only:
        end = args.end if args.end is not None else args.samples
        sdir = resolve(cfg["paths"]["samples_dir"])
        tasks = [t for t in plan_tasks(cfg, args.samples)
                 if args.start <= t[0] < end and not (sdir / f"sample_{t[0]:06d}.pkl").exists()]
        print(f"[data] simulating {len(tasks)} samples ({args.workers} workers, "
              f"meso={cfg.get('mesoscopic', False)}, demand {cfg['demand_range']})")
        done = fail = 0
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(cfg,)) as ex:
            for fut in as_completed([ex.submit(_work, t) for t in tasks]):
                idx, ok, err = fut.result()
                done += ok; fail += (not ok)
                if (done + fail) % 50 == 0 or not ok:
                    print(f"[data] {done+fail}/{len(tasks)} done={done} fail={fail}"
                          + ("" if ok else f"  idx={idx} FAIL({err})"), flush=True)
        print(f"[data] simulated: done={done} fail={fail}")

    # 4. manifest + packs
    rows = write_manifest(cfg, g["n_zones"])
    build_packs(cfg, g, rows)
    print(f"[data] DONE — next:  python -m src.odpipe.train")


if __name__ == "__main__":
    main()
