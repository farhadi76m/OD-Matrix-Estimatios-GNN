"""
resimulate.py — re-run SUMO on the EXISTING OD matrices to capture richer
measurements that make the OD more identifiable:

  * time-sliced edge data : flow/speed/density/traveltime per interval
                            (T intervals instead of one aggregate),
  * junction turn counts  : how many vehicles traverse each line-graph link
                            (u -> v), i.e. how through-traffic splits. Turn
                            counts are the classic remedy for OD under-
                            determination, and they live on edge_index, giving
                            the GNN real EDGE features.

OD + seed are read from od_dataset.h5, so targets are unchanged — only the
inputs get richer, enabling a clean old-vs-new comparison. Uses the version-
matched SUMO 1.27 binaries (sumolib.checkBinary), not /usr/bin (1.18).

Run:
    python -m src.data.resimulate --start 0 --end 200 --workers 4
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np

from src.config import load_yaml, resolve
from src.data.graph import load_graph
from src import sumo_export as sx

# time-sliced collection window (post-warmup) and number of intervals
WARMUP, SIM_END, N_INTERVALS = 300, 3600, 6
ATTRS = {"entered": "flow", "speed": "speed", "density": "density", "traveltime": "traveltime"}
OUT_DIR = "data/resim"


def _write_addl(path, out_abs):
    period = (SIM_END - WARMUP) / N_INTERVALS
    root = ET.Element("additional")
    ET.SubElement(root, "edgeData", {"id": "e", "file": str(out_abs), "begin": str(WARMUP),
                                     "end": str(SIM_END), "period": str(period), "excludeEmpty": "true"})
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def _write_cfg(path, net, routes, addl):
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=str(Path(net).resolve()))
    ET.SubElement(inp, "route-files", value=str(Path(routes).resolve()))
    ET.SubElement(inp, "additional-files", value=str(Path(addl).resolve()))
    tm = ET.SubElement(root, "time")
    ET.SubElement(tm, "begin", value="0"); ET.SubElement(tm, "end", value=str(SIM_END))
    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "ignore-route-errors", value="true")
    ET.SubElement(proc, "time-to-teleport", value="300")
    rep = ET.SubElement(root, "report")
    ET.SubElement(rep, "no-step-log", value="true"); ET.SubElement(rep, "no-warnings", value="true")
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def _parse_timesliced(xml_path, eid_idx, n_edges):
    """-> [T, n_edges, 4] flow/speed/density/traveltime."""
    x = np.zeros((N_INTERVALS, n_edges, 4), dtype=np.float32)
    keys = ["flow", "speed", "density", "traveltime"]
    try:
        intervals = ET.parse(xml_path).getroot().findall("interval")
    except (ET.ParseError, FileNotFoundError):
        return x
    for t, interval in enumerate(intervals[:N_INTERVALS]):
        for edge in interval.findall("edge"):
            i = eid_idx.get(edge.get("id", ""))
            if i is None:
                continue
            for c, (sumo_a, key) in enumerate(ATTRS.items()):
                v = edge.get(sumo_a)
                x[t, i, keys.index(key)] = float(v) if v not in (None, "nan", "") else 0.0
    return x


def _parse_turncounts(routes_path, eid_idx, link_idx):
    """-> [n_links] count of vehicles traversing each line-graph link u->v."""
    tc = np.zeros(len(link_idx), dtype=np.float32)
    try:
        root = ET.parse(routes_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return tc
    for route in root.iter("route"):
        edges = (route.get("edges") or "").split()
        for a, b in zip(edges, edges[1:]):
            u, v = eid_idx.get(a), eid_idx.get(b)
            if u is None or v is None:
                continue
            k = link_idx.get((u, v))
            if k is not None:
                tc[k] += 1
    return tc


def run_one(idx, od, seed, zone_ids, eid_idx, link_idx, n_edges, net, taz):
    work = Path("/tmp/resim") / f"s{idx:06d}"
    work.mkdir(parents=True, exist_ok=True)
    trips, routes = work / "trips.xml", work / "routes.xml"
    addl, edge_ts, cfg = work / "addl.xml", work / "edge_ts.xml", work / "sim.sumocfg"
    try:
        sx.write_trips_xml(od.astype(float), zone_ids, trips, seed=seed)  # exact for integer OD
        ok, err = sx.run_duarouter(net, taz, trips, routes, seed=seed)
        if not ok:
            return idx, None, f"duarouter: {err[:80]}"
        _write_addl(addl, edge_ts.resolve()); _write_cfg(cfg, net, routes, addl)
        r = subprocess.run([sx._bin("sumo"), "-c", str(cfg)], capture_output=True, text=True, timeout=360)
        if r.returncode != 0:
            return idx, None, f"sumo: {(r.stderr or '')[:80]}"
        x_ts = _parse_timesliced(edge_ts, eid_idx, n_edges)
        turn = _parse_turncounts(routes, eid_idx, link_idx)
        return idx, {"idx": idx, "od": od.astype(np.int32), "seed": seed,
                     "x_ts": x_ts, "turn": turn}, None
    except Exception as e:
        return idx, None, f"exc: {e}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    dcfg = load_yaml("data")
    graph = load_graph()
    eid_idx = {e: i for i, e in enumerate(graph["edge_ids"])}
    ei = graph["edge_index"]
    link_idx = {(int(ei[0, k]), int(ei[1, k])): k for k in range(ei.shape[1])}
    zone_ids = graph["zone_ids"]; n_edges = graph["n_nodes"]
    net = str(resolve(dcfg["paths"]["net_file"])); taz = str(resolve(dcfg["paths"]["taz_file"]))
    out = resolve(OUT_DIR); out.mkdir(parents=True, exist_ok=True)

    with h5py.File(resolve(dcfg["paths"]["source_hdf5"]), "r") as f:
        N = f["od_matrices"].shape[0]
        end = min(args.end, N)
        todo = [i for i in range(args.start, end) if not (out / f"resim_{i:06d}.pkl").exists()]
        ods = {i: f["od_matrices"][i] for i in todo}
        seeds = {i: int(f["sample_seeds"][i]) for i in todo}
    print(f"[resim] {len(todo)} samples to run (range {args.start}:{end}), workers={args.workers}")

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, i, ods[i], seeds[i], zone_ids, eid_idx, link_idx,
                            n_edges, net, taz): i for i in todo}
        for fut in as_completed(futs):
            idx, res, err = fut.result()
            if res is None:
                fail += 1
                if fail <= 5:
                    print(f"  [{idx}] FAIL {err}")
            else:
                with open(out / f"resim_{idx:06d}.pkl", "wb") as pf:
                    pickle.dump(res, pf)
                ok += 1
            if (ok + fail) % 25 == 0:
                print(f"  progress {ok+fail}/{len(todo)} (ok={ok} fail={fail})", flush=True)
    print(f"[resim] done: ok={ok} fail={fail} -> {out}")


if __name__ == "__main__":
    main()
