"""
match_edges.py — Stage B (2/2): map Neshan segments to SUMO edges and build the
per-edge observed travel time + coverage mask (data/neshan_matched.pkl).

Matching priority (most reliable first), since the SUMO net is OSM-derived:
  1. OSM id   — SUMO edge ids embed the OSM way id ("894548817#1" -> way 894548817).
  2. name     — street name attribute on the edge (e.g. "توحید").
  3. geometry — nearest edge to the segment's lon/lat (sumolib spatial lookup).

Travel time is length-normalised through SPEED: from each Neshan record we take
(or derive) a speed, average per edge, then set edge travel time = edge_length /
edge_speed. This makes Neshan segment lengths and SUMO edge lengths consistent.

Run:
    python -m src.realworld.match_edges                 # uses neshan_segments.pkl
    python -m src.realworld.match_edges --self-test     # validate matching on synthetic rows
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import load_yaml, resolve
from src.data.graph import load_graph
from src.realworld import common as C


def _way_id(edge_id: str) -> str:
    return edge_id.lstrip("-").split("#")[0]


def build_indexes(edge_ids, net_path):
    """way_id -> [edge idx], name -> [edge idx]."""
    by_way, by_name = {}, {}
    for i, e in enumerate(edge_ids):
        by_way.setdefault(_way_id(e), []).append(i)
    names = C.edge_names(net_path)
    for i, e in enumerate(edge_ids):
        nm = names.get(e)
        if nm:
            by_name.setdefault(nm, []).append(i)
    return by_way, by_name


def _has_geoproj(net) -> bool:
    """True if the net can convert lon/lat <-> xy (needs a projection + pyproj)."""
    try:
        net.getGeoProj(); return True
    except Exception:
        return False


def _record_speed(rec):
    if rec.get("speed_mps"):
        return float(rec["speed_mps"])
    if rec.get("traveltime_s") and rec.get("length_m"):
        tt = float(rec["traveltime_s"])
        return float(rec["length_m"]) / tt if tt > 0 else None
    return None


def match(recs, graph, net_path):
    edge_ids = graph["edge_ids"]; n = len(edge_ids)
    length = graph["static"][:, 0]
    by_way, by_name = build_indexes(edge_ids, net_path)

    # lazy geometry lookup (only if some records carry lon/lat AND the net is geo-projected)
    net = None
    if any(r.get("lon") is not None for r in recs):
        import sumolib
        net = sumolib.net.readNet(str(resolve(net_path)))
        if not _has_geoproj(net):
            print("[match] WARNING: net has no usable geo-projection (pyproj missing?) "
                  "— geometry matching disabled; relying on osm_id / name.")
            net = None

    speed_acc = [[] for _ in range(n)]
    hits = {"osm_id": 0, "name": 0, "geometry": 0, "unmatched": 0}
    for r in recs:
        spd = _record_speed(r)
        if not spd or spd <= 0:
            hits["unmatched"] += 1; continue
        idxs, how = [], None
        if r.get("osm_id") is not None and str(r["osm_id"]) in by_way:
            idxs, how = by_way[str(r["osm_id"])], "osm_id"
        elif r.get("name") and r["name"] in by_name:
            idxs, how = by_name[r["name"]], "name"
        elif net is not None and r.get("lon") is not None:
            try:
                x, y = net.convertLonLat2XY(float(r["lon"]), float(r["lat"]))
                near = net.getNeighboringEdges(x, y, r=50.0)
                if near:
                    eid = min(near, key=lambda t: t[1])[0].getID()
                    j = {e: i for i, e in enumerate(edge_ids)}.get(eid)
                    if j is not None:
                        idxs, how = [j], "geometry"
            except Exception:
                idxs = []
        if not idxs:
            hits["unmatched"] += 1; continue
        hits[how] += 1
        for j in idxs:
            speed_acc[j].append(spd)

    tt_obs = np.zeros(n); coverage = np.zeros(n, dtype=bool)
    for i in range(n):
        if speed_acc[i]:
            v = float(np.mean(speed_acc[i]))
            tt_obs[i] = length[i] / max(v, 0.1)
            coverage[i] = True
    return tt_obs, coverage, hits


def synth_records(graph, net_path, n_osm=40, n_name=40, n_geo=20, seed=0):
    """Make synthetic Neshan-like rows from known edges to validate matching."""
    rng = np.random.default_rng(seed)
    edge_ids = graph["edge_ids"]; length = graph["static"][:, 0]
    names = C.edge_names(net_path)
    named = [i for i, e in enumerate(edge_ids) if names.get(e)]
    import sumolib
    net = sumolib.net.readNet(str(resolve(net_path)))
    recs = []
    for i in rng.choice(len(edge_ids), n_osm, replace=False):
        recs.append({"osm_id": _way_id(edge_ids[i]), "name": None, "lon": None,
                     "lat": None, "length_m": float(length[i]), "traveltime_s": float(length[i] / 8.0)})
    for i in rng.choice(named, min(n_name, len(named)), replace=False):
        recs.append({"osm_id": None, "name": names[edge_ids[i]], "lon": None,
                     "lat": None, "length_m": float(length[i]), "speed_mps": 7.0})
    if _has_geoproj(net):
        for i in rng.choice(len(edge_ids), n_geo, replace=False):
            e = net.getEdge(edge_ids[i]); x, y = np.mean(e.getShape(), axis=0)
            lon, lat = net.convertXY2LonLat(x, y)
            recs.append({"osm_id": None, "name": None, "lon": lon, "lat": lat,
                         "length_m": float(length[i]), "speed_mps": 9.0})
    return recs


def main() -> None:
    rcfg = load_yaml("realworld")
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="data/neshan/neshan_segments.pkl")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    graph = load_graph(); net_path = rcfg["paths"]["net_file"]
    if args.self_test:
        recs = synth_records(graph, net_path)
        print(f"[match] self-test with {len(recs)} synthetic records")
    else:
        p = resolve(args.inp)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run neshan_ingest first (or --self-test).")
        recs = pickle.load(open(p, "rb"))

    tt_obs, coverage, hits = match(recs, graph, net_path)
    print(f"[match] matched by {hits}")
    print(f"[match] coverage: {int(coverage.sum())}/{len(graph['edge_ids'])} edges "
          f"({100*coverage.mean():.1f}%)")

    if args.self_test:
        ok = int(coverage.sum())
        print(f"[match] self-test {'PASS' if ok > 50 else 'CHECK'}: {ok} edges matched "
              f"(unmatched rows={hits['unmatched']})")
        return

    out = resolve("data/neshan_matched.pkl")
    with open(out, "wb") as f:
        pickle.dump({"tt_obs": tt_obs, "coverage_mask": coverage,
                     "regime": rcfg.get("neshan", {}).get("regime", "observed"),
                     "match_hits": hits}, f)
    print(f"[match] -> {out}  (now run: python -m src.realworld.build_observation)")


if __name__ == "__main__":
    main()
