"""
graph.py — Step 1b: build the road line-graph from the SUMO net + TAZ file.

  nodes = road edges, static features [length, num_lanes, speed_limit]
  links = downstream edge->edge connectivity
  node_zone[i] = TAZ index (clean 1:1 partition; raises if any edge is unmapped)

Zone count is taken from the TAZ file (any number). Cached to paths.graph_cache.

Run:
    python -m src.odpipe.graph
"""

from __future__ import annotations

import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sumolib

from src.config import load_yaml, resolve
from src.odpipe import CFG


def build_graph() -> dict:
    cfg = load_yaml(CFG)
    net_path = resolve(cfg["paths"]["net_file"])
    taz_path = resolve(cfg["paths"]["taz_file"])
    if not taz_path.exists():
        raise FileNotFoundError(f"{taz_path} missing — run `python -m src.odpipe.taz` first.")

    net = sumolib.net.readNet(str(net_path))
    edge_ids = sorted(e.getID() for e in net.getEdges())
    eid = {e: i for i, e in enumerate(edge_ids)}
    n_nodes = len(edge_ids)

    zone_ids = sorted(t.get("id") for t in ET.parse(taz_path).getroot().findall("taz"))
    zid = {z: i for i, z in enumerate(zone_ids)}
    print(f"[graph] nodes(edges)={n_nodes}  zones={len(zone_ids)}")

    node_zone = np.full(n_nodes, -1, dtype=np.int64)
    for taz in ET.parse(taz_path).getroot().findall("taz"):
        zi = zid[taz.get("id")]
        for e in (taz.get("edges") or "").split():
            j = eid.get(e)
            if j is not None:
                node_zone[j] = zi
    unmapped = int((node_zone < 0).sum())
    if unmapped:
        raise RuntimeError(f"{unmapped} edges have no TAZ; cannot pool to zones.")

    static = np.zeros((n_nodes, 3), dtype=np.float32)
    for e in net.getEdges():
        static[eid[e.getID()]] = [e.getLength(), len(e.getLanes()), e.getSpeed()]

    src, dst = [], []
    for e in net.getEdges():
        u = eid[e.getID()]
        for down in e.getOutgoing():
            v = eid.get(down.getID())
            if v is not None:
                src.append(u); dst.append(v)
    edge_index = np.array([src, dst], dtype=np.int64)
    print(f"[graph] links={edge_index.shape[1]}  zone sizes={np.bincount(node_zone).tolist()}")

    return {"edge_ids": edge_ids, "zone_ids": zone_ids, "node_zone": node_zone,
            "edge_index": edge_index, "static": static,
            "static_names": ["length", "num_lanes", "speed_limit"],
            "n_nodes": n_nodes, "n_zones": len(zone_ids)}


def load_graph() -> dict:
    cfg = load_yaml(CFG)
    cache = resolve(cfg["paths"]["graph_cache"])
    if not cache.exists():
        g = build_graph()
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(g, open(cache, "wb"))
        print(f"[graph] cached -> {cache}")
    return pickle.load(open(cache, "rb"))


if __name__ == "__main__":
    g = build_graph()
    cache = resolve(load_yaml(CFG)["paths"]["graph_cache"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(g, open(cache, "wb"))
    print(f"[graph] cached -> {cache}")
