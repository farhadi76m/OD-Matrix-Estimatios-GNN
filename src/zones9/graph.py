"""
graph.py — build the 9-zone road line-graph from the SUMO net + taz_9.xml.

Mirrors src/data/graph.py but is self-contained for the 9-zone study: edge ids
come straight from the network (no HDF5 dependency), zones from the new TAZ file.
  nodes = 1897 road edges, features [length, num_lanes, speed_limit]
  links = downstream edge->edge connectivity
  node_zone[i] = TAZ index (0..8), a clean 1:1 partition

Run:
    python -m src.zones9.graph
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


def build_graph() -> dict:
    cfg = load_yaml("zones9")
    net_path = resolve(cfg["paths"]["net_file"])
    taz_path = resolve(cfg["paths"]["taz_file"])

    net = sumolib.net.readNet(str(net_path))
    edge_ids = sorted(e.getID() for e in net.getEdges())   # non-internal, deterministic
    eid_to_idx = {e: i for i, e in enumerate(edge_ids)}
    n_nodes = len(edge_ids)

    zone_ids = sorted(t.get("id") for t in ET.parse(taz_path).getroot().findall("taz"))
    zid_to_idx = {z: i for i, z in enumerate(zone_ids)}
    print(f"[graph9] nodes(edges)={n_nodes}  zones={len(zone_ids)}: {zone_ids}")

    node_zone = np.full(n_nodes, -1, dtype=np.int64)
    for taz in ET.parse(taz_path).getroot().findall("taz"):
        zi = zid_to_idx[taz.get("id")]
        for e in (taz.get("edges") or "").split():
            j = eid_to_idx.get(e)
            if j is not None:
                node_zone[j] = zi
    unmapped = int((node_zone < 0).sum())
    if unmapped:
        raise RuntimeError(f"{unmapped} edges have no TAZ; cannot pool to zones.")

    static = np.zeros((n_nodes, 3), dtype=np.float32)
    for e in net.getEdges():
        static[eid_to_idx[e.getID()]] = [e.getLength(), len(e.getLanes()), e.getSpeed()]

    src, dst = [], []
    for e in net.getEdges():
        u = eid_to_idx[e.getID()]
        for down in e.getOutgoing():
            v = eid_to_idx.get(down.getID())
            if v is not None:
                src.append(u); dst.append(v)
    edge_index = np.array([src, dst], dtype=np.int64)
    print(f"[graph9] directed links={edge_index.shape[1]}  zone sizes="
          f"{np.bincount(node_zone).tolist()}")

    return {"edge_ids": edge_ids, "zone_ids": zone_ids, "node_zone": node_zone,
            "edge_index": edge_index, "static": static,
            "static_names": ["length", "num_lanes", "speed_limit"],
            "n_nodes": n_nodes, "n_zones": len(zone_ids)}


def load_graph() -> dict:
    cfg = load_yaml("zones9")
    cache = resolve(cfg["paths"]["graph_cache"])
    if not cache.exists():
        g = build_graph()
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(g, open(cache, "wb"))
        print(f"[graph9] cached -> {cache}")
    return pickle.load(open(cache, "rb"))


if __name__ == "__main__":
    g = build_graph()
    cache = resolve(load_yaml("zones9")["paths"]["graph_cache"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(g, open(cache, "wb"))
    print(f"[graph9] cached -> {cache}")
