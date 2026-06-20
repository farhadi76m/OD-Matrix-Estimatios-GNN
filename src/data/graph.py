"""
graph.py — build the static road *line-graph* once and cache it.

Nodes  = the 1897 non-internal road edges (where SUMO measurements live).
Links  = directed edge->edge connectivity through junctions (downstream);
         reverse links optionally added so context propagates both ways.
Static node features = [length, num_lanes, speed_limit] from the network.
node_zone[i] = TAZ index (0..35) the edge belongs to (clean 1:1 partition).

Node ordering is taken from the HDF5 `edge_ids` (sorted) so that the dynamic
per-edge measurements line up with these static features by row index.

Run:
    python -m src.data.graph
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root on path

import h5py
import numpy as np
import sumolib

from src.config import load_yaml, resolve


def build_graph() -> dict:
    dcfg = load_yaml("data")

    net_path = resolve(dcfg["paths"]["net_file"])
    taz_path = resolve(dcfg["paths"]["taz_file"])
    h5_path = resolve(dcfg["paths"]["source_hdf5"])

    # --- canonical node / zone ordering from the HDF5 (the data we must align to)
    with h5py.File(h5_path, "r") as f:
        edge_ids = [e.decode() for e in f["edge_ids"][:]]
        zone_ids = [z.decode() for z in f["zone_ids"][:]]
    eid_to_idx = {e: i for i, e in enumerate(edge_ids)}
    zid_to_idx = {z: i for i, z in enumerate(zone_ids)}
    n_nodes = len(edge_ids)
    print(f"[graph] nodes (edges)={n_nodes}  zones={len(zone_ids)}")

    # --- edge -> TAZ membership from the TAZ file
    import xml.etree.ElementTree as ET

    node_zone = np.full(n_nodes, -1, dtype=np.int64)
    conflicts = 0
    root = ET.parse(taz_path).getroot()
    for taz in root.findall("taz"):
        zi = zid_to_idx.get(taz.get("id"))
        if zi is None:
            continue
        for e in (taz.get("edges") or "").split():
            ni = eid_to_idx.get(e)
            if ni is None:
                continue
            if node_zone[ni] != -1 and node_zone[ni] != zi:
                conflicts += 1
            node_zone[ni] = zi
    unmapped = int((node_zone < 0).sum())
    print(f"[graph] edge->zone: unmapped={unmapped}  multi-zone conflicts={conflicts}")
    if unmapped:
        # Surface rather than silently work around (CLAUDE.md working agreement).
        raise RuntimeError(f"{unmapped} edges have no TAZ; cannot pool to zones.")

    # --- static node features + connectivity from the SUMO network
    net = sumolib.net.readNet(str(net_path))
    static = np.zeros((n_nodes, 3), dtype=np.float32)  # length, num_lanes, speed_limit
    for e in net.getEdges():
        ni = eid_to_idx.get(e.getID())
        if ni is None:
            continue
        static[ni] = [e.getLength(), len(e.getLanes()), e.getSpeed()]

    src_list, dst_list = [], []
    for e in net.getEdges():
        u = eid_to_idx.get(e.getID())
        if u is None:
            continue
        for down in e.getOutgoing():
            v = eid_to_idx.get(down.getID())
            if v is not None:
                src_list.append(u)
                dst_list.append(v)
    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    print(f"[graph] directed links={edge_index.shape[1]}")

    return {
        "edge_ids": edge_ids,
        "zone_ids": zone_ids,
        "node_zone": node_zone,                 # [N] int64, 0..35
        "edge_index": edge_index,               # [2, L] int64 (downstream direction)
        "static": static,                       # [N, 3] float32
        "static_names": ["length", "num_lanes", "speed_limit"],
        "n_nodes": n_nodes,
        "n_zones": len(zone_ids),
    }


def load_graph() -> dict:
    """Load the cached static graph, building it on first use."""
    dcfg = load_yaml("data")
    cache = resolve(dcfg["paths"]["graph_cache"])
    if not cache.exists():
        g = build_graph()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(g, f)
        print(f"[graph] cached -> {cache}")
    with open(cache, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    g = build_graph()
    dcfg = load_yaml("data")
    cache = resolve(dcfg["paths"]["graph_cache"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(g, f)
    print(f"[graph] cached -> {cache}")
    print(f"[graph] static feature ranges (length/lanes/speed):")
    s = g["static"]
    for j, name in enumerate(g["static_names"]):
        print(f"         {name:12s} min={s[:,j].min():.2f} mean={s[:,j].mean():.2f} max={s[:,j].max():.2f}")
