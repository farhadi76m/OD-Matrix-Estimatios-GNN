"""
whatif.py — Stage E (part 2): use the calibrated OD to evaluate a network change.

The calibrated demand is held FIXED; only the network (or a closure) changes, and
SUMO re-assigns routes. This is the project's end goal: "if we add/close a road,
what happens to traffic?".

Two modes:
  --alt-net PATH    simulate the calibrated OD on an edited network (the realistic
                    workflow: edit OSM/net in netedit -> netconvert -> alt.net.xml).
  --close-edges L   close edges on the BASE net via a SUMO rerouter (a quick,
                    self-contained closure what-if needing no net editing).

Caveat (honest): travel time under-determines the OD (Stage A), so the calibrated
demand is one of many that reproduce today's traffic. What-if results are most
trustworthy for moderate, local changes; large changes carry OD uncertainty.

Run:
    python -m src.realworld.whatif --close-edges <edgeId1>,<edgeId2>
    python -m src.realworld.whatif --alt-net sumo/with_new_road.net.xml
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import load_yaml, resolve
from src.data.graph import load_graph
from src.realworld import common as C
from src.realworld.sim import simulate_od


def write_closure_rerouter(path, edges) -> None:
    """A rerouter that closes the given edges for the whole simulation."""
    root = ET.Element("additional")
    rr = ET.SubElement(root, "rerouter", {"id": "whatif_close", "edges": " ".join(edges)})
    iv = ET.SubElement(rr, "interval", {"begin": "0", "end": "99999"})
    for e in edges:
        ET.SubElement(iv, "closingReroute", {"id": e})
    ET.indent(ET.ElementTree(root), space="  ")
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def network_stats(r, edge_ids):
    """Aggregate network performance from a sim result."""
    tt, flow = r["traveltime"], r["flow"]
    active = flow > 0
    return {"vehicle_hours": float((tt * flow).sum() / 3600.0),
            "active_edges": int(active.sum()),
            "mean_speed_active": float(r["speed"][active].mean()) if active.any() else 0.0,
            "total_entered": float(flow.sum())}


def main() -> None:
    rcfg = load_yaml("realworld")
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-net", type=str, help="alternate .net.xml (new/edited road)")
    ap.add_argument("--close-edges", type=str, help="comma-separated edge ids to close")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the calibrated OD (demonstrate the what-if at a "
                         "realistic congested demand; the synthetic demand range is low)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if not args.alt_net and not args.close_edges:
        ap.error("provide --alt-net PATH or --close-edges id1,id2")

    graph = load_graph(); zone_ids = graph["zone_ids"]; edge_ids = graph["edge_ids"]
    base_net = resolve(rcfg["paths"]["net_file"]); taz = resolve(rcfg["paths"]["taz_file"])
    cal = pickle.load(open(resolve(rcfg["calibrate"]["out_od"]), "rb"))
    od = cal["od_calibrated"] * args.scale
    sseed = rcfg["calibrate"]["sim_seed"]
    out_dir = resolve(args.out or rcfg["validate"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    scale_note = f" x{args.scale:g} = {od.sum():.0f}" if args.scale != 1.0 else ""
    print(f"[whatif] calibrated OD total={cal['od_calibrated'].sum():.0f}{scale_note} trips ({cal['source']})")

    base = simulate_od(od, zone_ids, edge_ids, base_net, taz, "/tmp/whatif_base", seed=sseed)
    if not base["ok"]:
        raise RuntimeError(f"baseline sim failed: {base.get('err')}")

    if args.alt_net:
        scenario = f"alt-net={Path(args.alt_net).name}"
        alt = simulate_od(od, zone_ids, edge_ids, resolve(args.alt_net), taz,
                          "/tmp/whatif_alt", seed=sseed)
    else:
        closed = [e.strip() for e in args.close_edges.split(",") if e.strip()]
        scenario = f"close={closed}"
        rr = Path("/tmp/whatif_alt"); rr.mkdir(parents=True, exist_ok=True)
        rr_file = rr / "closure.add.xml"; write_closure_rerouter(rr_file, closed)
        alt = simulate_od(od, zone_ids, edge_ids, base_net, taz, "/tmp/whatif_alt",
                          seed=sseed, extra_addl=[rr_file])
    if not alt["ok"]:
        raise RuntimeError(f"scenario sim failed: {alt.get('err')}")

    sb, sa = network_stats(base, edge_ids), network_stats(alt, edge_ids)
    dtt = alt["traveltime"] - base["traveltime"]
    order = np.argsort(-np.abs(dtt))[:10]
    movers = [{"edge": edge_ids[i], "base_tt": round(float(base["traveltime"][i]), 1),
               "alt_tt": round(float(alt["traveltime"][i]), 1),
               "delta_tt": round(float(dtt[i]), 1)} for i in order]

    report = {"scenario": scenario, "calibrated_total": float(od.sum()),
              "baseline": sb, "scenario_stats": sa,
              "delta_vehicle_hours": sa["vehicle_hours"] - sb["vehicle_hours"],
              "edges_with_tt_change": int((np.abs(dtt) > 1.0).sum()),
              "top_changed_edges": movers}
    resolve(out_dir / "whatif_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== what-if: {scenario} ===")
    print(f"  vehicle-hours: {sb['vehicle_hours']:.1f} -> {sa['vehicle_hours']:.1f} "
          f"(delta {report['delta_vehicle_hours']:+.1f})")
    print(f"  active edges:  {sb['active_edges']} -> {sa['active_edges']}")
    print(f"  mean speed:    {sb['mean_speed_active']:.2f} -> {sa['mean_speed_active']:.2f} m/s")
    print(f"  edges with travel-time change >1s: {report['edges_with_tt_change']}")
    print(f"  top changed edges:")
    for mv in movers[:5]:
        print(f"    {mv['edge']:>14s}  {mv['base_tt']:7.1f}s -> {mv['alt_tt']:7.1f}s  ({mv['delta_tt']:+.1f})")

    _plot(graph, base, alt, dtt, scenario, out_dir)
    print(f"\n[whatif] report + map -> {out_dir}")


def _plot(graph, base, alt, dtt, scenario, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from src.plot_graph import node_positions
    pos = node_positions(graph["edge_ids"], resolve(load_yaml("realworld")["paths"]["net_file"]))
    ei = graph["edge_index"]; segs = np.stack([pos[ei[0]], pos[ei[1]]], axis=1)
    chg = np.abs(dtt) > 1.0
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_collection(LineCollection(segs, colors="#eee", linewidths=0.3))
    vmax = max(np.abs(dtt[chg]).max(), 1.0) if chg.any() else 1.0
    sc = ax.scatter(pos[chg, 0], pos[chg, 1], c=dtt[chg], cmap="coolwarm",
                    vmin=-vmax, vmax=vmax, s=22, linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.04, label="Δ travel time (s): red=worse, blue=better")
    ax.set_title(f"What-if impact on travel time\n{scenario}")
    fig.tight_layout(); fig.savefig(out_dir / "whatif_delta.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
