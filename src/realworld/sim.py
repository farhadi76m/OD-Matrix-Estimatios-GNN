"""
sim.py — headless SUMO run for ONE OD matrix, returning per-edge measurements.

This is the forward operator used inside the calibration loop: given a candidate
OD, simulate it and read back per-edge travel time (and speed/flow/density). It
mirrors the synthetic generator (od_generator/sumo_runner.py) exactly — same
warm-up, same edgeData window, excludeEmpty — so simulated travel times are
comparable to the training distribution, but uses the version-matched
eclipse-sumo 1.27 binaries (src/sumo_export.py) instead of the system 1.18.

    OD -> trips.xml -> duarouter -> routes.xml -> edgeData -> {tt, speed, flow, density}
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.sumo_export import _bin, ensure_sumo_home, run_duarouter, write_trips_xml

WARMUP = 300
SIM_END = 3600


def _write_additional(path, edge_out, begin=WARMUP, end=SIM_END) -> None:
    root = ET.Element("additional")
    ET.SubElement(root, "edgeData", attrib={
        "id": "e_collector", "file": str(Path(edge_out).resolve()),
        "begin": str(begin), "end": str(end), "period": str(end - begin),
        "excludeEmpty": "true"})
    ET.indent(ET.ElementTree(root), space="  ")
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def _write_cfg(path, net, routes, addls, seed, begin=0, end=SIM_END, step=1.0) -> None:
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=str(Path(net).resolve()))
    ET.SubElement(inp, "route-files", value=str(Path(routes).resolve()))
    ET.SubElement(inp, "additional-files",
                  value=",".join(str(Path(a).resolve()) for a in addls))
    tm = ET.SubElement(root, "time")
    ET.SubElement(tm, "begin", value=str(begin)); ET.SubElement(tm, "end", value=str(end))
    ET.SubElement(tm, "step-length", value=str(step))
    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "ignore-route-errors", value="true")
    ET.SubElement(proc, "collision.action", value="warn")
    ET.SubElement(proc, "time-to-teleport", value="300")
    rnd = ET.SubElement(root, "random_number"); ET.SubElement(rnd, "seed", value=str(seed))
    rep = ET.SubElement(root, "report")
    ET.SubElement(rep, "no-step-log", value="true"); ET.SubElement(rep, "no-warnings", value="true")
    ET.indent(ET.ElementTree(root), space="  ")
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def _parse_edgedata(xml_path, eid_to_idx, n_edges):
    """Return arrays [N] for traveltime, speed, flow(entered), density."""
    tt = np.zeros(n_edges); spd = np.zeros(n_edges)
    flow = np.zeros(n_edges); den = np.zeros(n_edges)
    if not Path(xml_path).exists():
        return tt, spd, flow, den
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:                       # truncated edgeData (sim killed mid-write)
        return tt, spd, flow, den
    for interval in root.findall("interval"):
        for e in interval.findall("edge"):
            i = eid_to_idx.get(e.get("id", ""))
            if i is None:
                continue
            def g(a):
                v = e.get(a)
                return float(v) if v not in (None, "", "nan") else 0.0
            tt[i], spd[i], flow[i], den[i] = g("traveltime"), g("speed"), g("entered"), g("density")
    return tt, spd, flow, den


def simulate_od(od: np.ndarray, zone_ids, edge_ids, net, taz, workdir,
                seed: int = 7, timeout: int = 360, extra_addl=None, meso: bool = False) -> dict:
    """Simulate one OD; return {ok, n_trips, traveltime, speed, flow, density}[N].

    extra_addl: optional list of additional-file paths (e.g. a rerouter that
    closes edges) layered on top of the edgeData collector — used by what-if.
    meso: run the fast mesoscopic model (queue-based) — much faster at high demand."""
    ensure_sumo_home()
    work = Path(workdir); work.mkdir(parents=True, exist_ok=True)
    trips, routes = work / "trips.xml", work / "routes.xml"
    addl, edge_out, cfg = work / "addl.xml", work / "edge_data.xml", work / "sim.sumocfg"
    sumo_log = work / "sumo.log"

    n_trips = write_trips_xml(od, zone_ids, trips, seed=seed, prefix="cal_")
    eid_to_idx = {e: i for i, e in enumerate(edge_ids)}
    blank = {"ok": False, "n_trips": n_trips,
             "traveltime": np.zeros(len(edge_ids)), "speed": np.zeros(len(edge_ids)),
             "flow": np.zeros(len(edge_ids)), "density": np.zeros(len(edge_ids))}
    if n_trips == 0:
        return blank

    ok, err = run_duarouter(net, taz, trips, routes, seed=seed)
    if not ok:
        blank["err"] = f"duarouter: {err}"; return blank

    _write_additional(addl, edge_out)
    _write_cfg(cfg, net, routes, [addl] + list(extra_addl or []), seed)
    sumo = _bin("sumo")
    cmd = [sumo, "-c", str(cfg)] + (["--mesosim", "--meso-junction-control", "true"] if meso else [])
    try:
        with open(sumo_log, "w") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=lf, timeout=timeout)
    except subprocess.TimeoutExpired:
        blank["err"] = f"sumo TIMEOUT {timeout}s"; return blank
    if r.returncode != 0:
        blank["err"] = f"sumo rc={r.returncode} (see {sumo_log})"; return blank

    tt, spd, flow, den = _parse_edgedata(edge_out, eid_to_idx, len(edge_ids))
    return {"ok": True, "n_trips": n_trips,
            "traveltime": tt, "speed": spd, "flow": flow, "density": den}
