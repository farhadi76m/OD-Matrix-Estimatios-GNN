"""
sumo_runner.py – SUMO Subprocess Interface

Full pipeline for one OD matrix sample:

  1.  write_od_xml        → od.xml          (tazRelation demand format)
  2.  od2trips            → trips.xml        (randomised departure times)
  3.  duarouter           → routes.xml       (shortest-path route assignment)
  4.  write_additional    → addl.xml         (edgeData collector config)
  5.  write_sumo_cfg      → sim.sumocfg
  6.  sumo                → edge_data.xml    (per-edge aggregates)
  7.  parse_edge_data     → dict of metrics  (flow, traveltime, speed, …)

Each call works in an isolated temp directory under CFG["work_dir"]
so parallel runs never interfere.
"""

import logging
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# XML writers
# ─────────────────────────────────────────────────────────────────────────────
def write_trips_xml(
    od: np.ndarray,
    zone_ids: List[str],
    path: str,
    seed: int,
    prefix: str = ""
):
    rng = np.random.default_rng(seed)

    root = ET.Element("routes")

    trip_id = 0

    begin = CFG["sim_begin"]
    end = CFG["sim_end"]

    n = len(zone_ids)

    for i in range(n):
        for j in range(n):
            count = int(od[i, j])

            if count <= 0:
                continue

            departs = rng.uniform(begin, end, count)

            for depart in departs:
                ET.SubElement(
                    root,
                    "trip",
                    {
                        "id": f"{prefix}{trip_id}",
                        "fromTaz": zone_ids[i],
                        "toTaz": zone_ids[j],
                        "depart": f"{depart:.2f}",
                    }
                )
                trip_id += 1

    _write_xml(root, path)

def write_od_xml(od: np.ndarray, zone_ids: List[str], path: str) -> None:
    """
    Write OD matrix as SUMO tazRelation XML (od2trips -d input format).

    Only non-zero OD pairs are written to keep file size small.
    One interval covers the full simulation window.
    """
    begin = float(CFG["sim_begin"])
    end   = float(CFG["sim_end"])

    root     = ET.Element("data")
    interval = ET.SubElement(root, "interval",
                             begin=f"{begin:.1f}", end=f"{end:.1f}")

    n = len(zone_ids)
    for i in range(n):
        for j in range(n):
            v = int(od[i, j])
            if v > 0:
                ET.SubElement(interval, "tazRelation", attrib={
                    "from":  zone_ids[i],
                    "to":    zone_ids[j],
                    "count": str(v),
                })

    _write_xml(root, path)


def write_additional_xml(path: str, edge_out_abs: str) -> None:
    """
    Write SUMO additional file that declares the edgeData detector.

    Data is collected from (sim_begin + warm_up) → sim_end so the
    warm-up transient does not pollute the measurements.
    Collection period = full post-warmup window (one aggregate entry).
    """
    begin  = CFG["sim_begin"] + CFG["warm_up"]
    end    = CFG["sim_end"]
    period = end - begin

    root = ET.Element("additional")
    ET.SubElement(root, "edgeData", attrib={
        "id":           "e_collector",
        "file":         edge_out_abs,
        "begin":        str(begin),
        "end":          str(end),
        "period":       str(period),
        "excludeEmpty": "true",
    })
    _write_xml(root, path)


def write_sumo_cfg(
    path:        str,
    net_file:    str,
    routes_file: str,
    addl_file:   str,
    seed:        int,
) -> None:
    """Write SUMO .sumocfg (XML configuration file)."""
    root = ET.Element("configuration")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file",         value=net_file)
    ET.SubElement(inp, "route-files",      value=routes_file)
    ET.SubElement(inp, "additional-files", value=addl_file)

    tm = ET.SubElement(root, "time")
    ET.SubElement(tm, "begin",       value=str(CFG["sim_begin"]))
    ET.SubElement(tm, "end",         value=str(CFG["sim_end"]))
    ET.SubElement(tm, "step-length", value=str(CFG["step_length"]))

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "ignore-route-errors",           value="true")
    ET.SubElement(proc, "collision.action",               value="warn")
    ET.SubElement(proc, "time-to-teleport",               value="300")

    rnd = ET.SubElement(root, "random_number")
    ET.SubElement(rnd, "seed", value=str(seed))

    rep = ET.SubElement(root, "report")
    ET.SubElement(rep, "no-step-log", value="true")
    ET.SubElement(rep, "no-warnings", value="true")
    ET.SubElement(rep, "verbose",     value="false")

    _write_xml(root, path)


def _write_xml(root: ET.Element, path: str) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True
    )
# ─────────────────────────────────────────────────────────────────────────────
# Edge-data output parser
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from SUMO XML attribute → our canonical key
_SUMO_ATTRS = {
    "entered":     "flow",           # vehicles that entered the edge
    "traveltime":  "traveltime",     # mean travel time [s]
    "speed":       "speed",          # mean speed [m/s]
    "density":     "density",        # mean density [veh/km]
    "waitingTime": "waitingTime",    # cumulative waiting time [s]
    "TimeLoss":    "timeLoss",       # cumulative time-loss vs free-flow [s]
}


def parse_edge_data(xml_path: str) -> Dict[str, Dict[str, float]]:
    """
    Parse SUMO edgeData XML output.

    Returns
    -------
    {edge_id: {flow, traveltime, speed, density, waitingTime, timeLoss}}

    Internal edges (id starts with ':') are excluded.
    Returns empty dict on any error.
    """
    if not Path(xml_path).exists():
        return {}

    result: Dict[str, Dict[str, float]] = {}
    try:
        tree = ET.parse(xml_path)
        for interval in tree.getroot().findall("interval"):
            for edge in interval.findall("edge"):
                eid = edge.get("id", "")
                if eid.startswith(":"):   # skip internal junction edges
                    continue
                stats: Dict[str, float] = {}
                for sumo_attr, our_key in _SUMO_ATTRS.items():
                    raw = edge.get(sumo_attr)
                    stats[our_key] = float(raw) if raw not in (None, "nan", "") else 0.0
                result[eid] = stats
    except ET.ParseError as exc:
        log.warning(f"EdgeData XML parse error ({xml_path}): {exc}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SUMO subprocess wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: List[str], timeout: int, log_path: Optional[str] = None) -> tuple[bool, str]:
    """Run a command; return (success, stderr_snippet)."""
    try:
        if log_path:
            with open(log_path, "w") as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=lf, timeout=timeout)
        else:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        ok  = r.returncode == 0
        err = "" if log_path else (r.stderr or "")
        return ok, err[:400]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except FileNotFoundError as exc:
        return False, f"Binary not found: {exc}"


def _od2trips(taz: str, od_xml: str, trips_out: str, seed: int, prefix: str) -> tuple[bool, str]:
    cmd = [
        CFG["od2trips_bin"],
        "--taz-files",       taz,
        "--od-matrix-files", od_xml,
        "--output-file",     trips_out,
        "--seed",            str(seed),
        "--prefix",          prefix,
        "--spread.uniform",             # spread departures uniformly in window
        "--no-warnings",
    ]
    return _run(cmd, timeout=90)


def _duarouter(net: str, taz: str, trips: str, routes_out: str, seed: int) -> tuple[bool, str]:
    cmd = [
        CFG["duarouter_bin"],
        "--net-file",     net,
        "--taz-files",    taz,          # needed when trips reference TAZ ids
        "--route-files",  trips,
        "--output-file",  routes_out,
        "--seed",         str(seed),
        "--ignore-errors", "true",      # skip unroutable trips silently
        "--no-warnings",
        "--no-step-log",
    ]
    return _run(cmd, timeout=180)


def _sumo(cfg_path: str, log_path: str) -> tuple[bool, str]:
    cmd = [CFG["sumo_bin"], "--configuration-file", cfg_path]
    ok, err = _run(cmd, timeout=360, log_path=log_path)
    return ok, err


# ─────────────────────────────────────────────────────────────────────────────
# Single simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    sample_idx: int,
    od:         np.ndarray,
    zone_ids:   List[str],
    net_path:   str,
    taz_path:   str,
    seed:       int,
) -> Optional[Dict]:
    """
    Run the full SUMO pipeline for one OD matrix.

    Parameters
    ----------
    sample_idx : unique sample ID (used for temp dir name + vehicle prefix)
    od         : (n_zones, n_zones) int demand matrix
    zone_ids   : ordered list of TAZ IDs (must match od axis order)
    net_path   : absolute path to .net.xml
    taz_path   : absolute path to tehran_taz.xml
    seed       : RNG seed for this run

    Returns
    -------
    dict with keys: sample_idx, od_matrix, total_demand, edge_data, seed, ok
    None on any unrecoverable failure.
    """
    work = Path(CFG["work_dir"]) / f"s{sample_idx:06d}"
    work.mkdir(parents=True, exist_ok=True)

    p = {k: str(work / v) for k, v in {
        "od_xml":    "od.xml",
        "trips":     "trips.xml",
        "routes":    "routes.xml",
        "addl":      "addl.xml",
        "edge_out":  "edge_data.xml",
        "sumo_cfg":  "sim.sumocfg",
        "sumo_log":  "sumo.log",
    }.items()}

    try:
        # ── 1. OD XML ────────────────────────────────────────────────────────
        # 1. Trips XML
        write_trips_xml(
            od,
            zone_ids,
            p["trips"],
            seed,
            prefix=f"s{sample_idx}_"
        )
        # ── 3. duarouter ─────────────────────────────────────────────────────
        ok, err = _duarouter(net_path, taz_path, p["trips"], p["routes"], seed)
        if not ok or not Path(p["routes"]).exists():
            return _fail(sample_idx, f"duarouter failed: {err}")

        # ── 4–5. Additional config + SUMO cfg ────────────────────────────────
        write_additional_xml(p["addl"], p["edge_out"])
        write_sumo_cfg(p["sumo_cfg"], net_path, p["routes"], p["addl"], seed)

        # ── 6. SUMO ──────────────────────────────────────────────────────────
        ok, err = _sumo(p["sumo_cfg"], p["sumo_log"])
        if not ok:
            return _fail(sample_idx, f"sumo failed: {err}")

        # ── 7. Parse results ─────────────────────────────────────────────────
        edge_data = parse_edge_data(p["edge_out"])
        if not edge_data:
            return _fail(sample_idx, "Edge data output is empty")

        # Sanity: at least 5% of edges should have non-zero flow
        total_flow = sum(v["flow"] for v in edge_data.values())
        if total_flow < 1.0:
            return _fail(sample_idx, "All edges show zero flow – routing failure?")

    except Exception as exc:
        log.error(f"[{sample_idx:06d}] Unhandled exception: {exc}", exc_info=True)
        shutil.rmtree(work, ignore_errors=True)
        return None

    # Cleanup to keep disk usage bounded
    shutil.rmtree(work, ignore_errors=True)

    return {
        "sample_idx":   sample_idx,
        "od_matrix":    od.astype(np.int32),
        "total_demand": int(od.sum()),
        "edge_data":    edge_data,
        "seed":         seed,
        "ok":           True,
    }


def _fail(idx: int, reason: str) -> None:
    log.warning(f"[{idx:06d}] {reason}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Network / TAZ introspection
# ─────────────────────────────────────────────────────────────────────────────

def parse_taz_ids(taz_file: str) -> List[str]:
    """
    Extract TAZ IDs from tehran_taz.xml.
    Handles both <tazs> and <additional> root elements.
    """
    tree = ET.parse(taz_file)
    root = tree.getroot()
    # Root may be <tazs> or <additional>
    ids  = [t.get("id") for t in root.findall("taz") if t.get("id")]
    if not ids:
        raise ValueError(f"No <taz> elements found in {taz_file}")
    return sorted(ids)


def parse_edge_ids(net_file: str) -> List[str]:
    """
    Extract all non-internal edge IDs from a SUMO net file.
    Internal edges start with ':' and are skipped.
    """
    tree = ET.parse(net_file)
    ids  = [
        e.get("id") for e in tree.getroot().findall("edge")
        if not e.get("id", "").startswith(":")
    ]
    if not ids:
        raise ValueError(f"No edges found in {net_file}")
    return sorted(ids)