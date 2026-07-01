"""
sumo_export.py — turn a (predicted) OD matrix into SUMO inputs and, optionally,
launch sumo-gui to watch the resulting traffic.

Pipeline mirrors the forward generator:
    OD -> trips.xml (fromTaz/toTaz) -> duarouter -> routes.xml -> sim.sumocfg -> sumo-gui

SUMO 1.18 is installed at /usr/bin; SUMO_HOME is set here if unset.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import numpy as np

SUMO_HOME_DEFAULT = "/usr/share/sumo"


def _wheel_home() -> str | None:
    """SUMO_HOME of the eclipse-sumo pip wheel (1.27), if installed."""
    try:
        import sumo
        sh = Path(sumo.__file__).resolve().parent
        return str(sh) if (sh / "bin").exists() else None
    except Exception:
        return None


def ensure_sumo_home() -> str | None:
    """ALWAYS prefer the eclipse-sumo wheel (1.27, matches the net) over any
    pre-set/system SUMO_HOME (e.g. /usr/share/sumo = 1.18, which can't parse
    this net's vehicle classes)."""
    wheel = _wheel_home()
    if wheel:
        os.environ["SUMO_HOME"] = wheel
    elif not os.environ.get("SUMO_HOME") and Path(SUMO_HOME_DEFAULT).exists():
        os.environ["SUMO_HOME"] = SUMO_HOME_DEFAULT
    return os.environ.get("SUMO_HOME")


def _bin(name: str) -> str | None:
    """Resolve a SUMO binary, preferring the version-matched eclipse-sumo wheel."""
    ensure_sumo_home()
    wheel = _wheel_home()
    if wheel and (Path(wheel) / "bin" / name).exists():
        return str(Path(wheel) / "bin" / name)
    try:
        import sumolib
        return sumolib.checkBinary(name)
    except Exception:
        return shutil.which(name)


def _write_xml(root: ET.Element, path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


def write_trips_xml(od: np.ndarray, zone_ids: List[str], path,
                    seed: int = 0, begin: int = 0, end: int = 3600,
                    prefix: str = "pred_") -> int:
    """Write OD as SUMO TAZ trips with uniformly-spread departures.

    Uses *stochastic* rounding (floor + Bernoulli on the fractional part) so a
    fractional predicted OD whose cells are all < 0.5 still yields trips and the
    expected total is preserved (plain rounding would zero everything out).
    """
    rng = np.random.default_rng(seed)
    od = np.clip(od, 0, None)
    odi = (np.floor(od) + (rng.random(od.shape) < (od - np.floor(od)))).astype(int)
    np.fill_diagonal(odi, 0)
    n = len(zone_ids)

    # collect every trip then sort GLOBALLY by departure time. SUMO loads route
    # files as a stream and silently drops any vehicle that departs earlier than
    # the previous one ("Route file should be sorted ... ignoring"); writing
    # per-OD-pair would interleave departures and lose most of the demand.
    trips = []
    for i in range(n):
        for j in range(n):
            for dep in rng.uniform(begin, end, int(odi[i, j])):
                trips.append((float(dep), zone_ids[i], zone_ids[j]))
    trips.sort(key=lambda t: t[0])

    root = ET.Element("routes")
    for tid, (dep, frm, to) in enumerate(trips):
        ET.SubElement(root, "trip", {
            "id": f"{prefix}{tid}", "depart": f"{dep:.2f}", "fromTaz": frm, "toTaz": to})
    _write_xml(root, path)
    return len(trips)


def write_sumocfg(path, net, routes, begin: int = 0, end: int = 3600, step: float = 1.0) -> None:
    # absolute paths so SUMO resolves them regardless of the cfg's location
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=str(Path(net).resolve()))
    ET.SubElement(inp, "route-files", value=str(Path(routes).resolve()))
    tm = ET.SubElement(root, "time")
    ET.SubElement(tm, "begin", value=str(begin))
    ET.SubElement(tm, "end", value=str(end))
    ET.SubElement(tm, "step-length", value=str(step))
    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "ignore-route-errors", value="true")
    ET.SubElement(proc, "time-to-teleport", value="300")
    _write_xml(root, path)


def run_duarouter(net, taz, trips, routes, seed: int = 0) -> tuple[bool, str]:
    duarouter = _bin("duarouter")
    if not duarouter:
        return False, "duarouter not found"
    cmd = [duarouter, "--net-file", str(net), "--taz-files", str(taz),
           "--route-files", str(trips), "--output-file", str(routes),
           "--seed", str(seed), "--ignore-errors", "true",
           "--no-warnings", "true", "--no-step-log", "true"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and Path(routes).exists(), (r.stderr or "")[:600]


def launch_sumo_gui(cfg, delay: int = 80, start: bool = True, block: bool = False,
                    meso: bool = False) -> tuple[bool, str]:
    gui = _bin("sumo-gui")
    if not gui:
        return False, "sumo-gui not found"
    cmd = [gui, "-c", str(cfg), "--delay", str(delay)]
    if meso:
        cmd += ["--mesosim", "--meso-junction-control", "true"]
    if start:
        cmd.append("--start")
    if block:
        subprocess.run(cmd)
    else:
        # detach + drop the inherited stdout/stderr pipes so the GUI survives
        # this process exiting and never holds the parent's pipe open
        subprocess.Popen(cmd, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True, " ".join(cmd)
