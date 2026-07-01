"""
taz.py — Step 1: generate the TAZ (zone) file from the SUMO network.

Wraps SUMO's gridDistricts tool at the configured grid width (configs/odpipe.yaml
-> taz.width). The number of zones falls out of the width and the network extent.

Run:
    python -m src.odpipe.taz                 # width from config
    python -m src.odpipe.taz --width 2000    # override
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_yaml, resolve
from src.odpipe import CFG
from src.sumo_export import ensure_sumo_home, _wheel_home


def gridDistricts_path() -> Path:
    """Locate SUMO's gridDistricts.py (prefer the version-matched wheel)."""
    ensure_sumo_home()
    for base in (_wheel_home(), __import__("os").environ.get("SUMO_HOME")):
        if base:
            p = Path(base) / "tools" / "district" / "gridDistricts.py"
            if p.exists():
                return p
    raise FileNotFoundError("gridDistricts.py not found under SUMO_HOME/tools/district")


def main():
    cfg = load_yaml(CFG)
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=cfg["taz"]["width"])
    ap.add_argument("--out", type=str, default=cfg["paths"]["taz_file"])
    args = ap.parse_args()

    net = resolve(cfg["paths"]["net_file"]); out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tool = gridDistricts_path()
    cmd = [sys.executable, str(tool), "-n", str(net), "-o", str(out), "-w", str(args.width)]
    print(f"[taz] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"gridDistricts failed:\n{r.stderr[-800:]}")

    tazs = ET.parse(out).getroot().findall("taz")
    sizes = sorted(len((t.get("edges") or "").split()) for t in tazs)
    print(f"[taz] width={args.width}m -> {len(tazs)} zones  edge-counts={sizes}")
    tiny = [s for s in sizes if s < 5]
    if tiny:
        print(f"[taz] NOTE: {len(tiny)} tiny zone(s) with <5 edges {tiny} — their marginals "
              f"will be noisy (little traffic). Increase width for fewer, larger zones.")
    print(f"[taz] -> {out}")


if __name__ == "__main__":
    main()
