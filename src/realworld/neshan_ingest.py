"""
neshan_ingest.py — Stage B (1/2): load a Neshan traffic export into a normalised
per-segment table the matcher can consume.

Neshan exports vary; this reader is deliberately format-tolerant and
column-mapping is configurable (configs/realworld.yaml -> neshan.columns). It
accepts CSV or GeoJSON and produces records with whatever locator is available
(osm id / street name / a representative lon-lat point) plus a travel time:

    [{osm_id?, name?, lon?, lat?, traveltime_s, speed_mps?, length_m?}]

If only speed is present, travel time is derived as length / speed when a length
is given. Output: data/neshan/neshan_segments.pkl

Run:
    python -m src.realworld.neshan_ingest --in data/neshan/export.csv
    python -m src.realworld.neshan_ingest --in data/neshan/export.geojson
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_yaml, resolve

# Default column-name candidates (overridable in config). First match wins.
DEFAULTS = {
    "osm_id": ["osm_id", "osmid", "way_id", "wayId", "edge_id"],
    "name": ["name", "street", "road", "title"],
    "lon": ["lon", "lng", "longitude", "x"],
    "lat": ["lat", "latitude", "y"],
    "traveltime_s": ["traveltime", "travel_time", "tt", "duration_s", "duration"],
    "speed_mps": ["speed", "speed_mps", "v"],
    "speed_kmh": ["speed_kmh", "kmh", "speed_kph"],
    "length_m": ["length", "length_m", "dist", "distance_m"],
}


def _pick(row: dict, candidates: list[str]):
    for c in candidates:
        if c in row and row[c] not in (None, "", "nan"):
            return row[c]
    return None


def _to_record(row: dict, cols: dict) -> dict | None:
    def g(key):
        return _pick(row, cols.get(key, DEFAULTS[key]))

    rec = {"osm_id": g("osm_id"), "name": g("name")}
    for k in ("lon", "lat"):
        v = g(k)
        rec[k] = float(v) if v is not None else None
    length = g("length_m")
    rec["length_m"] = float(length) if length is not None else None

    tt = g("traveltime_s")
    if tt is not None:
        rec["traveltime_s"] = float(tt)
    else:  # derive from speed if possible
        spd = g("speed_mps")
        spd = float(spd) if spd is not None else (
            float(g("speed_kmh")) / 3.6 if g("speed_kmh") is not None else None)
        if spd and spd > 0 and rec["length_m"]:
            rec["traveltime_s"] = rec["length_m"] / spd
            rec["speed_mps"] = spd
        else:
            return None  # no usable travel time
    if rec["osm_id"] is None and rec["name"] is None and rec["lon"] is None:
        return None      # no locator at all
    return rec


def _read_csv(path: Path) -> list[dict]:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_geojson(path: Path) -> list[dict]:
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties", {}))
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if coords:                                   # representative point = midpoint
            flat = coords
            while isinstance(flat[0], (list, tuple)):
                flat = flat[len(flat) // 2]
            props.setdefault("lon", flat[0]); props.setdefault("lat", flat[1])
        rows.append(props)
    return rows


def main() -> None:
    rcfg = load_yaml("realworld")
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str,
                    default=rcfg["paths"]["neshan_raw"])
    ap.add_argument("--out", type=str, default="data/neshan/neshan_segments.pkl")
    args = ap.parse_args()

    path = resolve(args.inp)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Drop your Neshan export there (CSV or GeoJSON) or "
            f"pass --in PATH. See configs/realworld.yaml -> neshan.columns to map "
            f"your column names. For a no-data dry run use build_observation --demo-total.")
    cols = rcfg.get("neshan", {}).get("columns", {})
    rows = _read_geojson(path) if path.suffix.lower() == ".geojson" else _read_csv(path)
    recs = [r for r in (_to_record(row, cols) for row in rows) if r is not None]
    locator = {"osm_id": sum(r["osm_id"] is not None for r in recs),
               "name": sum(r["name"] is not None for r in recs),
               "lonlat": sum(r["lon"] is not None for r in recs)}
    print(f"[ingest] {path.name}: {len(rows)} rows -> {len(recs)} usable segments")
    print(f"[ingest] locators available: {locator}")

    out = resolve(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(recs, f)
    print(f"[ingest] -> {out}")


if __name__ == "__main__":
    main()
