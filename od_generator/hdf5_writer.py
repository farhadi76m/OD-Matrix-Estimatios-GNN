"""
hdf5_writer.py – Incremental HDF5 Dataset Writer

Writes simulation results batch-by-batch into a single HDF5 file
with chunked, compressed, resizable datasets.

HDF5 Schema
───────────
/od_matrices         float32  [N, Z, Z]   raw OD demand (trips)
/edge_flow           float32  [N, E]       vehicles entering edge per sim-period
/edge_traveltime     float32  [N, E]       mean travel time [s]
/edge_speed          float32  [N, E]       mean speed [m/s]
/edge_density        float32  [N, E]       density [veh/km]
/edge_waitingTime    float32  [N, E]       cumulative waiting time [s]
/edge_timeLoss       float32  [N, E]       cumulative time-loss vs free-flow [s]
/total_demand        int32    [N]          total trips for each sample
/sample_seeds        int64    [N]          RNG seed used in that run
/edge_ids            bytes    [E]          edge ID strings
/zone_ids            bytes    [Z]          zone ID strings

N  = samples written so far (resized on every batch write)
Z  = number of TAZ zones  (36 for Tehran district)
E  = number of edges in network

Usage
-----
    writer = ODDatasetWriter("od_dataset.h5", n_zones, zone_ids, edge_ids)
    writer.write_batch(results)       # call after each batch
    print(writer.n_written)
"""

import logging
from typing import Dict, List, Optional

import h5py
import numpy as np

log = logging.getLogger(__name__)

# Canonical field names returned by sumo_runner.parse_edge_data()
_EDGE_FIELDS = [
    "flow",
    "traveltime",
    "speed",
    "density",
    "waitingTime",
    "timeLoss",
]


class ODDatasetWriter:
    """
    Writes one batch at a time; safe to interrupt and resume is NOT
    supported (open the file fresh each run).  If you want resumability,
    wrap write_batch in a try/finally that closes the file on crash.
    """

    def __init__(
        self,
        path:      str,
        n_zones:   int,
        zone_ids:  List[str],
        edge_ids:  List[str],
    ) -> None:
        self.path    = path
        self.n_zones = n_zones
        self.n_edges = len(edge_ids)
        self.zone_ids = zone_ids
        self.edge_ids = edge_ids
        self._eid_idx = {e: i for i, e in enumerate(edge_ids)}

        # Chunk sizes – balance random-access vs compression efficiency
        C_sample = 128       # samples per chunk
        C_edge   = min(512, self.n_edges)

        with h5py.File(path, "w") as f:
            f.attrs["n_zones"] = n_zones
            f.attrs["n_edges"] = self.n_edges
            f.attrs["created"] = _now_str()

            # ID arrays (fixed size, written once)
            f.create_dataset("zone_ids",
                             data=np.array(zone_ids, dtype="S64"))
            f.create_dataset("edge_ids",
                             data=np.array(edge_ids, dtype="S64"))

            # OD matrices – int32 (no fractional demand)
            f.create_dataset(
                "od_matrices",
                shape=(0, n_zones, n_zones),
                maxshape=(None, n_zones, n_zones),
                dtype="int32",
                chunks=(C_sample, n_zones, n_zones),
                compression="gzip",
                compression_opts=4,
            )

            # Per-edge metrics
            for field in _EDGE_FIELDS:
                f.create_dataset(
                    f"edge_{field}",
                    shape=(0, self.n_edges),
                    maxshape=(None, self.n_edges),
                    dtype="float32",
                    chunks=(C_sample, C_edge),
                    compression="gzip",
                    compression_opts=4,
                )

            # Scalar per-sample metadata
            f.create_dataset("total_demand",
                             shape=(0,), maxshape=(None,),
                             dtype="int32", chunks=(1024,))
            f.create_dataset("sample_seeds",
                             shape=(0,), maxshape=(None,),
                             dtype="int64", chunks=(1024,))

        log.info(
            f"HDF5 created: {path}  |  zones={n_zones}  |  edges={self.n_edges}"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def write_batch(self, results: List[Optional[Dict]]) -> int:
        """
        Append valid simulation results to the HDF5 file.

        Parameters
        ----------
        results : list of dicts returned by run_simulation(); None entries
                  (failed runs) are silently skipped.

        Returns
        -------
        Number of rows actually written.
        """
        valid = [r for r in results if r is not None and r.get("ok")]
        if not valid:
            return 0

        B    = len(valid)
        n, e = self.n_zones, self.n_edges

        # ── Assemble batch tensors ───────────────────────────────────────────
        od_buf   = np.stack([r["od_matrix"] for r in valid], 0).astype(np.int32)
        dem_buf  = np.array([r["total_demand"] for r in valid], dtype=np.int32)
        seed_buf = np.array([r["seed"]         for r in valid], dtype=np.int64)

        field_bufs: Dict[str, np.ndarray] = {
            f: np.zeros((B, e), dtype=np.float32) for f in _EDGE_FIELDS
        }

        for b, r in enumerate(valid):
            for eid, stats in r["edge_data"].items():
                idx = self._eid_idx.get(eid)
                if idx is not None:
                    for field in _EDGE_FIELDS:
                        field_bufs[field][b, idx] = float(stats.get(field, 0.0))

        # ── Append to HDF5 ───────────────────────────────────────────────────
        with h5py.File(self.path, "a") as f:
            n0 = int(f["od_matrices"].shape[0])
            n1 = n0 + B

            f["od_matrices"].resize(n1, axis=0)
            f["od_matrices"][n0:n1] = od_buf

            for field in _EDGE_FIELDS:
                ds = f[f"edge_{field}"]
                ds.resize(n1, axis=0)
                ds[n0:n1] = field_bufs[field]

            for name, buf in [("total_demand", dem_buf),
                               ("sample_seeds", seed_buf)]:
                ds = f[name]
                ds.resize(n1, axis=0)
                ds[n0:n1] = buf

        log.debug(f"Batch written: {B} samples  |  total in file: {n1}")
        return B

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def n_written(self) -> int:
        """Return current number of samples stored in the file."""
        with h5py.File(self.path, "r") as f:
            return int(f["od_matrices"].shape[0])


# ─────────────────────────────────────────────────────────────────────────────

def _now_str() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"