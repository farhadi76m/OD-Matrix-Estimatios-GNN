"""
pipeline.py – Main Dataset Generation Pipeline

Entry point.  Reads CFG, pre-generates all OD matrices for
reproducibility, then runs SUMO simulations in parallel batches
and writes results incrementally to HDF5.

Usage
-----
    python pipeline.py                    # uses config.py defaults
    python pipeline.py --workers 8        # override worker count
    python pipeline.py --samples 500      # quick test run
    python pipeline.py --dry-run          # validate setup, skip SUMO
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import CFG
from hdf5_writer import ODDatasetWriter
from od_generator import generate_od_matrix, sample_demand
from sumo_runner import parse_edge_ids, parse_taz_ids, run_simulation

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup (file + console)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CFG["log_file"], mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate OD→Traffic SUMO dataset")
    p.add_argument("--workers",  type=int,  default=CFG["n_workers"],
                   help="Parallel SUMO workers")
    p.add_argument("--samples",  type=int,  default=CFG["n_samples"],
                   help="Override number of samples")
    p.add_argument("--batch",    type=int,  default=CFG["batch_size"],
                   help="HDF5 write frequency (samples)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Parse network and generate OD matrices; skip SUMO")
    p.add_argument("--net",      type=str,  default=CFG["sumo_net"])
    p.add_argument("--taz",      type=str,  default=CFG["taz_file"])
    p.add_argument("--output",   type=str,  default=CFG["output_hdf5"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_paths(*paths: str) -> None:
    for p in paths:
        if not Path(p).exists():
            log.error(f"Required file not found: {p}")
            sys.exit(1)


def save_metadata(path: str, zone_ids, edge_ids, n: int) -> None:
    meta = {
        "n_samples":       n,
        "n_zones":         len(zone_ids),
        "n_edges":         len(edge_ids),
        "zone_ids":        zone_ids,
        "demand_range":    [CFG["min_trips"], CFG["max_trips"]],
        "demand_strategy": CFG["demand_strategy"],
        "od_strategy":     CFG["od_strategy"],
        "mix_ratios":      CFG.get("mix_ratios", {}),
        "sim_window_s":    [CFG["sim_begin"], CFG["sim_end"]],
        "warm_up_s":       CFG["warm_up"],
        "master_seed":     CFG["master_seed"],
        "edge_fields": [
            "flow", "traveltime", "speed",
            "density", "waitingTime", "timeLoss",
        ],
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    n_samples  = args.samples
    n_workers  = args.workers
    batch_size = args.batch
    net_path   = str(Path(args.net).resolve())
    taz_path   = str(Path(args.taz).resolve())
    out_path   = args.output
    dry_run    = args.dry_run

    # ── Validate ─────────────────────────────────────────────────────────────
    validate_paths(net_path, taz_path)
    Path(CFG["work_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Parse network ─────────────────────────────────────────────────────────
    log.info("Parsing network and TAZ files …")
    zone_ids = parse_taz_ids(taz_path)
    edge_ids = parse_edge_ids(net_path)
    n_zones  = len(zone_ids)
    n_edges  = len(edge_ids)

    log.info(f"  Zones : {n_zones}  (expected {CFG['n_zones']})")
    log.info(f"  Edges : {n_edges}")

    if n_zones != CFG["n_zones"]:
        log.warning(
            f"Zone count mismatch: got {n_zones}, CFG says {CFG['n_zones']}. "
            "Proceeding with actual count."
        )

    # ── Pre-generate all OD matrices for reproducibility ─────────────────────
    log.info(f"Pre-generating {n_samples} OD matrices (seed={CFG['master_seed']}) …")
    t0  = time.time()
    rng = np.random.default_rng(CFG["master_seed"])

    od_list     : list[np.ndarray] = []
    demand_list : list[int]        = []
    seed_list   : list[int]        = []

    for _ in range(n_samples):
        demand = sample_demand(rng)
        od     = generate_od_matrix(rng, n_zones, demand)
        seed   = int(rng.integers(1, 2**31 - 1))
        od_list.append(od)
        demand_list.append(demand)
        seed_list.append(seed)

    log.info(f"  Generated in {time.time() - t0:.1f}s  |  "
             f"demand: min={min(demand_list)}  mean={sum(demand_list)//n_samples}  "
             f"max={max(demand_list)}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    save_metadata(CFG["meta_json"], zone_ids, edge_ids, n_samples)

    if dry_run:
        log.info("Dry-run mode – stopping before SUMO.  OD matrices look good.")
        return
    # ── Initialise HDF5 ───────────────────────────────────────────────────────
    writer = ODDatasetWriter(out_path, n_zones, zone_ids, edge_ids)

    # ── Run simulations ───────────────────────────────────────────────────────
    n_ok   = 0
    n_fail = 0
    t_sim  = time.time()

    log.info(
        f"Starting {n_samples} simulations  "
        f"(workers={n_workers}, batch={batch_size}) …"
    )

    with tqdm(total=n_samples, unit="sim", ncols=90,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for b_start in range(0, n_samples, batch_size):
            b_end = min(b_start + batch_size, n_samples)
            b_len = b_end - b_start

            # Build argument list for this batch
            batch_args = [
                (i, od_list[i], zone_ids, net_path, taz_path, seed_list[i])
                for i in range(b_start, b_end)
            ]

            # Run batch in parallel
            results = [None] * b_len

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                fut_map = {
                    pool.submit(run_simulation, *args): pos
                    for pos, args in enumerate(batch_args)
                }
                for fut in as_completed(fut_map):
                    pos = fut_map[fut]
                    try:
                        results[pos] = fut.result()
                    except Exception as exc:
                        log.error(
                            f"Future exception (sample {b_start + pos}): {exc}"
                        )
                    pbar.update(1)

            # Write batch to HDF5
            written   = writer.write_batch(results)
            n_ok     += written
            n_fail   += b_len - written

            pbar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_sim
    log.info("─" * 65)
    log.info(f"  ✓ Complete.  {n_ok}/{n_samples} successful  |  "
             f"{n_fail} failed  |  {elapsed/3600:.1f}h elapsed")
    log.info(f"  Dataset   → {out_path}")
    log.info(f"  Metadata  → {CFG['meta_json']}")
    log.info(f"  Schema    :")
    log.info(f"    od_matrices  [{n_ok}, {n_zones}, {n_zones}]  int32")
    log.info(f"    edge_flow    [{n_ok}, {n_edges}]  float32")
    log.info(f"    (+ traveltime, speed, density, waitingTime, timeLoss)")
    log.info("─" * 65)


if __name__ == "__main__":
    main()