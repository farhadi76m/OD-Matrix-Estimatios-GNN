# Real-data OD calibration from Neshan travel times (Tehran, District 2)

Branch: `mehdi.neshan-calibration`. This is the **real-world** counterpart to the
synthetic GNN pipeline (`README_pipeline.md`). Goal: take **observed per-segment
travel times** (Neshan) on the OSM-derived SUMO network and produce an OD matrix
that, when re-simulated in SUMO, **reproduces the observed traffic** — so we can
then simulate **what-ifs** (new/closed roads).

## The headline finding (why this is built the way it is)
We first measured, on synthetic data, how much OD signal travel time actually
carries (`src/realworld/diagnose_traveltime.py`). It is the **weakest** OD
observable:

| input signal (ridge, 77% edge coverage) | zone-marginal corr | reconstructed OD cell-corr |
|---|---|---|
| **travel time (what Neshan gives)** | **0.25** | **0.06** |
| speed | 0.23 | 0.05 |
| link flow (counts) | 0.35 | 0.14 |
| junction turn counts (node-agg.) | 0.47 | 0.19 |
| *gravity ceiling from TRUE marginals* | 1.00 | **0.87** |

So **no model — GNN, VAE, or ridge — can recover the OD from travel time alone.**
A learned inverse is therefore used only as a cheap *warm-start*; the real work is
a **SUMO-in-the-loop calibration** that adjusts demand until simulated travel
times match the observation. (Turn counts were the big lever in the synthetic
work; Neshan doesn't provide them — see `od-identifiability-turn-counts`.)

## Pipeline
```
Neshan export ─ingest─▶ segments ─match─▶ per-edge observed travel time + coverage mask
                                              │
   synthetic tt ─ridge─▶ warm-start marginals ▼   (Stage C)
                          └─Furness─▶ OD0  ─scale search (match congestion)─▶ total pinned
                                              │
                          SPSA on spatial shape, SUMO in the loop ◀───────────┘   (Stage D)
                                              │
                          calibrated OD ─re-simulate─▶ validate vs Neshan       (Stage E)
                                              └─edit network─▶ what-if
```

Modules (`src/realworld/`): `diagnose_traveltime` · `neshan_ingest` ·
`match_edges` · `build_observation` · `sim` (headless SUMO forward operator) ·
`calibrate` (warm-start + scale search + SPSA) · `validate` · `whatif`.
All knobs live in `configs/realworld.yaml`. SUMO uses the version-matched
eclipse-sumo 1.27 wheel via `src/sumo_export.py` (the system 1.18 can't parse this net).

## How the calibration is made identifiable
Travel time on a subset of edges does **not** constrain the total demand or the
spatial OD. We break the problem into the parts travel time *can* inform:
1. **Total demand** — pinned by matching the overall **congestion intensity**
   (median `tt/tt_free` over jointly-active edges), with a coverage guard. Robust;
   not biased toward flooding the network.
2. **Spatial shape** — SPSA optimises 2×36 production/attraction multipliers
   (total held fixed) against a **log-space travel-time loss** on observed edges
   (log1p compresses congestion outliers; an optional one-time global bias absorbs
   the sim-vs-probe speed offset — set `remove_global_bias: true` for real data).

## Demo (self-consistent, no real file needed)
`build_observation --demo-total T` plants a known synthetic OD, simulates it to
get the "observed" travel times, and stores the planted OD for scoring. Result for
a planted evening-peak sample (~1906 trips, 398 observed edges):

| | value | note |
|---|---|---|
| recovered total demand | 2322 vs **1906** | rel. err **0.22** — magnitude recovered |
| travel-time match (calibrated) | corr **0.72** | warm-start 0.68; SPSA loss 2.11→1.41 |
| OD cell-corr vs planted | **0.04** | the OD itself is *not* recovered (as Stage A predicts) |

**The honest trade-off:** forcing a tighter per-edge travel-time match (inflating
total to ~5800) reaches corr **0.90** but makes demand 3× too high. Travel time
lets you fit *either* the demand level *or* the per-edge pattern — not recover the
true OD. For trustworthy what-ifs, the demand-correct calibration is preferred.

**What-if** (closing a 3-edge corridor, demand scaled to a realistic ~46k trips
since the synthetic range is low): **104 edges** change travel time — a relieved
edge drops 155 s→33 s while detour edges rise ~+31 s. Classic rerouting; the end
goal works.

## Run it
```bash
conda activate traffic && cd /home/mehdi/Desktop/Model

# Stage A — travel-time identifiability ceiling (synthetic; instant)
python -m src.realworld.diagnose_traveltime

# DEMO end-to-end (no Neshan file required)
python -m src.realworld.build_observation --demo-total 1900
python -m src.realworld.calibrate --reset --max-iters 30 --max-seconds 1200   # resumable
python -m src.realworld.validate
python -m src.realworld.whatif --scale 20 --close-edges <edge1>,<edge2>
python -m src.realworld.whatif --alt-net sumo/with_new_road.net.xml           # add-a-road what-if

# REAL Neshan data
#  1) drop the export at data/neshan/ (CSV or GeoJSON); map columns in
#     configs/realworld.yaml -> neshan.columns if names differ
python -m src.realworld.neshan_ingest --in data/neshan/<export>
python -m src.realworld.match_edges          # osm_id > name > geometry; --self-test to validate
python -m src.realworld.build_observation    # (no --demo-total => uses the matched data)
python -m src.realworld.calibrate --reset --max-seconds 1200
python -m src.realworld.validate
```

## Figures (`docs/realworld/`)
- `traveltime_match.png` — observed vs calibrated-simulated travel times on the
  network + scatter.
- `convergence.png` — SPSA travel-time loss vs evaluation.
- `whatif_delta.png` — per-edge travel-time change from a road closure (red=worse,
  blue=better).

## Honest limitations
- **Travel time under-determines the OD** (cell-corr ceiling 0.06). The calibrated
  OD reproduces observed *conditions*, but is one of many demands that do so;
  what-if predictions are most reliable for moderate, local changes.
- **Sim-vs-probe gap**: SUMO travel times come from its car-following model, Neshan
  from probes — mitigated by matching *pattern* (log space + one-time bias removal),
  not absolute seconds.
- **Coverage**: Neshan covers arterials; calibration/validation are over matched
  edges only (~77% of edges carry a street name here).
- **Synthetic demand is low** (50–2000 trips/hr district-wide) so congestion — and
  thus what-if impact — is small without `--scale`; real District-2 demand is far
  higher.
- **Single period** — one calibrated OD. The 4-regime machinery exists for a
  multi-period extension.

## What would make the OD itself recoverable
Add a stronger observable: **junction turn counts** or **link counts** (then
SUMO's `routeSampler`/`cadyts` apply directly), or a **CVAE** to represent the
*distribution* of ODs consistent with the travel times rather than a single
point. These are the documented next steps.
