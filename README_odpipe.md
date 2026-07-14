# odpipe — general OD-estimation pipeline (any zone count / demand range)

A clean, config-driven rebuild of the forward+inverse pipeline. Change two knobs
in [configs/odpipe.yaml](configs/odpipe.yaml) — `taz.width` and `demand_range` —
and re-run; **nothing is hard-coded to a zone count** (it is read from the
generated TAZ file everywhere). Everything lives in `src/odpipe/`.

```
TAZ (gridDistricts) → line-graph → OD dataset (SUMO, mesoscopic) → marginal GNN
      → gravity/Furness OD → evaluate → inference (+sumo-gui) → visualize
```

## Run it — four commands
```bash
conda activate traffic && cd /home/mehdi/Desktop/Model

# 1. DATASET  — TAZ → graph → simulate OD samples (.pkl) → packs.  All in one.
python -m src.odpipe.dataset
#   --samples 300        override n_samples          --width 3000   override TAZ grid
#   --start 0 --end 500  simulate a chunk (resumable, safe to re-run)
#   --packs-only         just rebuild manifest + packs

# 2. TRAIN + EVAL
python -m src.odpipe.train
#   --mode train --max-seconds 25   resumable chunks (if long processes get killed)
#   --mode eval                     metrics only

# 3. TEST one sample — predicted vs true OD, optional sumo-gui for BASE and PRED
python -m src.odpipe.infer --max-trips                                # busiest test sample
python -m src.odpipe.infer --index 42 --sumo-gui --which both --meso  # animate base + predicted

# 4. VISUALIZE (6 figures → data_od/viz/)
python -m src.odpipe.visualize
```

## What each command produces
| command | output |
|---|---|
| `dataset` | `sumo/taz_od.xml` · `data_od/graph.pkl` · `data_od/samples/*.pkl` · `index.json` · `pack_{train,val,test}.pt` · `norm.json` |
| `train` | `data_od/best.pt` · `metrics.json` · `test_predictions.npz` · `history.json` |
| `infer` | `data_od/infer/sample_<idx>.png` (+ `sumo_<idx>_{base,pred}/` for the GUI) |
| `visualize` | `data_od/viz/` — zones_map, training_curve, marginals/cell scatter, od_examples, metrics_summary |

Each `.pkl` sample holds `{idx, od[Z,Z], seed, x_dyn[N,4], turn[L], tier, tod, total_demand}`
— the OD target plus per-edge `[flow, speed, density, traveltime]` and junction turn counts.

## Change the resolution or demand
Everything follows [configs/odpipe.yaml](configs/odpipe.yaml):
```yaml
taz:  {width: 2000}          # 4000→6 zones · 3000→9 · 2000→13 · 1000→36
demand_range: [2000, 30000]  # trips/hour spanned by the 4 demand tiers
n_samples: 1500              # dataset size (more zones → use more)
mesoscopic: true             # fast queue model — keep on for high demand
```
Then just re-run from step 1 (delete/rename `data_od/` first, since packs are
tied to the zone count).

## Notes & honest caveats
- **Mesoscopic is essential here.** At 2000–30000 trips/hr microsimulation is slow
  and gridlocks; the meso model runs ~2 s/sample so the full 1500-sample set is
  ~25 min on 8 workers. `--meso` also drives the GUI and the eval GEH re-sim.
- **Width 2000 → 13 zones, 3 of them tiny** (2–3 edges). Those zones carry almost
  no traffic, so their marginals are noisy and show as near-empty rows/cols in the
  OD (visible in `od_examples.png`). Use a larger width for fewer, balanced zones.
- **Trip files are globally sorted by departure** (`write_trips_xml`) so SUMO no
  longer drops out-of-order vehicles — the full demand actually loads.
- **The numbers you first get are smoke-test numbers.** A 160-sample shakedown
  gives cell-corr ~0.16; scale `n_samples` to 1500+ for representative results
  (the 9-zone study reached 0.66 at 2000 samples). Same recipe, same ceiling logic.
- Reuses the shared model (`src/models/gnn.py` ODMarginalGNN), gravity
  reconstruction (`src/od_reconstruct.py`) and SUMO layer (`src/sumo_export.py`,
  `src/realworld/sim.py`); only the data/config differ.
