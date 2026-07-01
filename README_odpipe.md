# odpipe — general OD-estimation pipeline (any zone count / demand range)

A clean, config-driven rebuild of the forward+inverse pipeline. Change two knobs
in [configs/odpipe.yaml](configs/odpipe.yaml) — `taz.width` and `demand_range` —
and re-run; **nothing is hard-coded to a zone count** (it is read from the
generated TAZ file everywhere). Everything lives in `src/odpipe/`.

```
TAZ (gridDistricts) → line-graph → OD dataset (SUMO, mesoscopic) → marginal GNN
      → gravity/Furness OD → evaluate → inference (+sumo-gui) → visualize
```

## Run it step by step
```bash
conda activate traffic && cd /home/mehdi/Desktop/Model

# 1. TAZ — generate the zones at the configured grid width (2000 m → 13 zones)
python -m src.odpipe.taz                 # writes sumo/taz_od.xml  (--width to override)
python -m src.odpipe.graph               # cache the road line-graph

# 2. OD dataset — demand 2000–30000 trips/hr, simulated MESOSCOPIC (fast) with turn counts
python -m src.odpipe.generate --workers 8            # all n_samples (resumable, chunkable)
#   or in chunks:  --start 0 --end 500   /   --start 500 --end 1000  ...
python -m src.odpipe.build_packs                     # → data_od/pack_*.pt + norm.json

# 3. TRAIN / EVAL
python -m src.odpipe.train_eval --mode train --device cuda --max-seconds 25   # resumable chunks
python -m src.odpipe.train_eval --mode eval  --device cpu                     # metrics + GEH
#   (no ~45 s process watchdog?  just: --mode all --device cpu)

# 4. TEST one sample — predicted vs true OD, optional sumo-gui for BASE and PRED
python -m src.odpipe.infer --max-trips                              # busiest test sample
python -m src.odpipe.infer --index 42 --sumo-gui --which both --meso  # animate base + predicted
python -m src.odpipe.infer --regime gridlock --sumo-gui --which pred --meso

# 5. VISUALIZE (6 figures → data_od/viz/)
python -m src.odpipe.visualize
```

## What each step produces
| step | output |
|---|---|
| `taz` | `sumo/taz_od.xml` (+ reports zone count & sizes) |
| `graph` | `data_od/graph.pkl` (edge line-graph + zone map) |
| `generate` | `data_od/samples/*.pkl` + `data_od/index.json` (stratified split) |
| `build_packs` | `data_od/pack_{train,val,test}.pt` + `norm.json` |
| `train_eval` | `data_od/best.pt`, `metrics.json`, `test_predictions.npz`, `history.json` |
| `infer` | `data_od/infer/sample_<idx>.png` (+ `sumo_<idx>_{base,pred}/` for the GUI) |
| `visualize` | `data_od/viz/` — zones_map, training_curve, marginals/cell scatter, od_examples, metrics_summary |

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
