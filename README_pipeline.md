# OD Matrix Estimation — GNN Pipeline (Tehran, District 2)

Inverse pipeline: **SUMO edge measurements → GNN → 36×36 OD matrix**.
The forward generator lives in `od_generator/` (unchanged).

## Environment
```bash
conda activate traffic       # Python 3.11, torch 2.6 (+cu124), torch_geometric 2.8
# one-time extra deps already installed: h5py pyyaml scipy scikit-learn tqdm pandas sumolib
```
CUDA is auto-selected if present, else CPU.

## Resolved data schema (verified against the files)
- Source: `od_dataset.h5` — **9997 samples** (3 SUMO runs failed of 10k).
- Target `od_matrices` `[9997, 36, 36]` int32, zero diagonal (1260 off-diag cells, ~84% zero).
- Edge measurements `[9997, 1897]` f32 each: `edge_flow` (SUMO `entered` count over the
  3300 s post-warmup window — **not** veh/h), `edge_speed` (m/s), `edge_density` (veh/km),
  `edge_traveltime` (s). **`edge_timeLoss` is all-zero and `edge_waitingTime` ~all-zero —
  excluded** (forward-pipeline attr-name bug: `sumo_runner.py` maps `"TimeLoss"` but SUMO
  emits `timeLoss`).
- 36 TAZs, 1897 edges; **every edge belongs to exactly one TAZ** (clean partition) →
  used for zone pooling.

## Known divergences from CLAUDE.md (flagged, not silently worked around)
1. **No clock time-of-day in the data.** The generator only varies total demand magnitude
   (50–2000 trips) across 4 balanced log tiers. We relabel those tiers as regimes
   (night→noon→morning_peak→evening_peak, free-flow→congested) and stratify splits by them.
2. **SUMO version mismatch** → system `/usr/bin/sumo` is 1.18 but the net is from netconvert
   1.26; install matching binaries with `pip install eclipse-sumo==1.27.0` (used by
   `src/sumo_export.py` via `sumolib.checkBinary`). GEH still uses a ridge linear-assignment
   surrogate (re-simulation not wired).
3. HDF5 → converted to per-sample `.pkl` + `index.json` manifest (the `.h5` stays read-only).

## Run order (exact commands)
```bash
conda activate traffic
cd /home/mehdi/Desktop/Model

# 1. Build the static road line-graph (cached to data/graph.pkl)
python -m src.data.graph

# 2. Convert HDF5 -> per-sample .pkl + manifest + train-only norm stats
python -m src.data.convert_h5_to_pkl            # smoke subset (1000, from configs/data.yaml)
python -m src.data.convert_h5_to_pkl --limit 0  # full 9997 samples

# 3. Train (CUDA auto). Checkpoints -> checkpoints/best.pt, last.pt
python -m src.train                  # full config (60 epochs, early stop)
python -m src.train --epochs 15      # quick run

# 4. Evaluate on test split (OD metrics per regime + GEH)
python -m src.evaluate --save-preds  # -> data/eval_report.json, data/test_predictions.npz

# 5. Inference on one test sample: estimate OD, visualise, optionally sumo-gui
python -m src.infer --random                       # -> data/infer/sample_<idx>.png + .npy/.json
python -m src.infer --regime evening_peak
python -m src.infer --index 839 --sumo-gui         # animate predicted OD in sumo-gui
```

## Layout
```
configs/            data.yaml · model.yaml · train.yaml   (all hyperparams/paths)
src/
  config.py         YAML loader + seed/device helpers
  data/graph.py     static line-graph builder (edge-node graph + TAZ map)
  data/convert_h5_to_pkl.py   HDF5 -> .pkl + index.json + norm_stats.json
  data/dataset.py   PyG dataset/dataloaders (train-only normalisation)
  models/gnn.py     ODLineGraphGNN: encode → message-pass → zone-pool → pairwise OD head
  metrics.py        MAE/RMSE/total-flow error + GEH
  train.py · evaluate.py
data/               samples/*.pkl · index.json · graph.pkl · norm_stats.json
checkpoints/        best.pt · last.pt · train_full.log
```

## Model
Edge-as-node line graph (1897 nodes, 3052 downstream links + reverse). Node features =
normalised `[length, num_lanes, speed_limit, flow, speed, density, traveltime]` + zone
embedding. K residual SAGE layers → mean/max/sum pool into 36 zone vectors (LayerNorm) →
pairwise MLP over `(z_i, z_j, z_i·z_j, |z_i−z_j|, global)` → `log1p(trips)`, diagonal forced 0.
Loss = SmoothL1 on log1p(OD) + 0.1·total-trips term. ~449k params.
