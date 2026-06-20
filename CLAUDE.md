# CLAUDE.md — OD Matrix Estimation (Tehran, District 2)

## Project goal
Build and train a **graph neural network that estimates Origin–Destination (OD)
matrices from SUMO traffic measurements**. This is the *inverse* of the
data-generation pipeline that already lives in `od_generator/`:

```
random OD matrix  →  SUMO simulation  →  edge-level traffic data     (forward — already built)
edge-level traffic data  →  GNN  →  estimated OD matrix             (inverse — what we are building)
```

Study area: **Tehran, District 2**. The training set must cover the full daily
demand range — **morning peak, noon, evening peak, and night** — so the model
generalises across free-flow and congested regimes, not just one time of day.

## Always do this first
1. **Read `od_generator/` before writing any model code.** Map out and report:
   - how OD matrices are sampled — confirm the random/Poisson assumption and how
     demand is scaled per time-of-day,
   - how SUMO is invoked and which files matter (`.sumocfg`, network, TAZ/zone
     definitions, route files),
   - exactly which traffic measurements are written (edge flow, mean speed,
     density, occupancy, …), in what units, and what their array shapes are.
2. **Summarise the current data schema back to me and flag any mismatches**
   before changing anything. Do not assume — verify against the actual files.

## Data conventions
- **Stop using HDF5.** Persist samples as **`.pkl`** (arrays / tensors / graph
  objects) or **`.json`** (metadata, configs, small records). One sample =
  one `(traffic_features, od_matrix)` pair plus metadata
  (time-of-day label, RNG seed, OD totals).
- Keep a **manifest** (`index.json`) listing every sample, its split, and its
  scenario label, so the dataset is browsable without loading everything.
- **Reproducibility:** every generated sample stores the seed used to create it.

## ML task definition
- **Input (X):** SUMO traffic measurements on District-2 edges, arranged as
  features on the road graph.
- **Output (y):** the OD matrix (origin × destination over District-2 TAZs).
- Build a graph of the network (nodes = junctions/TAZs, edges = road links) and
  attach measurements as node/edge features.
- Provide a clean **train / val / test split with no scenario leakage**;
  **stratify by time-of-day** so every split contains all four regimes.
- Report regression-appropriate metrics: per-cell MAE/RMSE, total-flow error,
  and **GEH** on the implied link flows (re-simulate or use the BPR mapping).

## Config & code style
- **All hyperparameters and paths go in YAML** (`configs/*.yaml`) — no magic
  numbers in source. One small config loader, used everywhere.
- Write **clean, complete, runnable code**, not fragments. Prefer one clear
  pipeline over clever abstraction.
- Use **PyTorch + PyTorch Geometric** for the GNN unless I say otherwise.
- Set seeds for Python / NumPy / Torch. **Auto-select CUDA if available, else
  fall back to CPU** (I may run this on a laptop without a GPU).

Suggested layout:
```
od_generator/        # existing forward pipeline — review, do NOT break
configs/             # YAML: data.yaml, model.yaml, train.yaml
src/
  data/              # dataset, graph builder, dataloaders
  models/            # GNN definition
  train.py           # training + evaluation loop
  evaluate.py
data/                # generated .pkl / .json samples + index.json manifest
checkpoints/
```

## Working agreement
- Show me the **plan and the resolved data schema before** generating large code.
- Make the **smallest change that works**; don't refactor `od_generator/` unless asked.
- After each major step, give me the **exact command to run it**.
- If a SUMO/TAZ id mismatch or subprocess failure appears, surface it explicitly
  rather than silently working around it.
