# 9-zone OD study (Tehran, District 2)

A compact, self-contained re-run of the **forward + inverse** pipeline at **9
zones** (down from 36), reusing the proven marginal-GNN + turn-count recipe.
Everything lives in `src/zones9/` and `configs/zones9.yaml`; the 36-zone pipeline
is untouched.

## Step 1 — TAZ (9 zones)
SUMO's `gridDistricts.py` partitions the network into a grid of TAZ cells.
Grid-width controls the count: **1000 → 36 zones** (the original), and here:

| grid width (m) | zones |
|---|---|
| 4000 | 6 |
| **3000** | **9** ← used |
| 2500 | 10 |

Width 4000 gives only 6 zones, so we used **3000** to hit the requested 9.
```bash
SUMO_HOME=$(python -c "import sumo,os;print(os.path.dirname(sumo.__file__))")
python "$SUMO_HOME/tools/district/gridDistricts.py" \
       -n sumo/prune_tab.net.xml -o sumo/taz_9.xml -w 3000
```
Zone ids are grid `row_col` (`0_0`…`3_2`); the 1897 road edges form a clean
partition (zone sizes 5/168/48/807/140/9/536/183/1 — some cells are tiny).
See [docs/zones9/zones_map.png](docs/zones9/zones_map.png).

## Step 2 — OD dataset (1998 samples)
For each sample: draw a total demand (4 stratified log tiers = night/noon/
morning/evening), sample a 9×9 OD with the gravity/dirichlet/sparse mix (gravity
uses the **real** zone grid positions), simulate it through SUMO (eclipse-sumo
1.27), and record per-edge `[flow, speed, density, traveltime]` **plus junction
turn counts** — the measurement that makes the OD identifiable. Stored as
`.pkl` + `index.json` manifest (stratified split 1398/300/300).
```bash
python -m src.zones9.generate                # parallel, resumable, ~1998/2000 ok
python -m src.zones9.build_packs             # -> data9/pack_{train,val,test}.pt
```

## Step 3 — train + evaluate
The marginal GNN (`ODMarginalGNN`, turn counts as GraphConv edge weights) predicts
per-zone production/attraction (standardised-MSE); the 9×9 OD is rebuilt with a
Furness gravity prior. Trained on GPU in resumable ~25 s chunks (the environment
kills long-running processes ~45 s; `--max-seconds` makes each chunk self-pause).
```bash
# repeat until "training complete"; each call is one safe chunk:
python -m src.zones9.train_eval --mode train --device cuda --max-seconds 25
python -m src.zones9.train_eval --mode eval  --device cpu     # metrics + GEH re-sim
```

### Results (test n=300)
| metric | value |
|---|---|
| production / attraction corr | **0.82 / 0.80** |
| reconstructed OD **cell-corr** | **0.66**  (gravity ceiling 0.88) |
| total-flow error | 0.19 |
| **GEH on re-simulated link flows** | **mean 2.15, 94 % of links < 5** |

Per regime cell-corr: noon 0.74, morning-peak 0.79, evening-peak 0.63, **night
0.17** (night ≈ 50 trips over 72 cells → mostly noise; it has little traffic
anyway). The **GEH 2.15 / 94 %** is the practically important number: when the
predicted OD is re-simulated, the resulting link flows match the true traffic
very closely (GEH < 5 is the standard "good match" threshold).

## Step 4 — visualizations (`docs/zones9/`, full set in `data9/viz/`)
- `zones_map.png` — the 9 TAZ zones on the road network.
- `training_curve.png` — loss + validation marginal correlations (→ ~0.88).
- `marginals_scatter.png` — predicted vs true production/attraction.
- `cell_scatter.png` — pooled off-diagonal OD cells, predicted vs true.
- `od_examples.png` — predicted vs true 9×9 OD, busiest sample per regime
  (totals match closely: 124→106, 313→313, 786→772, 1988→1998).
- `metrics_summary.png` — cell-corr per regime vs the gravity ceiling.

## Files
```
configs/zones9.yaml          all paths + hyperparameters
sumo/taz_9.xml               9-zone TAZ (gridDistricts -w 3000)
src/zones9/graph.py          9-zone line-graph (net + TAZ)
src/zones9/generate.py       OD sampling + SUMO sim + turn counts -> pkl + manifest
src/zones9/build_packs.py    per-sample pkls -> train/val/test packs
src/zones9/train_eval.py     marginal GNN train (resumable) + evaluate (+GEH)
src/zones9/visualize.py      the 6 figures
data9/                       generated dataset, packs, checkpoints, viz (git-ignored)
```
Reuses `src/models/gnn.py` (model factory), `src/train.py` (marginal loss/eval),
`src/od_reconstruct.py` (Furness), `src/realworld/sim.py` (1.27 SUMO runner) and
`src/data/resimulate.py` (turn-count parser) — only the zone count and data differ.

## Notes / honest caveats
- **9 zones is an easier inverse than 36** (72 vs 1260 cells); the recipe is the
  same, results are strong and the re-simulated GEH is excellent.
- The **night/low-demand regime** is intrinsically hard (sparse trips); it drags
  the pooled cell-corr down but carries little traffic.
- Dataset is **2000 samples** (vs 10k for 36 zones) — enough for this small
  problem, generated with the fast 1.27-wheel runner (the original
  `od_generator` targets the system SUMO 1.18, which cannot parse this net).
