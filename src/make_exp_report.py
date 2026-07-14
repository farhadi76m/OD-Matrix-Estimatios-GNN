"""
make_exp_report.py — build the full experiments report as a PDF.

Markdown -> HTML (python-markdown) -> PDF (WeasyPrint). Numbers are the saved
metrics of every experiment; figures are the committed docs/ outputs.

    python -m src.make_exp_report   ->  report/OD_experiments_report.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]

DOC = r"""
# Origin–Destination Matrix Estimation from Traffic Measurements
### Experiments report — Tehran, District 2 (1,897 road links) · SUMO + Graph Neural Networks

**Abstract.** We study the inverse traffic problem: recovering an origin–destination
(OD) trip matrix from road‑level traffic measurements. Across four experiments we
establish one central result — **the OD matrix is only as recoverable as the
measurement is informative**, and the bottleneck is the *data*, not the model. Aggregate
link counts leave the OD under‑determined (cell‑correlation 0.14 for an optimal linear
inverse); **junction turn counts** make the zone marginals recoverable (correlation ≈ 0.90)
and, combined with a doubly‑constrained gravity reconstruction, yield **OD
cell‑correlation 0.78** at 36 zones. Real‑world **travel‑time** data (Neshan) is the
weakest signal of all (ceiling 0.06): it can be used to *calibrate* a SUMO scenario that
reproduces observed conditions, but it cannot identify the true OD. We report results at
36, 9 and 13 zones, a real‑data calibration pipeline, and the engineering defects found
along the way.

---

## 1. Objective and the inverse problem

The forward pipeline (`od_generator/`, generalised in `src/odpipe/`) samples an OD matrix,
routes it through SUMO, and records per‑edge traffic. We build the **inverse**:

> per‑edge SUMO measurements → **GNN** → OD matrix (origin × destination over TAZ zones).

This is fundamentally **ill‑posed**: many different OD matrices produce nearly the same
link measurements. Every result below follows from how much of the OD a given measurement
actually constrains.

## 2. Method

The recipe that works, used identically in every experiment:

1. **Road line‑graph** — nodes = the 1,897 road edges (features: length, lanes, speed
   limit, plus measured flow / speed / density / travel time and turn aggregates);
   edges = downstream connectivity; **edge weights = junction turn counts**.
2. **Predict the recoverable quantity** — a GNN (`ODMarginalGNN`, GraphConv) pools the
   graph into per‑zone vectors and predicts **production/attraction marginals**, trained
   with a per‑zone *standardised MSE* (a log1p+Huber loss under‑disperses the marginals
   and collapses the reconstruction).
3. **Reconstruct the matrix** — doubly‑constrained gravity / **Furness (IPF)** from the
   predicted marginals with a distance‑decay deterrence.

Splits are stratified by demand tier so every split covers all demand regimes. Metrics:
marginal correlation, OD cell‑correlation (pooled, off‑diagonal), total‑flow error, and
**GEH** on re‑simulated link flows.

## 3. Experiment 1 — What measurement identifies the OD? (the central result)

We fixed the model class (ridge) and varied only the *input signal*, on synthetic data
where the truth is known, under a realistic 77 % edge coverage.

| input signal | zone‑marginal corr | reconstructed OD cell‑corr |
|---|---|---|
| static only (no traffic) | 0.00 | 0.02 |
| **travel time** (what Neshan provides) | 0.25 | **0.06** |
| speed | 0.24 | 0.05 |
| link flow (counts) | 0.36 | 0.14 |
| junction turn counts (node‑aggregated) | 0.46 | 0.19 |
| all dynamic features | 0.50 | 0.20 |
| *ceiling: gravity from TRUE marginals* | 1.00 | **0.87** |

Two conclusions drive the whole project. **(a)** No model — GNN, VAE or ridge — can
recover the OD from travel time alone; the information is not there. **(b)** The gravity
reconstruction from *true* marginals reaches 0.87, so **predicting marginals + gravity is
near‑optimal**; the remaining task is to predict marginals well, which turn counts enable.

## 4. Experiment 2 — 36 zones (turn counts + marginal GNN)

Re‑simulating each scenario to capture junction turn counts, and feeding them as GNN edge
weights, lifts marginal correlation from ≈ 0.5 to ≈ 0.90.

| approach (36 zones, test n=750) | OD cell‑corr | total‑flow err | RMSE |
|---|---|---|---|
| direct OD‑cell regression (baseline) | 0.06 | 0.44 | 6.78 |
| marginal GNN + gravity, **link counts** | 0.435 | 0.49 | 6.78 |
| marginal GNN + gravity, **turn counts** | **0.784** | **0.132** | **4.08** |

Marginals: production r = **0.904**, attraction r = **0.916**. The direct‑regression
baseline confirms the under‑determination; the turn‑count model is a 13× improvement in
cell‑correlation over it.

## 5. Experiment 3 — 9 zones, and 36 vs 9

The same recipe re‑run at a coarser resolution (TAZ grid width 3000 m → 9 zones,
2,000 samples).

![](docs/zones9/zones_map.png)

*Figure 1 — The 9 TAZ zones on the District‑2 network. Note the imbalance: the central
zones dominate while edge cells are tiny — a recurring source of noisy marginals.*

| metric | 36 zones | 9 zones |
|---|---|---|
| OD cells (off‑diagonal) | 1,260 | 72 |
| training samples | 3,500 | 2,000 |
| production / attraction corr | **0.904 / 0.916** | 0.816 / 0.800 |
| OD cell‑corr | **0.784** | 0.660 |
| total‑flow error | **0.132** | 0.186 |
| gravity ceiling | ~0.86 | 0.884 |
| **GEH < 5 (link flows)** | not scored | **94.4 %** |

![](docs/slides/cmp_metrics_bar.png)

*Figure 2 — Marginal and OD recovery at both resolutions, against the gravity ceiling.
Both models sit close to their ceiling: the limit is information, not capacity.*

![](docs/slides/cmp_cell_scatter.png)

*Figure 3 — Per‑cell predicted vs true OD, pooled over the test set.*

**Finer is not harder here.** The 36‑zone model scores *higher* on every correlation
metric despite having 17× more cells — largely because it was trained on 1.75× more data,
and because coarse grids still contain tiny, near‑empty zones that add noise. Conversely
the 9‑zone OD is the most **simulation‑faithful**: re‑simulating it reproduces observed
link flows with GEH < 5 on 94 % of links. Raw RMSE is *not* comparable across resolutions
(the same trips packed into 72 vs 1,260 cells make 9‑zone per‑cell values ~6× larger).

![](docs/slides/cmp_per_regime.png)

*Figure 4 — Accuracy by demand regime. Both improve with demand; low‑demand (night) is
the hardest, where few trips give little signal.*

## 6. Experiment 4 — General pipeline: 13 zones at high demand (mesoscopic)

To test generality we rebuilt the pipeline config‑driven (`src/odpipe/`, nothing
hard‑coded to a zone count) and ran it at **TAZ width 2,000 m → 13 zones** with a
**high demand range of 2,000–30,000 trips/hour**, simulated with SUMO's **mesoscopic**
model (~2 s/sample — microsimulation is impractical at this demand).

![](docs/odpipe/od_examples.png)

*Figure 5 — Predicted vs true 13×13 OD, busiest test sample per demand tier. The blank
rows/columns in the predictions are the three tiny zones (2–3 edges each) produced by the
2,000 m grid: they carry almost no traffic, so their marginals are unlearnable.*

| 13 zones, high demand (**smoke run**: 160 samples, test n=24) | value |
|---|---|
| production / attraction corr | 0.749 / 0.390 |
| OD cell‑corr | 0.164 |
| total‑flow error | 0.270 |
| gravity ceiling | 0.825 |
| GEH < 5 | 48.3 % |

These are **shakedown numbers from a deliberately tiny dataset** (160 samples, to validate
the plumbing) and should not be compared with the fully‑trained experiments above. Two
real lessons do carry: the **grid width matters more than the zone count** — a width that
produces near‑empty zones (attraction r = 0.39 here) caps achievable accuracy; and
**mesoscopic simulation is what makes high‑demand study feasible** at all.

## 7. Experiment 5 — Real data: OD calibration from Neshan travel times

The operational goal is a SUMO scenario that reproduces *observed* traffic so that road
changes can be simulated. The only real observable is **per‑segment travel time** — by
Experiment 1, the weakest signal (ceiling 0.06). We therefore do **not** learn the OD; we
**calibrate** it with SUMO in the loop.

![](docs/realworld/pipeline.png)

*Figure 6 — The real‑data calibration pipeline. A ridge model gives only a warm start;
the engine is a SUMO‑in‑the‑loop search that matches observed travel times.*

The demand is split into the parts travel time can inform: **total demand** is pinned by
matching the overall congestion intensity, then **SPSA** (gradient‑free) optimises the
spatial shape against a log‑space travel‑time loss on the observed edges.

On a self‑consistent benchmark (a known 1,906‑trip evening‑peak OD, 398 observed edges):

| | warm start | calibrated |
|---|---|---|
| travel‑time correlation | 0.677 | **0.722** |
| travel‑time log‑RMSE | 1.452 | **1.185** |
| observed edges actually reproduced | 58 % | **77 %** |
| SPSA loss (30 iterations) | 2.110 | **1.412** |
| **recovered total demand** | — | 2,322 vs 1,906 true (**+22 %**) |
| **OD cell‑corr vs truth** | — | **0.035** |

![](docs/realworld/traveltime_match.png)

*Figure 7 — Observed vs simulated travel times after calibration. Points on the horizontal
axis are observed edges the calibrated demand leaves empty — a direct symptom of the
non‑uniqueness.*

**The honest result.** We can recover the *demand magnitude* to ~20 % and reproduce
observed travel times, **but not the OD itself** (cell‑corr 0.035) — exactly as
Experiment 1 predicts. Travel time lets you fit *either* the demand level *or* the
per‑edge pattern; forcing a tighter per‑edge match (travel‑time corr 0.90) requires
inflating demand 3×. The calibrated matrix is therefore **one of many** demands consistent
with today's data.

![](docs/realworld/whatif_delta.png)

*Figure 8 — What‑if: closing a three‑edge corridor with the calibrated demand held fixed.
Travel time changes on 104 edges — relieved links in blue, detour links in red. This is
the end goal working; its reliability is highest for moderate, local changes.*

## 8. Engineering findings

Defects found and fixed — each one materially affected results:

| finding | impact |
|---|---|
| **Route files must be globally sorted by departure.** SUMO streams route files and *silently drops* any vehicle departing before the previous one. Trips were written per OD pair, so most were discarded. | Simulations ran on a fraction of the intended demand (sparse traffic, under‑counted congestion, biased calibration). Fixed by sorting all trips by depart time. |
| **Race in parallel generation.** Workers shared a temp directory keyed by `idx % n_workers`, so a freed worker could overwrite a running one's trip files. | ~40 % of samples failed. Fixed with a unique directory per sample → 0 failures. |
| **SUMO version mismatch.** System SUMO 1.18 cannot parse this network (built with 1.26). | Pinned to the `eclipse-sumo` 1.27 wheel and resolve binaries from it. |
| **Marginal loss choice.** log1p + Huber under‑disperses the marginals (std‑ratio 0.35), collapsing the Furness reconstruction. | Per‑zone standardised MSE restores dispersion; cell‑corr 0.44 → 0.78. |
| **Long processes are killed (~45 s).** | All training and calibration are checkpointed and resumable (`--max-seconds`). |

## 9. Conclusions

1. **The measurement, not the model, is the bottleneck.** Ranked by OD identifiability:
   turn counts > link counts > speed ≈ travel time. Adding model capacity or synthetic
   data cannot cross this ceiling.
2. **Predict marginals, reconstruct with gravity.** The full OD is under‑determined; the
   marginals are not. This reframing plus turn counts takes cell‑correlation from 0.06 to
   **0.78**.
3. **Choose resolution by purpose.** 36 zones for spatial detail and the best correlations;
   9 zones for a compact, simulation‑faithful OD (GEH < 5 on 94 % of links). Avoid grid
   widths that create near‑empty zones.
4. **For the real network, calibrate — don't learn.** Travel time cannot identify the OD,
   but SUMO‑in‑the‑loop calibration produces a scenario that reproduces observed conditions
   and supports what‑if road changes, with quantified uncertainty.

**Recommended next step:** obtain any *count‑like* measurement (junction turn counts or
link volumes) for the real network. That single change moves the real‑data problem from
the 0.06 regime to the 0.78 regime, and makes SUMO's own `routeSampler`/`cadyts`
calibration directly applicable.

## 10. Reproducibility

All hyperparameters and paths live in `configs/*.yaml`; seeds are fixed; CUDA is
auto‑selected with CPU fallback.

```bash
conda activate traffic

# general pipeline (any zone count / demand range) — see README_odpipe.md
python -m src.odpipe.taz                     # TAZ at configs/odpipe.yaml -> taz.width
python -m src.odpipe.graph
python -m src.odpipe.generate --workers 8    # mesoscopic, resumable
python -m src.odpipe.build_packs
python -m src.odpipe.train_eval --mode train --device cuda --max-seconds 25
python -m src.odpipe.train_eval --mode eval  --device cpu
python -m src.odpipe.infer --max-trips --sumo-gui --which both --meso
python -m src.odpipe.visualize

# real-data calibration from Neshan travel times — see README_realworld.md
python -m src.realworld.diagnose_traveltime  # the identifiability table (Experiment 1)
python -m src.realworld.calibrate --max-seconds 300
python -m src.realworld.validate
python -m src.realworld.whatif --close-edges <e1>,<e2>
```
"""

CSS = """
@page { size: A4; margin: 1.8cm 1.7cm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; color: #0b3d5c; margin-bottom: 2pt; }
h2 { font-size: 14pt; color: #0b3d5c; border-bottom: 1.5px solid #0b3d5c; padding-bottom: 2px; margin-top: 18px; }
h3 { font-size: 11.5pt; color: #444; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 4px 7px; text-align: left; }
th { background: #e8f0f5; }
img { max-width: 100%; display: block; margin: 8px auto 2px; border: 1px solid #ddd; }
code, pre { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; background: #f4f4f4; }
pre { padding: 8px; border-radius: 4px; white-space: pre-wrap; }
p { text-align: justify; }
p.cap { text-align: center; font-size: 9pt; color: #444; margin: 0 0 12px; }
"""


def main():
    html_body = markdown.markdown(DOC, extensions=["tables", "fenced_code"])
    html_body = html_body.replace("<p><em>Figure", '<p class="cap"><em>Figure')
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    out_dir = ROOT / "report"; out_dir.mkdir(exist_ok=True)
    pdf = out_dir / "OD_experiments_report.pdf"
    HTML(string=html, base_url=str(ROOT)).write_pdf(str(pdf))
    print(f"[report] wrote {pdf}  ({pdf.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
