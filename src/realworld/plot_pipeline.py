"""
plot_pipeline.py — render the real-data calibration pipeline as a diagram.

    python -m src.realworld.plot_pipeline   ->  docs/realworld/pipeline.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import resolve


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    C = {"in": "#cfe8f3", "A": "#f3e1cf", "B": "#d8f0d8", "C": "#e7d8f0",
         "D": "#f7d6d6", "E": "#fff0c2", "sumo": "#fdd9b5"}

    def box(x, y, w, h, text, color, fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                     fc=color, ec="#555", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, color="#333", style="-|>", lw=1.6):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=16, color=color, lw=lw,
                     connectionstyle="arc3,rad=0"))

    def label(x, y, t, fs=11, color="#222"):
        ax.text(x, y, t, ha="left", va="center", fontsize=fs, fontweight="bold", color=color)

    # ── inputs ────────────────────────────────────────────────────────────────
    box(2, 90, 28, 7, "Neshan\nper-segment TRAVEL TIME\n(one period)", C["in"], bold=True)
    box(36, 90, 28, 7, "OSM SUMO net + TAZ\n(prune_tab.net.xml)", C["in"], bold=True)
    box(70, 90, 28, 7, "synthetic od_dataset.h5\n(forward sims, for A & warm-start)", C["in"], bold=True)

    # ── Stage A ────────────────────────────────────────────────────────────────
    label(2, 84, "A  diagnose how much OD signal travel time carries", color="#a5641a")
    box(20, 76, 60, 6.5, "ridge: travel time → zone marginals → gravity OD\n"
        "→ cell-corr ceiling ≈ 0.06  (flow 0.14, turns 0.19)  ⇒ data is the bottleneck",
        C["A"])
    arrow(84, 90, 70, 82.5)   # synthetic -> A

    # ── Stage B ────────────────────────────────────────────────────────────────
    label(2, 70, "B  ingest Neshan + match to SUMO edges", color="#2e7d32")
    box(3, 61, 26, 6.5, "neshan_ingest\nCSV / GeoJSON → segments", C["B"])
    box(33, 61, 30, 6.5, "match_edges\nosm_id › name › geometry", C["B"])
    box(67, 61, 30, 6.5, "build_observation\ntt_obs[N] + coverage mask", C["B"])
    arrow(16, 90, 16, 67.5)          # Neshan -> ingest
    arrow(50, 90, 48, 67.5)          # net -> match
    arrow(29, 64.2, 33, 64.2); arrow(63, 64.2, 67, 64.2)

    # ── Stage C ────────────────────────────────────────────────────────────────
    label(2, 55, "C  learned warm-start (only a seed — travel time is weak)", color="#6a3d9a")
    box(20, 46, 60, 6.5, "ridge: tt → production/attraction marginals → Furness → OD₀", C["C"])
    arrow(82, 61, 78, 52.5)          # observation -> C
    arrow(50, 76, 50, 52.5, color="#aaa", style="-|>", lw=1.2)  # A informs C

    # ── Stage D — the engine ────────────────────────────────────────────────────
    label(2, 40, "D  SUMO-in-the-loop calibration  (the real engine)", color="#c0392b")
    box(8, 24, 34, 9, "① find_total\nmatch congestion intensity\n(median tt/tt_free) → pin demand",
        C["D"])
    box(48, 24, 30, 11, "② SPSA on 2×9 shape\nmultipliers vs log-space\ntravel-time loss", C["D"])
    box(82, 24, 15, 11, "SUMO 1.27\nsimulate\nOD → tt", C["sumo"], bold=True)
    arrow(50, 46, 25, 33)            # OD0 -> find_total
    arrow(42, 28.5, 48, 28.5)        # find_total -> SPSA
    # SPSA <-> SUMO loop
    arrow(78, 31, 82, 31)            # SPSA -> SUMO
    arrow(82, 26.5, 78, 26.5)        # SUMO -> SPSA (tt back)
    ax.text(80, 22.0, "loop / eval", fontsize=7, color="#c0392b", ha="center")

    # ── Stage E ────────────────────────────────────────────────────────────────
    label(2, 18, "E  validate + what-if", color="#9a7d0a")
    box(10, 8, 40, 7, "validate\ntravel-time corr / RMSE · GEH (re-sim flows)", C["E"])
    box(56, 8, 40, 7, "whatif\nclose / add road → traffic redistributes", C["E"])
    arrow(60, 24, 30, 15)            # calibrated OD -> validate
    arrow(63, 24, 76, 15)            # calibrated OD -> whatif

    ax.text(50, 4, "calibrated OD reproduces observed travel times; it is ONE of many demands "
            "consistent with them (travel time under-determines the OD).",
            ha="center", fontsize=9, style="italic", color="#444")
    ax.set_title("Real-data OD calibration pipeline (Neshan travel times → SUMO scenario)",
                 fontsize=13, fontweight="bold")

    out = resolve("docs/realworld/pipeline.png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[pipeline] -> {out}")


if __name__ == "__main__":
    main()
