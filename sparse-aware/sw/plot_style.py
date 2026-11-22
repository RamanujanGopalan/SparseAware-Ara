#!/usr/bin/env python3
"""Plot style and table helpers used by report.py (subset of the team project's plot_report.py).

Style follows CompArch-Project/scripts/plot_results.py (same kernel colours, same
sparsity axis). GFLOP/s assumes a 1 GHz clock (Ara's TVLSI'20 paper title: "1-GHz+").
"""
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, FIG, MATS = ROOT / "results", ROOT / "results" / "figures", ROOT / "mats"
FIG.mkdir(parents=True, exist_ok=True)
GHZ = 1.0          # assumed clock for time / GFLOP/s
VLMAX = 128        # floats per m1 register at VLEN 4096
LANES = 4
PEAK_FLOP = LANES * 2 * 2   # 4 lanes x 2 float32 per 64-bit lane x 2 FLOP (FMA)

# Fixed kernel -> colour/marker (validated categorical palette, slots 1-5 in order).
KSTYLE = {
    "rvv_csr": ("#2a78d6", "o", "RVV CSR"),
    "rvv_dense": ("#eb6834", "s", "RVV dense"),
    "scalar_csr": ("#1baf7a", "^", "Scalar CSR"),
    "scalar_dense": ("#eda100", "D", "Scalar dense"),
    "rvv_csr_vl": ("#e87ba4", "v", "RVV CSR, row-length VL"),
}
SSTYLE = {
    "uniform": ("#2a78d6", "o", "Uniform random"),
    "banded": ("#eb6834", "s", "Banded"),
    "lower_tri": ("#1baf7a", "^", "Lower triangular"),
    "half_rows": ("#eda100", "D", "Half rows empty"),
}
SIZES = [32, 64, 128, 256]
SIZE_COLORS = dict(zip(SIZES, ["#a9c9f0", "#5b9be3", "#2a78d6", "#123f78"]))  # one hue, light -> dark
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def sparsity_axis(ax, narrow=False):
    ax.set_xscale("function", functions=(lambda s: -np.log10(np.clip(100 - s, 0.05, None)),
                                         lambda t: 100 - 10 ** (-t)))
    ticks = [0, 50, 75, 90, 95, 99, 99.9]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}%" for t in ticks], rotation=45 if narrow else 0)
    ax.set_xlabel("Sparsity (% zeros)")


def unity(ax):
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)


def title(ax, text):
    ax.set_title(text, loc="left", color=INK)



def md(frame, index=True, digits=4):
    """Markdown table without the tabulate package."""
    f = frame.reset_index() if index else frame
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{digits}g}")
        return str(v)
    head = "| " + " | ".join(map(str, f.columns)) + " |\n|" + "---|" * len(f.columns)
    return head + "\n" + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in f.itertuples(index=False))


