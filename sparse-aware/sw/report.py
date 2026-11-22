#!/usr/bin/env python3
"""Tables and figures for the segmented-sum instruction (vfsegsum) evaluation.

Input (no new simulation):
  results/results_runs_eval_spmv.csv   run_eval.sh, SpMV: scalar_csr, rvv_dense, ara_dense, ara_csr, segred_hw
  results/results_runs_eval_spmm.csv   run_eval.sh, SpMM (K = 8): the same five kernels, *_mm
  mats/*.bin                           the 43 matrices (row lengths, padding, windows)
Output:
  results/summary.md                   every table of the report, generated (no number typed by hand)
  results/figures/speedup.png          2 x 3 panels (SpMV row, SpMM row)
  results/figures/utilization.png, beats.png

Definitions:
  speedup              = cycles of the other kernel / cycles of ours (> 1: ours is faster)
  best other kernel    = fastest of scalar_csr, ara_dense, ara_csr (all run on unmodified Ara)
  win / loss           = more than 3% faster / slower; otherwise a tie (code layout moves cycles by up to 3%)
  useful multiplies    = nnz (x K for SpMM)
  executed multiplies  = dense kernels n per row; CSR and scalar L per row; segred: nnz rounded up to 8
                         (zero padding, bin2S.py); x K for SpMM
  utilization          = 2 x multiplies / (cycles x 16), 16 = peak FLOP/cycle of Ara (4 lanes x 2 x FMA)
  time                 = cycles / 1000 us at 1 GHz
  L1 miss rate         = dc_misses / dc_loads (CVA6 data cache, scalar loads only; Ara has no vector cache)
"""
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import FIG, RES, MATS, md, title, unity, sparsity_axis, SIZE_COLORS, PEAK_FLOP

K = 8                                   # columns of B in SpMM
W = 128                                 # segmented-reduction window (VLEN 4096 / 32)
WL = {"spmv": dict(csv="results_runs_eval_spmv.csv", sfx="", f=1, name="SpMV"),
      "spmm": dict(csv="results_runs_eval_spmm.csv", sfx="_mm", f=K, name="SpMM")}
BASE = ["scalar_csr", "rvv_dense", "ara_dense", "ara_csr", "segred_hw"]   # SpMV names; SpMM adds _mm
OTHERS = ["scalar_csr", "ara_dense", "ara_csr"]                           # "best other kernel" set
VEC = ["rvv_dense", "ara_dense", "ara_csr", "segred_hw"]
STYLE = {"rvv_dense": ("#eb6834", "s"), "ara_dense": ("#c0392b", "P"), "ara_csr": ("#16a3a3", "*"),
         "scalar_csr": ("#52514e", "^"), "segred_hw": ("#1baf7a", "o")}
STRUCT_COLORS = {"uniform": "#2a78d6", "banded": "#eb6834", "lower_tri": "#1baf7a",
                 "half_rows": "#eda100", "mixed": "#7b5cd6"}
STRUCTS = list(STRUCT_COLORS)
LEVELS = [0, 50, 75, 90, 95, 99, 99.9]


def rowinfo(structure, n, sparsity):
    """Row lengths of a matrix (from its .bin)."""
    tag = f"{sparsity:g}".replace(".", "p")
    b = (MATS / f"{structure}_n{n}_s{tag}.bin").read_bytes()
    R = struct.unpack_from("<i", b, 0)[0]
    return np.diff(np.frombuffer(b, "<i4", R + 1, 12))


def load(wl):
    c = WL[wl]
    df = pd.read_csv(RES / c["csv"])
    assert (df.passed == 1).all(), "a kernel run failed"
    df["kernel"] = df.kernel.str.replace("_mm", "", regex=False)          # same names for both workloads
    info = {k: rowinfo(*k) for k in df[["structure", "n", "sparsity"]].drop_duplicates().itertuples(index=False)}
    rows = []
    for r in df.itertuples():
        L = info[(r.structure, r.n, r.sparsity)]
        nzp = -(-r.nnz // 8) * 8
        vec = {"rvv_dense": r.n * r.n, "ara_dense": r.n * r.n, "ara_csr": r.nnz, "segred_hw": nzp}.get(r.kernel, 0)
        sca = r.nnz if r.kernel == "scalar_csr" else 0
        rows.append(dict(vec_mults=c["f"] * vec, scalar_mults=c["f"] * sca, nonempty_rows=int((L > 0).sum()),
                         nnz_pad=nzp, windows=-(-nzp // W)))
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    df["useful_mults"] = c["f"] * df.nnz
    df["executed_mults"] = df.vec_mults + df.scalar_mults
    df["skipped_mults"] = c["f"] * df.n * df.n - df.executed_mults
    df["util_useful_pct"] = 100 * 2 * df.useful_mults / (df.cycles * PEAK_FLOP)
    df["util_issued_pct"] = 100 * 2 * df.vec_mults / (df.cycles * PEAK_FLOP)
    df["time_us"] = df.cycles / 1000.0
    df["l1_miss_pct"] = 100 * df.dc_misses / df.dc_loads.where(df.dc_loads > 0)
    return df


def pivot(df, col="cycles"):
    p = df.pivot_table(index=["structure", "n", "sparsity", "nnz"], columns="kernel", values=col).reset_index()
    if col == "cycles":
        p["best_other"] = p[OTHERS].min(axis=1); p["best_other_kernel"] = p[OTHERS].idxmin(axis=1)
        p["vs_best"] = p.best_other / p.segred_hw
        p["vs_base"] = p.rvv_dense / p.segred_hw
        for k in OTHERS:
            p[f"vs_{k}"] = p[k] / p.segred_hw
    return p


def gmean(x):
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def wl_count(x):
    x = np.asarray(x)
    return int((x > 1.03).sum()), int((x < 1 / 1.03).sum())


def band(s):
    return "0–50%" if s <= 50 else ("75–99%" if s <= 99 else "99.9%")


def ranges(t, col):
    """Sparsity ranges where ours wins / ties / loses against one kernel, as text."""
    t = t.sort_values("sparsity")
    cls = ["wins" if v > 1.03 else ("loses" if v < 1 / 1.03 else "tie") for v in t[col]]
    lv = list(t.sparsity)
    if all(c == "wins" for c in cls):
        return f"{lv[0]:g}–{lv[-1]:g}% (all)"
    runs, i = [], 0
    while i < len(cls):
        j = i
        while j + 1 < len(cls) and cls[j + 1] == cls[i]:
            j += 1
        span = f"{lv[i]:g}%" if i == j else f"{lv[i]:g}–{lv[j]:g}%"
        runs.append(f"{cls[i]} {span}")
        i = j + 1
    return "; ".join(runs)


def breakeven(t, dense_k, sparse_k):
    """First sparsity level where the sparse kernel is faster than the dense one ."""
    t = t.sort_values("sparsity")
    faster = t[t[sparse_k] < t[dense_k]]
    if faster.empty:
        return f"never (up to {t.sparsity.max():g}%)"
    first = faster.sparsity.min(); lower = t[t.sparsity < first].sparsity
    return f"{lower.max():g}–{first:g}%" if len(lower) else f"≤ {first:g}% (always)"


def matrix_label(s, n):
    return f"{s} n={n}"


# ---------------------------------------------------------------- figures
def fig_speedup(P):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    for row, wl in enumerate(["spmv", "spmm"]):
        p, nm, sfx = P[wl], WL[wl]["name"], WL[wl]["sfx"]
        for ax, col, ttl in [(axes[row, 0], "vs_best", f"A. segred_hw{sfx} over best other kernel, uniform"),
                             (axes[row, 2], "vs_base", f"C. segred_hw{sfx} over baseline rvv_dense{sfx}, uniform")]:
            for n in [32, 64, 128]:
                t = p[(p.structure == "uniform") & (p.n == n)].sort_values("sparsity")
                ax.plot(t.sparsity, t[col], color=SIZE_COLORS[n], marker="o", label=f"n={n}")
            unity(ax); sparsity_axis(ax, narrow=True); ax.set_ylim(bottom=0)
            ax.set_ylabel(f"{nm} speedup (×, linear)"); title(ax, ttl); ax.legend(fontsize=8, loc="upper left")
        ax = axes[row, 1]
        for s, c in STRUCT_COLORS.items():
            t = p[(p.structure == s) & (p.n == 128)].sort_values("sparsity")
            ax.plot(t.sparsity, t.vs_best, color=c, marker="o", label=s)
        unity(ax); sparsity_axis(ax, narrow=True); ax.set_ylim(bottom=0)
        ax.set_ylabel(f"{nm} speedup (×, linear)"); title(ax, f"B. segred_hw{sfx} over best other kernel, n=128")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "speedup.png"); plt.close(fig)


def fig_utilization(D):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, wl in zip(axes, ["spmv", "spmm"]):
        t = D[wl][(D[wl].structure == "uniform") & (D[wl].n == 128)]
        for k in VEC:
            s = t[t.kernel == k].sort_values("sparsity"); c, m = STYLE[k]
            ax.plot(s.sparsity, s.util_useful_pct, color=c, marker=m, label=k + WL[wl]["sfx"])
        ax.set_yscale("log"); ax.set_ylabel("Percent of peak 16 FLOP/cycle (log scale)")
        sparsity_axis(ax, narrow=True)
        title(ax, f"{'A' if wl == 'spmv' else 'B'}. {WL[wl]['name']} useful vector utilization, uniform n=128")
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "utilization.png"); plt.close(fig)


def fig_beats(D):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, wl in zip(axes, ["spmv", "spmm"]):
        t = D[wl][(D[wl].structure == "uniform") & (D[wl].n == 128)]
        for k in VEC:
            s = t[t.kernel == k].sort_values("sparsity"); c, m = STYLE[k]
            ax.plot(s.sparsity, s.ara_r_beats, color=c, marker=m, label=k + WL[wl]["sfx"])
        ax.set_yscale("log"); ax.set_ylabel("Ara memory beats, 16 B each (log scale)")
        sparsity_axis(ax, narrow=True)
        title(ax, f"{'A' if wl == 'spmv' else 'B'}. {WL[wl]['name']} memory traffic, uniform n=128")
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "beats.png"); plt.close(fig)


# ---------------------------------------------------------------- tables
def t_speedup_bands(P):
    rows = []
    for b in ["0–50%", "75–99%", "99.9%"]:
        r = {"sparsity": b}
        for wl in ["spmv", "spmm"]:
            p = P[wl]; t = p[p.sparsity.map(band) == b]; w, l = wl_count(t.vs_best)
            nm = WL[wl]["name"]
            r[f"{nm}: matrices"] = len(t)
            r[f"{nm} vs best other (wins / losses)"] = (f"{gmean(t.vs_best):.2f}× ({w} / {l}), range "
                                                        f"{t.vs_best.min():.2f}–{t.vs_best.max():.2f}×")
            r[f"{nm} vs baseline"] = f"{gmean(t.vs_base):.2f}×"
        rows.append(r)
    return md(pd.DataFrame(rows), index=False)


def t_speedup_structs(P):
    rows = []
    for wl in ["spmv", "spmm"]:
        p = P[wl]; r = {"": WL[wl]["name"] + " vs best other"}
        for s in STRUCTS:
            t = p[(p.structure == s) & (p.sparsity >= 75) & (p.sparsity <= 99)]
            r[s] = f"{gmean(t.vs_best):.2f}×"
        rows.append(r)
        r = {"": WL[wl]["name"] + " vs baseline"}
        for s in STRUCTS:
            t = p[(p.structure == s) & (p.sparsity >= 75) & (p.sparsity <= 99)]
            r[s] = f"{gmean(t.vs_base):.2f}×"
        rows.append(r)
    return md(pd.DataFrame(rows), index=False)


def t_best_cases(P):
    out = []
    for wl in ["spmv", "spmm"]:
        p = P[wl].sort_values("vs_best", ascending=False).head(5); sfx = WL[wl]["sfx"]
        f = pd.DataFrame({"matrix": [f"{r.structure} n={r.n} {r.sparsity:g}%" for r in p.itertuples()],
                          "best other (cycles)": [f"{r.best_other:,.0f} ({r.best_other_kernel}{sfx})" for r in p.itertuples()],
                          f"segred_hw{sfx}": [f"{v:,.0f}" for v in p.segred_hw],
                          "speedup": [f"{v:.2f}×" for v in p.vs_best]})
        out.append(f"**{WL[wl]['name']}**\n\n" + md(f, index=False))
    return "\n\n".join(out)


def t_faster_ranges(P):
    rows = []
    keys = P["spmv"][["structure", "n"]].drop_duplicates().sort_values(["structure", "n"])
    order = {s: i for i, s in enumerate(STRUCTS)}
    keys = keys.assign(o=keys.structure.map(order)).sort_values(["o", "n"])
    for s, n in zip(keys.structure, keys.n):
        r = {"matrix": matrix_label(s, n)}
        for wl in ["spmv", "spmm"]:
            t = P[wl][(P[wl].structure == s) & (P[wl].n == n)]; nm, sfx = WL[wl]["name"], WL[wl]["sfx"]
            r[f"{nm} vs baseline rvv_dense{sfx}"] = ranges(t, "vs_base")
            r[f"{nm} vs dense ara_dense{sfx}"] = ranges(t, "vs_ara_dense")
            r[f"{nm} vs vector CSR ara_csr{sfx}"] = ranges(t, "vs_ara_csr")
        rows.append(r)
    return md(pd.DataFrame(rows), index=False)


def t_breakeven(P):
    rows = []
    for s, n in [("uniform", 32), ("uniform", 64), ("uniform", 128), ("banded", 128), ("lower_tri", 128),
                 ("half_rows", 128), ("mixed", 64), ("mixed", 128)]:
        r = {"matrix": matrix_label(s, n)}
        for wl in ["spmv", "spmm"]:
            t = P[wl][(P[wl].structure == s) & (P[wl].n == n)]; nm = WL[wl]["name"]
            r[f"{nm}: CSR beats baseline"] = breakeven(t, "rvv_dense", "ara_csr")
            r[f"{nm}: CSR beats fair dense"] = breakeven(t, "ara_dense", "ara_csr")
            r[f"{nm}: ours beats baseline"] = breakeven(t, "rvv_dense", "segred_hw")
            r[f"{nm}: ours beats fair dense"] = breakeven(t, "ara_dense", "segred_hw")
        rows.append(r)
    return md(pd.DataFrame(rows), index=False)


def t_per_level(D, col, kernels, fmtf, n=128, s="uniform"):
    rows = []
    for lv in LEVELS:
        r = {"sparsity": f"{lv:g}%"}
        for wl in ["spmv", "spmm"]:
            d = D[wl]; t = d[(d.structure == s) & (d.n == n) & (d.sparsity == lv)].set_index("kernel")
            for k in kernels:
                r[f"{WL[wl]['name']}: {k}{WL[wl]['sfx']}"] = fmtf(t.loc[k]) if k in t.index else ""
        rows.append(r)
    return md(pd.DataFrame(rows), index=False)


def t_beats_extra(D):
    rows = []
    for s, lv in [("banded", 75), ("banded", 95), ("half_rows", 95), ("mixed", 90)]:
        r = {"matrix (n=128)": f"{s} {lv:g}%"}
        for wl in ["spmv", "spmm"]:
            d = D[wl]; t = d[(d.structure == s) & (d.n == 128) & (d.sparsity == lv)].set_index("kernel")
            for k in VEC:
                r[f"{WL[wl]['name']}: {k}{WL[wl]['sfx']}"] = f"{t.loc[k].ara_r_beats:,.0f}"
        rows.append(r)
    return md(pd.DataFrame(rows), index=False)


def t_ops(D):
    rows = []
    for lv in LEVELS:
        r = {"sparsity": f"{lv:g}%"}
        for wl in ["spmv", "spmm"]:
            d = D[wl]; t = d[(d.structure == "uniform") & (d.n == 128) & (d.sparsity == lv)].set_index("kernel")
            nm, f = WL[wl]["name"], WL[wl]["f"]; sg = t.loc["segred_hw"]
            dense = f * 128 * 128
            r[f"{nm}: dense multiplies"] = f"{dense:,}"
            r[f"{nm}: useful"] = f"{sg.useful_mults:,.0f}"
            r[f"{nm}: ours executed"] = f"{sg.executed_mults:,.0f}"
            r[f"{nm}: skipped vs dense"] = f"{100 * sg.skipped_mults / dense:.1f}%"
        rows.append(r)
    return md(pd.DataFrame(rows), index=False)


def t_cache(D):
    def cell(r):
        return f"{r.l1_miss_pct:.2f}% ({r.dc_misses:,.0f} / {r.dc_loads:,.0f})" if r.dc_loads > 0 else "no loads"
    return t_per_level(D, None, ["scalar_csr", "rvv_dense", "ara_csr", "segred_hw"], cell)


def t_fits(D):
    """Least-squares cost models over all 43 matrices (median and largest relative error)."""
    rows = []
    def fit(g, X, terms):
        c, *_ = np.linalg.lstsq(X, g.cycles.values.astype(float), rcond=None)
        e = np.abs(X @ c - g.cycles) / g.cycles
        return " + ".join(f"{v:.1f} {t}" for v, t in zip(c, terms)), f"{100 * np.median(e):.1f}%", f"{100 * e.max():.0f}%"
    for wl in ["spmv", "spmm"]:
        d = D[wl]; nm, sfx = WL[wl]["name"], WL[wl]["sfx"]
        g = d[d.kernel == "ara_csr"]
        m, med, mx = fit(g, np.c_[g.nonempty_rows, g.n - g.nonempty_rows, g.nnz],
                         ["per non-empty row", "per empty row", "per non-zero"])
        rows.append({"workload": nm, "kernel": "ara_csr" + sfx, "cost model (cycles)": m, "median error": med, "max error": mx})
        g = d[d.kernel == "segred_hw"]
        m, med, mx = fit(g, np.c_[g.n, g.nnz_pad], ["per row of the matrix", "per non-zero"])
        rows.append({"workload": nm, "kernel": "segred_hw" + sfx, "cost model (cycles)": m, "median error": med, "max error": mx})
        for n in [32, 64, 128]:
            t = d[(d.kernel == "ara_dense") & (d.n == n)]
            rows.append({"workload": nm, "kernel": f"ara_dense{sfx} (n={n})", "cost model (cycles)":
                         f"{(t.cycles / t.n).median():.0f} per row (any sparsity)", "median error": "–", "max error": "–"})
    return md(pd.DataFrame(rows), index=False)


def main():
    D = {wl: load(wl) for wl in WL}
    P = {wl: pivot(D[wl]) for wl in WL}
    fig_speedup(P); fig_utilization(D); fig_beats(D)
    tm = lambda r: f"{r.time_us:,.1f}"
    ut = lambda r: f"{r.util_useful_pct:.2f}%"
    ui = lambda r: f"{r.util_issued_pct:.2f}%"
    bt = lambda r: f"{r.ara_r_beats:,.0f}"
    s = ["# Segmented sum (vfsegsum): tables (generated by report.py)\n",
         f"Runs: SpMV {len(D['spmv'])} kernel runs, SpMM {len(D['spmm'])} kernel runs, all passed. "
         "Speedup > 1 means segred_hw is faster. Win / loss = more than 3% faster / slower.\n",
         "## Speedup by sparsity band\n", t_speedup_bands(P),
         "\n## Speedup by structure, 75–99%\n", t_speedup_structs(P),
         "\n## Best cases (over the best other kernel)\n", t_best_cases(P),
         "\n## Sparsity levels at which segred_hw is faster than each kernel\n", t_faster_ranges(P),
         "\n## Break-even: first sparsity level where the sparse kernel beats the dense one\n", t_breakeven(P),
         "\n## Execution time, µs at 1 GHz, uniform n=128\n", t_per_level(D, None, BASE, tm),
         "\n## Useful vector utilization, uniform n=128 (2 × useful multiplies / (cycles × 16))\n",
         t_per_level(D, None, VEC, ut),
         "\n## Issued vector utilization, uniform n=128 (every multiply Ara performs, zeros included)\n",
         t_per_level(D, None, VEC, ui),
         "\n## Memory traffic: beats read by Ara, uniform n=128\n", t_per_level(D, None, VEC, bt),
         "\n## Memory traffic: structured matrices, n=128\n", t_beats_extra(D),
         "\n## Useful vs skipped multiplies, uniform n=128 (ours vs a dense kernel)\n", t_ops(D),
         "\n## CVA6 L1 data-cache miss rate, uniform n=128: % (misses / loads)\n", t_cache(D),
         "\n## Cost models (least squares over all 43 matrices)\n", t_fits(D), ""]
    (RES / "summary.md").write_text("\n".join(s))
    print("wrote", RES / "summary.md", "and 3 figures in", FIG)


if __name__ == "__main__":
    main()
