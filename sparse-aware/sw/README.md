# Kernels, tests and evaluation for `vfsegsum`

Everything here runs on the Verilator model of this repository's Ara (with `vfsegsum`). The idea is described in [`../IDEA.md`](../IDEA.md).

## Files

| File | What it is |
|---|---|
| `main.c` | SpMV, y = A·x. Kernels: `scalar_csr`, `rvv_dense` (baseline), `ara_dense` (fair dense), `ara_csr` (row-by-row vector CSR), **`segred_hw`** (windows of non-zeros + `vfsegsum`) |
| `spmm.c` | SpMM, C = A·B with B dense (n × 8). The same five kernels, `*_mm`; **`segred_hw_mm`** is ours |
| `segsum_test.c` | Directed unit test of the instruction itself: 41 cases (vl = 1…128, 5 flag patterns) for `vsegsum` (int) and `vfsegsum` (float, bit-exact vs a C loop) |
| `bin2S.py` | Turns a matrix `.bin` into `data.S`. Also builds the one-time sparse metadata: tail-flag bitmap `SEG_TAIL`, row offsets in y `SEG_DOFF`, and per-window info `SEG_INFO` |
| `gen_matrices.py` | Synthetic CSR matrices with a SciPy reference result (uniform, banded, lower_tri, half_rows, mixed; seeded) |
| `segred_cases.py` | Edge-case matrices: rows crossing windows, rows longer than a window, empty rows, windows of 1-element rows |
| `build_run.sh` | Build one program for one matrix and run it on the simulator |
| `segsum_test.sh` | Build and run the unit test |
| `quick_check.sh` | A few minutes: unit tests plus both workloads on the edge cases and uniform n = 32 |
| `run_eval.sh` | Full evaluation: 43 matrices × 2 workloads = 86 simulations, then `summarize.py` |
| `summarize.py` | Collects the `RES` and `[roi-counters]` lines of each `sim.log` into a CSV |
| `report.py`, `plot_style.py` | Tables (`results/summary.md`) and figures (`results/figures/`) from the CSVs |
| `mats/` | The matrices used for the reported results |
| `results/` | CSVs of the final evaluation, the generated tables and figures, and sample simulator logs |

## Running

Prerequisite: the Ara toolchain and simulator, built as in the main Ara README (`make toolchain-llvm`, `make verilator`, then `make -C hardware checkout apply-patches verilate`). The Python scripts need numpy, scipy, pandas and matplotlib.

```bash
cd sparse-aware/sw

# unit test of the instruction (mode 1 = int vsegsum, 2 = float vfsegsum)
./segsum_test.sh 2 /tmp/segsum_float          # prints "SEGSUM ALL PASS"

# one matrix, SpMV and SpMM, all five kernels
EXTRA_CFLAGS=-DFINAL_EVAL ./build_run.sh mats/uniform_n128_s90.bin /tmp/spmv
PROG=spmm.c BIN2S_ARGS="--spmm 8" EXTRA_CFLAGS=-DFINAL_EVAL ./build_run.sh mats/uniform_n128_s90.bin /tmp/spmm

# quick check (absolute output path), then the full evaluation (hours) and the report
./quick_check.sh /tmp/quick
JOBS=8 ./run_eval.sh
python3 report.py
```

Every kernel prints one line, e.g. `RES kernel=segred_hw n=128 nnz=1638 cycles=9158 PASS`. Its output is checked against the SciPy reference.

`ARA_ROOT` points the scripts at another Ara checkout (default: this repository).

## Note on the stored results

The CSVs, tables, figures and logs in `results/` are the reported results of the final evaluation (430 kernel runs, all correct). They were produced by an earlier build of `main.c`/`bin2S.py` that also contained one extra kernel and its data tables, which are not part of this repository. Removing them moves the remaining code and data to slightly different addresses. That changes instruction fetch, cache behaviour and memory alignment, though not what any kernel computes.

Re-running with the code here gives the same correctness and the same conclusions, but individual cycle counts can differ by up to about 3%. A spot check of 40 kernel runs on 4 matrices gave 29 identical and 11 within −3.2% to +0.4%, mostly in the scalar and dense baselines. This is the same code-layout effect that the evaluation already accounts for: kernels are only compared within one program, and a win or loss means a difference of more than 3%.

## Credits

The baseline kernels in `main.c` (`scalar_csr`, `rvv_dense`, `ara_csr`, `ara_dense`), the SpMM program `spmm.c` (apart from `segred_hw_mm`) and `gen_matrices.py` come from the team project and are used for comparison. `segred_hw`, `segred_hw_mm`, `segsum_test.c`, the metadata generation in `bin2S.py` and the evaluation scripts belong to the `vfsegsum` work.
