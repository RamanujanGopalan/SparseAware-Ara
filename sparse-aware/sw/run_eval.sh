#!/usr/bin/env bash
# Final evaluation of the segmented-sum instruction (vfsegsum), both workloads, one run.
#   SpMV: main.c with -DFINAL_EVAL  -> scalar_csr, rvv_dense (baseline), ara_dense, ara_csr, segred_hw
#   SpMM: spmm.c (C = A x B, B dense n x 8) -> scalar_csr_mm, rvv_dense_mm, ara_dense_mm, ara_csr_mm, segred_hw_mm
# Matrices (43, gen_matrices.py): uniform n=32/64/128 at all 7 sparsity levels; banded, lower_tri,
# half_rows n=128 at 50/75/95/99; mixed n=64/128 at 50/75/90/95/99.  86 simulations in total.
# Runs JOBS (default 12) simulations at once, longest first. Finished runs (SUCCESS in sim.log) are
# skipped, so the script can be restarted after an interruption.
# Output: runs_eval_spmv/<matrix>/, runs_eval_spmm/<matrix>/ (data.S, app, sim.log)
#         results/results_runs_eval_spmv.csv, results/results_runs_eval_spmm.csv (summarize.py)
cd "$(dirname "$0")"
ARA=${ARA_ROOT:-$(cd ../.. && pwd)}; [ -f "$ARA/install/venv/bin/activate" ] && source "$ARA/install/venv/bin/activate"   # Python with numpy/scipy, if Ara has a venv
G=${GEN:-./gen_matrices.py}   # matrix generator
echo "START $(date '+%F %T')"
for n in 32 64 128; do python3 $G -n $n --structure uniform -o mats >/dev/null; done
for s in banded lower_tri half_rows; do python3 $G -n 128 --structure $s -s 50 75 95 99 -o mats >/dev/null; done
for n in 64 128; do python3 $G -n $n --structure mixed -s 50 75 90 95 99 -o mats >/dev/null; done
MATS=$(ls mats/uniform_n{32,64,128}_*.bin mats/{banded,lower_tri,half_rows}_n128_s{50,75,95,99}.bin mats/mixed_n{64,128}_*.bin)
echo "matrices: $(echo $MATS | wc -w)"

# Job list "<estimate> <workload> <matrix>", longest first. The estimate only orders the jobs:
# simulated time grows with n^2 x density, and an SpMM run is about 8 SpMV runs.
for m in $MATS; do
  b=$(basename $m .bin); n=${b#*_n}; n=${n%%_*}; s=${b##*_s}; s=${s/p/.}
  for w in spmv spmm; do
    f=1; [ $w = spmm ] && f=8
    echo "$(awk -v n=$n -v s=$s -v f=$f 'BEGIN{printf "%d", f*n*n*(1.05-s/100)}') $w $m"
  done
done | sort -rn | cut -d' ' -f2- | xargs -P "${JOBS:-12}" -L 1 bash -c '
  w=$0; m=$1; b=$(basename $m .bin); d=runs_eval_$w/$b
  if grep -q SUCCESS $d/sim.log 2>/dev/null; then echo "skip $w $b"; exit 0; fi
  if [ $w = spmv ]; then EXTRA_CFLAGS=-DFINAL_EVAL ./build_run.sh $m $d >/dev/null 2>&1
  else PROG=spmm.c BIN2S_ARGS="--spmm 8" EXTRA_CFLAGS=-DFINAL_EVAL ./build_run.sh $m $d >/dev/null 2>&1; fi
  echo "done $w $b $(date +%T) PASS=$(grep -c " PASS" $d/sim.log) FAIL=$(grep -c " FAIL" $d/sim.log) $(grep -o "SUCCESS\|FAILED" $d/sim.log | head -1)"'

python3 summarize.py runs_eval_spmv
python3 summarize.py runs_eval_spmm
echo "ALL DONE $(date '+%F %T')"
