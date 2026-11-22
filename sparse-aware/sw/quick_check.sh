#!/usr/bin/env bash
# Quick check of the vfsegsum instruction (minutes): the directed instruction tests and both workloads on small matrices.
#   ./quick_check.sh <outdir>          (env ARA_ROOT = Ara clone with the vfsegsum simulator in hardware/build)
# 1. segsum_test.sh 1 and 2: 41 directed cases of vsegsum.vv (int) and vfsegsum.vv (float, bit-exact vs a C loop)
# 2. SpMV (main.c, -DFINAL_EVAL) and SpMM (spmm.c) on the 7 segmented-reduction edge cases and uniform n=32
# Every kernel must print PASS. SpMM: the dense kernels need a square matrix, so the non-square edge case runs
# with -DSPMM_SPARSE_ONLY; the two edge cases with rows longer than 256 non-zeros are run as SpMV only, because
# the SpMM CSR kernel (ara_csr_mm, from the team's SpMM code) loads a row in one register group (at most 256 elements).
cd "$(dirname "$0")"; O=$(mkdir -p "$1" && cd "$1" && pwd)
ARA=${ARA_ROOT:-$(cd ../.. && pwd)}; [ -f "$ARA/install/venv/bin/activate" ] && source "$ARA/install/venv/bin/activate"
python3 segred_cases.py mats >/dev/null
python3 ./gen_matrices.py -n 32 --structure uniform -o mats >/dev/null
./segsum_test.sh 1 "$O/segsum_int" | grep -E "SEGSUM|SUCCESS|FAILED"
./segsum_test.sh 2 "$O/segsum_float" | grep -E "SEGSUM|SUCCESS|FAILED"
export O
ls mats/segred_*.bin mats/uniform_n32_*.bin | while read m; do echo "spmv $m"
  case $(basename $m) in segred_longer_than_w.bin|segred_mixed.bin) ;; *) echo "spmm $m";; esac; done | xargs -P "${JOBS:-4}" -L 1 bash -c '
  w=$0; m=$1; b=$(basename $m .bin); d=$O/${w}_$b
  case $b in segred_ends_at_edge) SO=-DSPMM_SPARSE_ONLY;; *) SO=;; esac
  if [ $w = spmv ]; then EXTRA_CFLAGS=-DFINAL_EVAL ./build_run.sh $m $d >/dev/null 2>&1
  else PROG=spmm.c BIN2S_ARGS="--spmm 8" EXTRA_CFLAGS="-DFINAL_EVAL $SO" ./build_run.sh $m $d >/dev/null 2>&1; fi'
echo "kernel results: PASS=$(cat $O/sp*/sim.log | grep -c '^RES.* PASS') FAIL=$(cat $O/sp*/sim.log | grep -c '^RES.* FAIL')  simulations=$(ls -d $O/sp* | wc -l)"
