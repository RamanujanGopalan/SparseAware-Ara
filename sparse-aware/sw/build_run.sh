#!/usr/bin/env bash
# usage: build_run.sh <matrix.bin> <outdir> [nr_lanes vlen simulator]
# env EXTRA_CFLAGS, e.g. EXTRA_CFLAGS=-DSKIP_SCALAR (vector kernels only)
#                       EXTRA_CFLAGS=-DSEGRED_HW (only scalar_csr, ara_csr and segred_hw; needs Ara with vfsegsum)
#                       EXTRA_CFLAGS=-DFINAL_EVAL (final evaluation: scalar_csr, rvv_dense, ara_dense, ara_csr, segred_hw)
# env ARA_ROOT = Ara clone (default: this repository)
# env PROG picks the program (default main.c = SpMV; spmm.c = second workload), BIN2S_ARGS adds data
#     options (spmm.c needs BIN2S_ARGS="--spmm 8"), ARA_SIM picks another simulator
set -e
ARA=${ARA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; M=$1; O=$2; L=${3:-4}; V=${4:-4096}; SIM=${5:-${ARA_SIM:-$ARA/hardware/build/verilator/Vara_tb_verilator}}
SRC=$(cd "$(dirname "$0")" && pwd); CC=$ARA/install/riscv-llvm/bin/clang; C=$ARA/apps/common; mkdir -p "$O"
F="$EXTRA_CFLAGS -march=rv64gcv_zfh_zvfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float -mcmodel=medany -I$C -O3 -ffast-math -fno-common -fno-builtin-printf -DNR_LANES=$L -DVLEN=$V -ffunction-sections -fdata-sections -std=gnu99 -w"
ND=; case "$EXTRA_CFLAGS" in *ONLY_OPT*|*SEGRED_HW\ *|*SEGRED_HW) ND=--no-dense;; esac   # these modes run no dense kernel
python3 "$SRC/bin2S.py" "$M" $ND $BIN2S_ARGS --window $((V / 32)) > "$O/data.S"   # segred window = VLMAX at e32, LMUL 1
$CC $F -o "$O/app" "$SRC/${PROG:-main.c}" "$O/data.S" $C/crt0.S $C/printf.c $C/string.c $C/serial.c $C/util.c -static -nostartfiles -lm -Wl,--gc-sections -T$C/link.ld
"$SIM" -l ram,"$O/app",elf > "$O/sim.log" 2>&1
grep -E "^RES|SUCCESS|FAILED|Executed cycles|Wallclock" "$O/sim.log"
