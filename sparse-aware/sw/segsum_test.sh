#!/usr/bin/env bash
# usage: segsum_test.sh <mode 0|1|2> <outdir> [simulator]
# env ARA_ROOT = Ara clone (default: this repository)
# Builds segsum_test.c (standalone test of the new segmented-sum instruction) and runs it on Ara RTL.
set -e
ARA=${ARA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; MODE=$1; O=$(mkdir -p "$2" && cd "$2" && pwd); SIM=${3:-${ARA_SIM:-$ARA/hardware/build/verilator/Vara_tb_verilator}}
SRC=$(cd "$(dirname "$0")" && pwd); CC=$ARA/install/riscv-llvm/bin/clang; C=$ARA/apps/common; mkdir -p "$O"
F="-DMODE=$MODE -march=rv64gcv_zfh_zvfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mno-implicit-float -mcmodel=medany -I$C -O2 -fno-common -fno-builtin-printf -DNR_LANES=4 -DVLEN=4096 -ffunction-sections -fdata-sections -std=gnu99 -w"
$CC $F -o "$O/app" "$SRC/segsum_test.c" $C/crt0.S $C/printf.c $C/string.c $C/serial.c $C/util.c -static -nostartfiles -lm -Wl,--gc-sections -T$C/link.ld
(cd "$O" && stdbuf -oL timeout ${TIMEOUT:-600} "$SIM" -l ram,"$O/app",elf > "$O/sim.log" 2>&1) || true   # run in $O so the trace file lands there
grep -E "SEGSUM|SUCCESS|FAILED|tohost" "$O/sim.log" | tail -50
