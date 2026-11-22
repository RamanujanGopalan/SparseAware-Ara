// Standalone test of the new segmented-sum instructions (hardware steps 1-3).
//   vsegsum.vv  vd, vs2, v0.t   (int32, OPMVV funct6 010001)
//   vfsegsum.vv vd, vs2, v0.t   (fp32,  OPFVV funct6 010001)
// Rule: acc = 0; for i < vl: acc += vs2[i]; vd[i] = acc; if v0[i]: acc = 0
// Build modes (-DMODE=...):  0 = step 1, instruction just copies vs2 to vd
//                            1 = step 2, integer segmented sum
//                            2 = step 3, float segmented sum (must match the C loop bit for bit)
// Build and run: ./segsum_test.sh <mode> <outdir>
#include <stdint.h>
#include <string.h>
#include "printf.h"
#include "runtime.h"

#ifndef MODE
#define MODE 1
#endif
#define N 128   // VLMAX at e32, m1, VLEN 4096

// Encoding: funct6 | vm | vs2 | vs1 | funct3 | vd | 0x57.  vm = 0 means "v0 is used".
#define ENC(f3, vd, vs2) ((0x11u << 26) | (0u << 25) | ((vs2) << 20) | (0u << 15) | ((f3) << 12) | ((vd) << 7) | 0x57u)
#define STR_(x) #x
#define STR(x) STR_(x)
#define VSEGSUM_V8_V12  ENC(2, 8, 12)   // OPMVV
#define VFSEGSUM_V8_V12 ENC(1, 8, 12)   // OPFVV

#define L2 __attribute__((section(".l2"), aligned(64)))
static uint32_t in[N] L2, out[N] L2, ref[N] L2;
static uint8_t flags[N / 8] L2;

static uint32_t lcg = 12345;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int flag(int i) { return (flags[i / 8] >> (i % 8)) & 1; }

static void reference(int vl) {
    if (MODE == 0) { for (int i = 0; i < vl; i++) ref[i] = in[i]; return; }
    if (MODE == 1) { int32_t acc = 0;
        for (int i = 0; i < vl; i++) { acc += (int32_t)in[i]; ref[i] = (uint32_t)acc; if (flag(i)) acc = 0; } return; }
    float acc = 0.0f;   // MODE 2
    for (int i = 0; i < vl; i++) { acc += u2f(in[i]); ref[i] = f2u(acc); if (flag(i)) acc = 0.0f; }
}

static void run_hw(int vl) {
    for (int i = 0; i < N; i++) out[i] = 0xDEADBEEF;   // sentinel: elements >= vl must stay untouched
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(N));
    asm volatile("vle32.v v8, (%0)" ::"r"(out));          // vd starts as all sentinels
    asm volatile("vsetvli zero, %0, e32, m1, tu, mu" ::"r"(vl));
    asm volatile("vlm.v v0, (%0)" ::"r"(flags));          // tail flags
    asm volatile("vle32.v v12, (%0)" ::"r"(in));          // data
#if MODE == 2
    asm volatile(".word " STR(VFSEGSUM_V8_V12));
#else
    asm volatile(".word " STR(VSEGSUM_V8_V12));
#endif
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(N));
    asm volatile("vse32.v v8, (%0)" ::"r"(out) : "memory");
    asm volatile("fence");
}

static int check(const char *name, int vl) {
    reference(vl);
    run_hw(vl);
    int bad = -1, tail_bad = -1;
    for (int i = 0; i < vl; i++) if (out[i] != ref[i]) { bad = i; break; }
    for (int i = vl; i < N; i++) if (out[i] != 0xDEADBEEF) { tail_bad = i; break; }
    if (bad >= 0) printf("SEGSUM %s vl=%d FAIL at %d: got %x want %x\n", name, vl, bad, out[bad], ref[bad]);
    else if (tail_bad >= 0) printf("SEGSUM %s vl=%d FAIL tail element %d changed to %x\n", name, vl, tail_bad, out[tail_bad]);
    else printf("SEGSUM %s vl=%d PASS\n", name, vl);
    return bad < 0 && tail_bad < 0;
}

static void fill_data(int kind) {
    for (int i = 0; i < N; i++) {
        if (MODE == 2) in[i] = f2u(kind ? (float)((int)(rnd() % 2001) - 1000) / 64.0f : (float)(i + 1));
        else in[i] = kind ? rnd() : (uint32_t)(i + 1);
    }
}
static void fill_flags(int kind) {   // 0 none, 1 all, 2 last only, 3 every 3rd, 4 random
    for (int i = 0; i < N / 8; i++) flags[i] = 0;
    for (int i = 0; i < N; i++) {
        int f = kind == 1 || (kind == 3 && i % 3 == 2) || (kind == 4 && (rnd() & 3) == 0);
        if (f) flags[i / 8] |= 1 << (i % 8);
    }
    if (kind == 2) flags[N / 8 - 1] = 0x80;
}

int main(void) {
    printf("segsum_test MODE=%d\n", MODE);
    int ok = 1;
    static const int vls[] = {1, 7, 8, 9, 31, 64, 127, 128};
    static const char *fname[] = {"no_flags", "all_flags", "last_flag", "every3", "random_flags"};
    for (int fk = 0; fk < 5; fk++)
        for (int v = 0; v < 8; v++) {
            fill_data(v & 1); fill_flags(fk);
            if (fk == 2) { for (int i = 0; i < N / 8; i++) flags[i] = 0; flags[(vls[v] - 1) / 8] = 1 << ((vls[v] - 1) % 8); }
            ok &= check(fname[fk], vls[v]);
        }
    // example from the plan: [2 3 4 | 5 | 6 1] -> [2 5 9 5 6 7]
    for (int i = 0; i < 6; i++) in[i] = MODE == 2 ? f2u((float)"\2\3\4\5\6\1"[i]) : (uint32_t)"\2\3\4\5\6\1"[i];
    for (int i = 0; i < N / 8; i++) flags[i] = 0;
    flags[0] = (1 << 2) | (1 << 3) | (1 << 5);
    ok &= check("plan_example", 6);
    printf(ok ? "SEGSUM ALL PASS\n" : "SEGSUM SOME FAILED\n");
    return ok ? 0 : 1;
}
