// Second workload: sparse x dense matrix multiply (SpMM) on bare-metal Ara.
// Taken from the sparsity-dispatch branch; its two vle32.bp kernels are left out (they need that
// branch's simulator) and our segred_hw_mm (vfsegsum) is added.
//   C = A x B,  A: n x n sparse (the same matrices as SpMV),  B: n x K dense,  C: n x K.
// Each output C[i][j] is the dot product of row i of A with column j of B. B is stored column by
// column (data from bin2S.py --spmm K), so every column is laid out like SpMV's x.
// What SpMM adds over SpMV: the A data (row of A, or a window of non-zeros) is loaded once and
// reused for the K = 8 columns of B.
// Build: PROG=spmm.c BIN2S_ARGS="--spmm 8" ./build_run.sh <matrix.bin> <outdir>
// Needs Ara with vfsegsum (hardware/build) for segred_hw_mm. n <= 128.
// The dense kernels assume a square A; -DSPMM_SPARSE_ONLY skips them (used for non-square tests).
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>
#include "printf.h"
#include "runtime.h"

extern int32_t R, C, NZ, K, ROW_PTR[], COL_IDX[], COL_OFF[];
extern float VAL[], DENSE[], SPMM_B[], SPMM_CREF[], SPMM_C[];
// Segmented-reduction tables (bin2S.py, the same as for segred_hw in main.c)
extern int32_t SEG_NNZ, SEG_W, SEG_NWIN, SEG_INFO[];
extern uint8_t SEG_TAIL[];
extern uint32_t SEG_DOFF[];

typedef struct {
    int n, k, cols;                   // rows, columns of B, columns of A (= rows of B)
    const int32_t *rp, *ci, *co;      // CSR: row pointers, column indices, column byte offsets
    const float *val, *dense, *b;     // CSR values, dense copy of A, B column by column
} mm_t;

// ---------------------------------------------------------------- helpers (as in main.c)
// Move the sum in element 0 of a vector register to the scalar core (CVA6 waits for it)
#define READ_SUM(SUM, dst) { float t; asm volatile("vfmv.f.s %0, " SUM : "=f"(t)); dst = t; }
// One output element: row of A already in v4 (vl elements), LOAD_B puts the matching elements of
// the B column into v0; multiply and reduce into element 0 of SUM (cleared with vl = 1)
#define DOT(SUM, LOAD_B)                                                     \
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(1));             \
    asm volatile("vmv.v.i " SUM ", 0");                                    \
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vl));            \
    LOAD_B;                                                                  \
    asm volatile("vfmul.vv v12, v4, v0");                                    \
    asm volatile("vfredusum.vs " SUM ", v12, " SUM);
// Deferred read across all outputs (as in SpMV, #1): output p reduces into v16 or v24, and the
// sum of output p-1 is read from the other register only after output p's vector work is issued
#define OUT(LOAD_B, DST)                                                     \
    { if ((p & 1) == 0) { DOT("v16", LOAD_B); READ_SUM("v24", *pend); }      \
      else              { DOT("v24", LOAD_B); READ_SUM("v16", *pend); }      \
      pend = (DST); p++; }
#define FINAL_READ() \
    { if (p) { if (p & 1) READ_SUM("v16", *pend) else READ_SUM("v24", *pend); } }

// ---------------------------------------------------------------- kernels
// 1. Scalar CSR on CVA6: for each output, walk row i's non-zeros
__attribute__((noinline)) static void scalar_csr_mm(const mm_t *M, float *c) {
    const int n = M->n, k = M->k;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++) {
            const float *bj = M->b + (size_t)j * M->cols; float s = 0.0f;
            for (int q = M->rp[i]; q < M->rp[i + 1]; q++) s += M->val[q] * bj[M->ci[q]];
            c[i * k + j] = s;
        }
}

// 2. Baseline RVV: K separate runs of the midsem dense kernel rvv_dense (one per B column):
//    LMUL=1 strips, vfmacc into an accumulator, full-width reduction, immediate read
__attribute__((noinline)) static void rvv_dense_mm(const mm_t *M, float *c) {
    const int n = M->n, k = M->k;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int j = 0; j < k; j++) {
        const float *x = M->b + (size_t)j * n;
        for (int r = 0; r < n; r++) {
            const float *row = M->dense + (size_t)r * n;
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
            for (int c0 = 0; c0 < n;) {
                size_t vl = __riscv_vsetvl_e32m1(n - c0);
                vfloat32m1_t a = __riscv_vle32_v_f32m1(row + c0, vl);
                vfloat32m1_t xv = __riscv_vle32_v_f32m1(x + c0, vl);
                acc = __riscv_vfmacc_vv_f32m1_tu(acc, a, xv, vl);
                c0 += vl;
            }
            vfloat32m1_t s = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vlmax);
            c[r * k + j] = __riscv_vfmv_f_s_f32m1_f32(s);
        }
    }
}

// 3. Fair dense: row i of A loaded once (one LMUL=2 strip, n <= 256), then K outputs, each a
//    unit-stride load of a B column, multiply, reduce, deferred read
__attribute__((noinline)) static void ara_dense_mm(const mm_t *M, float *c) {
    const int n = M->n, k = M->k; const size_t vl = n;
    float dummy, *pend = &dummy; unsigned p = 0;
    for (int i = 0; i < n; i++) {
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vl));
        asm volatile("vle32.v v4, (%0)" ::"r"(M->dense + (size_t)i * n));
        for (int j = 0; j < k; j++)
            OUT(asm volatile("vle32.v v0, (%0)" ::"r"(M->b + (size_t)j * n)), c + i * k + j)
    }
    FINAL_READ();
}

// 4. Fair CSR: row i's values and column offsets loaded once, then per output a gather of the
//    B column at those offsets (vluxei32), multiply, reduce, deferred read. Empty rows: zeros
__attribute__((noinline)) static void ara_csr_mm(const mm_t *M, float *c) {
    const int n = M->n, k = M->k;
    float dummy, *pend = &dummy; unsigned p = 0;
    for (int i = 0; i < n; i++) {
        const int start = M->rp[i]; const size_t vl = M->rp[i + 1] - start;
        if (vl == 0) { for (int j = 0; j < k; j++) c[i * k + j] = 0.0f; continue; }
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vl));
        asm volatile("vle32.v v4, (%0)" ::"r"(M->val + start));      // values
        asm volatile("vle32.v v8, (%0)" ::"r"(M->co + start));       // byte offsets
        for (int j = 0; j < k; j++)
            OUT(asm volatile("vluxei32.v v0, (%0), v8" ::"r"(M->b + (size_t)j * M->cols)), c + i * k + j)
    }
    FINAL_READ();
}

// 5. Ours: segmented reduction with the new instruction vfsegsum (as segred_hw in main.c).
//    The flat non-zero list is cut into windows of SEG_W elements, ignoring row boundaries.
//    Per window, the A data is loaded ONCE: values, column offsets, tail flags (v0) and the
//    output offsets (row x K x 4 bytes, one vsll of SEG_DOFF). Then, for each column j of B:
//      gather B[:, j] -> multiply -> carry-in -> vfsegsum -> masked scatter of the row totals to
//      C[row][j] -> the last element becomes column j's carry for the next window.
//    The 8 carries are 8 vector variables, so the column loop is unrolled (needs K = 8).
static inline vfloat32m1_t vfsegsum(vfloat32m1_t p, vbool32_t tail, size_t vl) {
    vfloat32m1_t r;   // vfsegsum.vv vd, vs2, v0.t  (OPFVV funct6 010001), emitted with .insn
    asm volatile("vsetvli zero, %3, e32, m1, ta, ma\n\t"
                 ".insn r 0x57, 0x1, 0x22, %0, v0, %1"
                 : "=&vr"(r) : "vr"(p), "vm"(tail), "r"(vl) : "vl", "vtype");
    return r;
}
#define SEG_COL(J, CARRY)                                                                     \
    { vfloat32m1_t p = __riscv_vfmul_vv_f32m1(                                                \
          v, __riscv_vluxei32_v_f32m1(M->b + (size_t)(J) * M->cols, off, vl), vl);                  \
      if (carry_in) p = __riscv_vfadd_vv_f32m1_tu(p, p, CARRY, 1);   /* carry-in, element 0 */ \
      if (info[1]) p = vfsegsum(p, tail, vl);                         /* running row sums   */ \
      if (info[0]) __riscv_vsuxei32_v_f32m1_m(tail, c + (J), coff, p, vl); /* row totals -> C */ \
      if (info[2]) CARRY = __riscv_vslidedown_vx_f32m1(p, vl - 1, 1); }   /* carry-out       */
__attribute__((noinline)) static void segred_hw_mm(const mm_t *M, float *c) {
    const int n = M->n; const size_t W = SEG_W, wb = SEG_W / 8, tot = (size_t)n * 8;
    for (size_t r = 0; r < tot;) {                       // empty rows produce no result: clear C
        size_t vl = __riscv_vsetvl_e32m1(tot - r);
        __riscv_vse32_v_f32m1(c + r, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl); r += vl; }
    const int32_t *info = SEG_INFO;
    vfloat32m1_t c0 = __riscv_vfmv_v_f_f32m1(0.0f, W), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    int carry_in = 0;
    for (int w = 0; w < SEG_NWIN; w++, info += 3) {
        size_t s = (size_t)w * W, vl = __riscv_vsetvl_e32m1(SEG_NNZ - s);
        vfloat32m1_t v   = __riscv_vle32_v_f32m1(M->val + s, vl);                    // A values
        vuint32m1_t off  = __riscv_vle32_v_u32m1((const uint32_t *)M->co + s, vl);   // column offsets
        vbool32_t tail   = __riscv_vlm_v_b32(SEG_TAIL + (size_t)w * wb, vl);         // row-end flags
        vuint32m1_t coff = __riscv_vsll_vx_u32m1(__riscv_vle32_v_u32m1(SEG_DOFF + s, vl), 3, vl); // row*4 -> row*8*4
        SEG_COL(0, c0) SEG_COL(1, c1) SEG_COL(2, c2) SEG_COL(3, c3)
        SEG_COL(4, c4) SEG_COL(5, c5) SEG_COL(6, c6) SEG_COL(7, c7)
        carry_in = info[2];
    }
}

// ---------------------------------------------------------------- check and driver
static float fa(float v) { return v < 0 ? -v : v; }
// per output: error relative to sum |a_ij b_jk| (the SpMV check of main.c, per element)
static int check(const mm_t *M, const float *c) {
    for (int i = 0; i < M->n; i++)
        for (int j = 0; j < M->k; j++) {
            const float *bj = M->b + (size_t)j * M->cols; float scale = 1e-30f;
            for (int q = M->rp[i]; q < M->rp[i + 1]; q++) scale += fa(M->val[q] * bj[M->ci[q]]);
            if (fa(c[i * M->k + j] - SPMM_CREF[i * M->k + j]) / scale >= 1e-4f) return 0;
        }
    return 1;
}

int main(void) {
    mm_t M = {R, K, C, ROW_PTR, COL_IDX, COL_OFF, VAL, DENSE, SPMM_B};
    static const struct { const char *n; void (*f)(const mm_t *, float *); } KER[] = {
        {"scalar_csr_mm", scalar_csr_mm}, {"rvv_dense_mm", rvv_dense_mm}, {"ara_dense_mm", ara_dense_mm},
        {"ara_csr_mm", ara_csr_mm}, {"segred_hw_mm", segred_hw_mm}};
    printf("SPMM n=%d K=%d nnz=%d\n", R, K, NZ);
    if (K != 8) { printf("FAIL: segred_hw_mm needs K = 8\n"); return 1; }
    for (int q = 0; q < 5; q++) {
#ifdef SKIP_SCALAR
        if (q == 0) continue;
#endif
#ifdef SPMM_SPARSE_ONLY
        if (q == 1 || q == 2) continue;   // dense kernels assume a square A (n <= 256); for the non-square tests
#endif
        KER[q].f(&M, SPMM_C);                                       // warm-up, as in main.c
        // Fill C with a wrong value before the timed run, so a kernel that skips an output
        // (e.g. of an empty row) fails the check instead of passing on leftover zeros.
        for (int i = 0; i < R * K; i++) SPMM_C[i] = 12345.0f;
        HW_CNT_READY; start_timer(); KER[q].f(&M, SPMM_C); stop_timer(); HW_CNT_NOT_READY;
        printf("RES kernel=%s n=%d nnz=%d cycles=%d %s\n", KER[q].n, R, NZ, (int)get_timer(),
               check(&M, SPMM_C) ? "PASS" : "FAIL");
    }
    return 0;
}
