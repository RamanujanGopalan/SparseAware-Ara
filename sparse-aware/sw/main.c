// Midsem kernels, copied unchanged from CompArch-Project/src/spmv.c, on bare-metal Ara.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>
#include "printf.h"
#include "runtime.h"
extern int32_t R, C, NZ, ROW_PTR[], COL_IDX[];
extern float VAL[], X[], Y_REF[], DENSE[], Y[];
extern int32_t COL_OFF[];   // COL_IDX * 4 (byte offsets), precomputed offline like Ara's apps/spmv data
typedef struct { int32_t rows, cols, nnz; int32_t *row_ptr, *col_idx; float *val, *x, *y_ref, *dense; } matrix_t;
static void scalar_csr(const matrix_t *A, float *y) {
    for (int r = 0; r < A->rows; r++) { float sum = 0.0f;
        for (int j = A->row_ptr[r]; j < A->row_ptr[r + 1]; j++) sum += A->val[j] * A->x[A->col_idx[j]];
        y[r] = sum; } }
static void rvv_csr(const matrix_t *A, float *y) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int r = 0; r < A->rows; r++) {
        int start = A->row_ptr[r], end = A->row_ptr[r + 1];
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        for (int j = start; j < end;) {
            size_t vl = __riscv_vsetvl_e32m1(end - j);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(A->val + j, vl);
            vuint32m1_t idx = __riscv_vle32_v_u32m1((const uint32_t *)A->col_idx + j, vl);
            vuint32m1_t off = __riscv_vsll_vx_u32m1(idx, 2, vl);
            vfloat32m1_t xv = __riscv_vluxei32_v_f32m1(A->x, off, vl);
            acc = __riscv_vfmacc_vv_f32m1_tu(acc, v, xv, vl);
            j += vl; }
        vfloat32m1_t s = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vlmax);
        y[r] = __riscv_vfmv_f_s_f32m1_f32(s); } }
static void scalar_dense(const matrix_t *A, float *y) {
    for (int r = 0; r < A->rows; r++) { const float *row = A->dense + (size_t)r * A->cols; float sum = 0.0f;
        for (int c = 0; c < A->cols; c++) sum += row[c] * A->x[c];
        y[r] = sum; } }
static void rvv_dense(const matrix_t *A, float *y) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int r = 0; r < A->rows; r++) {
        const float *row = A->dense + (size_t)r * A->cols;
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        for (int c = 0; c < A->cols;) {
            size_t vl = __riscv_vsetvl_e32m1(A->cols - c);
            vfloat32m1_t a = __riscv_vle32_v_f32m1(row + c, vl);
            vfloat32m1_t xv = __riscv_vle32_v_f32m1(A->x + c, vl);
            acc = __riscv_vfmacc_vv_f32m1_tu(acc, a, xv, vl);
            c += vl; }
        vfloat32m1_t s = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vlmax);
        y[r] = __riscv_vfmv_f_s_f32m1_f32(s); } }
// Same as rvv_csr, but clear and reduce only row-length elements instead of vlmax.
static void rvv_csr_vl(const matrix_t *A, float *y) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int r = 0; r < A->rows; r++) {
        int start = A->row_ptr[r], end = A->row_ptr[r + 1];
        if (start == end) { y[r] = 0.0f; continue; }
        size_t vl0 = __riscv_vsetvl_e32m1(end - start);
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl0);
        for (int j = start; j < end;) {
            size_t vl = __riscv_vsetvl_e32m1(end - j);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(A->val + j, vl);
            vuint32m1_t idx = __riscv_vle32_v_u32m1((const uint32_t *)A->col_idx + j, vl);
            vuint32m1_t off = __riscv_vsll_vx_u32m1(idx, 2, vl);
            vfloat32m1_t xv = __riscv_vluxei32_v_f32m1(A->x, off, vl);
            acc = __riscv_vfmacc_vv_f32m1_tu(acc, v, xv, vl);
            j += vl; }
        vfloat32m1_t s = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vl0);
        y[r] = __riscv_vfmv_f_s_f32m1_f32(s); } }
// ---- Optimizations taken from Ara's own kernel (ara/apps/spmv/kernel/spmv.c), on float32 ----
// 1. column indices stored as byte offsets (COL_OFF), so no vsll before the gather
// 2. LMUL=2: one strip covers up to 256 nonzeros (VLEN 4096, e32)
// 3. the row sum lives in element 0 of one register, cleared with vl=1 (not vlmax);
//    each strip is multiplied (vfmul) and reduced straight into it with vl = strip length
// 4. software pipelining: two sum registers (v16/v24) alternate between rows, and the scalar
//    read (vfmv.f.s, which stalls CVA6 until the reduction finishes) for row i-1 is issued only
//    after all of row i's vector instructions, so that reduction overlaps row i's loads.
// Written in inline asm, like the original, so the compiler cannot move the deferred read.
#define ROW_STRIPS(SUM)                                                              \
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(1));                     \
    asm volatile("vmv.v.i " SUM ", 0");                                            \
    while (len > 0) {                                                                \
        size_t vl;                                                                   \
        asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(vl) : "r"(len));       \
        asm volatile("vle32.v v4, (%0)" ::"r"(d));         /* values         */      \
        asm volatile("vle32.v v8, (%0)" ::"r"(ix));        /* byte offsets   */      \
        asm volatile("vluxei32.v v0, (%0), v8" ::"r"(x));  /* gather x       */      \
        asm volatile("vfmul.vv v12, v4, v0");                                        \
        asm volatile("vfredusum.vs " SUM ", v12, " SUM);   /* sum += strip   */      \
        len -= vl; d += vl; ix += vl;                                                \
    }
#define READ_SUM(SUM, dst) { float t; asm volatile("vfmv.f.s %0, " SUM : "=f"(t)); dst = t; }
// all four optimizations
__attribute__((noinline)) static void ara_csr(const matrix_t *A, float *y) {
    const float *x = A->x;
    for (int i = 0; i < A->rows; i++) {
        int len = A->row_ptr[i + 1] - A->row_ptr[i];
        const float *d = A->val + A->row_ptr[i]; const int32_t *ix = COL_OFF + A->row_ptr[i];
        if ((i & 1) == 0) { ROW_STRIPS("v16"); if (i) READ_SUM("v24", y[i - 1]); }
        else              { ROW_STRIPS("v24");        READ_SUM("v16", y[i - 1]); }
    }
    if (A->rows & 1) READ_SUM("v16", y[A->rows - 1]) else READ_SUM("v24", y[A->rows - 1]);
}
// optimizations 1-3 only: each row's sum is read right after its own reduction
__attribute__((noinline)) static void ara_csr_nopipe(const matrix_t *A, float *y) {
    const float *x = A->x;
    for (int i = 0; i < A->rows; i++) {
        int len = A->row_ptr[i + 1] - A->row_ptr[i];
        const float *d = A->val + A->row_ptr[i]; const int32_t *ix = COL_OFF + A->row_ptr[i];
        ROW_STRIPS("v16"); READ_SUM("v16", y[i]);
    }
}
// ---- Fair dense reference (ported from the sparsity-dispatch branch) ----
// Dense row as LMUL=2 strips: unit-stride loads of the row and of x (no gather), same
// multiply + reduce-into-element-0 as ROW_STRIPS.
#define DENSE_STRIPS(SUM)                                                            \
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(1));                     \
    asm volatile("vmv.v.i " SUM ", 0");                                            \
    for (size_t c = 0, vl; c < n; c += vl) {                                         \
        asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(vl) : "r"(n - c));     \
        asm volatile("vle32.v v4, (%0)" ::"r"(row + c));   /* A row strip    */      \
        asm volatile("vle32.v v0, (%0)" ::"r"(x + c));     /* x strip        */      \
        asm volatile("vfmul.vv v12, v4, v0");                                        \
        asm volatile("vfredusum.vs " SUM ", v12, " SUM);   /* sum += strip   */      \
    }
// rvv_dense rewritten Ara-style: LMUL=2 strips, reduce into element 0, and the same deferred
// read (two sum registers) as ara_csr. A dense kernel as well optimized as the best CSR kernel.
__attribute__((noinline)) static void ara_dense(const matrix_t *A, float *y) {
    const float *x = A->x; const size_t n = A->cols;   // local, so it is not reloaded every strip
    for (int i = 0; i < A->rows; i++) {
        const float *row = A->dense + i * n;
        if ((i & 1) == 0) { DENSE_STRIPS("v16"); if (i) READ_SUM("v24", y[i - 1]); }
        else              { DENSE_STRIPS("v24");        READ_SUM("v16", y[i - 1]); }
    }
    if (A->rows & 1) READ_SUM("v16", y[A->rows - 1]) else READ_SUM("v24", y[A->rows - 1]);
}
// ---- Segmented reduction with the new vfsegsum instruction ----
// The flat nonzero list is processed in windows of SEG_W elements, ignoring row boundaries.
// Inside a window, vfsegsum leaves each row's total at its last element; a scatter masked by
// the tail flags writes those totals to their rows. A row cut by the window end passes its
// partial sum (carry) to the next window.
// Tables come from bin2S.py (one-time preprocessing, not timed):
//   SEG_TAIL  1 bit per nonzero: last nonzero of its row
//   SEG_INFO  per window: {results k, non-zero if some row piece has 2+ elements, carry out 0/1}
//   SEG_DOFF  per nonzero: byte offset in y of its row (32-bit; 16-bit made Ara fault)
//   SEG_NNZ   nnz rounded up to a multiple of 8 (zero padding; odd vl made Ara fault)
extern int32_t SEG_NNZ, SEG_W, SEG_NWIN, SEG_INFO[];
extern uint8_t SEG_TAIL[];
extern uint32_t SEG_DOFF[];
// The new Ara instruction:
//   vfsegsum.vv vd, vs2, v0.t   (OPFVV, funct6 010001; acc += vs2[i]; vd[i] = acc; if v0[i]: acc = 0)
// The compiler does not know it, so it is emitted with .insn; the compiler still picks the registers
// (vd, vs2) and puts the tail flags in v0. vl is set inside the same asm so it cannot be stale.
#if (defined(SEGRED_HW) || defined(FINAL_EVAL)) && !defined(SEGRED_HW_ON)
#define SEGRED_HW_ON
#endif
static inline vfloat32m1_t vfsegsum(vfloat32m1_t p, vbool32_t tail, size_t vl) {
    vfloat32m1_t r;
    asm volatile("vsetvli zero, %3, e32, m1, ta, ma\n\t"
                 ".insn r 0x57, 0x1, 0x22, %0, v0, %1"
                 : "=&vr"(r) : "vr"(p), "vm"(tail), "r"(vl) : "vl", "vtype");
    return r;
}
__attribute__((noinline)) static void segred_hw(const matrix_t *A, float *y) {
    const size_t W = SEG_W, wb = SEG_W / 8;
    for (int r = 0; r < A->rows;) {                       // 1. empty rows produce no result: clear y
        size_t vl = __riscv_vsetvl_e32m1(A->rows - r);
        __riscv_vse32_v_f32m1(y + r, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl); r += vl; }
    const int32_t *info = SEG_INFO;
    vfloat32m1_t carry = __riscv_vfmv_v_f_f32m1(0.0f, W);
    int carry_in = 0;
    for (int w = 0; w < SEG_NWIN; w++, info += 3) {
        size_t s = (size_t)w * W, vl = __riscv_vsetvl_e32m1(SEG_NNZ - s);
        vfloat32m1_t v  = __riscv_vle32_v_f32m1(A->val + s, vl);           // 2. products
        vuint32m1_t off = __riscv_vle32_v_u32m1((const uint32_t *)COL_OFF + s, vl);
        vfloat32m1_t p  = __riscv_vfmul_vv_f32m1(v, __riscv_vluxei32_v_f32m1(A->x, off, vl), vl);
        if (carry_in) p = __riscv_vfadd_vv_f32m1_tu(p, p, carry, 1);        // 3. carry-in
        vbool32_t tail = __riscv_vlm_v_b32(SEG_TAIL + (size_t)w * wb, vl);
        if (info[1]) p = vfsegsum(p, tail, vl);                            // 4. segmented sum: one instruction
        if (info[0]) {                                                     // 5. row totals -> y
            vuint32m1_t doff = __riscv_vle32_v_u32m1(SEG_DOFF + s, vl);
            __riscv_vsuxei32_v_f32m1_m(tail, y, doff, p, vl); }
        carry_in = info[2];                                                // 6. carry-out
        if (carry_in) carry = __riscv_vslidedown_vx_f32m1(p, vl - 1, 1);
    }
}
static float fa(float v) { return v < 0 ? -v : v; }
// same check as midsem: per-row error relative to sum|a_ij x_j|, in ppm of that scale
static int check(const matrix_t *A, const float *y) {
    for (int r = 0; r < A->rows; r++) { float scale = 1e-30f;
        for (int j = A->row_ptr[r]; j < A->row_ptr[r + 1]; j++) scale += fa(A->val[j] * A->x[A->col_idx[j]]);
        if (fa(y[r] - A->y_ref[r]) / scale >= 1e-4f) return 0; }
    return 1; }
int main(void) {
    matrix_t A = {R, C, NZ, ROW_PTR, COL_IDX, VAL, X, Y_REF, DENSE};
    static const struct { const char *n; void (*f)(const matrix_t *, float *); } K[] = {
        {"scalar_csr", scalar_csr}, {"rvv_csr", rvv_csr}, {"scalar_dense", scalar_dense}, {"rvv_dense", rvv_dense}, {"rvv_csr_vl", rvv_csr_vl},
        {"ara_csr_nopipe", ara_csr_nopipe}, {"ara_csr", ara_csr}, {"segred_hw", segred_hw},
        {"ara_dense", ara_dense}};
    for (int k = 0; k < (int)(sizeof(K) / sizeof(K[0])); k++) {
#ifdef ONLY_OPT
        if (k < 4) continue;          // only rvv_csr_vl (reference) and the Ara-style kernels
#endif
#ifdef SKIP_SCALAR
        if (k == 0 || k == 2) continue;   // large n: vector kernels only
#endif
#ifdef SEGRED_HW
        if (k != 0 && k != 6 && k != 7) continue;   // scalar_csr, ara_csr, segred_hw
#endif
#ifdef FINAL_EVAL
        if (k != 0 && k != 3 && k != 6 && k != 7 && k != 8) continue;   // scalar_csr, rvv_dense (baseline), ara_csr, segred_hw, ara_dense
#endif
#ifndef SEGRED_HW_ON
        if (k == 7) continue;   // segred_hw needs the patched Ara; off unless built with -DSEGRED_HW_ON
#endif
        K[k].f(&A, Y);              // warm-up, as in midsem
        // Fill y with a wrong value before the timed run, so a kernel that skips a row
        // (e.g. an empty row) fails the check instead of passing on leftover zeros.
        for (int r = 0; r < A.rows; r++) Y[r] = 12345.0f;
        HW_CNT_READY; start_timer(); K[k].f(&A, Y); stop_timer(); HW_CNT_NOT_READY;  // flag gates the testbench counters
        printf("RES kernel=%s n=%d nnz=%d cycles=%d %s\n", K[k].n, A.rows, A.nnz, (int)get_timer(), check(&A, Y) ? "PASS" : "FAIL");
    }
    return 0;
}
