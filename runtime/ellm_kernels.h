#ifndef ELLM_KERNELS_H
#define ELLM_KERNELS_H
#include <stddef.h>
#include <stdint.h>

#ifdef ESP_PLATFORM
#include "esp_attr.h"
#define ELLM_HOT IRAM_ATTR
#else
#define ELLM_HOT
#endif

typedef enum { ELLM_F32 = 0, ELLM_I8 = 1, ELLM_T158 = 2, ELLM_Q2 = 3 } ellm_wtype;

typedef struct {
    int32_t type, rows, cols, cols_pad;
    uint32_t row_bytes;
    const uint8_t *data; /* 64-byte aligned (pack invariant); T158: row_bytes = cols_pad/5, cols_pad % 20 == 0;
                            Q2: row_bytes = cols_pad/4, cols_pad % 16 == 0 (both: rows multiple of 4 bytes) */
    const float *scale; /* NULL for F32 */
} ellm_tensor;

typedef struct {
    int n, n_pad;
    float scale;
    int8_t *q;   /* caller-allocated, n_pad = pad20(n) entries (zero past n) */
    int16_t *lut; /* caller-allocated, keep 4-byte aligned: ellm_lut_bytes(n_pad) bytes for T158,
                     ellm_lut_q2_bytes(n) for Q2 (one LUT per activation vector, for one type) */
    const float *f;
    int32_t lut_type; /* ellm_wtype the table in lut was built for (ELLM_T158 / ELLM_Q2) or
                         ELLM_LUT_NONE: set by the full LUT builders, cleared by ellm_act_quant */
} ellm_act;

#define ELLM_LUT_NONE 0

/* 1 if ellm_matvec may run w on a: I8/F32 always; T158 needs a T158 table of a vector with
 * n_pad == w->cols_pad, Q2 a Q2 table with n == w->cols. ellm_matvec asserts it once per call
 * (not per row) unless ELLM_LUT_CHECK is 0 (default: 0 under NDEBUG, else 1). */
static inline int ellm_lut_ok(const ellm_tensor *w, const ellm_act *a) {
    if (w->type == ELLM_T158) return a->lut_type == ELLM_T158 && a->n_pad == w->cols_pad;
    if (w->type == ELLM_Q2) return a->lut_type == ELLM_Q2 && a->n == w->cols;
    return 1;
}

/* Matvec output (I8, T158, Q2): out[r] = (float)acc * w->scale[r] * a->scale, acc = exact int32
 * sum over the row of (integer weight) * a->q. Integer weights: I8 the byte, T158 trit t in
 * {-1,0,1}, Q2 2c-3 in {-3,-1,1,3} for the 2-bit code c (so Q2's per-row scale is half the
 * quantizer's absmean s: (2c-3) * s/2 = (c-1.5) * s).
 *
 * LUT kernels (T158, Q2): a row is read as aligned little-endian 32-bit words; each weight byte b
 * of group g (one byte per group: 5 activations for T158, 4 for Q2) adds one int16 table entry
 * T_g[b], so acc = sum_g T_g[row byte g]. The tables of the four groups of one word are
 * interleaved: entry b of group g lives at
 *     lut[256 * (g & ~3) + 4 * b + (g & 3)]
 * (a 1024-entry / 2 KiB block per 4 groups; on Xtensa one addx8 per byte plus a constant
 * 0/2/4/6 load offset reaches it). The same kernel loop serves both types; only the tables differ:
 *  - T158 (ellm_act_build_lut, groups = n_pad/5): T_g[b] = sum_i (d_i - 1) * q[5g+i], b = sum_i d_i 3^i
 *    for b < 243; entries 243..255 are 0.
 *  - Q2 (ellm_act_build_lut_q2, groups = pad16(n)/4): T_g[b] = sum_{i<4} (2 c_i - 3) * x_i with
 *    c_i = (b >> 2i) & 3 and x_i = q[4g+i] if 4g+i < n else 0 (activations past n are zero, so Q2
 *    padding contributes exactly 0 whatever its codes). Built per group from two 16-entry
 *    partial tables (x0,x1 and x2,x3) by repeated +2x steps: T_g[lo + 16 hi] = A[lo] + B[hi].
 *    |T| <= 4 * 3 * 127 = 1524. The stored entry carries a lane bias: lut[...] = T_g[b] + K for
 *    g & 1 == 0 and T_g[b] - K for g & 1 == 1 (K = 4096). Every weight word looks up each of its
 *    four lanes exactly once, so the biases cancel in each word and acc is exact; the bias lets
 *    the builder write two entries with one 32-bit add (no borrow between the int16 fields).
 *    ellm_matvec needs the Q2 LUT of a vector with a->n == w->cols. */
int ellm_pad20(int n);
int ellm_pad16(int n);
size_t ellm_lut_bytes(int n_pad);  /* T158 LUT for n_pad = pad20(n) activations */
size_t ellm_lut_q2_bytes(int n);   /* Q2 LUT for n activations: pad16(n)/4 groups * 512 bytes */
uint32_t ellm_t158_row_bytes(int cols_pad);
uint32_t ellm_q2_row_bytes(int cols_pad);
void ellm_act_quant(ellm_act *a, const float *x, int n); /* also clears a->lut_type */
void ellm_act_build_lut(ellm_act *a);    /* sets a->lut_type = ELLM_T158 */
void ellm_act_build_lut_q2(ellm_act *a); /* sets a->lut_type = ELLM_Q2 */
/* Q2 groups [g_lo, g_hi) only (a split of ellm_act_build_lut_q2, e.g. across cores; disjoint
 * ranges write disjoint entries; split on multiples of 4 to keep each core on whole blocks).
 * Does not tag a: the caller sets a->lut_type = ELLM_Q2 once all groups are built. */
void ellm_act_build_lut_q2_groups(const ellm_act *a, int g_lo, int g_hi);
void ellm_matvec(const ellm_tensor *w, const ellm_act *a, float *out, int row_lo, int row_hi);
void ellm_matvec_naive(const ellm_tensor *w, const ellm_act *a, float *out, int row_lo, int row_hi);

#endif
