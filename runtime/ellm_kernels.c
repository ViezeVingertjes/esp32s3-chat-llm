#include "ellm_kernels.h"
#include <assert.h>
#include <math.h>
#include <string.h>

#ifndef ELLM_LUT_CHECK
#ifdef NDEBUG
#define ELLM_LUT_CHECK 0
#else
#define ELLM_LUT_CHECK 1
#endif
#endif

int ellm_pad20(int n) { return (n + 19) / 20 * 20; }
int ellm_pad16(int n) { return (n + 15) / 16 * 16; }
size_t ellm_lut_bytes(int n_pad) { return (size_t)(n_pad / 5) * 256 * sizeof(int16_t); }
size_t ellm_lut_q2_bytes(int n) { return (size_t)(ellm_pad16(n) / 4) * 256 * sizeof(int16_t); }
uint32_t ellm_t158_row_bytes(int cols_pad) { return (uint32_t)(cols_pad / 5); }
uint32_t ellm_q2_row_bytes(int cols_pad) { return (uint32_t)(cols_pad / 4); }

static float round_half_away(float v) {
    return v < 0 ? -floorf(-v + 0.5f) : floorf(v + 0.5f);
}

void ellm_act_quant(ellm_act *a, const float *x, int n) {
    a->n = n;
    a->n_pad = ellm_pad20(n);
    a->f = x;
    a->lut_type = ELLM_LUT_NONE;
    float absmax = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(x[i]);
        if (v > absmax) absmax = v;
    }
    if (absmax == 0.0f) {
        a->scale = 0.0f;
        memset(a->q, 0, (size_t)a->n_pad);
        return;
    }
    float inv = 127.0f / absmax;
    for (int i = 0; i < n; i++) {
        float r = round_half_away(x[i] * inv);
        if (r > 127.0f) r = 127.0f;
        if (r < -127.0f) r = -127.0f;
        a->q[i] = (int8_t)r;
    }
    for (int i = n; i < a->n_pad; i++) a->q[i] = 0;
    a->scale = absmax / 127.0f;
}

/* LUT layout: the four groups packed in one 32-bit weight word are interleaved,
 * entry c of group g lives at lut[256*(g & ~3) + 4*c + (g & 3)], so the kernel
 * reaches each lane with one addx8 plus a constant load offset.
 * Entry c = lo + 9*hi (lo = d0 + 3*d1, hi = d2 + 3*d3 + 9*d4) is A[lo] + B[hi]:
 * a 9-entry table over x0,x1 plus a 27-entry table over x2..x4. */
ELLM_HOT void ellm_act_build_lut(ellm_act *a) {
    int groups = a->n_pad / 5;
    for (int g = 0; g < groups; g++) {
        const int8_t *x = a->q + 5 * g;
        int16_t *T = a->lut + 256 * (g & ~3) + (g & 3);
        int x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4];
        int A0 = -x0 - x1, A1 = A0 + x0, A2 = A1 + x0;
        int A3 = A0 + x1, A4 = A1 + x1, A5 = A2 + x1;
        int A6 = A3 + x1, A7 = A4 + x1, A8 = A5 + x1;
        int B[27];
        B[0] = -x2 - x3 - x4;
        B[1] = B[0] + x2;
        B[2] = B[1] + x2;
        for (int j = 0; j < 3; j++) B[3 + j] = B[j] + x3;
        for (int j = 0; j < 3; j++) B[6 + j] = B[3 + j] + x3;
        for (int j = 0; j < 9; j++) B[9 + j] = B[j] + x4;
        for (int j = 0; j < 9; j++) B[18 + j] = B[9 + j] + x4;
        for (int hi = 0; hi < 27; hi++, T += 36) {
            int b = B[hi];
            T[0] = (int16_t)(b + A0); T[4] = (int16_t)(b + A1); T[8] = (int16_t)(b + A2);
            T[12] = (int16_t)(b + A3); T[16] = (int16_t)(b + A4); T[20] = (int16_t)(b + A5);
            T[24] = (int16_t)(b + A6); T[28] = (int16_t)(b + A7); T[32] = (int16_t)(b + A8);
        }
        for (int c = 0; c < 13; c++) T[4 * c] = 0; /* entries 243..255 */
    }
    a->lut_type = ELLM_T158;
}

/* Weight rows (and the Q2 LUT blocks below) are accessed as aligned 32-bit little-endian words:
 * T158/Q2 data must be 4-byte aligned (row_bytes = cols_pad/5 resp. cols_pad/4 is always a
 * multiple of 4), and so must ellm_act.lut. */
typedef uint32_t ellm_u32a __attribute__((may_alias));

/* Q2 LUT (layout in ellm_kernels.h): group g, byte b = c0 | c1 << 2 | c2 << 4 | c3 << 6,
 * T[b] = A[b & 15] + B[b >> 4] with A[c0 + 4 c1] = (2c0-3) x0 + (2c1-3) x1 and B likewise over
 * x2, x3; each partial table is built from its all-zero-code entry by +2x steps. */
static ELLM_HOT void q2_partials(const ellm_act *a, int g, int A[16], int B[16]) {
    const int n = a->n;
    int x[4];
    for (int i = 0; i < 4; i++) x[i] = 4 * g + i < n ? a->q[4 * g + i] : 0;
    A[0] = -3 * x[0] - 3 * x[1];
    B[0] = -3 * x[2] - 3 * x[3];
    for (int c = 1; c < 4; c++) {
        A[c] = A[c - 1] + 2 * x[0];
        B[c] = B[c - 1] + 2 * x[2];
    }
    for (int j = 4; j < 16; j++) {
        A[j] = A[j - 4] + 2 * x[1];
        B[j] = B[j - 4] + 2 * x[3];
    }
}

/* Lane bias of the Q2 tables (see ellm_kernels.h): lanes 0 and 2 hold T + K, lanes 1 and 3
 * hold T - K. A weight word looks up each lane once, so the biases cancel in every word. */
#define Q2_LANE_BIAS 4096
static inline int q2_bias(int g) { return (g & 1) ? -Q2_LANE_BIAS : Q2_LANE_BIAS; }

/* One group on its own (the ends of a range that does not start/stop on a 4-group block). */
static ELLM_HOT void q2_group(const ellm_act *a, int g) {
    int A[16], B[16];
    q2_partials(a, g, A, B);
    const int k = q2_bias(g);
    int16_t *T = a->lut + 256 * (g & ~3) + (g & 3);
    for (int hi = 0; hi < 16; hi++, T += 64) {
        int b = B[hi] + k;
        for (int lo = 0; lo < 16; lo++) T[4 * lo] = (int16_t)(b + A[lo]);
    }
}

/* A whole block of 4 groups g..g+3 (g % 4 == 0): entry b of the block is 4 int16 lanes, i.e.
 * two little-endian 32-bit words holding lanes (0,1) and (2,3). Each word is one 32-bit add of
 * packed partials: PA[p][lo] + PB[p][hi] = (v_even + K) + 65536 * (v_odd - K) mod 2^32. With
 * |v| <= 1524 < K = 4096 the low field v_even + K is in [0, 32767], so it never borrows from the
 * high field, and both fields are the int16 entries exactly. Four PA values stay in registers
 * while hi runs: ~2.25 instructions per 32-bit word (2 entries) instead of ~7 per entry for the
 * per-group loop. */
static ELLM_HOT void q2_block(const ellm_act *a, int g) {
    uint32_t PA[2][16], PB[2][16];
    for (int p = 0; p < 2; p++) {
        int A0[16], B0[16], A1[16], B1[16];
        q2_partials(a, g + 2 * p, A0, B0);
        q2_partials(a, g + 2 * p + 1, A1, B1);
        for (int j = 0; j < 16; j++) {
            PA[p][j] = (uint32_t)(A0[j] + Q2_LANE_BIAS) + ((uint32_t)(A1[j] - Q2_LANE_BIAS) << 16);
            PB[p][j] = (uint32_t)B0[j] + ((uint32_t)B1[j] << 16);
        }
    }
    ellm_u32a *T = (ellm_u32a *)(void *)(a->lut + 256 * g); /* word 2*b + p: entry b, lane pair p */
    for (int p = 0; p < 2; p++)
        for (int lq = 0; lq < 16; lq += 4) {
            const uint32_t a0 = PA[p][lq], a1 = PA[p][lq + 1], a2 = PA[p][lq + 2], a3 = PA[p][lq + 3];
            const uint32_t *pb = PB[p];
            ellm_u32a *t = T + 2 * lq + p;
            for (int hi = 0; hi < 16; hi++, t += 32) {
                uint32_t b = pb[hi];
                t[0] = a0 + b;
                t[2] = a1 + b;
                t[4] = a2 + b;
                t[6] = a3 + b;
            }
        }
}

ELLM_HOT void ellm_act_build_lut_q2_groups(const ellm_act *a, int g_lo, int g_hi) {
    int g = g_lo;
    for (; g < g_hi && (g & 3); g++) q2_group(a, g);
    for (; g + 4 <= g_hi; g += 4) q2_block(a, g);
    for (; g < g_hi; g++) q2_group(a, g);
}

void ellm_act_build_lut_q2(ellm_act *a) {
    ellm_act_build_lut_q2_groups(a, 0, ellm_pad16(a->n) / 4);
    a->lut_type = ELLM_Q2;
}

/* LUT sum for one weight word v (4 groups) with T at that word's LUT block. */
#define LUT_WORD(T, v) \
    ((T)[4 * ((v) & 0xFF)] + (T)[4 * (((v) >> 8) & 0xFF) + 1] + \
     (T)[4 * (((v) >> 16) & 0xFF) + 2] + (T)[4 * ((v) >> 24) + 3])

#if defined(__XTENSA__)
/* Xtensa inner loop, 8 weight bytes (40 T158 / 32 Q2 weights) per iteration: l32i the word,
 * extui each byte, addx8 into the interleaved LUT, l16si with the lane offset,
 * add. Two LUT bases (words 2i and 2i+1); loads are scheduled away from their
 * uses. GCC never wraps a hardware loop around inline asm, so loopgtz is safe.
 * Returns acc0 + acc1 and advances *pp / *pT past the processed words. */
static inline __attribute__((always_inline)) int32_t
lut_words8(const ellm_u32a **pp, const int16_t **pT, int n8) {
    const ellm_u32a *p = *pp;
    const int16_t *T = *pT, *T2 = T + 1024;
    int32_t acc0 = 0, acc1 = 0;
    uint32_t v0, v1, ta, tb, tc, td;
    __asm__ volatile(
        "loopgtz %[n], 1f\n\t"
        "l32i    %[v0], %[p], 0\n\t"
        "l32i    %[v1], %[p], 4\n\t"
        "extui   %[ta], %[v0], 0, 8\n\t"
        "extui   %[tb], %[v0], 8, 8\n\t"
        "addx8   %[ta], %[ta], %[T]\n\t"
        "addx8   %[tb], %[tb], %[T]\n\t"
        "l16si   %[ta], %[ta], 0\n\t"
        "l16si   %[tb], %[tb], 2\n\t"
        "extui   %[tc], %[v0], 16, 8\n\t"
        "extui   %[td], %[v0], 24, 8\n\t"
        "add     %[a0], %[a0], %[ta]\n\t"
        "add     %[a1], %[a1], %[tb]\n\t"
        "addx8   %[tc], %[tc], %[T]\n\t"
        "addx8   %[td], %[td], %[T]\n\t"
        "l16si   %[tc], %[tc], 4\n\t"
        "l16si   %[td], %[td], 6\n\t"
        "extui   %[ta], %[v1], 0, 8\n\t"
        "extui   %[tb], %[v1], 8, 8\n\t"
        "add     %[a0], %[a0], %[tc]\n\t"
        "add     %[a1], %[a1], %[td]\n\t"
        "addx8   %[ta], %[ta], %[T2]\n\t"
        "addx8   %[tb], %[tb], %[T2]\n\t"
        "l16si   %[ta], %[ta], 0\n\t"
        "l16si   %[tb], %[tb], 2\n\t"
        "extui   %[tc], %[v1], 16, 8\n\t"
        "extui   %[td], %[v1], 24, 8\n\t"
        "add     %[a0], %[a0], %[ta]\n\t"
        "add     %[a1], %[a1], %[tb]\n\t"
        "addx8   %[tc], %[tc], %[T2]\n\t"
        "addx8   %[td], %[td], %[T2]\n\t"
        "l16si   %[tc], %[tc], 4\n\t"
        "l16si   %[td], %[td], 6\n\t"
        "addi    %[p], %[p], 8\n\t"
        "addmi   %[T], %[T], 4096\n\t"
        "add     %[a0], %[a0], %[tc]\n\t"
        "add     %[a1], %[a1], %[td]\n\t"
        "addmi   %[T2], %[T2], 4096\n"
        "1:\n\t"
        : [p] "+r"(p), [T] "+r"(T), [T2] "+r"(T2), [a0] "+r"(acc0), [a1] "+r"(acc1),
          [v0] "=&r"(v0), [v1] "=&r"(v1), [ta] "=&r"(ta), [tb] "=&r"(tb),
          [tc] "=&r"(tc), [td] "=&r"(td)
        : [n] "r"(n8)
        : "memory");
    *pp = p;
    *pT = T;
    return acc0 + acc1;
}
#endif

/* T158 and Q2: one table lookup per weight byte (the LUT type matches w->type, see ellm_kernels.h). */
static ELLM_HOT void matvec_lut(const ellm_tensor *w, const ellm_act *a, float *out, int lo, int hi) {
    int nw = (int)(w->row_bytes / 4);
    int n8 = nw / 2;
    for (int r = lo; r < hi; r++) {
        const ellm_u32a *p = (const ellm_u32a *)(w->data + (size_t)r * w->row_bytes);
        const int16_t *T = a->lut;
#if defined(__XTENSA__)
        int32_t acc = lut_words8(&p, &T, n8);
#else
        int32_t acc0 = 0, acc1 = 0;
        for (int i = 0; i < n8; i++) {
            uint32_t v0 = p[0], v1 = p[1];
            p += 2;
            acc0 += LUT_WORD(T, v0);
            acc1 += LUT_WORD(T + 1024, v1);
            T += 2048;
        }
        int32_t acc = acc0 + acc1;
#endif
        if (nw & 1) acc += LUT_WORD(T, p[0]);
        out[r] = (float)acc * w->scale[r] * a->scale;
    }
}

static ELLM_HOT void matvec_i8(const ellm_tensor *w, const ellm_act *a, float *out, int lo, int hi) {
    for (int r = lo; r < hi; r++) {
        const int8_t *p = (const int8_t *)(w->data + (size_t)r * w->row_bytes);
        int32_t acc = 0;
        for (int i = 0; i < a->n_pad; i++) acc += (int32_t)p[i] * a->q[i];
        out[r] = (float)acc * w->scale[r] * a->scale;
    }
}

/* Four rows per pass: four independent accumulator chains (FP add latency) sharing each
 * activation load. Per-row summation order is still sequential over columns. */
static ELLM_HOT void matvec_f32(const ellm_tensor *w, const ellm_act *a, float *out, int lo, int hi) {
    const int n = w->cols;
    const float *x = a->f;
    int r = lo;
    for (; r + 4 <= hi; r += 4) {
        const float *p0 = (const float *)w->data + (size_t)r * n;
        const float *p1 = p0 + n, *p2 = p1 + n, *p3 = p2 + n;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < n; i++) {
            float xi = x[i];
            s0 += p0[i] * xi;
            s1 += p1[i] * xi;
            s2 += p2[i] * xi;
            s3 += p3[i] * xi;
        }
        out[r] = s0;
        out[r + 1] = s1;
        out[r + 2] = s2;
        out[r + 3] = s3;
    }
    for (; r < hi; r++) {
        const float *p = (const float *)w->data + (size_t)r * n;
        float acc = 0.0f;
        for (int i = 0; i < n; i++) acc += p[i] * x[i];
        out[r] = acc;
    }
}

ELLM_HOT void ellm_matvec(const ellm_tensor *w, const ellm_act *a, float *out, int row_lo, int row_hi) {
#if ELLM_LUT_CHECK
    assert(ellm_lut_ok(w, a)); /* once per call: a T158 table cannot serve a Q2 matrix, or vice versa */
#endif
    switch (w->type) {
    case ELLM_T158:
    case ELLM_Q2: matvec_lut(w, a, out, row_lo, row_hi); break;
    case ELLM_I8: matvec_i8(w, a, out, row_lo, row_hi); break;
    default: matvec_f32(w, a, out, row_lo, row_hi); break;
    }
}

void ellm_matvec_naive(const ellm_tensor *w, const ellm_act *a, float *out, int row_lo, int row_hi) {
    for (int r = row_lo; r < row_hi; r++) {
        if (w->type == ELLM_T158) {
            const uint8_t *p = w->data + (size_t)r * w->row_bytes;
            int32_t acc = 0;
            for (int c = 0; c < a->n_pad; c++) {
                int d = p[c / 5];
                for (int k = 0; k < c % 5; k++) d /= 3;
                d %= 3;
                acc += (d - 1) * (int32_t)a->q[c];
            }
            out[r] = (float)acc * w->scale[r] * a->scale;
        } else if (w->type == ELLM_Q2) {
            const uint8_t *p = w->data + (size_t)r * w->row_bytes;
            int32_t acc = 0;
            for (int c = 0; c < w->cols_pad; c++) {
                int code = (p[c / 4] >> (2 * (c % 4))) & 3;
                int32_t x = c < a->n ? a->q[c] : 0;
                acc += (2 * code - 3) * x;
            }
            out[r] = (float)acc * w->scale[r] * a->scale;
        } else if (w->type == ELLM_I8) {
            const int8_t *p = (const int8_t *)(w->data + (size_t)r * w->row_bytes);
            int32_t acc = 0;
            for (int c = 0; c < a->n_pad; c++) acc += (int32_t)p[c] * a->q[c];
            out[r] = (float)acc * w->scale[r] * a->scale;
        } else {
            matvec_f32(w, a, out, r, r + 1);
        }
    }
}
