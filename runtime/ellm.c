/* Llama forward pass (plain C99, host + device). See ellm.h. */
#include "ellm.h"
#include <math.h>
#include <string.h>
#include "ellm_tok.h"
#ifdef ESP_PLATFORM
#include "sdkconfig.h"
#endif

/* ESP32-S3 PIE (SIMD) int8 dot product in the attention scores; portable C elsewhere. */
#if defined(__XTENSA__) && defined(CONFIG_IDF_TARGET_ESP32S3)
#define ELLM_PIE 1
#else
#define ELLM_PIE 0
#endif

void (*ellm_matvec_hook)(const ellm_tensor *, const ellm_act *, float *) = NULL;
void (*ellm_attn_hook)(const ellm_model *, ellm_state *, int, int) = NULL;
void (*ellm_lut_hook)(ellm_act *, int) = NULL;

#define TBIT(t) (1u << (unsigned)(t).type)
#define QUANT_TYPES ((1u << ELLM_I8) | (1u << ELLM_T158) | (1u << ELLM_Q2)) /* read ellm_act.q */
#define LUT_TYPES ((1u << ELLM_T158) | (1u << ELLM_Q2))                    /* read ellm_act.lut */

/* ---------------------------------------------------------------- loader */

#define ELLM_MAX_EXTENT (1 << 24) /* bound on dims/rows/cols: keeps all int arithmetic safe */

static const char *const slot_names[EL_N] = {"attn_norm", "wq", "wk", "wv", "wo",
                                             "ffn_norm", "w_gate", "w_up", "w_down"};

/* name -> tensor slot in m (NULL if unknown). */
static ellm_tensor *slot_for(ellm_model *m, const char *name, int n_layers) {
    if (strcmp(name, "embed") == 0) return &m->embed;
    if (strcmp(name, "norm_final") == 0) return &m->norm_final;
    if (name[0] != 'L' || name[1] < '0' || name[1] > '9') return NULL;
    int l = 0;
    const char *p = name + 1;
    while (*p >= '0' && *p <= '9') {
        l = l * 10 + (*p++ - '0');
        if (l >= n_layers) return NULL;
    }
    if (*p++ != '.') return NULL;
    for (int k = 0; k < EL_N; k++)
        if (strcmp(p, slot_names[k]) == 0) return &m->layer[l][k];
    return NULL;
}

/* Per-entry format invariants (ellm_pack.h) and payload bounds. */
static int set_tensor(ellm_tensor *t, const ellm_tensor_entry *e, const uint8_t *image,
                      const ellm_header *h) {
    uint64_t lo = h->payload_off, hi = (uint64_t)h->payload_off + h->payload_len;
    int max_type = h->version >= 2 ? ELLM_Q2 : ELLM_T158; /* Q2 from pack v2 on; BLOB: set_blob */
    if (e->type < ELLM_F32 || e->type > max_type || e->rows <= 0 || e->cols <= 0 ||
        e->rows > ELLM_MAX_EXTENT || e->cols > ELLM_MAX_EXTENT || e->cols_pad < 0)
        return ELLM_ERR_SHAPE;
    if (e->data_off % 64) return ELLM_ERR_SHAPE;
    switch (e->type) {
    case ELLM_F32:
        if (e->cols_pad != e->cols || e->row_bytes != 4ull * (uint64_t)e->cols || e->scale_off != 0)
            return ELLM_ERR_SHAPE;
        break;
    case ELLM_I8:
        if (e->cols_pad != ellm_pad20(e->cols) || e->row_bytes != (uint32_t)e->cols_pad) return ELLM_ERR_SHAPE;
        break;
    case ELLM_T158:
        if (e->cols_pad % 20 != 0 || e->cols_pad != ellm_pad20(e->cols) ||
            e->row_bytes != ellm_t158_row_bytes(e->cols_pad))
            return ELLM_ERR_SHAPE;
        break;
    default: /* Q2 */
        if (e->cols_pad % 16 != 0 || e->cols_pad != ellm_pad16(e->cols) ||
            e->row_bytes != ellm_q2_row_bytes(e->cols_pad))
            return ELLM_ERR_SHAPE;
        break;
    }
    if (e->type != ELLM_F32 && (e->scale_off == 0 || e->scale_off % 64)) return ELLM_ERR_SHAPE;
    if (e->data_off < lo || (uint64_t)e->data_off + (uint64_t)e->rows * e->row_bytes > hi)
        return ELLM_ERR_BOUNDS;
    if (e->scale_off && (e->scale_off < lo || (uint64_t)e->scale_off + 4ull * (uint64_t)e->rows > hi))
        return ELLM_ERR_BOUNDS;
    t->type = e->type;
    t->rows = e->rows;
    t->cols = e->cols;
    t->cols_pad = e->cols_pad;
    t->row_bytes = e->row_bytes;
    t->data = image + e->data_off;
    t->scale = e->scale_off ? (const float *)(const void *)(image + e->scale_off) : NULL;
    return 0;
}

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* The "tokenizer" entry (ellm_pack.h invariant 7): a v3 BLOB row inside the payload, framed as an
 * etok blob of this vocab (ellm_tok_init validates its contents). */
static int set_blob(ellm_model *m, const ellm_tensor_entry *e, const uint8_t *image, const ellm_header *h) {
    uint64_t lo = h->payload_off, hi = (uint64_t)h->payload_off + h->payload_len;
    if (h->version < 3 || e->type != ELLM_BLOB || e->rows != 1 || e->cols < ELLM_TOK_HDR_BYTES ||
        e->cols > ELLM_MAX_EXTENT || e->cols_pad != e->cols || e->row_bytes != (uint32_t)e->cols ||
        e->scale_off != 0 || e->data_off % 64)
        return ELLM_ERR_SHAPE;
    if (e->data_off < lo || (uint64_t)e->data_off + e->row_bytes > hi) return ELLM_ERR_BOUNDS;
    const uint8_t *b = image + e->data_off;
    if (memcmp(b, ELLM_TOK_MAGIC, 4) != 0 || rd_u32(b + 8) != e->row_bytes || rd_u32(b + 12) != (uint32_t)h->vocab)
        return ELLM_ERR_SHAPE;
    m->tok_blob = b;
    m->tok_len = e->row_bytes;
    return 0;
}

static int shape_is(const ellm_tensor *t, int rows, int cols) {
    return t->data != NULL && t->rows == rows && t->cols == cols;
}

int ellm_model_init(ellm_model *m, const uint8_t *image, size_t len) {
    memset(m, 0, sizeof *m);
    if (len < sizeof(ellm_header) || ((uintptr_t)image & 63u)) return ELLM_ERR_BOUNDS;
    ellm_header *h = &m->h;
    memcpy(h, image, sizeof *h);
    if (h->magic != ELLM_MAGIC) return ELLM_ERR_MAGIC;
    if (h->version < 1 || h->version > ELLM_VERSION) return ELLM_ERR_VERSION; /* v1, v2 or v3 */
    uint64_t payload_end = (uint64_t)h->payload_off + h->payload_len;
    if (h->payload_off < sizeof(ellm_header) || payload_end > len) return ELLM_ERR_BOUNDS;
    /* CRC over [0, payload_end) with the header's crc32 field zeroed (ellm_pack.h). */
    ellm_header h0 = *h;
    h0.crc32 = 0;
    uint32_t crc = ellm_crc32(0, (const uint8_t *)&h0, sizeof h0);
    crc = ellm_crc32(crc, image + sizeof h0, (size_t)(payload_end - sizeof h0));
    if (crc != h->crc32) return ELLM_ERR_CRC;

    if (h->n_layers < 1 || h->n_layers > ELLM_MAX_LAYERS || h->dim <= 0 || h->dim > ELLM_MAX_EXTENT ||
        h->hidden <= 0 || h->hidden > ELLM_MAX_EXTENT || h->vocab <= 0 || h->vocab > ELLM_MAX_EXTENT ||
        h->seq_len <= 0 || h->seq_len > ELLM_MAX_EXTENT || h->n_heads <= 0 || h->n_heads > h->dim ||
        h->n_kv_heads <= 0 || h->n_kv_heads > h->n_heads || h->n_heads % h->n_kv_heads != 0 ||
        h->head_dim <= 0 || h->head_dim % 2 != 0 || h->head_dim > ELLM_MAX_HEAD_DIM ||
        (h->kv_type != 0 && h->kv_type != 1) || !(h->norm_eps > 0.0f) || !(h->rope_theta > 0.0f) ||
        (h->n_tensors != 2 + 9 * h->n_layers && !(h->version >= 3 && h->n_tensors == 3 + 9 * h->n_layers)))
        return ELLM_ERR_SHAPE;
    int64_t q_dim64 = (int64_t)h->n_heads * h->head_dim, kv_dim64 = (int64_t)h->n_kv_heads * h->head_dim;
    if (q_dim64 > ELLM_MAX_EXTENT || kv_dim64 > ELLM_MAX_EXTENT) return ELLM_ERR_SHAPE;
    uint64_t table_end = (uint64_t)h->table_off + (uint64_t)h->n_tensors * sizeof(ellm_tensor_entry);
    if (h->table_off < sizeof(ellm_header) || table_end > h->payload_off) return ELLM_ERR_BOUNDS;

    for (int i = 0; i < h->n_tensors; i++) {
        ellm_tensor_entry e;
        memcpy(&e, image + h->table_off + (size_t)i * sizeof e, sizeof e);
        char name[sizeof e.name + 1];
        memcpy(name, e.name, sizeof e.name);
        name[sizeof e.name] = '\0';
        int r;
        if (strcmp(name, "tokenizer") == 0) {
            if (m->tok_blob != NULL) return ELLM_ERR_SHAPE; /* duplicate */
            r = set_blob(m, &e, image, h);
        } else {
            ellm_tensor *t = slot_for(m, name, h->n_layers);
            if (t == NULL || t->data != NULL) return ELLM_ERR_SHAPE; /* unknown or duplicate */
            r = set_tensor(t, &e, image, h);
        }
        if (r) return r;
    }

    int d = h->dim, q_dim = (int)q_dim64, kv_dim = (int)kv_dim64;
    if (!shape_is(&m->embed, h->vocab, d) || !shape_is(&m->norm_final, 1, d) ||
        m->norm_final.type != ELLM_F32)
        return ELLM_ERR_SHAPE;
    for (int l = 0; l < h->n_layers; l++) {
        const ellm_tensor *L = m->layer[l];
        if (!shape_is(&L[EL_ATTN_NORM], 1, d) || L[EL_ATTN_NORM].type != ELLM_F32 ||
            !shape_is(&L[EL_FFN_NORM], 1, d) || L[EL_FFN_NORM].type != ELLM_F32 ||
            !shape_is(&L[EL_WQ], q_dim, d) || !shape_is(&L[EL_WK], kv_dim, d) ||
            !shape_is(&L[EL_WV], kv_dim, d) || !shape_is(&L[EL_WO], d, q_dim) ||
            !shape_is(&L[EL_W_GATE], h->hidden, d) || !shape_is(&L[EL_W_UP], h->hidden, d) ||
            !shape_is(&L[EL_W_DOWN], d, h->hidden))
            return ELLM_ERR_SHAPE;
        /* q/k/v and gate/up each read one activation vector with one LUT: T158 and Q2 can't share */
        unsigned qkv = TBIT(L[EL_WQ]) | TBIT(L[EL_WK]) | TBIT(L[EL_WV]), gu = TBIT(L[EL_W_GATE]) | TBIT(L[EL_W_UP]);
        if ((qkv & LUT_TYPES) == LUT_TYPES || (gu & LUT_TYPES) == LUT_TYPES) return ELLM_ERR_SHAPE;
    }

    int mx = d > q_dim ? d : q_dim;
    if (h->hidden > mx) mx = h->hidden;
    m->act_npad = ellm_pad20(mx);
    unsigned types = TBIT(m->embed); /* matvec weights (norms are F32) */
    for (int l = 0; l < h->n_layers; l++)
        for (int k = 0; k < EL_N; k++) types |= TBIT(m->layer[l][k]);
    m->need_q = (types & QUANT_TYPES) != 0;
    m->need_lut = (types & LUT_TYPES) != 0;
    size_t lb = (types & (1u << ELLM_T158)) ? ellm_lut_bytes(m->act_npad) : 0;
    size_t lb2 = (types & (1u << ELLM_Q2)) ? ellm_lut_q2_bytes(mx) : 0;
    m->lut_bytes = lb > lb2 ? lb : lb2;

    /* HF LlamaRotaryEmbedding (default rope): inv_freq[i] = 1 / theta^(2i/head_dim), float32.
     * pow in double, rounded once: platform-independent (no float libm), mirrored by quantsim. */
    for (int i = 0; i < h->head_dim / 2; i++)
        m->inv_freq[i] = 1.0f / (float)pow((double)h->rope_theta, (double)((float)(2 * i) / (float)h->head_dim));
    return 0;
}

/* ---------------------------------------------------------------- forward */

/* exp(x) in float32 from +, -, *, floorf and exponent bits only (Cephes expf polynomial, ~1 ulp;
 * 0 below -87, +inf above 88), so it is bit-identical on any IEEE float32 target without FMA
 * contraction and is mirrored exactly by ellm_py.quantsim.expf_c (libm expf differs between glibc,
 * newlib and correctly rounded exp at the ulp level, which the int8 model amplifies). */
float ellm_expf(float x) {
    if (x != x) return x; /* NaN */
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return INFINITY;
    /* k = floorf(x / ln2 + 0.5) in [-125, 127], as an exact integer floor (newlib's floorf is a
     * slow libm call on the device; the value is identical) */
    float y0 = x * 1.44269504088896341f + 0.5f;
    int ki = (int)y0;
    if ((float)ki > y0) ki--;
    float k = (float)ki;
    float r = x - k * 0.693359375f; /* Cody-Waite: ln2 = C1 + C2 */
    r = r - k * -2.12194440e-4f;
    float z = r * r;
    float y = ((((1.9875691500e-4f * r + 1.3981999507e-3f) * r + 8.3334519073e-3f) * r +
                4.1665795894e-2f) * r + 1.6666665459e-1f) * r + 5.0000001201e-1f;
    y = y * z + r + 1.0f;
    union { uint32_t u; float f; } p2;
    p2.u = (uint32_t)(ki + 127) << 23; /* 2^k, k normal */
    return y * p2.f;
}

/* ellm_expf of v[0..4) in place, the four lanes interleaved operation by operation (the FPU's
 * latency hides behind the independent lanes); every lane is the scalar's exact op sequence, so
 * the values are ellm_expf's bit for bit. Lanes outside the polynomial's range (or NaN) take the
 * scalar path. */
static void expf4(float *v) {
    float x0 = v[0], x1 = v[1], x2 = v[2], x3 = v[3];
    if (!(x0 >= -87.0f && x0 <= 88.0f && x1 >= -87.0f && x1 <= 88.0f && x2 >= -87.0f && x2 <= 88.0f &&
          x3 >= -87.0f && x3 <= 88.0f)) {
        v[0] = ellm_expf(x0);
        v[1] = ellm_expf(x1);
        v[2] = ellm_expf(x2);
        v[3] = ellm_expf(x3);
        return;
    }
    const float L2E = 1.44269504088896341f, C1 = 0.693359375f, C2 = -2.12194440e-4f;
    float y0 = x0 * L2E + 0.5f, y1 = x1 * L2E + 0.5f, y2 = x2 * L2E + 0.5f, y3 = x3 * L2E + 0.5f;
    int k0 = (int)y0, k1 = (int)y1, k2 = (int)y2, k3 = (int)y3;
    if ((float)k0 > y0) k0--;
    if ((float)k1 > y1) k1--;
    if ((float)k2 > y2) k2--;
    if ((float)k3 > y3) k3--;
    float f0 = (float)k0, f1 = (float)k1, f2 = (float)k2, f3 = (float)k3;
    float r0 = x0 - f0 * C1, r1 = x1 - f1 * C1, r2 = x2 - f2 * C1, r3 = x3 - f3 * C1;
    r0 = r0 - f0 * C2;
    r1 = r1 - f1 * C2;
    r2 = r2 - f2 * C2;
    r3 = r3 - f3 * C2;
    float z0 = r0 * r0, z1 = r1 * r1, z2 = r2 * r2, z3 = r3 * r3;
    float p0 = 1.9875691500e-4f * r0 + 1.3981999507e-3f, p1 = 1.9875691500e-4f * r1 + 1.3981999507e-3f;
    float p2 = 1.9875691500e-4f * r2 + 1.3981999507e-3f, p3 = 1.9875691500e-4f * r3 + 1.3981999507e-3f;
    p0 = p0 * r0 + 8.3334519073e-3f;
    p1 = p1 * r1 + 8.3334519073e-3f;
    p2 = p2 * r2 + 8.3334519073e-3f;
    p3 = p3 * r3 + 8.3334519073e-3f;
    p0 = p0 * r0 + 4.1665795894e-2f;
    p1 = p1 * r1 + 4.1665795894e-2f;
    p2 = p2 * r2 + 4.1665795894e-2f;
    p3 = p3 * r3 + 4.1665795894e-2f;
    p0 = p0 * r0 + 1.6666665459e-1f;
    p1 = p1 * r1 + 1.6666665459e-1f;
    p2 = p2 * r2 + 1.6666665459e-1f;
    p3 = p3 * r3 + 1.6666665459e-1f;
    p0 = p0 * r0 + 5.0000001201e-1f;
    p1 = p1 * r1 + 5.0000001201e-1f;
    p2 = p2 * r2 + 5.0000001201e-1f;
    p3 = p3 * r3 + 5.0000001201e-1f;
    p0 = p0 * z0 + r0 + 1.0f;
    p1 = p1 * z1 + r1 + 1.0f;
    p2 = p2 * z2 + r2 + 1.0f;
    p3 = p3 * z3 + r3 + 1.0f;
    union { uint32_t u; float f; } e0, e1, e2, e3;
    e0.u = (uint32_t)(k0 + 127) << 23;
    e1.u = (uint32_t)(k1 + 127) << 23;
    e2.u = (uint32_t)(k2 + 127) << 23;
    e3.u = (uint32_t)(k3 + 127) << 23;
    v[0] = p0 * e0.f;
    v[1] = p1 * e1.f;
    v[2] = p2 * e2.f;
    v[3] = p3 * e3.f;
}

/* Prepare the matvec input x[0..n) for weights of the given types (types: bit (1 << ellm_wtype)
 * per consumer). F32 consumers read a->f = x; if any consumer is I8/T158/Q2, x is quantized once
 * into s->xq (and the T158 or Q2 LUT built into s->lut if a consumer needs it; the loader
 * rejects packs where one vector feeds both), shared by all consumers. */
static void prep_act(ellm_act *a, ellm_state *s, const float *x, int n, unsigned types) {
    if (types & QUANT_TYPES) {
        a->q = s->xq;
        a->lut = s->lut;
        ellm_act_quant(a, x, n);
        int lt = (types & (1u << ELLM_T158)) ? ELLM_T158 : (types & (1u << ELLM_Q2)) ? ELLM_Q2 : ELLM_LUT_NONE;
        if (lt != ELLM_LUT_NONE && ellm_lut_hook)
            ellm_lut_hook(a, lt);
        else if (lt == ELLM_T158)
            ellm_act_build_lut(a);
        else if (lt == ELLM_Q2)
            ellm_act_build_lut_q2(a);
    } else {
        a->n = n;
        a->n_pad = n;
        a->scale = 0.0f;
        a->q = NULL;
        a->lut = NULL;
        a->f = x;
    }
}

static void matvec(const ellm_tensor *w, const ellm_act *a, float *out) {
    if (ellm_matvec_hook)
        ellm_matvec_hook(w, a, out);
    else
        ellm_matvec(w, a, out, 0, w->rows);
}

/* HF LlamaRMSNorm: x * rsqrt(mean(x^2) + eps) * w (out may alias x). */
static void rmsnorm(float *out, const float *x, const ellm_tensor *w, int n, float eps) {
    const float *g = (const float *)(const void *)w->data;
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = g[i] * (x[i] * r);
}

/* HF rotate_half: pairs (i, i + hd/2): v_i' = v_i c - v_{i+h} s, v_{i+h}' = v_{i+h} c + v_i s. */
static void rope(float *v, int n_heads, int hd, const float *c, const float *s) {
    int half = hd / 2;
    for (int h = 0; h < n_heads; h++, v += hd) {
        for (int i = 0; i < half; i++) {
            float x0 = v[i], x1 = v[i + half];
            v[i] = x0 * c[i] - x1 * s[i];
            v[i + half] = x1 * c[i] + x0 * s[i];
        }
    }
}

/* Embedding row `token` as float: F32 copy, I8/T158/Q2 dequantized as (float)q * scale[token]
 * (Q2: q = 2c - 3 for the 2-bit code c). */
static void embed_row(const ellm_tensor *e, int token, float *x) {
    const uint8_t *row = e->data + (size_t)token * e->row_bytes;
    int n = e->cols;
    if (e->type == ELLM_F32) {
        memcpy(x, row, sizeof(float) * (size_t)n);
    } else if (e->type == ELLM_I8) {
        float sc = e->scale[token];
        for (int i = 0; i < n; i++) x[i] = (float)((const int8_t *)row)[i] * sc;
    } else if (e->type == ELLM_Q2) {
        float sc = e->scale[token];
        for (int i = 0; i < n; i++) x[i] = (float)(2 * ((row[i / 4] >> (2 * (i % 4))) & 3) - 3) * sc;
    } else {
        float sc = e->scale[token];
        for (int g = 0; g * 5 < n; g++) {
            int d = row[g];
            for (int i = 0; i < 5 && g * 5 + i < n; i++, d /= 3) x[g * 5 + i] = (float)(d % 3 - 1) * sc;
        }
    }
}

/* Per-(pos, kv_head) int8 KV: quantize n_kv slices of hd floats (activation formula). */
static void kv_quant(int8_t *q, float *scale, const float *x, int n_kv, int hd) {
    int8_t tmp[ELLM_MAX_HEAD_DIM + 20]; /* ellm_act_quant zero-fills up to pad20(hd) */
    ellm_act a;
    a.q = tmp;
    a.lut = NULL;
    for (int h = 0; h < n_kv; h++) {
        ellm_act_quant(&a, x + h * hd, hd);
        memcpy(q + h * hd, tmp, (size_t)hd);
        scale[h] = a.scale;
    }
}

/* ---------------------------------------------------------------- attention */

/* Query heads per pass: heads of one pass share a KV head, so each K row is read once and each V
 * row converted to float once for all of them. Bounds the stack (rows, int8 q, per-head scalars). */
#define ATT_CHUNK 8

/* Bytes per head of the int8 query buffer: pad20(head_dim) (ellm_act_quant's zero padding),
 * rounded up to 16 so every head's vector stays 16-byte aligned. */
#define ATT_QSTRIDE ((((ELLM_MAX_HEAD_DIM + 19) / 20 * 20) + 15) / 16 * 16)

static void row_to_float(float *d, const int8_t *v, int n) {
    for (int i = 0; i < n; i++) d[i] = (float)v[i];
}

/* Exact int8 dot product sum_i a[i] * b[i] (|sum| <= n * 127^2 < 2^31). On the ESP32-S3 the
 * PIE SIMD unit does 16 multiply-accumulates per instruction into its 40-bit ACCX register (16-byte
 * aligned vectors, n % 16 == 0); integer sums are exact in any order, so the result is the
 * portable loop's, bit for bit. */
static int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
#if ELLM_PIE
    if (((((uintptr_t)a | (uintptr_t)b) & 15u) | ((unsigned)n & 15u)) == 0) {
        int32_t r;
        /* GCC never wraps a hardware loop around inline asm, so loopgtz is safe; q0/q1/ACCX are
         * per-task PIE state that IDF saves on context switches. Both are toolchain/IDF facts:
         * after an upgrade, re-check the disassembly or re-run compare_device (bit-exact). */
        __asm__ volatile("ee.zero.accx\n\t"
                         "loopgtz %[n], 1f\n\t"
                         "ee.vld.128.ip q0, %[a], 16\n\t"
                         "ee.vld.128.ip q1, %[b], 16\n\t"
                         "ee.vmulas.s8.accx q0, q1\n"
                         "1:\n\t"
                         "rur.accx_0 %[r]\n\t"
                         : [r] "=r"(r), [a] "+r"(a), [b] "+r"(b)
                         : [n] "r"(n / 16)
                         : "memory");
        return r;
    }
#endif
    int32_t acc = 0;
    for (int i = 0; i < n; i++) acc += (int32_t)a[i] * b[i];
    return acc;
}

const char *ellm_attn_simd_off_reason(const ellm_model *m, const ellm_state *s) {
    if (m->h.kv_type != 1) return "float KV cache (kv_type 0): no int8 q.k";
    if (m->h.head_dim % 16 != 0) return "head_dim not a multiple of 16";
    if (s->key_q == NULL || ((uintptr_t)s->key_q & 15u) != 0) return "key_q not 16-byte aligned";
    if (!ELLM_PIE) return "not an ESP32-S3 build (portable C dot product)";
    return NULL;
}

int ellm_attn_simd_ok(const ellm_model *m, const ellm_state *s) { return ellm_attn_simd_off_reason(m, s) == NULL; }

/* Scores of heads [ha, ha + nc) (all reading KV head kh) for positions 0..n-1 into att rows. */
static void attn_scores(const ellm_model *m, const ellm_state *s, size_t lrow, int kh, int ha, int nc, int n) {
    const ellm_header *h = &m->h;
    const int hd = h->head_dim, n_kv = h->n_kv_heads, kv_dim = n_kv * hd, seq = s->seq_len;
    const float att_scale = 1.0f / sqrtf((float)hd);
    if (h->kv_type == 0) {
        const float *kc = s->key_cache + lrow * kv_dim + (size_t)kh * hd;
        for (int j = 0; j < nc; j++) {
            const float *qh = s->q + (size_t)(ha + j) * hd;
            float *att = s->att + (size_t)(ha + j) * seq;
            for (int t = 0; t < n; t++) {
                const float *k = kc + (size_t)t * kv_dim;
                float dot = 0.0f;
                for (int i = 0; i < hd; i++) dot += qh[i] * k[i];
                att[t] = dot * att_scale;
            }
        }
        return;
    }
    /* int8 K: each query head is quantized to int8 like a matvec input (ellm_act_quant over
     * head_dim: absmax / 127, round half away), the dot product is exact in int32, and
     * score = ((float)acc * key_s) * (q_s * att_scale). */
    unsigned char qbuf[ATT_CHUNK * ATT_QSTRIDE + 15];
    int8_t *qq = (int8_t *)(void *)(qbuf + ((16u - ((uintptr_t)qbuf & 15u)) & 15u)); /* 16-aligned */
    float cq[ATT_CHUNK];
    for (int j = 0; j < nc; j++) {
        ellm_act a;
        a.q = qq + j * ATT_QSTRIDE;
        a.lut = NULL;
        ellm_act_quant(&a, s->q + (size_t)(ha + j) * hd, hd);
        cq[j] = a.scale * att_scale;
    }
    const int8_t *kq = s->key_q + lrow * kv_dim + (size_t)kh * hd;
    const float *ks = s->key_s + lrow * n_kv + kh;
    for (int t = 0; t < n; t++) {
        const int8_t *k = kq + (size_t)t * kv_dim;
        const float st = ks[(size_t)t * n_kv];
        for (int j = 0; j < nc; j++) {
            const int32_t acc = dot_i8(qq + j * ATT_QSTRIDE, k, hd);
            s->att[(size_t)(ha + j) * seq + t] = ((float)acc * st) * cq[j];
        }
    }
}

/* Softmax numerator in place over att[0..n): exp(att[t] - max); returns 1 / sum (sum sequential). */
static float softmax_exp(float *att, int n) {
    float mx = -INFINITY;
    for (int t = 0; t < n; t++)
        if (att[t] > mx) mx = att[t];
    float sum = 0.0f;
    int t = 0;
    for (; t + 4 <= n; t += 4) {
        att[t] -= mx;
        att[t + 1] -= mx;
        att[t + 2] -= mx;
        att[t + 3] -= mx;
        expf4(att + t);
        sum += att[t];
        sum += att[t + 1];
        sum += att[t + 2];
        sum += att[t + 3];
    }
    for (; t < n; t++) {
        att[t] = ellm_expf(att[t] - mx);
        sum += att[t];
    }
    return 1.0f / sum;
}

/* o[i] += p * v[i] over positions 0..n-1 in order, for heads [ha, ha + nc); p = att[t] * inv
 * (then *= value_s for int8 V). Four positions per pass, each element's additions in t order. */
static void attn_values(const ellm_model *m, ellm_state *s, size_t lrow, int kh, int ha, int nc, int n,
                        const float *inv, float *rows) {
    const ellm_header *h = &m->h;
    const int hd = h->head_dim, n_kv = h->n_kv_heads, kv_dim = n_kv * hd, seq = s->seq_len;
    for (int j = 0; j < nc; j++) {
        float *o = s->xb + (size_t)(ha + j) * hd;
        for (int i = 0; i < hd; i++) o[i] = 0.0f;
    }
    if (h->kv_type == 0) {
        const float *vc = s->value_cache + lrow * kv_dim + (size_t)kh * hd;
        for (int j = 0; j < nc; j++) {
            const float *att = s->att + (size_t)(ha + j) * seq;
            float *o = s->xb + (size_t)(ha + j) * hd;
            for (int t = 0; t < n; t++) {
                const float p = att[t] * inv[j];
                const float *v = vc + (size_t)t * kv_dim;
                for (int i = 0; i < hd; i++) o[i] += p * v[i];
            }
        }
        return;
    }
    const int8_t *vq = s->value_q + lrow * kv_dim + (size_t)kh * hd;
    const float *vs = s->value_s + lrow * n_kv + kh;
    int t = 0;
    for (; t + 4 <= n; t += 4) {
        const float *f0 = rows, *f1 = rows + hd, *f2 = rows + 2 * hd, *f3 = rows + 3 * hd;
        for (int r = 0; r < 4; r++) row_to_float(rows + r * hd, vq + (size_t)(t + r) * kv_dim, hd);
        const float s0 = vs[(size_t)t * n_kv], s1 = vs[(size_t)(t + 1) * n_kv];
        const float s2 = vs[(size_t)(t + 2) * n_kv], s3 = vs[(size_t)(t + 3) * n_kv];
        for (int j = 0; j < nc; j++) {
            const float *att = s->att + (size_t)(ha + j) * seq + t;
            float p0 = att[0] * inv[j], p1 = att[1] * inv[j], p2 = att[2] * inv[j], p3 = att[3] * inv[j];
            p0 *= s0;
            p1 *= s1;
            p2 *= s2;
            p3 *= s3;
            float *o = s->xb + (size_t)(ha + j) * hd;
            int i = 0;
            for (; i + 4 <= hd; i += 4) { /* four independent element chains */
                float a = o[i], b = o[i + 1], c = o[i + 2], d = o[i + 3];
                a = a + p0 * f0[i];
                b = b + p0 * f0[i + 1];
                c = c + p0 * f0[i + 2];
                d = d + p0 * f0[i + 3];
                a = a + p1 * f1[i];
                b = b + p1 * f1[i + 1];
                c = c + p1 * f1[i + 2];
                d = d + p1 * f1[i + 3];
                a = a + p2 * f2[i];
                b = b + p2 * f2[i + 1];
                c = c + p2 * f2[i + 2];
                d = d + p2 * f2[i + 3];
                a = a + p3 * f3[i];
                b = b + p3 * f3[i + 1];
                c = c + p3 * f3[i + 2];
                d = d + p3 * f3[i + 3];
                o[i] = a;
                o[i + 1] = b;
                o[i + 2] = c;
                o[i + 3] = d;
            }
            for (; i < hd; i++) {
                float a = o[i];
                a = a + p0 * f0[i];
                a = a + p1 * f1[i];
                a = a + p2 * f2[i];
                o[i] = a + p3 * f3[i];
            }
        }
    }
    for (; t < n; t++) {
        row_to_float(rows, vq + (size_t)t * kv_dim, hd);
        const float st = vs[(size_t)t * n_kv];
        for (int j = 0; j < nc; j++) {
            float p = s->att[(size_t)(ha + j) * seq + t] * inv[j];
            p *= st;
            float *o = s->xb + (size_t)(ha + j) * hd;
            for (int i = 0; i < hd; i++) o[i] += p * rows[i];
        }
    }
}

void ellm_attn_heads(const ellm_model *m, ellm_state *s, int l, int pos, int h_lo, int h_hi) {
    const ellm_header *h = &m->h;
    const int group = h->n_heads / h->n_kv_heads, seq = s->seq_len, n = pos + 1;
    const size_t lrow = (size_t)l * (size_t)seq;
    float rows[4 * ELLM_MAX_HEAD_DIM]; /* four V rows as float */
    float inv[ATT_CHUNK];
    for (int ha = h_lo; ha < h_hi;) {
        const int kh = ha / group;
        int hb = (kh + 1) * group;
        if (hb > h_hi) hb = h_hi;
        if (hb > ha + ATT_CHUNK) hb = ha + ATT_CHUNK;
        const int nc = hb - ha;
        attn_scores(m, s, lrow, kh, ha, nc, n);
        for (int j = 0; j < nc; j++) inv[j] = softmax_exp(s->att + (size_t)(ha + j) * seq, n);
        attn_values(m, s, lrow, kh, ha, nc, n, inv, rows);
        ha = hb;
    }
}

static int state_ok(const ellm_model *m, const ellm_state *s) {
    if (!s->x || !s->xb || !s->xb2 || !s->q || !s->att || !s->hb || !s->hb2 || !s->logits) return 0;
    if (m->need_q && !s->xq) return 0;
    if (m->need_lut && !s->lut) return 0;
    if (m->h.kv_type == 0) return s->key_cache && s->value_cache;
    return s->kf && s->vf && s->key_q && s->value_q && s->key_s && s->value_s;
}

/* The decoder step; want_logits = 0 stops right after the last layer's KV write for pos (the
 * rest of that layer, the final norm and the LM head only feed the logits). */
static int forward(const ellm_model *m, ellm_state *s, int token, int pos, int want_logits) {
    const ellm_header *h = &m->h;
    if (pos < 0 || pos >= s->seq_len) return ELLM_ERR_POS;
    if (token < 0 || token >= h->vocab) return ELLM_ERR_TOKEN;
    if (!state_ok(m, s)) return ELLM_ERR_UNSUPPORTED;

    const int dim = h->dim, hd = h->head_dim, nh = h->n_heads, n_kv = h->n_kv_heads, hidden = h->hidden;
    const int kv_dim = n_kv * hd, q_dim = nh * hd, kv8 = h->kv_type == 1;
    const float eps = h->norm_eps;
    float *x = s->x, *xb = s->xb, *xb2 = s->xb2, *q = s->q;
    float *hb = s->hb, *hb2 = s->hb2;
    ellm_act a;

    embed_row(&m->embed, token, x);

    /* RoPE tables: float32 angle, cos/sin in double rounded once to float (hd per token, so the
     * device's soft double is cheap); unlike cosf/sinf this does not depend on the libm, which
     * keeps host, device and quantsim (c_exact) bit-identical. */
    float rc[ELLM_MAX_HEAD_DIM / 2], rs[ELLM_MAX_HEAD_DIM / 2];
    for (int i = 0; i < hd / 2; i++) {
        float ang = (float)pos * m->inv_freq[i];
        rc[i] = (float)cos((double)ang);
        rs[i] = (float)sin((double)ang);
    }

    for (int l = 0; l < h->n_layers; l++) {
        const ellm_tensor *L = m->layer[l];
        size_t lrow = (size_t)l * s->seq_len;               /* first (layer, pos) row */
        size_t loff = lrow * kv_dim;
        float *kc = kv8 ? s->kf : s->key_cache + loff + (size_t)pos * kv_dim;
        float *vc = kv8 ? s->vf : s->value_cache + loff + (size_t)pos * kv_dim;

        /* attention */
        rmsnorm(xb, x, &L[EL_ATTN_NORM], dim, eps);
        prep_act(&a, s, xb, dim, TBIT(L[EL_WQ]) | TBIT(L[EL_WK]) | TBIT(L[EL_WV]));
        matvec(&L[EL_WQ], &a, q);
        matvec(&L[EL_WK], &a, kc);
        matvec(&L[EL_WV], &a, vc);
        rope(q, nh, hd, rc, rs);
        rope(kc, n_kv, hd, rc, rs);
        if (kv8) {
            size_t row = lrow + (size_t)pos;
            kv_quant(s->key_q + row * kv_dim, s->key_s + row * n_kv, kc, n_kv, hd);
            kv_quant(s->value_q + row * kv_dim, s->value_s + row * n_kv, vc, n_kv, hd);
        }
        if (!want_logits && l == h->n_layers - 1) return 0; /* KV of every layer is written */

        if (ellm_attn_hook)
            ellm_attn_hook(m, s, l, pos);
        else
            ellm_attn_heads(m, s, l, pos, 0, nh);
        prep_act(&a, s, xb, q_dim, TBIT(L[EL_WO]));
        matvec(&L[EL_WO], &a, xb2);
        for (int i = 0; i < dim; i++) x[i] += xb2[i];

        /* SwiGLU MLP */
        rmsnorm(xb, x, &L[EL_FFN_NORM], dim, eps);
        prep_act(&a, s, xb, dim, TBIT(L[EL_W_GATE]) | TBIT(L[EL_W_UP]));
        matvec(&L[EL_W_GATE], &a, hb);
        matvec(&L[EL_W_UP], &a, hb2);
        for (int i = 0; i < hidden; i++) {
            float g = hb[i];
            hb[i] = g / (1.0f + ellm_expf(-g)) * hb2[i];
        }
        prep_act(&a, s, hb, hidden, TBIT(L[EL_W_DOWN]));
        matvec(&L[EL_W_DOWN], &a, xb2);
        for (int i = 0; i < dim; i++) x[i] += xb2[i];
    }

    rmsnorm(x, x, &m->norm_final, dim, eps);
    prep_act(&a, s, x, dim, TBIT(m->embed));
    matvec(&m->embed, &a, s->logits); /* tied LM head */
    return 0;
}

int ellm_forward(const ellm_model *m, ellm_state *s, int token, int pos) {
    return forward(m, s, token, pos, 1);
}

int ellm_forward_nologits(const ellm_model *m, ellm_state *s, int token, int pos) {
    return forward(m, s, token, pos, 0);
}
