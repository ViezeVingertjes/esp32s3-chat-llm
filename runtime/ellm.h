#ifndef ELLM_H
#define ELLM_H
/* Llama forward pass over a pack v1/v2/v3 image (runtime/ellm_pack.h). Plain C99, shared by the host
 * (libellm.so via ellm_capi.c) and the ESP32-S3 firmware. No allocation, no IO in ellm.c:
 * the model points into the caller's image, the state buffers are provided by the caller
 * (ellm_state_alloc on the host, the firmware fills the fields itself). */
#include <stddef.h>
#include <stdint.h>
#include "ellm_kernels.h"
#include "ellm_pack.h"

#define ELLM_MAX_LAYERS 32
#define ELLM_MAX_HEAD_DIM 128

#define ELLM_ERR_MAGIC -1
#define ELLM_ERR_VERSION -2
#define ELLM_ERR_BOUNDS -3   /* offsets/sizes outside the image, or image not 64-byte aligned */
#define ELLM_ERR_CRC -4
#define ELLM_ERR_SHAPE -5    /* header/table invariant violated (alignment, geometry, shapes) */
#define ELLM_ERR_POS -6
#define ELLM_ERR_TOKEN -7
#define ELLM_ERR_UNSUPPORTED -8 /* forward with a state lacking the buffers the model needs */
#define ELLM_ERR_IO -9          /* host C API only */
#define ELLM_ERR_NOMEM -10      /* host C API only */

enum { EL_ATTN_NORM, EL_WQ, EL_WK, EL_WV, EL_WO, EL_FFN_NORM, EL_W_GATE, EL_W_UP, EL_W_DOWN, EL_N };

typedef struct {
    ellm_header h;
    ellm_tensor embed, norm_final;
    ellm_tensor layer[ELLM_MAX_LAYERS][9];
    const uint8_t *tok_blob; /* the "tokenizer" etok blob (pack v3; for ellm_tok_init), NULL if absent */
    size_t tok_len;          /* its length in bytes, 0 if absent */
    float inv_freq[ELLM_MAX_HEAD_DIM / 2]; /* RoPE: rope_theta^(-2i/head_dim), set by init */
    int act_npad;  /* pad20(max(dim, q_dim, hidden)): size of ellm_state.xq (set by init) */
    int need_q;    /* some matvec weight is I8/T158/Q2: ellm_state.xq required */
    int need_lut;  /* some matvec weight is T158/Q2: ellm_state.lut required (lut_bytes) */
    size_t lut_bytes; /* size of ellm_state.lut: max of ellm_lut_bytes(act_npad) if any weight is T158
                         and ellm_lut_q2_bytes(max(dim, q_dim, hidden)) if any is Q2; 0 if neither */
} ellm_model;

/* Validates the image (CRC32 over header+table+payload, alignment, geometry, shapes) and points
 * the tensors into it. image must be 64-byte aligned (pack data_off values are 64-aligned from
 * the image start; the device must map/copy the image accordingly), and must stay alive and
 * unmodified while m is used. Dims/rows/cols are bounded by 1<<24, head_dim by
 * ELLM_MAX_HEAD_DIM. 0 or ELLM_ERR_*. */
int ellm_model_init(ellm_model *m, const uint8_t *image, size_t len);

typedef struct ellm_state ellm_state;
/* Activation quantization (weights I8/T158/Q2): each matvec input vector (attn_norm out, attention
 * out, ffn_norm out, SwiGLU out, final-norm out) is quantized once into xq (+ the T158 or Q2 lut)
 * and shared by the matvecs that read it (the loader rejects a T158 and a Q2 weight reading the
 * same vector: q/k/v, gate/up). KV cache: kv_type 0 uses key_cache/value_cache (float);
 * kv_type 1 computes k/v into kf/vf, applies RoPE to kf, then quantizes each (pos, kv_head)
 * slice of head_dim values with the activation formula into key_q/value_q + key_s/value_s;
 * attention then quantizes each query head the same way and takes exact int8 q.k dot products
 * (score = ((float)acc * key_s) * (q_s * att_scale), att_scale = 1 / sqrtf(head_dim)); the weighted
 * V sum is float over the int8 values (p * value_s * v). Buffers the model does not need may
 * be NULL. */
struct ellm_state {
    int seq_len;          /* KV cache capacity: valid pos are [0, seq_len) */
    float *x;             /* [dim] residual stream */
    float *xb;            /* [max(dim, q_dim)] normed input / attention output */
    float *xb2;           /* [dim] matvec output */
    float *q;             /* [q_dim] */
    float *att;           /* [n_heads][seq_len] attention scores (one row per query head) */
    float *hb, *hb2;      /* [hidden] */
    float *logits;        /* [vocab] */
    int8_t *xq;           /* [m->act_npad] if m->need_q */
    int16_t *lut;         /* m->lut_bytes bytes, 4-byte aligned, if m->need_lut */
    /* kv_type 0 */
    float *key_cache;     /* [n_layers][seq_len][kv_dim], K stored after RoPE */
    float *value_cache;   /* [n_layers][seq_len][kv_dim] */
    /* kv_type 1 */
    float *kf, *vf;       /* [kv_dim] scratch */
    int8_t *key_q;        /* [n_layers][seq_len][kv_dim], K quantized after RoPE */
    int8_t *value_q;      /* [n_layers][seq_len][kv_dim] */
    float *key_s;         /* [n_layers][seq_len][n_kv_heads] per-(pos, kv_head) scale */
    float *value_s;       /* [n_layers][seq_len][n_kv_heads] */
};

/* Host only (ellm_capi.c): malloc every buffer the model needs (others NULL). 0 or
 * ELLM_ERR_NOMEM / ELLM_ERR_POS (seq_len < 1). */
int ellm_state_alloc(ellm_state *s, const ellm_model *m, int seq_len);
void ellm_state_free(ellm_state *s);

/* One decoder step for token at pos; KV for positions < pos must already be in the cache.
 * Logits in s->logits[0..vocab). 0, ELLM_ERR_POS, ELLM_ERR_TOKEN or ELLM_ERR_UNSUPPORTED (a
 * buffer the model needs is NULL in s). */
int ellm_forward(const ellm_model *m, ellm_state *s, int token, int pos);

/* ellm_forward without the logits, for prompt tokens whose logits are not used (chat prefill):
 * runs every layer up to and including its KV write for pos (the KV cache for pos ends up
 * byte-identical to ellm_forward's), then stops: the last layer's attention/MLP, the final norm
 * and the LM head are skipped. s->logits is left untouched; the other scratch buffers (x, xb,
 * q, att, ...) hold intermediate values, as after ellm_forward. Same checks and return codes as
 * ellm_forward. */
int ellm_forward_nologits(const ellm_model *m, ellm_state *s, int token, int pos);

/* exp used by the forward pass (softmax, SiLU): deterministic float32 arithmetic, see ellm.c. */
float ellm_expf(float x);

/* If set, every matvec of the forward pass calls this (full row range) instead of ellm_matvec,
 * e.g. to split rows across cores. NULL on the host. */
extern void (*ellm_matvec_hook)(const ellm_tensor *, const ellm_act *, float *);

/* Attention of query heads [h_lo, h_hi) for layer l at position pos (K/V of positions 0..pos
 * already in the cache): reads s->q and the KV cache, writes those heads' slices of s->xb
 * ([n_heads][head_dim]) and their rows of s->att. Calls on disjoint head ranges touch disjoint
 * memory, so they may run concurrently (one per core); every head's arithmetic is independent of
 * the split, so any split gives bit-identical results. */
void ellm_attn_heads(const ellm_model *m, ellm_state *s, int l, int pos, int h_lo, int h_hi);

/* 1 if the attention scores take the ESP32-S3 PIE (SIMD) int8 dot product for every K row: int8
 * KV, head_dim % 16 == 0, s->key_q 16-byte aligned, and an ESP32-S3 build. Otherwise the
 * portable C loop runs (same results, several times slower on the device), and
 * ellm_attn_simd_off_reason says why (NULL when ok). Call once after setting up the state. */
int ellm_attn_simd_ok(const ellm_model *m, const ellm_state *s);
const char *ellm_attn_simd_off_reason(const ellm_model *m, const ellm_state *s);

/* If set, the forward pass calls this for each layer's attention instead of
 * ellm_attn_heads(m, s, l, pos, 0, n_heads), e.g. to split the heads across cores. NULL on the
 * host. */
extern void (*ellm_attn_hook)(const ellm_model *m, ellm_state *s, int l, int pos);

/* If set, the forward pass builds each activation LUT with this (type ELLM_T158 or ELLM_Q2; a was
 * just quantized by ellm_act_quant) instead of ellm_act_build_lut / ellm_act_build_lut_q2,
 * e.g. to split the groups across cores. It must leave the same table and set a->lut_type = type,
 * like the builder it replaces. NULL on the host. */
extern void (*ellm_lut_hook)(ellm_act *a, int type);

#endif
