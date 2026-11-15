// Core staging policy (see ellm_stage.h). Portable C99: no allocation, no IO.
#include "ellm_stage.h"
#include <string.h>

static size_t al64(size_t n) { return (n + 63) & ~(size_t)63; }

static const ellm_tensor *tensor_at(const ellm_model *m, int l, int k) {
    return l < 0 ? &m->norm_final : &m->layer[l][k];
}

static size_t unit_bytes(const ellm_tensor *t, int is_scale) {
    if (is_scale) return t->scale ? al64((size_t)t->rows * sizeof(float)) : 0;
    return al64((size_t)t->rows * t->row_bytes);
}

// Visits every non-empty unit in priority order (l = -1: norm_final); stops when visit returns
// nonzero. groups: bit g-1 set = include priority group g (1..4).
typedef int (*visit_fn)(void *ctx, int l, int k, int is_scale, size_t bytes);

static int is_norm(int k) { return k == EL_ATTN_NORM || k == EL_FFN_NORM; }
static int is_attn(int k) { return k >= EL_WQ && k <= EL_WO; }

static void walk(const ellm_model *m, unsigned groups, visit_fn f, void *ctx) {
    const int L = m->h.n_layers;
#define VISIT(l, k, s)                                                    \
    do {                                                                  \
        size_t b_ = unit_bytes(tensor_at(m, (l), (k)), (s));              \
        if (b_ && f(ctx, (l), (k), (s), b_)) return;                      \
    } while (0)
    if (groups & 1u) {
        VISIT(-1, 0, 0);
        VISIT(-1, 0, 1);
        for (int l = 0; l < L; l++)
            for (int k = 0; k < EL_N; k++)
                if (is_norm(k)) {
                    VISIT(l, k, 0);
                    VISIT(l, k, 1);
                }
    }
    if (groups & 2u)
        for (int l = 0; l < L; l++)
            for (int k = 0; k < EL_N; k++)
                if (!is_norm(k)) VISIT(l, k, 1);
    if (groups & 4u)
        for (int l = 0; l < L; l++)
            for (int k = 0; k < EL_N; k++)
                if (is_attn(k)) VISIT(l, k, 0);
    if (groups & 8u)
        for (int l = 0; l < L; l++)
            for (int k = EL_W_GATE; k <= EL_W_DOWN; k++) VISIT(l, k, 0);
#undef VISIT
}

static int sum_visit(void *ctx, int l, int k, int is_scale, size_t bytes) {
    (void)l, (void)k, (void)is_scale;
    *(size_t *)ctx += bytes;
    return 0;
}

size_t ellm_stage_core_bytes(const ellm_model *m) {
    size_t n = 0;
    walk(m, 15u, sum_visit, &n);
    return n;
}

size_t ellm_stage_must_bytes(const ellm_model *m) {
    size_t n = 0;
    walk(m, 7u, sum_visit, &n);
    return n;
}

#define TBITS (ELLM_MAX_LAYERS * EL_N + 1) /* tensor index: 0 = norm_final, 1 + l*EL_N + k */

typedef struct {
    size_t budget;
    ellm_stage_plan *p;
    int stopped;
    uint32_t spilled[(TBITS + 31) / 32];
} plan_ctx;

static int plan_visit(void *ctx, int l, int k, int is_scale, size_t bytes) {
    plan_ctx *c = ctx;
    (void)is_scale;
    if (!c->stopped && c->p->psram_bytes + bytes <= c->budget) {
        c->p->psram_bytes += bytes;
        c->p->n_units++;
        return 0;
    }
    c->stopped = 1; // this unit and every later one stay in flash
    int ti = l < 0 ? 0 : 1 + l * EL_N + k;
    c->spilled[ti / 32] |= 1u << (ti % 32);
    return 0;
}

void ellm_stage_plan_for(const ellm_model *m, size_t budget, ellm_stage_plan *p) {
    memset(p, 0, sizeof *p);
    plan_ctx c;
    memset(&c, 0, sizeof c);
    c.budget = budget;
    c.p = p;
    walk(m, 15u, plan_visit, &c);
    p->core_bytes = ellm_stage_core_bytes(m);
    p->flash_bytes = p->core_bytes - p->psram_bytes;
    for (int i = 0; i < TBITS; i++) p->spilled += (int)((c.spilled[i / 32] >> (i % 32)) & 1u);
}

typedef struct {
    ellm_model *m;
    uint8_t *dst;
    size_t used;
    int left;
} apply_ctx;

static int apply_visit(void *ctx, int l, int k, int is_scale, size_t bytes) {
    apply_ctx *c = ctx;
    if (c->left == 0) return 1;
    ellm_tensor *t = l < 0 ? &c->m->norm_final : &c->m->layer[l][k];
    uint8_t *d = c->dst + c->used;
    if (is_scale) {
        memcpy(d, t->scale, (size_t)t->rows * sizeof(float));
        t->scale = (const float *)(const void *)d;
    } else {
        memcpy(d, t->data, (size_t)t->rows * t->row_bytes);
        t->data = d;
    }
    c->used += bytes;
    c->left--;
    return 0;
}

size_t ellm_stage_apply(ellm_model *m, const ellm_stage_plan *p, uint8_t *dst) {
    apply_ctx c = {m, dst, 0, p->n_units};
    walk(m, 15u, apply_visit, &c);
    return c.used;
}
