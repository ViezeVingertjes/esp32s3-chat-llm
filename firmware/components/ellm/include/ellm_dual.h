#ifndef ELLM_DUAL_H
#define ELLM_DUAL_H
#include "ellm.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* Starts the core-1 worker task. Called lazily by ellm_matvec_dual; call it
 * early to keep task creation out of timed regions. Idempotent. */
void ellm_dual_init(void);

/* The core-1 worker task (NULL before ellm_dual_init), e.g. for its stack high-water mark. */
TaskHandle_t ellm_dual_worker(void);

/* out[0..rows) = W·a, rows split between the calling core and a worker pinned
 * to core 1 (any type: F32, I8, T158, Q2). Bit-identical to ellm_matvec(w, a, out, 0, w->rows).
 * Not reentrant: call from a single task. */
void ellm_matvec_dual(const ellm_tensor *w, const ellm_act *a, float *out);

/* ellm_act_build_lut with the groups split between the two cores. Same result (table and
 * a->lut_type). Short vectors (< 16 groups) are built on the calling core only. Not reentrant. */
void ellm_act_build_lut_dual(ellm_act *a);

/* ellm_act_build_lut_q2 with the groups split between the two cores. Same result (table and
 * a->lut_type). Short vectors (< 16 groups) are built on the calling core only. Not reentrant. */
void ellm_act_build_lut_q2_dual(ellm_act *a);

/* ellm_lut_hook for the forward pass: ellm_act_build_lut_q2_dual (type ELLM_Q2) or
 * ellm_act_build_lut_dual (ELLM_T158). Bit-identical to the 1-core builders. Not reentrant. */
void ellm_lut_dual(ellm_act *a, int type);

/* Attention of one layer with the query heads split between the calling core (heads
 * [0, ceil(n_heads/2))) and the core-1 worker (the rest): ellm_attn_heads on each half, so the
 * result is bit-identical to the single-core forward. Use as ellm_attn_hook. Not reentrant. */
void ellm_attn_dual(const ellm_model *m, ellm_state *s, int l, int pos);

#endif
