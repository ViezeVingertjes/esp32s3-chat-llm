#ifndef ELLM_STAGE_H
#define ELLM_STAGE_H
/* Core staging policy (portable C99, host-tested in host_test/): which core tensors are copied from
 * the mmapped pack into the PSRAM arena and which stay mapped from flash ("flash spill").
 *
 * A unit is a tensor's data or its per-row scale, each 64-aligned in the arena (al64 of its bytes).
 * Units are staged in one fixed priority order until the next unit would not fit the budget; that
 * unit and every later one stay in flash (no skipping ahead), so the same pack and budget always
 * give the same split:
 *   1. the norms (norm_final, then per layer attn_norm, ffn_norm), data and scale;
 *   2. every other scale (per layer: wq, wk, wv, wo, w_gate, w_up, w_down);
 *   3. attention weights (per layer: wq, wk, wv, wo);
 *   4. FFN weights (per layer: w_gate, w_up, w_down).
 * Groups 1-3 are the "must" part (ellm_stage_must_bytes): the firmware shortens the KV cache before
 * it spills any of them; only a forced debug spill reaches into them (from the end: last layers'
 * attention weights first). The forward reads the same bytes either way: placement never changes
 * the outputs. */
#include <stddef.h>
#include <stdint.h>
#include "ellm.h"

typedef struct {
    size_t core_bytes;  /* all core units (= psram_bytes + flash_bytes) */
    size_t psram_bytes; /* units staged in the arena (a 64-aligned prefix of the priority order) */
    size_t flash_bytes; /* units left mapped from flash */
    int n_units;        /* number of staged units */
    int spilled;        /* tensors (norm_final + layer tensors) with their data or scale left in flash */
} ellm_stage_plan;

/* Bytes of all core units (norm_final + every layer tensor, data and scale, each al64). */
size_t ellm_stage_core_bytes(const ellm_model *m);

/* Bytes of priority groups 1-3 (norms, scales, attention weights). */
size_t ellm_stage_must_bytes(const ellm_model *m);

/* Plan for a PSRAM budget: units in priority order while the next one fits. */
void ellm_stage_plan_for(const ellm_model *m, size_t budget, ellm_stage_plan *p);

/* Copy the planned units into dst (64-aligned, >= p->psram_bytes) in priority order and repoint
 * the tensors there; unstaged tensors keep pointing at the image. Returns the bytes used
 * (== p->psram_bytes). */
size_t ellm_stage_apply(ellm_model *m, const ellm_stage_plan *p, uint8_t *dst);

#endif
