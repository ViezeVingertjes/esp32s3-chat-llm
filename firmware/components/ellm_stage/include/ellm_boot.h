#ifndef ELLM_BOOT_H
#define ELLM_BOOT_H
/* Device boot shared by firmware/app and firmware/chat (ESP32-S3 only): mmap the pack, stage the
 * core into one PSRAM arena (ellm_stage.h policy, flash spill when it does not all fit), allocate
 * the state, and the DCache autoload over exactly that arena. No printing: each firmware reports
 * from ellm_boot_info and handles failures its own way (err holds one line, no "ELLM FAIL" prefix).
 *
 * PSRAM layout: ONE never-freed 64-aligned arena = [staged core units][K int8][V int8][K scales]
 * [V scales][logits]. Internal RAM: one block = LUT, int8 activations, floats, attention scores.
 *
 * Sizing (room(r) = min(PSRAM free - r, largest free PSRAM block)):
 *   - the KV cache gets the pack header's seq_len; only if even the "must" part of the core
 *     (norms, scales, attention weights) would not fit beside it in room(cfg.reserve) is seq_len
 *     shortened (clamped=1);
 *   - the core gets the rest of room(cfg.core_reserve) (minus cfg.force_spill bytes, a debug
 *     knob), staged in the ellm_stage priority order; the units that do not fit stay mapped from
 *     flash. core_reserve > reserve: a model that spills leaves more PSRAM free (the chat's
 *     margin) than one that only clamps its KV cache. */
#include <stddef.h>
#include <stdint.h>
#include "ellm.h"
#include "ellm_stage.h"

typedef struct {
    size_t reserve;      /* PSRAM kept free when sizing the KV cache (headroom the caller checks + slack) */
    size_t core_reserve; /* PSRAM kept free when sizing the staged core (>= reserve) */
    size_t force_spill; /* debug: at least this many core bytes stay in flash (0 = off) */
} ellm_boot_cfg;

typedef struct {
    /* ellm_boot_load */
    const void *img;
    size_t img_len;
    uint32_t part_addr;
    int64_t t_map_us, t_init_us;
    /* ellm_boot_stage */
    size_t ps_free, ps_big;  /* PSRAM free / largest block before staging */
    ellm_stage_plan plan;    /* core: psram_bytes staged, flash_bytes spilled */
    uint8_t *arena;          /* 64-aligned */
    size_t arena_bytes;      /* multiple of 64 */
    size_t kv_bytes;         /* each of S.key_q / S.value_q */
    size_t kvs_bytes;        /* each of S.key_s / S.value_s */
    size_t per_pos;          /* KV bytes per position (all layers, K + V + scales) */
    size_t logit_bytes;      /* vocab floats */
    size_t in_bytes;         /* the internal block */
    int seq_header, seq;     /* pack header seq_len; staged seq_len (S.seq_len) */
    int clamped;             /* seq < seq_header */
    int64_t t_copy_us;       /* core copy flash -> PSRAM */
} ellm_boot_info;

/* mmap the whole pack from the `model` partition and ellm_model_init (full-image CRC32). */
int ellm_boot_load(ellm_model *m, ellm_boot_info *bi, char *err, size_t err_n);

/* Stage the core and fill every buffer of s (seq_len, LUT, activations, floats, KV, logits);
 * ellm_attn_simd_ok must hold. 0 or -1 (err set). */
int ellm_boot_stage(ellm_model *m, ellm_state *s, const ellm_boot_cfg *cfg, ellm_boot_info *bi, char *err,
                    size_t err_n);

/* DCache autoload (miss trigger, size 3) over exactly the arena (on), or disabled (off). Returns the
 * ROM region-config result (0 ok). */
int ellm_boot_autoload(const ellm_boot_info *bi, int on);

#endif
