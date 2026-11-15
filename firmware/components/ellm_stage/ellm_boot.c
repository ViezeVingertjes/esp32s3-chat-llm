// Device boot shared by firmware/app and firmware/chat (see ellm_boot.h).
#include "ellm_boot.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "esp32s3/rom/cache.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"

static size_t al16(size_t n) { return (n + 15) & ~(size_t)15; }
static size_t al64(size_t n) { return (n + 63) & ~(size_t)63; }

static int fail(char *err, size_t n, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, n, fmt, ap);
    va_end(ap);
    return -1;
}

int ellm_boot_load(ellm_model *m, ellm_boot_info *bi, char *err, size_t err_n) {
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!part) return fail(err, err_n, "no 'model' partition");
    bi->part_addr = part->address;
    ellm_header hdr;
    if (esp_partition_read(part, 0, &hdr, sizeof hdr) != ESP_OK || hdr.magic != ELLM_MAGIC)
        return fail(err, err_n, "no pack in 'model' partition @0x%" PRIx32 " (magic 0x%08" PRIx32 ")", part->address,
                    hdr.magic);
    bi->img_len = (size_t)hdr.payload_off + hdr.payload_len;
    if (bi->img_len > part->size)
        return fail(err, err_n, "pack length %u exceeds partition %u", (unsigned)bi->img_len, (unsigned)part->size);
    esp_partition_mmap_handle_t mh;
    int64_t t0 = esp_timer_get_time();
    esp_err_t e = esp_partition_mmap(part, 0, bi->img_len, ESP_PARTITION_MMAP_DATA, &bi->img, &mh);
    bi->t_map_us = esp_timer_get_time() - t0;
    if (e != ESP_OK) return fail(err, err_n, "mmap %u B: %s", (unsigned)bi->img_len, esp_err_to_name(e));
    t0 = esp_timer_get_time();
    int r = ellm_model_init(m, bi->img, bi->img_len);
    bi->t_init_us = esp_timer_get_time() - t0;
    if (r) return fail(err, err_n, "ellm_model_init = %d", r);
    return 0;
}

// Bump allocator over one block (never freed); an overflow sets bad and returns the base.
typedef struct {
    uint8_t *base;
    size_t used, size;
    int bad;
} bump;

static void *take(bump *a, size_t n, size_t align) {
    size_t off = (a->used + align - 1) & ~(align - 1);
    if (off + n > a->size) {
        a->bad = 1;
        return a->base;
    }
    a->used = off + n;
    return a->base + off;
}

int ellm_boot_stage(ellm_model *m, ellm_state *s, const ellm_boot_cfg *cfg, ellm_boot_info *bi, char *err,
                    size_t err_n) {
    const ellm_header *h = &m->h;
    size_t q_dim = (size_t)h->n_heads * h->head_dim, kv_dim = (size_t)h->n_kv_heads * h->head_dim;
    size_t xb = q_dim > (size_t)h->dim ? q_dim : (size_t)h->dim;

    size_t core = ellm_stage_core_bytes(m), must = ellm_stage_must_bytes(m);
    bi->logit_bytes = (size_t)h->vocab * sizeof(float);
    size_t other = al64(bi->logit_bytes) + 4 * 64; /* logits + KV section alignment */
    // KV per position: K + V int8 (kv_dim each) + 2 float scales per KV head, all layers
    bi->per_pos = (size_t)h->n_layers * (2 * kv_dim + 2 * (size_t)h->n_kv_heads * sizeof(float));
    bi->ps_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    bi->ps_big = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    size_t room = bi->ps_free > cfg->reserve ? bi->ps_free - cfg->reserve : 0;
    if (room > bi->ps_big) room = bi->ps_big;
    size_t room_core = bi->ps_free > cfg->core_reserve ? bi->ps_free - cfg->core_reserve : 0;
    if (room_core > room) room_core = room;

    // KV first (the header's seq_len), shortened only if the must part would not fit beside it
    size_t seq = (size_t)h->seq_len;
    bi->seq_header = (int)seq;
    size_t cap = room > must + other ? (room - must - other) / bi->per_pos : 0;
    bi->clamped = seq > cap;
    if (bi->clamped) seq = cap;
    bi->seq = (int)seq;
    if (seq < 1) return fail(err, err_n, "PSRAM free %u (largest %u) holds no KV position", (unsigned)bi->ps_free,
                             (unsigned)bi->ps_big);
    // the core gets the rest; units that do not fit stay in flash
    size_t used = other + seq * bi->per_pos;
    size_t budget = room_core > used ? room_core - used : 0;
    size_t must_fit = room > used ? room - used : 0; // the must part never spills for the margin
    if (must_fit > must) must_fit = must;
    if (budget < must_fit) budget = must_fit;
    size_t forced = cfg->force_spill < core ? core - cfg->force_spill : 0;
    if (budget > forced) budget = forced;
    ellm_stage_plan_for(m, budget, &bi->plan);

    size_t rows = (size_t)h->n_layers * seq;
    bi->kv_bytes = rows * kv_dim;
    bi->kvs_bytes = rows * h->n_kv_heads * sizeof(float);
    size_t fixed = bi->plan.psram_bytes + other;
    bi->arena_bytes = al64(fixed + 2 * al64(bi->kv_bytes) + 2 * al64(bi->kvs_bytes));

    size_t lut_b = m->lut_bytes;
    size_t f_n = (size_t)h->dim * 2 + xb + q_dim + (size_t)h->n_heads * seq + (size_t)h->hidden * 2 + kv_dim * 2;
    bi->in_bytes = al16(lut_b) + al16((size_t)m->act_npad) + f_n * sizeof(float) + 16 * 12;
    bump in = {heap_caps_malloc(bi->in_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT), 0, bi->in_bytes, 0};
    bi->arena = heap_caps_aligned_alloc(64, bi->arena_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    bump ps = {bi->arena, 0, bi->arena_bytes, 0};
    if (!in.base || !ps.base)
        return fail(err, err_n, "alloc internal %u B (%s), psram arena %u B (%s)", (unsigned)bi->in_bytes,
                    in.base ? "ok" : "failed", (unsigned)bi->arena_bytes, ps.base ? "ok" : "failed");
    memset(in.base, 0, bi->in_bytes);

    // core units: flash mmap -> PSRAM arena (priority order), the model points there; the rest
    // (and the tied embedding/head) stays mapped from flash
    int64_t t0 = esp_timer_get_time();
    size_t staged = ellm_stage_apply(m, &bi->plan, bi->arena);
    bi->t_copy_us = esp_timer_get_time() - t0;
    if (staged != bi->plan.psram_bytes)
        return fail(err, err_n, "core copy %u B != planned %u B", (unsigned)staged, (unsigned)bi->plan.psram_bytes);
    ps.used = staged;
    memset(bi->arena + staged, 0, bi->arena_bytes - staged);

    s->seq_len = (int)seq;
    s->lut = take(&in, lut_b, 16);
    s->xq = take(&in, (size_t)m->act_npad, 16);
    s->x = take(&in, (size_t)h->dim * sizeof(float), 16);
    s->xb = take(&in, xb * sizeof(float), 16);
    s->xb2 = take(&in, (size_t)h->dim * sizeof(float), 16);
    s->q = take(&in, q_dim * sizeof(float), 16);
    s->att = take(&in, (size_t)h->n_heads * seq * sizeof(float), 16);
    s->hb = take(&in, (size_t)h->hidden * sizeof(float), 16);
    s->hb2 = take(&in, (size_t)h->hidden * sizeof(float), 16);
    s->kf = take(&in, kv_dim * sizeof(float), 16);
    s->vf = take(&in, kv_dim * sizeof(float), 16);
    s->key_q = take(&ps, bi->kv_bytes, 64);
    s->value_q = take(&ps, bi->kv_bytes, 64);
    s->key_s = take(&ps, bi->kvs_bytes, 64);
    s->value_s = take(&ps, bi->kvs_bytes, 64);
    s->logits = take(&ps, bi->logit_bytes, 64);
    if (in.bad || ps.bad)
        return fail(err, err_n, "%s overflow (internal %u/%u, arena %u/%u)", in.bad ? "internal block" : "arena",
                    (unsigned)in.used, (unsigned)in.size, (unsigned)ps.used, (unsigned)ps.size);
    if (!ellm_attn_simd_ok(m, s)) // never run the slow portable attention silently
        return fail(err, err_n, "attn SIMD off: %s", ellm_attn_simd_off_reason(m, s));
    return 0;
}

int ellm_boot_autoload(const ellm_boot_info *bi, int on) {
    Cache_Disable_DCache_Autoload();
    if (!on) return 0;
    // the region is the arena itself (64-aligned start and size): it never covers unmapped space
    // or the flash-mapped spill. ROM functions IDF marks "do not call in SDK application".
    struct autoload_config cfg = {.ena = 1, .order = 0, .trigger = 0 /* miss */, .size = 3};
    Cache_Config_DCache_Autoload(&cfg);
    struct autoload_region_config r0 = {.region = 0, .ena = 1, .addr = (uint32_t)(uintptr_t)bi->arena,
                                        .size = (uint32_t)bi->arena_bytes};
    int e = Cache_Config_DCache_Region_Autoload(&r0);
    Cache_Enable_DCache_Autoload();
    return e;
}
