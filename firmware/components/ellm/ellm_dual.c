#include "ellm_dual.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

typedef enum { JOB_MATVEC, JOB_LUT, JOB_LUT_Q2, JOB_ATTN } job_kind;

typedef struct {
    job_kind kind;
    const ellm_tensor *w;
    const ellm_act *a;
    ellm_act lut_part; /* JOB_LUT: view of the upper half of the groups; JOB_LUT_Q2: the act */
    float *out;
    int lo, hi;          /* JOB_MATVEC: rows; JOB_LUT_Q2: groups */
    const ellm_model *m; /* JOB_ATTN: heads [lo, hi) of layer `layer` at `pos` */
    ellm_state *s;
    int layer, pos;
} dual_job;

static dual_job s_job;
static SemaphoreHandle_t s_go, s_done;
static TaskHandle_t s_task;

static void worker(void *arg) {
    (void)arg;
    for (;;) {
        xSemaphoreTake(s_go, portMAX_DELAY);
        if (s_job.kind == JOB_LUT) ellm_act_build_lut(&s_job.lut_part);
        else if (s_job.kind == JOB_LUT_Q2) ellm_act_build_lut_q2_groups(&s_job.lut_part, s_job.lo, s_job.hi);
        else if (s_job.kind == JOB_ATTN) ellm_attn_heads(s_job.m, s_job.s, s_job.layer, s_job.pos, s_job.lo, s_job.hi);
        else ellm_matvec(s_job.w, s_job.a, s_job.out, s_job.lo, s_job.hi);
        xSemaphoreGive(s_done);
    }
}

void ellm_dual_init(void) {
    if (s_task) return;
    s_go = xSemaphoreCreateBinary();
    s_done = xSemaphoreCreateBinary();
    configASSERT(s_go && s_done);
    /* stack: ellm_attn_heads' frame is ~3.5 KiB (4 V rows as float, 2 KiB at head_dim 128, plus
     * the int8 q buffer of up to 8 heads, ~1.2 KiB); dprobe prints the high-water mark */
    BaseType_t ok = xTaskCreatePinnedToCore(worker, "ellm_w1", 6144, NULL,
                                            configMAX_PRIORITIES - 2, &s_task, 1);
    configASSERT(ok == pdPASS);
}

TaskHandle_t ellm_dual_worker(void) { return s_task; }

void ellm_matvec_dual(const ellm_tensor *w, const ellm_act *a, float *out) {
    ellm_dual_init();
    int mid = w->rows / 2;
    s_job.kind = JOB_MATVEC;
    s_job.w = w;
    s_job.a = a;
    s_job.out = out;
    s_job.lo = mid;
    s_job.hi = w->rows;
    xSemaphoreGive(s_go);
    ellm_matvec(w, a, out, 0, mid);
    xSemaphoreTake(s_done, portMAX_DELAY);
}

/* Below this many groups (2 KiB of LUT per 4 groups) one core builds the whole table: the
 * ~20 us handoff would cost more than the split saves, and groups / 8 * 4 would give core 0
 * nothing. */
#define LUT_DUAL_MIN_GROUPS 16

void ellm_act_build_lut_dual(ellm_act *a) {
    int groups = a->n_pad / 5;
    if (groups < LUT_DUAL_MIN_GROUPS) {
        ellm_act_build_lut(a);
        return;
    }
    ellm_dual_init();
    /* split on a 4-group boundary: the LUT interleaves groups 4k..4k+3 */
    int gmid = groups / 8 * 4;
    ellm_act lo = *a, *hi = &s_job.lut_part;
    *hi = *a;
    hi->q += 5 * gmid;
    hi->lut += 256 * gmid;
    hi->n_pad -= 5 * gmid;
    lo.n_pad = 5 * gmid;
    s_job.kind = JOB_LUT;
    xSemaphoreGive(s_go);
    ellm_act_build_lut(&lo);
    xSemaphoreTake(s_done, portMAX_DELAY);
    a->lut_type = ELLM_T158; /* the halves were built through views of a */
}

void ellm_act_build_lut_q2_dual(ellm_act *a) {
    int groups = ellm_pad16(a->n) / 4;
    if (groups < LUT_DUAL_MIN_GROUPS) {
        ellm_act_build_lut_q2(a);
        return;
    }
    ellm_dual_init();
    /* split on a 4-group boundary: the LUT interleaves groups 4k..4k+3 */
    int gmid = groups / 8 * 4;
    s_job.kind = JOB_LUT_Q2;
    s_job.lut_part = *a;
    s_job.lo = gmid;
    s_job.hi = groups;
    xSemaphoreGive(s_go);
    ellm_act_build_lut_q2_groups(a, 0, gmid);
    xSemaphoreTake(s_done, portMAX_DELAY);
    a->lut_type = ELLM_Q2; /* _groups does not tag */
}

void ellm_lut_dual(ellm_act *a, int type) {
    if (type == ELLM_Q2) ellm_act_build_lut_q2_dual(a);
    else ellm_act_build_lut_dual(a);
}

void ellm_attn_dual(const ellm_model *m, ellm_state *s, int l, int pos) {
    ellm_dual_init();
    int nh = m->h.n_heads, mid = (nh + 1) / 2; /* core 0 (caller) takes the larger half */
    s_job.kind = JOB_ATTN;
    s_job.m = m;
    s_job.s = s;
    s_job.layer = l;
    s_job.pos = pos;
    s_job.lo = mid;
    s_job.hi = nh;
    xSemaphoreGive(s_go);
    ellm_attn_heads(m, s, l, pos, 0, mid);
    xSemaphoreTake(s_done, portMAX_DELAY);
}
