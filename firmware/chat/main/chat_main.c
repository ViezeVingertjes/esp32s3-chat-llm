// Plan 4 chat firmware (v1, no settings commands): the shared chat engine (runtime/ellm_chat.c)
// over a v3 pack (model + tokenizer) in the `model` flash partition, talking over the
// USB-Serial-JTAG console. firmware/app stays the trace and bench tool; both boot through the shared
// firmware/components/ellm_stage (pack mmap, PSRAM arena + flash spill, autoload): same placement,
// same hooks.
//
// Boot: mmap the pack, ellm_model_init (full-image CRC32); the pack must carry the tokenizer
// (else the ELLM FAIL line below, display 0, idle: no reboot loop); stage the core into one PSRAM
// arena (KV + logits there too; the core units that do not fit with 392 KiB kept free stay mapped
// from flash, FFN weights of the last layers first: "ELLM stage: psram=<B> flash_spill=<B>
// spilled=<n> tensors"), LUT/activations/floats in one internal block; dual-core
// matvec/LUT/attention hooks; DCache autoload (prefetch) on over the arena (CONFIG_CHAT_PREFETCH,
// default y; n disables it explicitly); display 0; ellm_chat_init (prefills the system turn
// without the LM head; boot ~3.4 s, no tick); one banner line; ESP-IDF logging down to ERROR (an ESP_LOGW/I line, e.g. the display's
// "draw skipped", would otherwise land inside a streamed reply); the prompt.
//
// Serial protocol (device -> host is LF only, no CR; reply bytes are passed through raw):
//   boot      "ELLM chat: loading ...\n" "ELLM stage: psram=<B> flash_spill=<B> spilled=<n> tensors\n"  then
//             "ELLM chat ready: pack fnv1a=0x<8 hex> seq_len=<n> prefetch=<on|off> display=<on|off>
//              psram_free=<B> internal_free=<B> boot_ms=<ms since boot>\n"  then the prompt "> " (no newline)
//   input     one line per prompt, terminated by CR or LF (CRLF = a line + an ignored empty
//             line). The turn text is exactly the bytes before the terminator: only CR/LF are
//             stripped, every other byte (spaces, tabs, NUL, invalid UTF-8) is kept. At most
//             ELLM_CHAT_MAX_USER (1024) bytes; longer: "! line too long (max 1024 bytes)\n> ".
//             An empty line is ignored (no output, still at the prompt). "/reset" (exactly) starts
//             a new conversation: "(new conversation)\n> ". 0x03 at the prompt drops the partial
//             line. Input is read only at the prompt: everything after a line's terminator up to
//             the next "> " is discarded, except 0x03 (Ctrl-C), which stops the running reply.
//   reply     <reply bytes, complete UTF-8 only> "\n" "[<new> tok, <tok/s> tok/s" [", dropped <d>
//             earlier turns"] [", serial_lost=<B>" (only if output bytes were ever dropped since
//             boot)] [", stack_free=<B> worker_stack_free=<B>" (CONFIG_CHAT_DEBUG_STACK
//             builds only)] "]" [" (context full)" | " (stopped)"] "\n> "
//             <tok/s> = new / (end of reply - end of prompt prefill), "%.2f"; 0.00 if new = 0.
//             The reply ends on <|im_end|> (no note), 256 tokens (no note; new = 256), context
//             full (CTX) or Ctrl-C (ABORT).
//   too long  a message that cannot fit the context even after dropping every earlier turn:
//             "! message too long\n> " (no stats line; history unchanged).
//   error     a forward failure: "ELLM FAIL chat turn = <err> (new conversation)\n> ".
// Output never drops bytes while a host is attached (ruling P4-R13): writes block (a slow reader
// stalls generation) and give up only when the host is gone (no USB SOF) or has not read anything
// for 10 s; dropped bytes are counted (serial_lost in the stats line, normally absent).
// Display (docs/stages/screen-design.md): drawn only from the tick callback (after a forward,
// never during one), >= 500 ms apart, and between turns. CONFIG_CHAT_SCREEN_TEXT (default): the
// chat screen: state (thinking = prefill, writing = decode, idle), context fill (n_kv of seq_len,
// bar + numbers), the user line and the streaming reply (last 9 rows), live tok/s (0.0 when idle).
// CONFIG_CHAT_SCREEN_TEXT=n: only the live tok/s number (0 when idle). CONFIG_CHAT_DISPLAY=n:
// never drawn.
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include "ellm.h"
#include "ellm_boot.h"
#include "ellm_chat.h"
#include "ellm_dual.h"
#include "ellm_tok.h"
#include "status_display.h"

#define ELLM_PSRAM_HEADROOM (256u * 1024u) /* PSRAM that must stay free after all allocations */
#define PSRAM_SLACK (64u * 1024u)          /* kept back when sizing KV (display, worker, heap) */
/* kept free when sizing the staged core: a model whose core spills to flash still leaves >= 384 KiB
 * PSRAM free after all allocations (256 KiB headroom + 128 KiB chat margin; + 8 KiB for the
 * allocations after staging, ~0.6 KB measured) */
#define ELLM_SPILL_RESERVE (392u * 1024u)
#define LINE_MAX_BYTES ELLM_CHAT_MAX_USER  /* longest input line (1024) */
#define RX_BUF 2048                        /* USB-Serial-JTAG driver ring buffers */
#define TX_BUF 1024
#define NO_TOK_MSG "ELLM FAIL: pack has no tokenizer (need a v3 pack: tools/flash_pack.sh " \
                   "artifacts/packs/model-a-chat-512.ellm)"

static ellm_model M;
static ellm_state S;
static ellm_tok T;
static ellm_chat C;          /* ~3.4 KB, internal BSS */
static uint16_t *hist;       /* [S.seq_len], internal */
static ellm_boot_info B;     /* pack mapping, PSRAM arena (staged core, KV, logits), flash spill */

/* ---------------------------------------------------------------- serial out */

#define TX_CHUNK 256             /* <= TX_BUF: the driver's ring buffer takes a chunk whole */
#define TX_PATIENCE_US 10000000  /* a host that reads nothing for 10 s is gone */
static uint32_t tx_lost;         /* output bytes dropped since boot (host gone); normally 0 */
static int tx_gave_up;           /* the last chunk timed out: don't wait again until one goes */

// Write to the host without ever dropping bytes while it reads (ruling P4-R13; the VFS path gives
// up after ~50 ms and then drops). Blocks while the ring buffer is full (generation stalls, which
// is fine); drops only when the host is gone: no SOF (unplugged) or nothing read for 10 s.
static void out(const void *b, size_t n) {
    fflush(stdout); /* text printf'd by shared code (boot) goes first */
    const uint8_t *p = b;
    while (n > 0) {
        size_t k = n < TX_CHUNK ? n : TX_CHUNK;
        int ok = usb_serial_jtag_write_bytes(p, k, 0) > 0;
        if (!ok && !tx_gave_up) {
            int64_t t0 = esp_timer_get_time();
            while (!ok && usb_serial_jtag_is_connected() && esp_timer_get_time() - t0 < TX_PATIENCE_US)
                ok = usb_serial_jtag_write_bytes(p, k, pdMS_TO_TICKS(20)) > 0;
            tx_gave_up = !ok;
        }
        if (ok) tx_gave_up = 0;
        else tx_lost += (uint32_t)k;
        p += k;
        n -= k;
    }
}

__attribute__((format(printf, 1, 2))) static void outf(const char *fmt, ...) {
    char b[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (n > 0) out(b, (size_t)n < sizeof b ? (size_t)n : sizeof b - 1);
}

static void prompt(void) { out("> ", 2); }

/* ---------------------------------------------------------------- display */

#if CONFIG_CHAT_DISPLAY && CONFIG_CHAT_SCREEN_TEXT
#define SCREEN_TEXT 1 /* the chat screen (text, state, context) */
#else
#define SCREEN_TEXT 0 /* the tok/s number only, or no display */
#endif

#if CONFIG_CHAT_DISPLAY
static int disp_ok; /* 0 not yet, 1 initialized, -1 init failed */
#endif

// The screen: state, live tok/s, context fill (ctx_used of seq_len); draws what changed.
static void show(disp_state st, float rate, int ctx_used) {
#if CONFIG_CHAT_DISPLAY
    if (!disp_ok) {
        disp_ok = disp_init() == ESP_OK ? 1 : -1;
        if (disp_ok < 0) outf("ELLM display init failed (chat continues)\n");
    }
    if (disp_ok < 0) return;
#if SCREEN_TEXT
    disp_chat_status(st, rate, ctx_used, S.seq_len);
    disp_chat_draw();
#else
    (void)st;
    (void)ctx_used;
    disp_number(rate);
#endif
#else
    (void)st;
    (void)rate;
    (void)ctx_used;
#endif
}

// Text on the chat screen (memory only; drawn by the next show). No-ops without SCREEN_TEXT.
static void screen_turn(const char *user, size_t len) {
#if SCREEN_TEXT
    disp_chat_turn(user, len);
#else
    (void)user;
    (void)len;
#endif
}

static void screen_text(const void *b, size_t n) {
#if SCREEN_TEXT
    disp_chat_reply(b, n);
#else
    (void)b;
    (void)n;
#endif
}

// Live tok/s: completion times of the last forwards; rate over the last ~1 s (or since the start).
#define LIVE_N 32
static int64_t live_t[LIVE_N];
static int live_n;
static int64_t live_last_draw;

static void live_reset(void) {
    live_t[0] = esp_timer_get_time();
    live_n = 1;
    live_last_draw = live_t[0];
}

// One finished forward (called from the tick, so never during a forward); draws >= 500 ms apart.
static void live_forward_done(disp_state st) {
    int64_t now = esp_timer_get_time();
    live_t[live_n % LIVE_N] = now;
    live_n++;
    if (now - live_last_draw < 500000) return;
    int newest = live_n - 1, oldest = live_n > LIVE_N ? live_n - LIVE_N : 0, k = oldest;
    for (int j = oldest; j < newest; j++)
        if (now - live_t[j % LIVE_N] >= 1000000) k = j;
    show(st, (float)((double)(newest - k) * 1e6 / (double)(now - live_t[k % LIVE_N])), C.n_kv);
    live_last_draw = esp_timer_get_time();
}

/* ---------------------------------------------------------------- serial in */

// Discard everything received so far (input is read only at the prompt).
static void rx_drain(void) {
    uint8_t b[64];
    while (usb_serial_jtag_read_bytes(b, sizeof b, 0) > 0) {
    }
}

// Idle forever after a failure: display 0; the FAIL line again for every line the host sends
// (a client that connects late still sees why the board does nothing). Never returns.
__attribute__((noreturn)) static void fail_idle(const char *msg) {
    screen_turn(NULL, 0);
    screen_text(msg, strlen(msg)); /* the reason on the screen, too */
    show(DISP_IDLE, 0, 0);
    outf("%s\n", msg);
    for (;;) {
        uint8_t b[64];
        int n = usb_serial_jtag_read_bytes(b, sizeof b, portMAX_DELAY);
        if (n > 0 && (memchr(b, '\n', (size_t)n) || memchr(b, '\r', (size_t)n))) outf("%s\n", msg);
    }
}

static char fail_msg[256];
#define FAIL(...) (snprintf(fail_msg, sizeof fail_msg, "ELLM FAIL " __VA_ARGS__), fail_idle(fail_msg))

/* ---------------------------------------------------------------- boot (shared: ellm_boot.h) */

#if CONFIG_CHAT_PREFETCH
#define PREFETCH 1
#else
#define PREFETCH 0
#endif
#if CONFIG_CHAT_DISPLAY
#define DISPLAY_ON 1 /* banner display=on|off */
#else
#define DISPLAY_ON 0
#endif

static char err[160];

static void load_pack(void) {
    if (ellm_boot_load(&M, &B, err, sizeof err)) FAIL("%s", err);
    int r;
    if (!M.tok_blob) fail_idle(NO_TOK_MSG);
    if ((r = ellm_tok_init(&T, M.tok_blob, M.tok_len)) != 0) FAIL("ellm_tok_init = %d", r);
    if (M.h.kv_type != 1 || !M.need_q || !M.need_lut)
        FAIL("this firmware expects an int8-KV, T158/Q2 pack (kv_type %d)", (int)M.h.kv_type);
}

// Arena (PSRAM): staged core units, KV, logits; the rest of the core stays in flash (spill).
// Internal block: LUT, xq, attention scores, floats.
static void state_setup(void) {
    ellm_boot_cfg cfg = {.reserve = ELLM_PSRAM_HEADROOM + PSRAM_SLACK, .core_reserve = ELLM_SPILL_RESERVE,
                         .force_spill = (size_t)CONFIG_ELLM_FORCE_SPILL_KB * 1024u /* debug, default 0 */};
    int r = ellm_boot_stage(&M, &S, &cfg, &B, err, sizeof err);
    if (B.clamped)
        outf("ELLM WARN seq_len clamped %d -> %d (PSRAM free %u)\n", B.seq_header, B.seq, (unsigned)B.ps_free);
    if (r) FAIL("%s", err);
    outf("ELLM stage: psram=%u flash_spill=%u spilled=%d tensors\n", (unsigned)B.plan.psram_bytes,
           (unsigned)B.plan.flash_bytes, B.plan.spilled);
}

/* ---------------------------------------------------------------- chat turn */

typedef struct {
    ellm_turn_stats *st;
    int abort_pending; /* 0x03 seen before the first tick */
    int64_t t_prefill_end;
} turn_ctx;

static void on_emit(void *ctx, const uint8_t *b, int n) {
    (void)ctx;
    out(b, (size_t)n);
    screen_text(b, (size_t)n);
}

// After every forward: the display, then the input (0x03 stops the reply; the rest is discarded).
static int on_tick(void *ctx) {
    turn_ctx *tc = ctx;
    int64_t now = esp_timer_get_time();
    if (tc->st->new_tokens == 0) tc->t_prefill_end = now;
    live_forward_done(tc->st->new_tokens == 0 ? DISP_THINKING : DISP_WRITING);
    uint8_t b[64];
    int n, stop = tc->abort_pending;
    while ((n = usb_serial_jtag_read_bytes(b, sizeof b, 0)) > 0)
        if (memchr(b, 0x03, (size_t)n)) stop = 1;
    return stop;
}

static void run_turn(const char *line, size_t len, int abort_pending) {
    ellm_turn_stats st = {0};
    turn_ctx tc = {.st = &st, .abort_pending = abort_pending, .t_prefill_end = esp_timer_get_time()};
    screen_turn(line, len);
    show(DISP_THINKING, 0, C.n_kv); /* between turns: before the first forward */
    live_reset();
    int r = ellm_chat_turn(&C, line, len, on_emit, on_tick, &tc, &st);
    int64_t t_end = esp_timer_get_time();
    if (r < 0) screen_text("(error)", 7);
    else if (st.stop == ELLM_STOP_TOOLONG) screen_text("(message too long)", 18);
    if (r < 0) ellm_chat_reset(&C);
    show(DISP_IDLE, 0, C.n_hist);
    rx_drain();
    if (r < 0) {
        outf("ELLM FAIL chat turn = %d (new conversation)\n", r);
    } else if (st.stop == ELLM_STOP_TOOLONG) {
        outf("! message too long\n");
    } else {
        double dt = (double)(t_end - tc.t_prefill_end) / 1e6;
        double rate = st.new_tokens > 0 && dt > 0 ? st.new_tokens / dt : 0.0;
        outf("\n[%d tok, %.2f tok/s", st.new_tokens, rate);
        if (st.dropped_turns) outf(", dropped %d earlier turns", st.dropped_turns);
        if (tx_lost) outf(", serial_lost=%" PRIu32, tx_lost);
#if CONFIG_CHAT_DEBUG_STACK // lifetime minimum free stack (bytes) of the main task and core-1 worker
        outf(", stack_free=%u worker_stack_free=%u", (unsigned)uxTaskGetStackHighWaterMark(NULL),
             (unsigned)uxTaskGetStackHighWaterMark(ellm_dual_worker()));
#endif
        outf("]%s\n", st.stop == ELLM_STOP_CTX ? " (context full)" : st.stop == ELLM_STOP_ABORT ? " (stopped)" : "");
    }
    prompt();
}

/* ---------------------------------------------------------------- main */

void app_main(void) {
    usb_serial_jtag_driver_config_t ucfg = {.tx_buffer_size = TX_BUF, .rx_buffer_size = RX_BUF};
    esp_err_t ue = usb_serial_jtag_driver_install(&ucfg);
    if (ue == ESP_OK) {
        usb_serial_jtag_vfs_use_driver();
        usb_serial_jtag_vfs_set_tx_line_endings(ESP_LINE_ENDINGS_LF); /* raw bytes: no CR added */
    } else {
        printf("ELLM FAIL usb_serial_jtag_driver_install: %s\n", esp_err_to_name(ue)); /* no driver: VFS */
        show(DISP_IDLE, 0, 0);
        for (;;) vTaskDelay(pdMS_TO_TICKS(10000));
    }
    outf("\nELLM chat: loading ...\n");
    load_pack();
    state_setup(); // the model's internal RAM before the display's
    int r;
    ellm_dual_init();
    ellm_matvec_hook = ellm_matvec_dual;
    ellm_lut_hook = ellm_lut_dual;
    ellm_attn_hook = ellm_attn_dual;
    if ((r = ellm_boot_autoload(&B, PREFETCH)) != 0) FAIL("autoload region config ret %d", r);
    show(DISP_IDLE, 0, 0); /* the display's internal RAM after the model's */
    hist = heap_caps_malloc((size_t)S.seq_len * sizeof(uint16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!hist) FAIL("alloc history (%d ids)", S.seq_len);
    r = ellm_chat_init(&C, &M, &S, &T, hist, S.seq_len);
    if (r) FAIL("ellm_chat_init = %d (seq_len %d)", r, S.seq_len);
    size_t ps_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (ps_free < ELLM_PSRAM_HEADROOM)
        FAIL("PSRAM headroom %u B < %u B", (unsigned)ps_free, (unsigned)ELLM_PSRAM_HEADROOM);
    show(DISP_IDLE, 0, C.n_hist);
    outf("ELLM chat ready: pack fnv1a=0x%08" PRIx32 " seq_len=%d prefetch=%s display=%s psram_free=%u "
           "internal_free=%u boot_ms=%lld\n", M.h.fnv1a, S.seq_len, PREFETCH ? "on" : "off",
           DISPLAY_ON ? "on" : "off", (unsigned)ps_free,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (long long)(esp_timer_get_time() / 1000));
    esp_log_level_set("*", ESP_LOG_ERROR); // no INFO/WARN lines inside the reply protocol from here on
    rx_drain();
    prompt();

    static char line[LINE_MAX_BYTES];
    size_t len = 0;
    int overflow = 0;
    for (;;) {
        uint8_t b[128];
        int n = usb_serial_jtag_read_bytes(b, sizeof b, portMAX_DELAY);
        for (int i = 0; i < n; i++) {
            uint8_t c = b[i];
            if (c == 0x03) { // Ctrl-C at the prompt: drop the partial line
                len = 0;
                overflow = 0;
                continue;
            }
            if (c != '\r' && c != '\n') {
                if (len < LINE_MAX_BYTES) line[len++] = (char)c;
                else overflow = 1;
                continue;
            }
            if (overflow) {
                outf("! line too long (max %d bytes)\n", LINE_MAX_BYTES);
            } else if (len == 0) {
                continue; // empty line: ignored, still at the prompt
            } else if (len == 6 && memcmp(line, "/reset", 6) == 0) {
                ellm_chat_reset(&C);
                screen_turn(NULL, 0);
                show(DISP_IDLE, 0, C.n_hist);
                outf("(new conversation)\n");
            } else {
                // the rest of this chunk arrived after the line: discarded, but a 0x03 in it stops
                int ap = memchr(b + i + 1, 0x03, (size_t)(n - i - 1)) != NULL;
                run_turn(line, len, ap);
                len = 0;
                overflow = 0;
                break; // run_turn drained the input and printed the prompt
            }
            len = 0;
            overflow = 0;
            rx_drain();
            prompt();
            break; // the rest of this chunk is discarded (one line per prompt)
        }
    }
}
