#ifndef ELLM_CHAT_H
#define ELLM_CHAT_H
/* Chat engine: SmolLM2 chat template, token history, context policy, greedy decoding with the
 * output-only n-gram loop breaker and the prior-reply n-gram ban, UTF-8-safe streaming. Plain C99, shared by the host
 * (libellm.so, tools/chat_host.py) and the ESP32-S3 firmware. No allocation and no IO: the
 * history is caller-provided, every other buffer (encode scratch, token bytes) lives in the
 * ellm_chat struct (~3.5 KB; make it static on the device, not a stack local). Stack use per call
 * is small (ellm_tok_encode's ~1.1 KB plus the forward pass).
 *
 * History invariant: hist[0..n_hist) holds the ACTUAL ids the model consumed (or will consume
 * at the next turn's prefill), in order:
 *   system turn  <|im_start|> E("system\n" ELLM_CHAT_SYSTEM) <|im_end|> E("\n")
 *   per pair     <|im_start|> E("user\n" text) <|im_end|> E("\n") <|im_start|> E("assistant\n")
 *                <the generated reply ids> <|im_end|> E("\n")
 * where E is the plain-text encoder (ellm_tok_encode; special strings in user text are plain
 * text) and "user\n" text is one string, exactly how `tokenizers` segments the rendered template
 * between special tokens. The reply part is the generated ids as generated, closed with
 * <|im_end|> E("\n") after every stop (EOT, MAXNEW, CTX, ABORT). Re-prefill after a drop reuses
 * these stored ids; reply text is never re-encoded. With this tokenizer E("\n") = [198] and
 * E("assistant\n") = [520, 9472, 198], so every reply ends with [2, 198]. Consequently the
 * history equals ellm_py.etok.encode_chat(messages, add_generation_prompt=False) for the
 * template and user parts always, and for a reply whenever its ids are the canonical encoding
 * of its text (true of greedy output in practice, not guaranteed).
 *
 * KV cache: positions [0, n_kv) hold the KV of hist[0..n_kv); n_kv <= n_hist. Forward work is kept to what is needed:
 * the closing <|im_end|> "\n" (and the last reply token) are
 * forwarded lazily at the start of the next turn. Prefill forwards whose logits are not used
 * (every prompt id but the last of a turn, and the whole system turn) run without the LM head
 * (ellm_forward_nologits: the same KV, so the same replies). The system turn is prefilled once by
 * ellm_chat_init; positions below sys_len are never rewritten, so ellm_chat_reset needs no
 * forward. */
#include <stddef.h>
#include <stdint.h>
#include "ellm.h"
#include "ellm_tok.h"

#define ELLM_CHAT_MAX_NEW 256      /* reply cap (new tokens) */
#define ELLM_CHAT_NGRAM 4          /* loop breaker n (output-only no-repeat n-gram ban); 0 = off */
#ifndef ELLM_CHAT_PRIOR_NGRAM
#define ELLM_CHAT_PRIOR_NGRAM 4    /* prior-reply ban m: no m-gram of an earlier reply; 0 = off.
                                    * 4 with the shipped n = 4: the 8-prompt repetition sweep in the
                                    * README scored 4/4 best. The earlier default, m = 3 with
                                    * n = 8 (ruling P4-R16, host basic chat check), passed that
                                    * check too. */
#endif
#define ELLM_CHAT_RESERVE 128      /* context room kept free for the reply (context policy) */
#define ELLM_CHAT_KEEP(seq_len) ((seq_len) / 4) /* most ids of kept turns after an overflow */
#define ELLM_CHAT_MAX_USER 1024    /* longest user message in bytes (longer: ELLM_STOP_TOOLONG) */
#define ELLM_CHAT_TOK_BYTES 64     /* token byte strings the engine can stream (>= max_tok_bytes) */
#define ELLM_CHAT_PIECE 16         /* longest template piece (E("\n"), E("assistant\n")) in ids */
#define ELLM_CHAT_SYSTEM "You are a helpful AI assistant named SmolLM, trained by Hugging Face"

#define ELLM_ERR_LOGITS -11        /* the forward produced a NaN or infinite logit */

enum { ELLM_STOP_EOT = 0, ELLM_STOP_MAXNEW = 1, ELLM_STOP_CTX = 2, ELLM_STOP_TOOLONG = 3,
       ELLM_STOP_ABORT = 4 };

/* Per turn. Updated live while the turn runs, so a tick callback may read it through its ctx.
 * prompt_tokens: forwards before the reply this turn (pending ids of the previous turn,
 * re-prefill after a drop, the new user message); new_tokens: reply ids; dropped_turns: oldest
 * user+assistant pairs dropped by the context policy this turn; stop: ELLM_STOP_*. */
typedef struct { int prompt_tokens, new_tokens, dropped_turns, stop; } ellm_turn_stats;

/* Receives reply bytes, only complete UTF-8 sequences (a character split across tokens is held
 * back until it is complete). Bytes that are not valid UTF-8 lead/continuation structure pass
 * one by one. An incomplete sequence still held when the reply stops (MAXNEW, CTX, ABORT or a
 * special-token stop) is dropped, never emitted (its ids stay in the history), so the
 * concatenated emits equal the reply ids' bytes up to ellm_chat_utf8_complete of them. */
typedef void (*ellm_emit_fn)(void *ctx, const uint8_t *b, int n);
/* Called after every forward of a turn (prefill and decode; the display hook). Nonzero aborts
 * the reply (ELLM_STOP_ABORT); the history stays consistent (partial reply, then closed). */
typedef int (*ellm_tick_fn)(void *ctx);

typedef struct {
    const ellm_model *m;
    ellm_state *s;
    const ellm_tok *t;
    uint16_t *hist;            /* caller's history buffer, hist_cap >= s->seq_len ids */
    int hist_cap;
    int n_hist;                /* ids in hist */
    int n_kv;                  /* ids of hist whose KV is in the cache (prefix) */
    int sys_len;               /* ids of the system turn */
    int n_pairs;               /* user+assistant pairs in hist */
    int reply_start;           /* hist index of the current/last reply's first id */
    int ngram;                 /* loop breaker n, ELLM_CHAT_NGRAM at init (tests may change it) */
    int prior_ngram;           /* prior-reply ban n, ELLM_CHAT_PRIOR_NGRAM at init (tests may change it) */
    int max_new;               /* reply cap, ELLM_CHAT_MAX_NEW at init (tests may change it) */
    int nl_n, asst_n;
    uint16_t nl[ELLM_CHAT_PIECE];    /* E("\n") */
    uint16_t asst[ELLM_CHAT_PIECE];  /* E("assistant\n") */
    int held;                        /* bytes of an incomplete UTF-8 sequence in obuf[0..held) */
    uint8_t obuf[4 + ELLM_CHAT_TOK_BYTES];
    char ubuf[ELLM_CHAT_MAX_USER + 16];            /* role "\n" content */
    uint16_t uids[ELLM_CHAT_MAX_USER + 16 + 3 * ELLM_CHAT_PIECE]; /* encode scratch (>= bytes) */
} ellm_chat;

/* Sets up the engine over a model, its state (s->seq_len = context) and its tokenizer, and
 * prefills the default system turn (sys_len forwards without logits, no tick). 0, ELLM_ERR_SHAPE (hist_cap <
 * seq_len, vocab mismatch, a template piece or token too long, context too small for the system
 * turn + ELLM_CHAT_RESERVE) or a forward error. */
int ellm_chat_init(ellm_chat *c, const ellm_model *m, ellm_state *s, const ellm_tok *t,
                   uint16_t *hist, int hist_cap);

/* One user turn: user[0..len) (UTF-8, plain text; may contain NUL or invalid bytes) is encoded,
 * the context policy applied, the prompt prefilled and the reply decoded (greedy over the
 * logits after the loop breaker and the prior-reply ban over the kept earlier replies,
 * ellm_chat_pick_prior) and streamed through emit (may be NULL). The
 * reply stops on a special token (ELLM_STOP_EOT; <|im_end|>, <|endoftext|>, <|im_start|> or any
 * other added token, which is not kept), after max_new ids (MAXNEW), when the next id and the
 * closing ids would not fit the context (CTX), or when tick returns nonzero (ABORT).
 * Context policy (ruling P4-R2): when n_hist + message ids + ELLM_CHAT_RESERVE > seq_len, the
 * oldest whole user+assistant pairs are dropped until the kept turns (after the system turn)
 * total <= ELLM_CHAT_KEEP(seq_len) ids and the message + reserve fit; the kept ids are moved
 * down to sys_len and re-prefilled (<= seq_len/4 forwards, then room for several more turns). A message that does not fit even after dropping every pair (or
 * longer than ELLM_CHAT_MAX_USER bytes) gives ELLM_STOP_TOOLONG and leaves the history and the
 * cache untouched. Returns 0 with *st filled, or a negative ELLM_ERR_* (forward failure,
 * ELLM_ERR_LOGITS); after an error call ellm_chat_reset. */
int ellm_chat_turn(ellm_chat *c, const char *user, size_t len, ellm_emit_fn emit,
                   ellm_tick_fn tick, void *ctx, ellm_turn_stats *st);

/* New conversation: n_hist = n_kv = sys_len, no pairs. No forward needed. */
void ellm_chat_reset(ellm_chat *c);

/* The decoding rule: checks that logits[0..vocab) are finite (else ELLM_ERR_LOGITS), bans (sets
 * to -INFINITY, in place) every id that would complete an ngram-gram already present in
 * out[0..n_out) (the reply so far: ellm_py.decode.apply_decoding with no_repeat_ngram=ngram,
 * ngram_scope="output"; ngram 0 = off), and returns the greedy argmax (lowest id on ties). */
int ellm_chat_pick(float *logits, int vocab, const uint16_t *out, int n_out, int ngram);

/* ellm_chat_pick plus the prior-reply ban (ruling P4-R16; ellm_py.decode rule step 4, prior_ngram =
 * m, 0 = off): also bans every id that would complete an m-gram of an EARLIER assistant reply,
 * the prefix being the last m-1 ids of out (nothing until n_out >= m-1). turns[0..n_turns) are
 * whole user+assistant pairs in the history layout (hist[sys_len..) up to the current user
 * block); the replies are the ids between each assistant header (every second <|im_start|>, then
 * asst_n ids) and the next <|im_end|>. The system turn and the user turns are never scanned, so a
 * reply may quote the user. One pass over turns per call (<= seq_len ids), no allocation. */
int ellm_chat_pick_prior(float *logits, int vocab, const uint16_t *out, int n_out, int ngram,
                         const uint16_t *turns, int n_turns, int asst_n, int prior_ngram);

/* Length of the longest prefix of b[0..n) that ends on a complete UTF-8 sequence, by the
 * streaming rule above (the rest is an incomplete sequence of at most 3 bytes). */
int ellm_chat_utf8_complete(const uint8_t *b, int n);

/* FNV-1a 32 over hist[0..n_hist) as little-endian u16 bytes. */
uint32_t ellm_chat_hist_fnv1a(const ellm_chat *c);

#endif
