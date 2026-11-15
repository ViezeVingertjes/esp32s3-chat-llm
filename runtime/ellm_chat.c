/* Chat engine (runtime/ellm_chat.h). Plain C99: no allocation, no IO. */
#include "ellm_chat.h"

#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------- decoding rule */

/* Bans every id seq[i + n - 1] with seq[i .. i + n - 1) == prefix[0 .. n - 1) (it would complete
 * an n-gram of seq; n = 1: every id of seq, prefix unused). */
static void ban_ngrams(float *logits, int vocab, const uint16_t *seq, int len, const uint16_t *prefix,
                       int n) {
    for (int i = 0; i + n <= len; i++)
        if (n == 1 || memcmp(seq + i, prefix, sizeof(uint16_t) * (size_t)(n - 1)) == 0) {
            int id = seq[i + n - 1];
            if (id < vocab) logits[id] = -INFINITY;
        }
}

int ellm_chat_pick_prior(float *logits, int vocab, const uint16_t *out, int n_out, int ngram,
                         const uint16_t *turns, int n_turns, int asst_n, int prior_ngram) {
    for (int i = 0; i < vocab; i++)
        if (!isfinite(logits[i])) return ELLM_ERR_LOGITS;
    /* output-only no-repeat n-gram ban (ellm_py.decode, rule step 3, S = out[0..n_out)) */
    if (ngram > 0 && n_out >= ngram - 1)
        ban_ngrams(logits, vocab, out, n_out, ngram > 1 ? out + n_out - (ngram - 1) : out, ngram);
    /* prior-reply ban (rule step 4): the same prefix, scanned in each earlier reply. The replies are
     * the ids after every second <|im_start|> (the assistant header) and its asst_n header ids, up
     * to the next <|im_end|>; one pass over turns[0..n_turns), no n-gram straddles two replies. */
    if (prior_ngram > 0 && n_out >= prior_ngram - 1) {
        const uint16_t *prefix = prior_ngram > 1 ? out + n_out - (prior_ngram - 1) : out;
        int k = 0, i = 0;
        while (i < n_turns) {
            if (turns[i] != ELLM_TOK_IM_START || k++ % 2 == 0) { i++; continue; }
            int a = i + 1 + asst_n, b = a;
            while (b < n_turns && turns[b] != ELLM_TOK_IM_END) b++;
            if (b > a) ban_ngrams(logits, vocab, turns + a, b - a, prefix, prior_ngram);
            i = b;
        }
    }
    int best = 0;
    for (int i = 1; i < vocab; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

int ellm_chat_pick(float *logits, int vocab, const uint16_t *out, int n_out, int ngram) {
    return ellm_chat_pick_prior(logits, vocab, out, n_out, ngram, NULL, 0, 0, 0);
}

/* ---------------------------------------------------------------- UTF-8 streaming */

static int utf8_need(uint8_t x) {
    if (x >= 0xC2 && x <= 0xDF) return 2;
    if (x >= 0xE0 && x <= 0xEF) return 3;
    if (x >= 0xF0 && x <= 0xF4) return 4;
    return 1; /* ASCII, or a byte that cannot start a sequence: passes alone */
}

int ellm_chat_utf8_complete(const uint8_t *b, int n) {
    int i = 0, last = 0;
    while (i < n) {
        int need = utf8_need(b[i]), j = 1;
        while (j < need && i + j < n && (b[i + j] & 0xC0) == 0x80) j++;
        if (j == need) {
            i += need;
            last = i;
        } else if (i + j == n) {
            break; /* incomplete sequence at the end: hold it */
        } else {
            i++; /* lead byte not followed by its continuations: passes alone */
            last = i;
        }
    }
    return last;
}

/* Appends token id's bytes after the held ones and emits the complete part. */
static void emit_token(ellm_chat *c, int id, ellm_emit_fn emit, void *ctx) {
    int r = ellm_tok_decode(c->t, id, c->obuf + c->held, (int)sizeof c->obuf - c->held);
    if (r <= 0) return; /* special (no bytes); larger tokens are excluded by init */
    int n = c->held + r, k = ellm_chat_utf8_complete(c->obuf, n);
    if (k > 0 && emit) emit(ctx, c->obuf, k);
    memmove(c->obuf, c->obuf + k, (size_t)(n - k));
    c->held = n - k;
}

/* ---------------------------------------------------------------- template */

/* ids of E(role "\n" text) into dst[0..cap); -1 if it does not fit. */
static int encode_turn(ellm_chat *c, const char *role, const char *text, size_t len, uint16_t *dst,
                       int cap) {
    size_t rl = strlen(role);
    if (rl + 1 + len > sizeof c->ubuf) return -1;
    memcpy(c->ubuf, role, rl);
    c->ubuf[rl] = '\n';
    if (len) memcpy(c->ubuf + rl + 1, text, len);
    return ellm_tok_encode(c->t, c->ubuf, rl + 1 + len, dst, cap);
}

static int encode_piece(const ellm_tok *t, const char *s, uint16_t *dst) {
    return ellm_tok_encode(t, s, strlen(s), dst, ELLM_CHAT_PIECE);
}

static int is_special(const ellm_chat *c, int id) {
    return c->t->tok_off[id + 1] == c->t->tok_off[id]; /* added tokens have no bytes */
}

/* Forwards hist[n_kv..n_hist) (tick after each). Only the last id's logits can be used (they
 * pick the first reply token), so every other id runs without the LM head
 * (ellm_forward_nologits: same KV), and the last one too unless last_logits. 1 if tick aborted,
 * 0, or a forward error. */
static int prefill(ellm_chat *c, int last_logits, ellm_tick_fn tick, void *ctx, ellm_turn_stats *st) {
    while (c->n_kv < c->n_hist) {
        int pos = c->n_kv, id = c->hist[pos];
        int r = last_logits && pos == c->n_hist - 1 ? ellm_forward(c->m, c->s, id, pos)
                                                    : ellm_forward_nologits(c->m, c->s, id, pos);
        if (r) return r;
        c->n_kv++;
        if (st) st->prompt_tokens++;
        if (tick && tick(ctx)) return 1;
    }
    return 0;
}

int ellm_chat_init(ellm_chat *c, const ellm_model *m, ellm_state *s, const ellm_tok *t,
                   uint16_t *hist, int hist_cap) {
    memset(c, 0, sizeof *c);
    c->m = m;
    c->s = s;
    c->t = t;
    c->hist = hist;
    c->hist_cap = hist_cap;
    c->ngram = ELLM_CHAT_NGRAM;
    c->prior_ngram = ELLM_CHAT_PRIOR_NGRAM;
    c->max_new = ELLM_CHAT_MAX_NEW;
    if (!hist || hist_cap < s->seq_len || t->vocab != m->h.vocab) return ELLM_ERR_SHAPE;
    if (t->max_tok_bytes > ELLM_CHAT_TOK_BYTES) return ELLM_ERR_SHAPE;
    c->nl_n = encode_piece(t, "\n", c->nl);
    c->asst_n = encode_piece(t, "assistant\n", c->asst);
    if (c->nl_n < 1 || c->asst_n < 1) return ELLM_ERR_SHAPE;
    int n = encode_turn(c, "system", ELLM_CHAT_SYSTEM, strlen(ELLM_CHAT_SYSTEM), c->uids,
                        (int)(sizeof c->uids / sizeof c->uids[0]));
    if (n < 0) return ELLM_ERR_SHAPE;
    int sys_len = 1 + n + 1 + c->nl_n;
    if (sys_len + ELLM_CHAT_RESERVE > s->seq_len) return ELLM_ERR_SHAPE;
    hist[0] = ELLM_TOK_IM_START;
    memcpy(hist + 1, c->uids, sizeof(uint16_t) * (size_t)n);
    hist[1 + n] = ELLM_TOK_IM_END;
    memcpy(hist + 2 + n, c->nl, sizeof(uint16_t) * (size_t)c->nl_n);
    c->n_hist = c->sys_len = c->reply_start = sys_len;
    return prefill(c, 0, NULL, NULL, NULL); /* the next turn appends ids: no logits needed */
}

void ellm_chat_reset(ellm_chat *c) {
    c->n_hist = c->reply_start = c->sys_len;
    if (c->n_kv > c->sys_len) c->n_kv = c->sys_len;
    c->n_pairs = 0;
    c->held = 0;
}

uint32_t ellm_chat_hist_fnv1a(const ellm_chat *c) {
    uint32_t h = 0x811C9DC5u;
    for (int i = 0; i < c->n_hist; i++) {
        h = (h ^ (uint32_t)(c->hist[i] & 0xFF)) * 0x01000193u;
        h = (h ^ (uint32_t)(c->hist[i] >> 8)) * 0x01000193u;
    }
    return h;
}

/* Index in hist of the start of pair k (0 = oldest), or n_hist for k == n_pairs. Pairs start at
 * every other <|im_start|> after the system turn: user content never holds special ids (plain
 * encode) and replies never keep one (a special stops the reply). */
static int pair_start(const ellm_chat *c, int k) {
    int seen = 0;
    for (int i = c->sys_len; i < c->n_hist; i++)
        if (c->hist[i] == ELLM_TOK_IM_START && seen++ == 2 * k) return i;
    return c->n_hist;
}

int ellm_chat_turn(ellm_chat *c, const char *user, size_t len, ellm_emit_fn emit,
                   ellm_tick_fn tick, void *ctx, ellm_turn_stats *st) {
    ellm_turn_stats dummy;
    if (!st) st = &dummy;
    memset(st, 0, sizeof *st);
    const int seq = c->s->seq_len, vocab = c->m->h.vocab, close_n = 1 + c->nl_n;
    const int ucap = (int)(sizeof c->uids / sizeof c->uids[0]);

    /* the new block: <|im_start|> E("user\n" text) <|im_end|> E("\n") <|im_start|> E("assistant\n") */
    int n = len > ELLM_CHAT_MAX_USER ? -1
                                     : encode_turn(c, "user", user, len, c->uids + 1,
                                                   ucap - 1 - close_n - 1 - c->asst_n);
    if (n < 0) {
        st->stop = ELLM_STOP_TOOLONG;
        return 0;
    }
    uint16_t *u = c->uids;
    int nb = 1 + n;
    u[0] = ELLM_TOK_IM_START;
    u[nb++] = ELLM_TOK_IM_END;
    memcpy(u + nb, c->nl, sizeof(uint16_t) * (size_t)c->nl_n);
    nb += c->nl_n;
    u[nb++] = ELLM_TOK_IM_START;
    memcpy(u + nb, c->asst, sizeof(uint16_t) * (size_t)c->asst_n);
    nb += c->asst_n;
    if (c->sys_len + nb + ELLM_CHAT_RESERVE > seq) {
        st->stop = ELLM_STOP_TOOLONG;
        return 0;
    }

    /* context policy (ruling P4-R2): drop the oldest pairs until the kept turns are at most
     * ELLM_CHAT_KEEP(seq) ids and the block plus the reserve fit; re-prefill only those (their
     * stored ids, moved down: replies are never re-encoded) */
    if (c->n_hist + nb + ELLM_CHAT_RESERVE > seq) {
        int k = 0, cut = c->sys_len;
        while (c->n_hist - cut > ELLM_CHAT_KEEP(seq) ||
               c->sys_len + (c->n_hist - cut) + nb + ELLM_CHAT_RESERVE > seq)
            cut = pair_start(c, ++k);
        int kept = c->n_hist - cut;
        memmove(c->hist + c->sys_len, c->hist + cut, sizeof(uint16_t) * (size_t)kept);
        c->n_hist = c->sys_len + kept;
        c->n_kv = c->sys_len;
        c->n_pairs -= k;
        st->dropped_turns = k;
    }
    const int blk = c->n_hist; /* hist[sys_len..blk): the earlier pairs (the prior-reply ban scans them) */
    memcpy(c->hist + c->n_hist, u, sizeof(uint16_t) * (size_t)nb);
    c->n_hist += nb;
    c->n_pairs++;
    c->reply_start = c->n_hist;
    c->held = 0;

    int r = prefill(c, 1, tick, ctx, st);
    if (r < 0) return r;
    if (r == 1) st->stop = ELLM_STOP_ABORT;

    /* decode: logits are those of hist[n_hist - 1] (n_kv == n_hist) whenever a pick runs */
    while (r == 0) {
        if (st->new_tokens >= c->max_new) { st->stop = ELLM_STOP_MAXNEW; break; }
        if (c->n_hist + 1 + close_n > seq) { st->stop = ELLM_STOP_CTX; break; }
        int t = ellm_chat_pick_prior(c->s->logits, vocab, c->hist + c->reply_start,
                                     c->n_hist - c->reply_start, c->ngram, c->hist + c->sys_len,
                                     blk - c->sys_len, c->asst_n, c->prior_ngram);
        if (t < 0) return t;
        if (is_special(c, t)) { st->stop = ELLM_STOP_EOT; break; }
        c->hist[c->n_hist++] = (uint16_t)t;
        st->new_tokens++;
        emit_token(c, t, emit, ctx);
        /* forward only if another pick will follow (else it is prefilled lazily next turn) */
        if (st->new_tokens >= c->max_new || c->n_hist + 1 + close_n > seq) continue;
        if ((r = ellm_forward(c->m, c->s, t, c->n_kv)) != 0) return r;
        c->n_kv++;
        if (tick && tick(ctx)) { st->stop = ELLM_STOP_ABORT; break; }
    }
    c->held = 0; /* an incomplete UTF-8 tail at the stop is never emitted (its ids stay) */
    c->hist[c->n_hist++] = ELLM_TOK_IM_END;
    memcpy(c->hist + c->n_hist, c->nl, sizeof(uint16_t) * (size_t)c->nl_n);
    c->n_hist += c->nl_n;
    return 0;
}
