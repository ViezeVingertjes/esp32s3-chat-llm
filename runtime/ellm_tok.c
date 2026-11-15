/* Byte-level BPE tokenizer over an etok blob (layout and algorithm: ellm_tok.h). */
#include "ellm_tok.h"

#include <string.h>

#include "ellm.h"

#define NO_ID 0xFFFFu
#define NO_RANK 0xFFFFu
#define EMPTY_KEY 0xFFFFFFFFu
#define BAD_CP 0xFFFFFFFFu
#define HASH_MUL 0x9E3779B1u

enum { C_O, C_L, C_N, C_S };

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* [off, off + size) inside the blob (past the header) and 4-aligned when align. */
static int section(uint64_t off, uint64_t size, size_t len, int align) {
    if (off < ELLM_TOK_HDR_BYTES || off > len || size > len - off) return ELLM_ERR_BOUNDS;
    if (align && (off & 3)) return ELLM_ERR_SHAPE;
    return 0;
}

static int ranges_ok(const uint32_t *r, int n) {
    for (int i = 0; i < n; i++) {
        if (r[2 * i] > r[2 * i + 1] || r[2 * i + 1] > 0x10FFFF) return 0;
        if (i > 0 && r[2 * i] <= r[2 * i - 1]) return 0;
    }
    return 1;
}

int ellm_tok_init(ellm_tok *t, const uint8_t *b, size_t len) {
    memset(t, 0, sizeof *t);
    if (!b || len < ELLM_TOK_HDR_BYTES || ((uintptr_t)b & 3)) return ELLM_ERR_BOUNDS;
    if (memcmp(b, ELLM_TOK_MAGIC, 4) != 0 || rd32(b + 4) != ELLM_TOK_VERSION) return ELLM_ERR_SHAPE;
    if (rd32(b + 8) != len) return ELLM_ERR_BOUNDS;
    uint32_t vocab = rd32(b + 12), n_merges = rd32(b + 16), bits = rd32(b + 20);
    uint32_t max_tok = rd32(b + 24), n_l = rd32(b + 40), n_n = rd32(b + 44), n_s = rd32(b + 48);
    uint32_t n_tok_bytes = rd32(b + 60);
    if (vocab < 1 || vocab > 0xFFFF || bits < 1 || bits > 16 || n_merges >= (1u << bits) ||
        n_merges >= NO_RANK)
        return ELLM_ERR_SHAPE;
    uint64_t n_ranges = (uint64_t)n_l + n_n + n_s;
    int r;
    if ((r = section(rd32(b + 28), 512, len, 1)) || (r = section(rd32(b + 32), (uint64_t)8 << bits, len, 1)) ||
        (r = section(rd32(b + 36), 8 * n_ranges, len, 1)) ||
        (r = section(rd32(b + 52), 4 * ((uint64_t)vocab + 1), len, 1)) ||
        (r = section(rd32(b + 56), n_tok_bytes, len, 0)))
        return r;
    const uint16_t *byte_ids = (const uint16_t *)(b + rd32(b + 28));
    const ellm_tok_merge *hash = (const ellm_tok_merge *)(b + rd32(b + 32));
    const uint32_t *ranges = (const uint32_t *)(b + rd32(b + 36));
    const uint32_t *tok_off = (const uint32_t *)(b + rd32(b + 52));

    uint32_t longest = 0;
    if (tok_off[0] != 0 || tok_off[vocab] != n_tok_bytes) return ELLM_ERR_SHAPE;
    for (uint32_t i = 0; i < vocab; i++) {
        if (tok_off[i + 1] < tok_off[i]) return ELLM_ERR_SHAPE;
        if (tok_off[i + 1] - tok_off[i] > longest) longest = tok_off[i + 1] - tok_off[i];
    }
    if (longest != max_tok) return ELLM_ERR_SHAPE;
    for (int i = 0; i < 256; i++)
        if (byte_ids[i] >= vocab && byte_ids[i] != NO_ID) return ELLM_ERR_SHAPE;
    if (!ranges_ok(ranges, (int)n_l) || !ranges_ok(ranges + 2 * n_l, (int)n_n) ||
        !ranges_ok(ranges + 2 * (n_l + n_n), (int)n_s))
        return ELLM_ERR_SHAPE;
    uint32_t used = 0;
    for (uint32_t i = 0; i < (1u << bits); i++) {
        const ellm_tok_merge *e = &hash[i];
        if (e->key == EMPTY_KEY) continue;
        if ((e->key >> 16) >= vocab || (e->key & 0xFFFF) >= vocab || e->id >= vocab || e->rank == NO_RANK)
            return ELLM_ERR_SHAPE;
        used++;
    }
    if (used != n_merges) return ELLM_ERR_SHAPE; /* also guarantees an empty slot: probes end */

    t->vocab = (int)vocab;
    t->n_merges = (int)n_merges;
    t->hash_bits = (int)bits;
    t->max_tok_bytes = (int)max_tok;
    t->n_l = (int)n_l;
    t->n_n = (int)n_n;
    t->n_s = (int)n_s;
    t->unicode_version = rd32(b + 64);
    t->byte_ids = byte_ids;
    t->hash = hash;
    t->r_l = ranges;
    t->r_n = ranges + 2 * n_l;
    t->r_s = ranges + 2 * (n_l + n_n);
    t->tok_off = tok_off;
    t->tok_bytes = b + rd32(b + 56);
    return 0;
}

/* ---- codepoints ---- */

/* Strict UTF-8 decode at s[i] (i < n): codepoint (BAD_CP for an invalid byte) and its length. */
static size_t utf8_next(const uint8_t *s, size_t n, size_t i, uint32_t *cp) {
    uint8_t c = s[i];
    *cp = BAD_CP;
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    size_t k;
    uint32_t v, lo = 0x80, hi = 0xBF; /* allowed range of the first continuation byte */
    if (c >= 0xC2 && c <= 0xDF) { k = 1; v = c & 0x1F; }
    else if (c >= 0xE0 && c <= 0xEF) {
        k = 2; v = c & 0x0F;
        if (c == 0xE0) lo = 0xA0;
        if (c == 0xED) hi = 0x9F; /* no surrogates */
    } else if (c >= 0xF0 && c <= 0xF4) {
        k = 3; v = c & 0x07;
        if (c == 0xF0) lo = 0x90;
        if (c == 0xF4) hi = 0x8F;
    } else
        return 1;
    if (n - i <= k) return 1;
    for (size_t j = 1; j <= k; j++) {
        uint8_t d = s[i + j];
        if (j == 1 ? (d < lo || d > hi) : (d & 0xC0) != 0x80) return 1;
        v = v << 6 | (d & 0x3F);
    }
    *cp = v;
    return k + 1;
}

static int in_ranges(const uint32_t *r, int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (cp < r[2 * mid]) hi = mid - 1;
        else if (cp > r[2 * mid + 1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

static int cp_class(const ellm_tok *t, uint32_t cp) {
    if (cp < 0x80) { /* ASCII fast path (same classes as the blob's tables) */
        if ((cp | 0x20) >= 'a' && (cp | 0x20) <= 'z') return C_L;
        if (cp >= '0' && cp <= '9') return C_N;
        if (cp == ' ' || (cp >= 9 && cp <= 13)) return C_S;
        return C_O;
    }
    if (cp == BAD_CP) return C_O;
    if (in_ranges(t->r_s, t->n_s, cp)) return C_S;
    if (in_ranges(t->r_l, t->n_l, cp)) return C_L;
    if (in_ranges(t->r_n, t->n_n, cp)) return C_N;
    return C_O;
}

/* End of the run of class k starting at i. */
static size_t run_end(const ellm_tok *t, const uint8_t *s, size_t n, size_t i, int k) {
    while (i < n) {
        uint32_t c;
        size_t l = utf8_next(s, n, i, &c);
        if (cp_class(t, c) != k) break;
        i += l;
    }
    return i;
}

/* End of the pre-token starting at i (< n). */
static size_t pretok_end(const ellm_tok *t, const uint8_t *s, size_t n, size_t i) {
    uint32_t c;
    size_t l = utf8_next(s, n, i, &c);
    int k = cp_class(t, c);
    if (c == '\'' && i + 1 < n) { /* 's 't 're 've 'm 'll 'd */
        uint8_t a = s[i + 1], b = i + 2 < n ? s[i + 2] : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
    if (c == ' ' && i + 1 < n) { /* ' ?\p{L}+' and ' ?[^\s\p{L}\p{N}]+' */
        uint32_t c1;
        utf8_next(s, n, i + 1, &c1);
        int k1 = cp_class(t, c1);
        if (k1 == C_L || k1 == C_O) return run_end(t, s, n, i + 1, k1);
    }
    if (k == C_N) return i + l; /* Digits(individual_digits): each \p{N} alone */
    if (k == C_L || k == C_O) return run_end(t, s, n, i, k);
    /* whitespace: \s+(?!\S) | \s+ */
    size_t j = i, last = i;
    while (j < n) {
        uint32_t cc;
        size_t ll = utf8_next(s, n, j, &cc);
        if (cp_class(t, cc) != C_S) break;
        last = j;
        j += ll;
    }
    if (j == n) return j;
    uint32_t cn;
    utf8_next(s, n, j, &cn);
    if (cp_class(t, cn) == C_N) return j; /* segment ends before the digit */
    return last > i ? last : j;            /* leave the last space for the next pre-token */
}

/* ---- BPE ---- */

static uint16_t merge_rank(const ellm_tok *t, uint16_t a, uint16_t b, uint16_t *id) {
    uint32_t key = (uint32_t)a << 16 | b, mask = (1u << t->hash_bits) - 1;
    uint32_t h = (uint32_t)(key * HASH_MUL) >> (32 - t->hash_bits);
    for (uint32_t probe = 0; probe <= mask; probe++) {
        const ellm_tok_merge *e = &t->hash[h];
        if (e->key == key) {
            *id = e->id;
            return e->rank;
        }
        if (e->key == EMPTY_KEY) break;
        h = (h + 1) & mask;
    }
    return NO_RANK;
}

/* Byte tokens of p[0..n) -> sym (bytes missing from the vocab are dropped). */
static int load_bytes(const ellm_tok *t, const uint8_t *p, size_t n, uint16_t *sym) {
    int m = 0;
    for (size_t i = 0; i < n; i++) {
        uint16_t id = t->byte_ids[p[i]];
        if (id != NO_ID) sym[m++] = id;
    }
    return m;
}

/* Merge sym[0..m) in place (rk: m-1 scratch slots): lowest rank first, ties leftmost. */
static int bpe(const ellm_tok *t, uint16_t *sym, uint16_t *rk, int m) {
    uint16_t id;
    for (int i = 0; i + 1 < m; i++) rk[i] = merge_rank(t, sym[i], sym[i + 1], &id);
    while (m > 1) {
        int bi = -1;
        uint16_t best = NO_RANK;
        for (int i = 0; i + 1 < m; i++)
            if (rk[i] < best) {
                best = rk[i];
                bi = i;
            }
        if (bi < 0) break;
        merge_rank(t, sym[bi], sym[bi + 1], &id);
        sym[bi] = id;
        memmove(sym + bi + 1, sym + bi + 2, sizeof *sym * (size_t)(m - bi - 2));
        if (m - bi - 3 > 0) memmove(rk + bi + 1, rk + bi + 2, sizeof *rk * (size_t)(m - bi - 3));
        m--;
        if (bi > 0) rk[bi - 1] = merge_rank(t, sym[bi - 1], sym[bi], &id);
        if (bi + 1 < m) rk[bi] = merge_rank(t, sym[bi], sym[bi + 1], &id);
    }
    return m;
}

/* Encode scratch on the stack (1 KB): sym + rk of a short pre-token, or the block minima of a
 * long one. */
typedef union {
    uint16_t h[2 * ELLM_TOK_STACK];
    uint32_t w[ELLM_TOK_STACK];
} tok_scratch;

#define DEAD 0xFFFFu    /* merged-away symbol in bpe_long (never a token id: vocab <= 65535) */
#define LONG_BLOCKS 128 /* bpe_long keeps the minimum pair rank of LONG_BLOCKS position blocks */

static int next_live(const uint16_t *sym, int m, int i) {
    for (i++; i < m && sym[i] == DEAD; i++) {}
    return i;
}

static int prev_live(const uint16_t *sym, int i) {
    for (i--; i >= 0 && sym[i] == DEAD; i--) {}
    return i;
}

/* Lowest-rank (leftmost on ties) pair whose left symbol lies in positions [b*B, b*B+B). */
static void block_scan(const ellm_tok *t, const uint16_t *sym, int m, int B, int b, uint16_t *brank,
                       uint32_t *bpos) {
    uint16_t best = NO_RANK, id;
    uint32_t pos = 0;
    int lo = b * B, end = m - lo > B ? lo + B : m;
    for (int i = lo; i < end; i++) {
        if (sym[i] == DEAD) continue;
        int j = next_live(sym, m, i);
        if (j >= m) break;
        uint16_t r = merge_rank(t, sym[i], sym[j], &id);
        if (r < best) {
            best = r;
            pos = (uint32_t)i;
        }
    }
    brank[b] = best;
    bpos[b] = pos;
}

/* Same merges as bpe() for any m >= 1 with no per-pair cache: merged-away symbols become DEAD in
 * place and each of LONG_BLOCKS position blocks caches its best pair, so a merge rescans only
 * the (at most 3) blocks holding the changed pairs. Compacts sym and returns the new length. */
static int bpe_long(const ellm_tok *t, uint16_t *sym, int m, tok_scratch *sc) {
    uint16_t *brank = sc->h;                    /* [LONG_BLOCKS], bytes [0, 256) */
    uint32_t *bpos = sc->w + LONG_BLOCKS / 2;   /* [LONG_BLOCKS], bytes [256, 768) */
    int B = (m - 1) / LONG_BLOCKS + 1, nb = (m - 1) / B + 1;
    for (int b = 0; b < nb; b++) block_scan(t, sym, m, B, b, brank, bpos);
    for (;;) {
        int bb = -1;
        uint16_t best = NO_RANK, id;
        for (int b = 0; b < nb; b++)
            if (brank[b] < best) {
                best = brank[b];
                bb = b;
            }
        if (bb < 0) break;
        int i = (int)bpos[bb], j = next_live(sym, m, i), pi = prev_live(sym, i);
        merge_rank(t, sym[i], sym[j], &id);
        sym[i] = id;
        sym[j] = DEAD;
        block_scan(t, sym, m, B, bb, brank, bpos);
        if (j / B != bb) block_scan(t, sym, m, B, j / B, brank, bpos);
        if (pi >= 0 && pi / B != bb) block_scan(t, sym, m, B, pi / B, brank, bpos);
    }
    int k = 0;
    for (int i = 0; i < m; i++)
        if (sym[i] != DEAD) sym[k++] = sym[i];
    return k;
}

/* BPE of the pre-token p[0..n), appended to out. 0 or -1 (cap). Short pre-tokens run on the
 * stack; longer ones run in place in out's free tail, which needs one slot per representable
 * byte (always available when cap >= the input's byte count). Exact either way. */
static int emit(const ellm_tok *t, const uint8_t *p, size_t n, uint16_t *out, int cap, int *used,
                tok_scratch *sc) {
    int m, free_slots = cap - *used;
    if (n <= ELLM_TOK_STACK) {
        uint16_t *sym = sc->h, *rk = sc->h + ELLM_TOK_STACK;
        m = bpe(t, sym, rk, load_bytes(t, p, n, sym));
        if (m > free_slots) return -1;
        if (m) memcpy(out + *used, sym, sizeof *sym * (size_t)m);
    } else {
        size_t need = 0;
        for (size_t i = 0; i < n; i++) need += t->byte_ids[p[i]] != NO_ID;
        if (need > (size_t)free_slots) return -1;
        m = need ? bpe_long(t, out + *used, load_bytes(t, p, n, out + *used), sc) : 0;
    }
    *used += m;
    return 0;
}

int ellm_tok_encode(const ellm_tok *t, const char *text, size_t len, uint16_t *out, int cap) {
    const uint8_t *s = (const uint8_t *)text;
    tok_scratch sc;
    int used = 0;
    if (cap < 0 || (len && !text) || (cap && !out)) return -1;
    for (size_t i = 0; i < len;) {
        size_t e = pretok_end(t, s, len, i);
        if (emit(t, s + i, e - i, out, cap, &used, &sc)) return -1;
        i = e;
    }
    return used;
}

int ellm_tok_decode(const ellm_tok *t, int id, uint8_t *out, int cap) {
    if (id < 0 || id >= t->vocab) return ELLM_ERR_TOKEN;
    uint32_t a = t->tok_off[id], n = t->tok_off[id + 1] - a;
    if (cap < 0 || n > (uint32_t)cap) return -1;
    if (n) memcpy(out, t->tok_bytes + a, n);
    return (int)n;
}
