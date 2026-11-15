#ifndef ELLM_TOK_H
#define ELLM_TOK_H
/* Byte-level BPE tokenizer (encode/decode) over an etok blob. Plain C99, shared by the host
 * (libellm.so) and the ESP32-S3 firmware. No allocation and no RAM tables: ellm_tok only points
 * into the blob, which may live in memory-mapped flash. Encode uses ~1.1 KB of stack.
 *
 * Reference: `tokenizers` (tokenizer.json) with encode_special_tokens=True and
 * add_special_tokens=False (ellm_py.etok.encode_plain): special-token strings are tokenized as
 * ordinary text, so ellm_tok_encode never emits special ids. Pipeline: no normalizer;
 * pre-tokenizer Sequence[Digits(individual_digits), ByteLevel(add_prefix_space=false, GPT-2
 * regex)]; BPE without dropout / continuing_subword_prefix / end_of_word_suffix /
 * byte_fallback / ignore_merges (ellm_py.etok.build_etok rejects tokenizers that set them).
 *
 * Blob format "etok" v1 (little-endian; written by ellm_py/etok.py build_etok; <= 1 MiB):
 *   header (68 bytes), u32 fields after the 4-byte magic "ETOK":
 *     [ 4] version = 1           [ 8] total_len (== blob length)
 *     [12] vocab (<= 65535)      [16] n_merges (< 1<<hash_bits, < 65535)
 *     [20] hash_bits (1..16)     [24] max_tok_bytes (longest token byte string)
 *     [28] off_byte_ids          [32] off_hash            [36] off_ranges
 *     [40] n_l  [44] n_n  [48] n_s   (range counts)
 *     [52] off_tok_off           [56] off_tok_bytes       [60] n_tok_bytes
 *     [64] unicode_version (major<<16 | minor<<8 | patch of the \p{L}/\p{N} tables)
 *   offsets are from the blob start; every section except tok_bytes is 4-byte aligned:
 *   byte_ids  u16[256]: byte b -> the token of ByteLevel char(b); 0xFFFF if the vocab lacks it
 *             (that byte is dropped, as `tokenizers` does without an unk token).
 *   hash      ellm_tok_merge[1 << hash_bits]: open addressing, linear probing; key = left<<16 |
 *             right, home slot = (key * 0x9E3779B1) >> (32 - hash_bits) (u32 arithmetic); empty
 *             slots have key 0xFFFFFFFF. rank = merge index in tokenizer.json (lower merges
 *             first), id = the merged token.
 *   ranges    u32[n_l + n_n + n_s][2]: inclusive, sorted, disjoint codepoint ranges of \p{L},
 *             then \p{N} (Python unicodedata general categories L* / N*), then \s (Unicode
 *             White_Space).
 *   tok_off   u32[vocab + 1]: token i's bytes are tok_bytes[tok_off[i], tok_off[i+1]);
 *             tok_off[0] = 0, non-decreasing, tok_off[vocab] = n_tok_bytes. Special (added)
 *             tokens have empty byte strings.
 *   tok_bytes u8[n_tok_bytes] (padded with zeros to a multiple of 4 at the blob end).
 *
 * Pre-tokenization (one pass, equivalent to Digits then the GPT-2 regex
 * 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ applied to each
 * digit-free segment): every \p{N} codepoint is its own pre-token and ends the segment (so a
 * whitespace run before it keeps its last space). Invalid UTF-8 bytes are single codepoints
 * outside L/N/\s. BPE per pre-token: repeatedly merge the adjacent pair with the lowest rank,
 * ties to the leftmost (identical to `tokenizers`' priority-queue merge). Encode is exact for
 * every pre-token length: pre-tokens up to ELLM_TOK_STACK bytes run in a stack buffer; longer
 * ones run in place in out's free tail, which needs one free slot per (representable) byte of
 * that pre-token, so cap >= len always suffices (natural sizing). Below that, encode returns -1
 * rather than different ids.
 */
#include <stddef.h>
#include <stdint.h>

#define ELLM_TOK_EOT 0       /* <|endoftext|> */
#define ELLM_TOK_IM_START 1  /* <|im_start|> */
#define ELLM_TOK_IM_END 2    /* <|im_end|> */

#define ELLM_TOK_MAGIC "ETOK"
#define ELLM_TOK_VERSION 1
#define ELLM_TOK_HDR_BYTES 68
#define ELLM_TOK_STACK 256   /* pre-token bytes handled in the on-stack BPE buffer */

typedef struct {
    uint32_t key;   /* left << 16 | right; 0xFFFFFFFF = empty */
    uint16_t rank;  /* merge priority (lower first) */
    uint16_t id;    /* merged token */
} ellm_tok_merge;

typedef struct {
    int vocab, n_merges, hash_bits, max_tok_bytes;
    int n_l, n_n, n_s;
    uint32_t unicode_version;
    const uint16_t *byte_ids;      /* [256] */
    const ellm_tok_merge *hash;    /* [1 << hash_bits] */
    const uint32_t *r_l, *r_n, *r_s; /* [n][2] inclusive ranges */
    const uint32_t *tok_off;       /* [vocab + 1] */
    const uint8_t *tok_bytes;
} ellm_tok;

/* Validates the blob (header, section bounds, offsets, ranges, hash contents) and points t into
 * it. blob must be 4-byte aligned and stay alive and unmodified while t is used. 0, or
 * ELLM_ERR_BOUNDS (too short, misaligned blob, total_len != len, a section outside the blob)
 * or ELLM_ERR_SHAPE (bad magic/version or any other invariant). */
int ellm_tok_init(ellm_tok *t, const uint8_t *blob, size_t len);

/* Plain text (UTF-8, len bytes; never emits special ids) -> ids in out[0..n), identical to
 * `tokenizers`. Returns n, or -1 if cap is too small (out is then partially written): cap >=
 * len is always enough; a smaller cap works when it holds the ids, except that a pre-token
 * longer than ELLM_TOK_STACK bytes needs a free slot per byte while it is merged. out may be
 * NULL only when cap == 0. */
int ellm_tok_encode(const ellm_tok *t, const char *text, size_t len, uint16_t *out, int cap);

/* Token id's raw bytes -> out[0..n) (n = 0 for special tokens; n <= t->max_tok_bytes). Returns
 * n, -1 if cap < n, or ELLM_ERR_TOKEN for an id outside [0, vocab). Bytes of a multi-byte UTF-8
 * character may be split across tokens. */
int ellm_tok_decode(const ellm_tok *t, int id, uint8_t *out, int cap);

#endif
