#ifndef ELLM_PACK_H
#define ELLM_PACK_H
#include <stddef.h>
#include <stdint.h>

/* Pack format v3 (little-endian). Written by ellm_py/pack.py.
 *
 * Versions: v2 = v1 + the Q2 weight type (type 3); v3 = v2 + the BLOB type (type 4), used only by
 * the "tokenizer" tensor; the layout is otherwise identical. The loader accepts versions 1..3; Q2
 * needs v2 or later, BLOB needs v3. The writer emits the lowest version that can hold the
 * tensors: v3 if the tokenizer is included, else v2 if some tensor is Q2, else v1 (so packs
 * without a tokenizer stay byte-identical to v2 / v1 and load on older runtimes).
 *
 * File: [ellm_header][pad to 64][ellm_tensor_entry x n_tensors][pad to 64][payload]
 * table_off / payload_off / data_off / scale_off are byte offsets from the START OF FILE.
 * payload covers [payload_off, payload_off + payload_len) = end of file.
 * Checksums:
 *  - crc32 (zlib/IEEE): integrity of the WHOLE image [0, payload_off + payload_len) -- header,
 *    tensor table, padding and payload -- computed with the header's crc32 field set to 0.
 *    The loader rejects a mismatch with ELLM_ERR_CRC.
 *  - fnv1a (32-bit FNV-1a, offset 0x811C9DC5, prime 0x01000193) over the payload only: the
 *    "model fingerprint" (identifies the weights independent of header fields like seq_len).
 *
 * Invariants (the writer asserts them; the loader may rely on them):
 *  1. every data_off (and every non-zero scale_off) is 64-byte aligned.
 *  2. T158: cols_pad = next multiple of 20 >= cols; row_bytes = cols_pad/5, hence a multiple
 *     of 4 bytes (the LUT kernel reads rows as aligned 32-bit words). Trit packing:
 *     byte g = sum_i (t[5g+i]+1)*3^i, padding trits = 0 (digit 1).
 *  3. I8: cols_pad = pad20(cols), row_bytes = cols_pad, padding bytes are 0.
 *  4. F32: cols_pad = cols, row_bytes = 4*cols, scale_off = 0 (no scale).
 *  5. T158/I8/Q2 scale_off points at float32 scale[rows] (per-row).
 *  6. Q2 (v2 only): cols_pad = next multiple of 16 >= cols; row_bytes = cols_pad/4, hence a
 *     multiple of 4 bytes. 2-bit codes c in 0..3: element 4g+i of a row sits in byte g, bits
 *     2i..2i+1 (byte g = sum_i c[4g+i] << 2i); padding codes are 0 (the runtime zero-pads the
 *     matching activations, so padding contributes exactly 0). Weight value = (2c - 3) * scale[r]
 *     with scale[r] = s/2 for the quantizer's per-row absmean s (ellm_py.quant.quant_2bit:
 *     levels (c - 1.5) * s).
 *  7. BLOB (v3 only, the "tokenizer" tensor only): nbytes opaque bytes; rows = 1,
 *     cols = cols_pad = row_bytes = nbytes, scale_off = 0. The bytes are an etok blob
 *     (runtime/ellm_tok.h); the loader checks only its frame: nbytes >= ELLM_TOK_HDR_BYTES, magic
 *     "ETOK", total_len == nbytes and vocab == header.vocab (ellm_tok_init validates the rest).
 *     The writer puts it after every weight, so the weights' payload bytes (and their offsets
 *     relative to payload_off) are those of the same pack without the tokenizer.
 * Tensors: "embed" [vocab, dim] (also the tied LM head), "norm_final" [1, dim] F32, and per
 * layer l: L{l}.attn_norm, .wq, .wk, .wv, .wo, .ffn_norm, .w_gate, .w_up, .w_down
 * (norms are F32 rows=1); n_tensors = 2 + 9 * n_layers, plus 1 when "tokenizer" is present (v3).
 * type values match ellm_wtype: 0 F32, 1 I8, 2 T158, 3 Q2 (v2); and 4 BLOB (v3).
 */
#define ELLM_MAGIC 0x4D4C4C45u /* "ELLM" little-endian */
#define ELLM_VERSION 3 /* newest version read and written; v1 and v2 images load too */
#define ELLM_BLOB 4    /* tensor type of the "tokenizer" blob (v3) */

typedef struct {
    uint32_t magic, version;
    int32_t dim, hidden, n_layers, n_heads, n_kv_heads, head_dim, vocab, seq_len;
    float rope_theta, norm_eps;
    int32_t kv_type; /* 0 f32, 1 i8 */
    int32_t n_tensors;
    uint32_t table_off, payload_off, payload_len, crc32, fnv1a;
} ellm_header; /* 76 bytes */

typedef struct {
    char name[24];
    int32_t type, rows, cols, cols_pad;
    uint32_t row_bytes, data_off, scale_off;
} ellm_tensor_entry; /* 52 bytes */

typedef char ellm_hdr_size_check[(sizeof(ellm_header) == 76) ? 1 : -1];
typedef char ellm_ent_size_check[(sizeof(ellm_tensor_entry) == 52) ? 1 : -1];

/* 32-bit FNV-1a over n bytes (header.fnv1a is computed over the payload). */
uint32_t ellm_fnv1a32(const uint8_t *p, size_t n);
/* CRC-32 (zlib/IEEE, reflected poly 0xEDB88320), zlib chaining form: start with crc = 0,
 * ellm_crc32(ellm_crc32(0, a, na), b, nb) == crc of a||b (same as Python zlib.crc32(b, crc)).
 * header.crc32 covers the image [0, payload_off + payload_len) with the crc32 field zeroed. */
uint32_t ellm_crc32(uint32_t crc, const uint8_t *p, size_t n);

#endif
