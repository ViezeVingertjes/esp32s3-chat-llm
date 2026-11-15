// Portable core of the chat screen; see sd_chat.h and docs/stages/screen-design.md.
#include "sd_chat.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static const sd_rect SCREEN = {0, 0, SD_SCREEN_W, SD_SCREEN_H};

// Regions (x, y, w, h).
static const sd_rect R_STATE = {SC_M, 40, 200, 28};          // dot + state word
static const sd_rect R_CTX = {SC_M + 200, 40, 200, 28};      // "<used>/<len>", right-aligned
static const sd_rect R_BAR = {SC_M, 78, SD_SCREEN_W - 2 * SC_M, 6};
#define DOT_CX (SC_M + 8)
#define DOT_CY 54
#define DOT_R 7
#define STATE_X (SC_M + 24)
#define ROW_Y 40 // state line top
#define NUM_GAP 12
#define NUM_BOX_W (3 * 46 + 24) // "99.9": three digits and the point
#define LABEL "tok/s"
#define LABEL_W (5 * 13)
#define NUM_X ((SD_SCREEN_W - NUM_BOX_W - NUM_GAP - LABEL_W) / 2 & ~1)
#define NUM_Y 376
static const sd_rect R_NUM = {NUM_X, NUM_Y, NUM_BOX_W, 50};
// the label's baseline sits on the digits' baseline (NUM_Y + num line_h)
static const sd_rect R_LABEL = {NUM_X + NUM_BOX_W + NUM_GAP, NUM_Y + 49 - 21, LABEL_W + 1, 28};

static const char *const STATE_WORD[3] = {"idle", "thinking", "writing"};
static const int STATE_COLOR[3] = {SC_C_IDLE, SC_C_THINK, SC_C_WRITE};

static int imin(int a, int b) { return a < b ? a : b; }
static int imax(int a, int b) { return a > b ? a : b; }
static int empty(sd_rect r) { return r.w <= 0 || r.h <= 0; }
static uint16_t swap16(uint16_t c) { return (uint16_t)((c << 8) | (c >> 8)); }

/* ------------------------------------------------------------------ model: text */

static void clear_text(sc_screen *s) {
    memset(s->cur.len, 0, sizeof s->cur.len);
    memset(s->cur.attr, 0, sizeof s->cur.attr);
    s->rows = 0;
    s->wrapped = 0;
    s->block_start = 1;
    s->limit = s->cut = 0;
    s->need = 0;
}

// Start a new row at the bottom (scrolling the top row out when full). 0: the user-line row
// limit is reached (nothing changed).
static int new_row(sc_screen *s, int wrapped) {
    sc_view *v = &s->cur;
    if (s->limit && s->rows >= s->limit) return 0;
    if (s->rows == SC_ROWS) {
        memmove(v->ch[0], v->ch[1], sizeof v->ch[0] * (SC_ROWS - 1));
        memmove(v->len, v->len + 1, SC_ROWS - 1);
        memmove(v->attr, v->attr + 1, SC_ROWS - 1);
        s->rows--;
    }
    v->len[s->rows] = 0;
    v->attr[s->rows] = (uint8_t)s->attr;
    s->rows++;
    s->wrapped = wrapped;
    return 1;
}

// One Latin-1 character (0x20..0xFF) or '\n'.
static void put(sc_screen *s, uint8_t c) {
    sc_view *v = &s->cur;
    if (s->cut) return;
    if (s->rows == 0 && !new_row(s, 0)) return;
    int r = s->rows - 1;
    if (c == '\n') {
        if (v->len[r] == 0 && (s->block_start || s->wrapped || (r > 0 && v->len[r - 1] == 0))) return;
        if (!new_row(s, 0)) s->cut = 1;
        return;
    }
    if (c == ' ') {
        if (v->len[r] == 0 && s->wrapped) return; // no leading space on a wrapped row
        if (v->len[r] == SC_COLS) {               // the space is the line break
            if (!new_row(s, 1)) s->cut = 1;
            return;
        }
    } else if (v->len[r] == SC_COLS) {
        int sp = SC_COLS - 1;
        while (sp > 0 && v->ch[r][sp] != ' ') sp--;
        uint8_t word[SC_COLS];
        int wl = sp > 0 ? SC_COLS - 1 - sp : 0;
        memcpy(word, v->ch[r] + sp + 1, (size_t)wl);
        if (!new_row(s, 1)) {
            s->cut = 1;
            return;
        }
        r = s->rows - 1;
        if (wl) {
            v->len[r - 1] = (uint8_t)sp; // the word moves down; the space before it goes
            memcpy(v->ch[r], word, (size_t)wl);
            v->len[r] = (uint8_t)wl;
        }
    }
    v->ch[r][v->len[r]++] = c;
    s->block_start = 0;
}

static void puts_(sc_screen *s, const char *t) {
    while (*t) put(s, (uint8_t)*t++);
}

static void put_cp(sc_screen *s, uint32_t cp) {
    if (cp == '\n') put(s, '\n');
    else if (cp == '\t' || cp == 0xA0) put(s, ' ');
    else if (cp < 0x20 || (cp >= 0x7F && cp < 0xA0)) return; // controls: skipped
    else if (cp <= 0xFF) put(s, (uint8_t)cp);
    else if (cp == 0x2018 || cp == 0x2019 || cp == 0x201A || cp == 0x2032) put(s, '\'');
    else if (cp == 0x201C || cp == 0x201D || cp == 0x201E || cp == 0x2033) put(s, '"');
    else if ((cp >= 0x2010 && cp <= 0x2015) || cp == 0x2212) put(s, '-');
    else if (cp == 0x2026) puts_(s, "...");
    else if (cp == 0x2022 || cp == 0x2023 || cp == 0x25CF || cp == 0x25E6 || cp == 0x2219) put(s, 0xB7);
    else if (cp == 0x2192) puts_(s, "->");
    else put(s, '?');
}

static void feed(sc_screen *s, const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t x = b[i];
        if (s->need) {
            if ((x & 0xC0) == 0x80) {
                s->cp = (s->cp << 6) | (x & 0x3Fu);
                if (--s->need == 0) put_cp(s, s->cp);
                continue;
            }
            s->need = 0; // truncated sequence: one '?', then x as a new start
            put_cp(s, '?');
        }
        if (x < 0x80) put_cp(s, x);
        else if ((x & 0xE0) == 0xC0) s->cp = x & 0x1Fu, s->need = 1;
        else if ((x & 0xF0) == 0xE0) s->cp = x & 0x0Fu, s->need = 2;
        else if ((x & 0xF8) == 0xF0) s->cp = x & 0x07u, s->need = 3;
        else put_cp(s, '?'); // stray continuation or invalid lead byte
    }
}

void sc_turn(sc_screen *s, const char *user, size_t len) {
    clear_text(s);
    if (user) {
        s->attr = SC_USER;
        s->limit = 2;
        puts_(s, "> ");
        feed(s, (const uint8_t *)user, len);
        if (s->need) put_cp(s, '?');
        if (s->cut) { // full last row: end it with "..."
            int r = s->rows - 1;
            memset(s->cur.ch[r] + SC_COLS - 3, '.', 3);
            s->cur.len[r] = SC_COLS;
        }
        s->limit = s->cut = s->need = 0;
        s->attr = SC_REPLY;
        new_row(s, 0);
        s->block_start = 1;
    }
    s->attr = SC_REPLY;
}

void sc_reply(sc_screen *s, const uint8_t *b, size_t n) { feed(s, b, n); }

/* ------------------------------------------------------------------ model: status */

void sc_fmt_rate(float v, char out[6]) {
    if (!(v > 0.0f)) v = 0.0f;
    if (v > 999.0f) v = 999.0f;
    unsigned t = (unsigned)lroundf(v * 10.0f), w = t < 1000 ? t / 10 : (unsigned)lroundf(v);
    char *o = out;
    if (w >= 100) *o++ = (char)('0' + w / 100);
    if (w >= 10) *o++ = (char)('0' + w / 10 % 10);
    *o++ = (char)('0' + w % 10);
    if (t < 1000) *o++ = '.', *o++ = (char)('0' + t % 10);
    *o = 0;
}

void sc_status(sc_screen *s, int state, float tok_s, int used, int len) {
    sc_view *v = &s->cur;
    v->state = (int8_t)(state >= SC_IDLE && state <= SC_WRITING ? state : SC_IDLE);
    sc_fmt_rate(tok_s, v->num);
    snprintf(v->ctx, sizeof v->ctx, "%d/%d", used, len);
    int u = imax(0, imin(used, len));
    v->bar = (int16_t)(len > 0 ? (int)((long)u * R_BAR.w / len) : 0);
}

void sc_init(sc_screen *s) {
    memset(s, 0, sizeof *s);
    static const uint8_t rgb[SC_NCOLORS][3] = {
        {230, 230, 230}, // text
        {120, 150, 190}, // user line
        {120, 120, 120}, // labels
        {100, 100, 100}, // idle
        {255, 176, 32},  // thinking (amber)
        {64, 216, 120},  // writing (green)
        {40, 40, 40},    // bar track
        {80, 140, 240},  // bar fill
    };
    for (int c = 0; c < SC_NCOLORS; c++)
        for (int k = 0; k < 16; k++)
            s->pal[c][k] = swap16(sd_blend(sd_rgb(rgb[c][0], rgb[c][1], rgb[c][2]), 0, k, 15));
    sc_blank(s);
    clear_text(s);
    s->attr = SC_REPLY;
    s->cur.label = 1;
    sc_status(s, SC_IDLE, 0.0f, 0, 0);
}

void sc_commit(sc_screen *s) { s->shown = s->cur; }

void sc_blank(sc_screen *s) {
    memset(&s->shown, 0, sizeof s->shown);
    s->shown.state = -1;
    s->shown.bar = -1;
}

/* ------------------------------------------------------------------ dirty rects */

static sd_rect cells(int r, int c0, int c1) {
    return (sd_rect){SC_TEXT_X + c0 * sc_font_text.cell, SC_TEXT_Y + r * SC_PITCH, (c1 - c0) * sc_font_text.cell,
                     SC_PITCH};
}

int sc_dirty(const sc_screen *s, sd_rect out[SC_MAX_DIRTY]) {
    const sc_view *a = &s->shown, *b = &s->cur;
    int n = 0;
    if (a->state != b->state) out[n++] = R_STATE;
    if (strcmp(a->ctx, b->ctx) != 0) out[n++] = R_CTX;
    if (a->bar != b->bar) {
        int x0 = a->bar < 0 ? 0 : imin(a->bar, b->bar), x1 = a->bar < 0 ? R_BAR.w : imax(a->bar, b->bar);
        out[n++] = (sd_rect){R_BAR.x + x0, R_BAR.y, x1 - x0, R_BAR.h};
    }
    if (strcmp(a->num, b->num) != 0) out[n++] = R_NUM;
    if (a->label != b->label) out[n++] = R_LABEL;
    int last_row = -2;
    for (int r = 0; r < SC_ROWS; r++) {
        int m = imax(a->len[r], b->len[r]), c0 = m, c1 = 0;
        for (int c = 0; c < m; c++) {
            uint8_t x = c < a->len[r] ? a->ch[r][c] : ' ', y = c < b->len[r] ? b->ch[r][c] : ' ';
            if (x != y || (a->attr[r] != b->attr[r] && y != ' ')) {
                c0 = imin(c0, c);
                c1 = c + 1;
            }
        }
        if (c0 >= c1) continue;
        sd_rect d = cells(r, c0, c1);
        if (last_row == r - 1) out[n - 1] = sd_unite(out[n - 1], d);
        else out[n++] = d;
        last_row = r;
    }
    int k = 0;
    for (int i = 0; i < n; i++) {
        sd_rect e = sd_even(out[i], SCREEN);
        if (!empty(e)) out[k++] = e;
    }
    return k;
}

/* ------------------------------------------------------------------ rendering */

static void fill(uint16_t *buf, sd_rect band, sd_rect clip, sd_rect r, uint16_t px) {
    sd_rect v = sd_intersect(r, clip);
    for (int y = v.y; y < v.y + v.h; y++) {
        uint16_t *row = buf + (size_t)(y - band.y) * band.w + (v.x - band.x);
        for (int x = 0; x < v.w; x++) row[x] = px;
    }
}

// Draws glyph `code` with its cell's top left at (x, y); returns its advance.
static int glyph(uint16_t *buf, sd_rect band, sd_rect clip, const sc_font *f, int code, int x, int y,
                 const uint16_t *pal) {
    if (code < f->first || code >= f->first + f->count) return f->cell;
    const int i = code - f->first;
    const int adv = f->adv ? f->adv[i] : f->cell;
    const sd_glyph *g = &f->glyphs[i];
    if (!g->w) return adv;
    const sd_rect ink = {x + g->xo, y + g->yo, g->w, g->h};
    const sd_rect vis = sd_intersect(ink, clip);
    if (empty(vis)) return adv;
    const int stride = (g->w + 1) / 2; // 4 bpp (both fonts)
    for (int yy = vis.y; yy < vis.y + vis.h; yy++) {
        const uint8_t *src = f->bits + g->off + (size_t)(yy - ink.y) * stride;
        uint16_t *dst = buf + (size_t)(yy - band.y) * band.w - band.x;
        for (int xx = vis.x; xx < vis.x + vis.w; xx++) {
            int dx = xx - ink.x;
            int v = (src[dx >> 1] >> ((dx & 1) << 2)) & 15;
            if (v) dst[xx] = pal[v];
        }
    }
    return adv;
}

static int str_w(const sc_font *f, const char *t) {
    int w = 0;
    for (; *t; t++) {
        int c = (uint8_t)*t;
        w += (f->adv && c >= f->first && c < f->first + f->count) ? f->adv[c - f->first] : f->cell;
    }
    return w;
}

static void text(uint16_t *buf, sd_rect band, sd_rect clip, const sc_font *f, int x, int y, const char *t,
                 const uint16_t *pal) {
    for (; *t; t++) x += glyph(buf, band, clip, f, (uint8_t)*t, x, y, pal);
}

static void dot(uint16_t *buf, sd_rect band, sd_rect clip, const uint16_t *pal) {
    sd_rect v = sd_intersect((sd_rect){DOT_CX - DOT_R, DOT_CY - DOT_R, 2 * DOT_R, 2 * DOT_R}, clip);
    for (int y = v.y; y < v.y + v.h; y++)
        for (int x = v.x; x < v.x + v.w; x++) {
            int k = 0; // 2x2 samples at quarter-pixel offsets, centre at (DOT_CX, DOT_CY)
            for (int sy = 0; sy < 2; sy++)
                for (int sx = 0; sx < 2; sx++) {
                    int dx = 4 * (x - DOT_CX) + 1 + 2 * sx, dy = 4 * (y - DOT_CY) + 1 + 2 * sy;
                    k += dx * dx + dy * dy <= 16 * DOT_R * DOT_R;
                }
            if (k) buf[(size_t)(y - band.y) * band.w + (x - band.x)] = pal[(k * 15 + 2) / 4];
        }
}

void sc_render(uint16_t *buf, sd_rect band, sd_rect area, const sc_screen *s) {
    const sd_rect clip = sd_intersect(band, area);
    if (empty(clip)) return;
    const sc_view *v = &s->cur;
    fill(buf, band, clip, clip, 0); // black
    if (!empty(sd_intersect(clip, R_STATE)) && v->state >= 0) {
        const uint16_t *pal = s->pal[STATE_COLOR[v->state]];
        dot(buf, band, clip, pal);
        text(buf, band, clip, &sc_font_text, STATE_X, ROW_Y, STATE_WORD[v->state], pal);
    }
    if (!empty(sd_intersect(clip, R_CTX)))
        text(buf, band, clip, &sc_font_text, R_CTX.x + R_CTX.w - str_w(&sc_font_text, v->ctx), ROW_Y, v->ctx,
             s->pal[SC_C_LABEL]);
    if (!empty(sd_intersect(clip, R_BAR)) && v->bar >= 0) {
        fill(buf, band, clip, (sd_rect){R_BAR.x, R_BAR.y, v->bar, R_BAR.h}, s->pal[SC_C_FILL][15]);
        fill(buf, band, clip, (sd_rect){R_BAR.x + v->bar, R_BAR.y, R_BAR.w - v->bar, R_BAR.h}, s->pal[SC_C_TRACK][15]);
    }
    if (!empty(sd_intersect(clip, R_NUM)))
        text(buf, band, clip, &sc_font_num, R_NUM.x + R_NUM.w - str_w(&sc_font_num, v->num), NUM_Y, v->num,
             s->pal[SC_C_TEXT]);
    if (v->label && !empty(sd_intersect(clip, R_LABEL)))
        text(buf, band, clip, &sc_font_text, R_LABEL.x, R_LABEL.y, LABEL, s->pal[SC_C_LABEL]);
    const int cell = sc_font_text.cell;
    int r0 = imax(0, (clip.y - SC_TEXT_Y) / SC_PITCH), r1 = imin(SC_ROWS - 1, (clip.y + clip.h - 1 - SC_TEXT_Y) / SC_PITCH);
    if (clip.y + clip.h <= SC_TEXT_Y) r1 = -1;
    int c0 = imax(0, (clip.x - SC_TEXT_X) / cell);
    for (int r = r0; r <= r1; r++) {
        const uint16_t *pal = s->pal[v->attr[r] == SC_USER ? SC_C_USER : SC_C_TEXT];
        int y = SC_TEXT_Y + r * SC_PITCH;
        for (int c = c0; c < v->len[r]; c++) {
            int x = SC_TEXT_X + c * cell;
            if (x >= clip.x + clip.w) break;
            if (v->ch[r][c] != ' ') glyph(buf, band, clip, &sc_font_text, v->ch[r][c], x, y, pal);
        }
    }
}
