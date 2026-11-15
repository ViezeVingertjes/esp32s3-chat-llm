// Portable core of the tok/s display; see sd_core.h.
#include "sd_core.h"

#include <math.h>
#include <string.h>

static const sd_rect SCREEN = {0, 0, SD_SCREEN_W, SD_SCREEN_H};

static int imin(int a, int b) { return a < b ? a : b; }
static int imax(int a, int b) { return a > b ? a : b; }
static int is_empty(sd_rect r) { return r.w <= 0 || r.h <= 0; }

sd_rect sd_intersect(sd_rect a, sd_rect b) {
    int x0 = imax(a.x, b.x), y0 = imax(a.y, b.y);
    int x1 = imin(a.x + a.w, b.x + b.w), y1 = imin(a.y + a.h, b.y + b.h);
    if (x1 <= x0 || y1 <= y0) return (sd_rect){0, 0, 0, 0};
    return (sd_rect){x0, y0, x1 - x0, y1 - y0};
}

sd_rect sd_unite(sd_rect a, sd_rect b) {
    if (is_empty(a)) return b;
    if (is_empty(b)) return a;
    int x0 = imin(a.x, b.x), y0 = imin(a.y, b.y);
    int x1 = imax(a.x + a.w, b.x + b.w), y1 = imax(a.y + a.h, b.y + b.h);
    return (sd_rect){x0, y0, x1 - x0, y1 - y0};
}

sd_rect sd_even(sd_rect r, sd_rect bounds) {
    int x0 = r.x & ~1, y0 = r.y & ~1;
    int x1 = (r.x + r.w + 1) & ~1, y1 = (r.y + r.h + 1) & ~1;
    return sd_intersect((sd_rect){x0, y0, x1 - x0, y1 - y0}, bounds);
}

void sd_format(float v, char out[SD_TEXT_MAX]) {
    if (!(v > 0.0f)) v = 0.0f; // also NaN
    if (v > 999.0f) v = 999.0f;
    int n = (int)lroundf(v);
    char *o = out;
    if (n >= 100) *o++ = (char)('0' + n / 100);
    if (n >= 10) *o++ = (char)('0' + n / 10 % 10);
    *o++ = (char)('0' + n % 10);
    *o = 0;
}

static const sd_glyph *glyph_of(const sd_font *f, char c) {
    return c >= '0' && c <= '9' ? &f->glyphs[c - '0'] : NULL;
}

static sd_rect cell_of(const sd_font *f, const sd_num *num, int i) {
    return (sd_rect){num->x[i], num->y, f->digit_w, f->line_h};
}

static sd_rect box_of(const sd_font *f, const sd_num *num) {
    return num->n ? (sd_rect){num->x[0], num->y, num->n * f->digit_w, f->line_h} : (sd_rect){0, 0, 0, 0};
}

void sd_layout(sd_num *num, const sd_font *f, const char *text) {
    memset(num, 0, sizeof *num);
    strncpy(num->text, text, SD_TEXT_MAX - 1);
    num->n = (int)strlen(num->text);
    num->y = (SD_SCREEN_H - f->line_h) / 2;
    int x0 = ((SD_SCREEN_W - num->n * f->digit_w) / 2) & ~1;
    for (int i = 0; i < num->n; i++) num->x[i] = x0 + i * f->digit_w;
}

int sd_num_set(sd_num *num, const sd_font *f, const char *text, sd_rect dirty[SD_SLOTS]) {
    sd_num next;
    sd_layout(&next, f, text);
    if (strcmp(next.text, num->text) == 0) return 0;
    int n = 0;
    if (next.n != num->n) {
        sd_rect all = sd_even(sd_unite(box_of(f, num), box_of(f, &next)), SCREEN);
        if (all.w > 0) dirty[n++] = all;
    } else {
        for (int i = 0; i < next.n; i++) {
            if (next.text[i] == num->text[i]) continue;
            sd_rect c = sd_even(cell_of(f, &next, i), SCREEN);
            if (n > 0 && next.text[i - 1] != num->text[i - 1]) dirty[n - 1] = sd_unite(dirty[n - 1], c);
            else dirty[n++] = c;
        }
    }
    *num = next;
    return n;
}

static uint16_t swap16(uint16_t c) { return (uint16_t)((c << 8) | (c >> 8)); }

uint16_t sd_blend(uint16_t fg, uint16_t bg, int v, int levels) {
    int r = (((fg >> 11) * v + (bg >> 11) * (levels - v)) * 2 + levels) / (2 * levels);
    int g = ((((fg >> 5) & 63) * v + ((bg >> 5) & 63) * (levels - v)) * 2 + levels) / (2 * levels);
    int b = (((fg & 31) * v + (bg & 31) * (levels - v)) * 2 + levels) / (2 * levels);
    return (uint16_t)((r << 11) | (g << 5) | b);
}

void sd_render(uint16_t *buf, sd_rect band, sd_rect area, const sd_font *f, const sd_num *num,
               uint16_t fg, uint16_t bg) {
    sd_rect clip = sd_intersect(band, area);
    if (is_empty(clip)) return;
    const int levels = (1 << f->bpp) - 1;
    uint16_t pal[16];
    for (int v = 0; v <= levels; v++) pal[v] = swap16(sd_blend(fg, bg, v, levels));
    for (int y = clip.y; y < clip.y + clip.h; y++) {
        uint16_t *row = buf + (size_t)(y - band.y) * band.w + (clip.x - band.x);
        for (int x = 0; x < clip.w; x++) row[x] = pal[0];
    }
    for (int i = 0; i < num->n; i++) {
        const sd_glyph *g = glyph_of(f, num->text[i]);
        if (!g) continue;
        sd_rect ink = {num->x[i] + g->xo, num->y + g->yo, g->w, g->h};
        sd_rect vis = sd_intersect(ink, clip);
        if (is_empty(vis)) continue;
        const int stride = (g->w * f->bpp + 7) / 8;
        for (int y = vis.y; y < vis.y + vis.h; y++) {
            const uint8_t *src = f->bits + g->off + (size_t)(y - ink.y) * stride;
            uint16_t *dst = buf + (size_t)(y - band.y) * band.w;
            for (int x = vis.x; x < vis.x + vis.w; x++) {
                int bit = (x - ink.x) * f->bpp;
                int v = (src[bit >> 3] >> (bit & 7)) & levels;
                if (v) dst[x - band.x] = pal[v];
            }
        }
    }
}
