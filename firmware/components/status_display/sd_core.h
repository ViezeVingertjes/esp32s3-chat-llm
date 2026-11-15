// Portable core of the tok/s display: rounding, fixed digit layout, band rendering and
// per-digit dirty rects. No ESP-IDF dependency, so it builds and is tested on the host
// (host_test/). The panel side (QSPI, CO5300) lives in status_display.c.
#pragma once

#include <stdint.h>

#define SD_SCREEN_W 480
#define SD_SCREEN_H 480
#define SD_TEXT_MAX 4 // "999" + NUL

typedef struct { int x, y, w, h; } sd_rect;

// One glyph: bitmap at bits + off, w x h pixels, bpp bits each (LSB-first in a byte, each
// row starting on a byte); xo/yo place the ink inside its cell, from the cell's top left.
typedef struct {
    uint16_t off;
    uint8_t w, h;
    uint8_t xo, yo;
} sd_glyph;

// Glyphs for '0'..'9'. Tabular: every digit owns a cell digit_w wide (even), so numbers
// with the same digit count sit in identical positions.
typedef struct {
    const uint8_t *bits;
    const sd_glyph *glyphs;
    int bpp; // 1, 2 or 4
    int line_h;
    int digit_w;
} sd_font;

#define SD_SLOTS 3 // at most 3 digits

// What is on screen: the text (n digits, centred) and each digit's cell x.
typedef struct {
    char text[SD_TEXT_MAX];
    int n, y;
    int x[SD_SLOTS];
} sd_num;

sd_rect sd_intersect(sd_rect a, sd_rect b);
sd_rect sd_unite(sd_rect a, sd_rect b);     // empty inputs are ignored
sd_rect sd_even(sd_rect r, sd_rect bounds); // widen to even edges, clipped to bounds

// Rounded to the nearest whole number (halves away from zero), clamped to 0..999;
// NaN and negatives give "0" (the idle value).
void sd_format(float v, char out[SD_TEXT_MAX]);

// Lay `text` ("0" .. "999", digits only) out centred on the screen, horizontally and
// vertically, with even cell positions.
void sd_layout(sd_num *num, const sd_font *f, const char *text);

// Lay `text` out into *num. Returns how many rects in dirty[] must be redrawn to turn the
// old screen into the new one (0 if unchanged). Same digit count: only the changed digit
// cells, neighbouring ones merged. Digit count changed: the old and new number areas as one
// rect. Even-aligned, inside the screen.
int sd_num_set(sd_num *num, const sd_font *f, const char *text, sd_rect dirty[SD_SLOTS]);

// Render the part of `area` inside `band` into buf (band.w x band.h pixels, RGB565
// byte-swapped for the panel). Pixels of buf outside `area` are left untouched.
void sd_render(uint16_t *buf, sd_rect band, sd_rect area, const sd_font *f, const sd_num *num,
               uint16_t fg, uint16_t bg);

// fg over bg at coverage v / levels (RGB565, per channel, rounded).
uint16_t sd_blend(uint16_t fg, uint16_t bg, int v, int levels);

static inline uint16_t sd_rgb(int r, int g, int b) {
    return (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}
