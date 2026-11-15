// Portable core of the chat screen (Plan 4 Task 14, docs/stages/screen-design.md): the state
// line (idle / thinking / writing), the context bar, the last rows of the conversation (the user
// line, then the streaming reply, word-wrapped) and the live tok/s. No ESP-IDF dependency: built
// and tested on the host (host_test/). The panel side lives in status_display.c.
//
// Model: the caller updates the wanted view (sc_turn / sc_reply / sc_status: memory only, no
// drawing), then sc_dirty gives the rects that differ from what the panel shows, sc_render
// renders any band of the wanted view, and sc_commit records it as shown once it is on the panel.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sd_core.h"

// A bitmap font: glyph i is code point first + i (w == 0: no ink); bpp-bit bitmaps (the renderer
// supports bpp 4 only), ink offset
// (xo, yo) from the glyph cell's top left; adv[i] its advance (NULL: every glyph advances cell).
typedef struct {
    const uint8_t *bits;
    const sd_glyph *glyphs;
    const uint8_t *adv;
    int bpp, line_h, base, first, count, cell;
} sc_font;

extern const sc_font sc_font_text; // DejaVu Sans Mono 22 px, 0x20-0xFF, cell 13, line 27
extern const sc_font sc_font_num;  // DejaVu Sans Bold 64 px, '.' '0'-'9', digits 46 wide, line 49

// Layout (480 x 480; content kept >= 40 px from the edges: the panel's corners are rounded).
#define SC_M 40
#define SC_ROWS 9
#define SC_COLS 31
#define SC_TEXT_X SC_M
#define SC_TEXT_Y 100
#define SC_PITCH 28 // row pitch (text line 27)
#define SC_MAX_DIRTY (6 + SC_ROWS)

enum { SC_IDLE = 0, SC_THINKING = 1, SC_WRITING = 2 };
enum { SC_REPLY = 0, SC_USER = 1 };
enum { SC_C_TEXT, SC_C_USER, SC_C_LABEL, SC_C_IDLE, SC_C_THINK, SC_C_WRITE, SC_C_TRACK, SC_C_FILL, SC_NCOLORS };

// Everything that is drawn. Text: Latin-1 bytes; a cell at or beyond len[r] is blank.
typedef struct {
    uint8_t ch[SC_ROWS][SC_COLS];
    uint8_t len[SC_ROWS];
    uint8_t attr[SC_ROWS]; // SC_REPLY / SC_USER
    int8_t state;          // SC_IDLE.. (-1: not drawn)
    int16_t bar;           // context bar fill in px (-1: not drawn)
    uint8_t label;         // the "tok/s" label is drawn
    char num[6];           // the tok/s text ("" : not drawn)
    char ctx[24];          // "<used>/<len>"
} sc_view;

typedef struct {
    sc_view cur, shown;
    uint16_t pal[SC_NCOLORS][16]; // per colour, 4-bit coverage -> panel pixel (byte-swapped)
    int rows;                     // rows of cur in use; the cursor is on rows - 1
    int attr;                     // attribute of text being added
    int wrapped;                  // the cursor row began at a soft wrap (drop its leading spaces)
    int block_start;              // nothing added since the block (user line / reply) began
    int limit, cut;               // user line: at most `limit` rows; cut = truncated
    uint32_t cp;                  // UTF-8 decoder
    int need;
} sc_screen;

// Blank panel shown, wanted view: idle, 0.0 tok/s, no text.
void sc_init(sc_screen *s);
// The panel was cleared: the shown view is blank (the next sc_dirty covers everything drawn).
void sc_blank(sc_screen *s);
// A new turn: clears the text and shows "> " + the user's message (UTF-8, at most 2 rows, cut
// with "..."), in the user colour; the reply follows on the next row. user == NULL: clear only.
void sc_turn(sc_screen *s, const char *user, size_t len);
// Append reply bytes (UTF-8). Latin-1 is shown as is; a few common typographic characters map
// to ASCII (quotes, dashes, ellipsis, bullet); any other character shows as '?'. Tab = space,
// other controls are skipped; at most one blank row in a row. Words wrap at SC_COLS; the text
// scrolls up a row at a time.
void sc_reply(sc_screen *s, const uint8_t *b, size_t n);
// State, live tok/s (one decimal below 100, whole above, 0..999; NaN/negative: 0.0) and the
// context fill (used of len tokens).
void sc_status(sc_screen *s, int state, float tok_s, int used, int len);
// Rects (even-aligned, inside the screen) where the wanted view differs from the shown one.
int sc_dirty(const sc_screen *s, sd_rect out[SC_MAX_DIRTY]);
// Render the wanted view's part of `area` inside `band` into buf (band.w x band.h, RGB565
// byte-swapped). Pixels outside `area` are left untouched.
void sc_render(uint16_t *buf, sd_rect band, sd_rect area, const sc_screen *s);
// The wanted view is now on the panel.
void sc_commit(sc_screen *s);

// The tok/s text (exposed for tests).
void sc_fmt_rate(float v, char out[6]);
