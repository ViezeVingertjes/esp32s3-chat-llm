// AMOLED display for esp-llm (spec 5.6): either one number, the tokens/second (disp_number:
// firmware/app, and firmware/chat with CHAT_SCREEN_TEXT=n), or the chat screen (disp_chat_*:
// state, context bar, the conversation's last rows, live tok/s; docs/stages/screen-design.md).
// Use one or the other after disp_init, not both.
// Call from one task, between compute phases only, at most ~1-2 times per second.
// No background task, no timers: no CPU use while not drawing. Draw buffers (2 x 4 KiB,
// internal DMA RAM) exist only during a draw; fonts are in flash.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Power up and initialize the CO5300 panel (QSPI, SPI2), clear to black, moderate brightness.
// Does not touch the SD card or GPIO1/2/3/41.
esp_err_t disp_init(void);
// Show `value` rounded to a whole number (0-999; negative or NaN: 0, the idle value), centred,
// in tabular digits. Redraws only the digits that changed (the whole number area when the
// digit count changes). If draw memory can't be had, the update is skipped (the model wins).
void disp_number(float value);

// Chat screen. The first four only update what should be shown (memory, no panel traffic, cheap:
// call them any time, e.g. from the reply stream); disp_chat_draw sends the changes.
typedef enum { DISP_IDLE = 0, DISP_THINKING = 1, DISP_WRITING = 2 } disp_state;
// New turn: clear the text, show the user's message (UTF-8; at most 2 rows); user == NULL: clear.
void disp_chat_turn(const char *user, size_t len);
// Reply bytes (UTF-8; Latin-1 shown, other characters as '?'), word-wrapped, scrolling.
void disp_chat_reply(const uint8_t *b, size_t n);
// State, live tok/s (0 when idle) and the context fill (ctx_used of ctx_len tokens).
void disp_chat_status(disp_state state, float tok_s, int ctx_used, int ctx_len);
// Draw what changed since the last draw (only the changed rows/cells/regions). Returns the number
// of rects sent (0: nothing changed or the panel is off), or -1 if draw memory couldn't be had
// (skipped: the model wins; the next call draws the accumulated changes).
int disp_chat_draw(void);

// Panel off (DISPOFF + sleep in). Later draw calls do nothing.
void disp_off(void);

#ifdef __cplusplus
}
#endif
