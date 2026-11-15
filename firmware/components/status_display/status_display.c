// CO5300 panel side of the tok/s display: QSPI bring-up and band-by-band dirty-rect sends.
//
// Pins, panel init sequence, orientation and QSPI command encoding follow the Waveshare BSP
// esp32_s3_touch_amoled_2_16 v2.0.1, the CO5300 V0.01 datasheet and esp_lcd_co5300 v2.2.0.
// The band technique (draw a dirty rect band by band into small internal DMA buffers, RAMWR
// then RAMWRC, no framebuffer) is a fresh minimal C version.
#include "status_display.h"

#include <string.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_lcd_io_spi.h"
#include "esp_lcd_panel_io.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sd_chat.h"
#include "sd_core.h"

extern const sd_font sd_font_digits;

static const char *TAG = "disp";

// Board schematic pins
#define PIN_CS 12
#define PIN_SCLK 38
#define PIN_D0 4
#define PIN_D1 5
#define PIN_D2 6
#define PIN_D3 7
#define PIN_RST 39
#define LCD_HOST SPI2_HOST
#define PCLK_HZ (40 * 1000 * 1000) // GPIO-matrix SPI tops out at 40 MHz on these pins

// Two draw buffers of BAND_BYTES each, allocated per draw (internal DMA RAM).
#define BAND_BYTES 4096
#define BAND_PX (BAND_BYTES / 2)
#define BRIGHTNESS 0x80 // 0x51 WRDISBV, 0..0xFF: moderate
#define FG 0xE73C       // RGB565 of (230, 230, 230): soft white on black
#define BG 0x0000

// QSPI framing (esp_lcd_co5300 v2.2.0): opcode in bits 31..24, DCS command in bits 15..8.
#define OP_WRITE_CMD 0x02
#define OP_WRITE_COLOR 0x32
#define QCMD(op, c) ((int)(((uint32_t)(op) << 24) | ((uint32_t)(c) << 8)))

static esp_lcd_panel_io_handle_t s_io;
static bool s_on;
static sd_num s_num;
static sc_screen s_chat; // the chat screen (disp_chat_*): wanted and shown view, ~1 KB
static bool s_chat_init;

static sc_screen *chat(void) {
    if (!s_chat_init) sc_init(&s_chat);
    s_chat_init = true;
    return &s_chat;
}

static esp_err_t send(uint8_t c, const uint8_t *p, size_t n) {
    return esp_lcd_panel_io_tx_param(s_io, QCMD(OP_WRITE_CMD, c), p, n);
}
static esp_err_t send1(uint8_t c, uint8_t v) { return send(c, &v, 1); }

// CASET/RASET take the first and last pixel, big-endian 16-bit.
static esp_err_t send_range(uint8_t c, int first, int count) {
    int last = first + count - 1;
    uint8_t p[4] = {(uint8_t)(first >> 8), (uint8_t)first, (uint8_t)(last >> 8), (uint8_t)last};
    return send(c, p, 4);
}

typedef void (*render_fn)(uint16_t *buf, sd_rect band, sd_rect area);

static void render_num(uint16_t *buf, sd_rect band, sd_rect area) {
    sd_render(buf, band, area, &sd_font_digits, &s_num, FG, BG);
}

static void render_black(uint16_t *buf, sd_rect band, sd_rect area) {
    sd_rect c = sd_intersect(band, area);
    for (int y = c.y; y < c.y + c.h; y++) memset(buf + (size_t)(y - band.y) * band.w + (c.x - band.x), 0, (size_t)c.w * 2);
}

// Draw rects band by band: render into one buffer while DMA sends the other. Every
// tx_color with a command first waits for the transfer in flight, so a buffer is never
// rendered into while it is still being sent; the final drain waits for the last one.
static esp_err_t flush(const sd_rect *rects, int n, render_fn render) {
    uint16_t *buf[2];
    for (int i = 0; i < 2; i++) buf[i] = heap_caps_malloc(BAND_BYTES, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    esp_err_t err = (buf[0] && buf[1]) ? ESP_OK : ESP_ERR_NO_MEM;
    int k = 0;
    for (int i = 0; i < n && err == ESP_OK; i++) {
        const sd_rect r = rects[i];
        const int rows = BAND_PX / r.w;
        for (int y = r.y; y < r.y + r.h && err == ESP_OK; y += rows) {
            sd_rect band = {r.x, y, r.w, r.y + r.h - y < rows ? r.y + r.h - y : rows};
            uint16_t *b = buf[k++ & 1];
            render(b, band, band);
            if (y == r.y) {
                err = send_range(0x2A, r.x, r.w); // CASET
                if (err == ESP_OK) err = send_range(0x2B, r.y, r.h); // RASET
            }
            if (err == ESP_OK)
                err = esp_lcd_panel_io_tx_color(s_io, QCMD(OP_WRITE_COLOR, y == r.y ? 0x2C : 0x3C), b,
                                                (size_t)band.w * band.h * 2); // RAMWR / RAMWRC
        }
    }
    if (s_io) esp_lcd_panel_io_tx_param(s_io, -1, NULL, 0); // drain: wait for queued sends
    free(buf[0]);
    free(buf[1]);
    return err;
}

esp_err_t disp_init(void) {
    if (s_on) return ESP_OK;
    const int64_t t0 = esp_timer_get_time();
    gpio_config_t rst = {.pin_bit_mask = 1ULL << PIN_RST, .mode = GPIO_MODE_OUTPUT};
    ESP_RETURN_ON_ERROR(gpio_config(&rst), TAG, "reset pin");

    spi_bus_config_t bus;
    memset(&bus, 0, sizeof bus);
    bus.sclk_io_num = PIN_SCLK;
    bus.data0_io_num = PIN_D0;
    bus.data1_io_num = PIN_D1;
    bus.data2_io_num = PIN_D2;
    bus.data3_io_num = PIN_D3;
    bus.data4_io_num = bus.data5_io_num = bus.data6_io_num = bus.data7_io_num = -1;
    bus.max_transfer_sz = BAND_BYTES;
    ESP_RETURN_ON_ERROR(spi_bus_initialize(LCD_HOST, &bus, SPI_DMA_CH_AUTO), TAG, "spi bus");

    esp_lcd_panel_io_spi_config_t io = {
        .cs_gpio_num = PIN_CS,
        .dc_gpio_num = -1,
        .pclk_hz = PCLK_HZ,
        .trans_queue_depth = 2,
        .lcd_cmd_bits = 32,
        .lcd_param_bits = 8,
        .flags.quad_mode = 1,
    };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io, &s_io), TAG, "panel io");

    // Hardware reset (esp_lcd_co5300: 10 ms low, 150 ms settle)
    gpio_set_level(PIN_RST, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(PIN_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(150));

    esp_err_t err = ESP_OK;
    err |= send1(0x36, 0xA0); // MADCTL: swap XY + mirror Y (screen orientation)
    err |= send1(0x3A, 0x55); // COLMOD: RGB565
    err |= send(0x11, NULL, 0); // SLPOUT; booster on 120 ms later (CO5300 §7.5.12)
    vTaskDelay(pdMS_TO_TICKS(120));
    // Waveshare BSP v2.0.1 lcd_init_cmds
    err |= send1(0xFE, 0x20);
    err |= send1(0x19, 0x10);
    err |= send1(0x1C, 0xA0);
    err |= send1(0xFE, 0x00);
    err |= send1(0xC4, 0x80);
    err |= send1(0x35, 0x00);
    err |= send1(0x53, 0x20);
    err |= send1(0x51, BRIGHTNESS);
    err |= send1(0x63, 0xFF);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "init commands failed");
        return ESP_FAIL;
    }
    // Clear the panel's RAM before DISPON so no power-on garbage shows.
    memset(&s_num, 0, sizeof s_num);
    sc_blank(chat());
    const sd_rect screen = {0, 0, SD_SCREEN_W, SD_SCREEN_H};
    ESP_RETURN_ON_ERROR(flush(&screen, 1, render_black), TAG, "clear");
    ESP_RETURN_ON_ERROR(send(0x29, NULL, 0), TAG, "DISPON");
    s_on = true;

    ESP_LOGI(TAG, "CO5300 ready in %lld ms; draw buffers 2 x %d B, allocated per draw",
             (long long)((esp_timer_get_time() - t0) / 1000), BAND_BYTES);
    return ESP_OK;
}

void disp_number(float value) {
    if (!s_on) return;
    char text[SD_TEXT_MAX];
    sd_format(value, text);
    if (strcmp(text, s_num.text) == 0) return;
    sd_num prev = s_num;
    sd_rect dirty[SD_SLOTS];
    int n = sd_num_set(&s_num, &sd_font_digits, text, dirty);
    if (n > 0 && flush(dirty, n, render_num) != ESP_OK) {
        // Draw skipped (e.g. no internal DMA RAM): keep the old state so the next call retries.
        s_num = prev;
        ESP_LOGW(TAG, "draw '%s' skipped", text);
    }
}

/* ------------------------------------------------------------------ chat screen */

static void render_chat(uint16_t *buf, sd_rect band, sd_rect area) { sc_render(buf, band, area, &s_chat); }

void disp_chat_turn(const char *user, size_t len) { sc_turn(chat(), user, len); }

void disp_chat_reply(const uint8_t *b, size_t n) { sc_reply(chat(), b, n); }

void disp_chat_status(disp_state state, float tok_s, int ctx_used, int ctx_len) {
    sc_status(chat(), (int)state, tok_s, ctx_used, ctx_len);
}

int disp_chat_draw(void) {
    if (!s_on) return 0;
    sd_rect dirty[SC_MAX_DIRTY];
    int n = sc_dirty(chat(), dirty);
    if (n == 0) return 0;
    if (flush(dirty, n, render_chat) != ESP_OK) return -1; // shown view unchanged: the next draw retries
    sc_commit(&s_chat);
    return n;
}

void disp_off(void) {
    if (!s_on) return;
    esp_lcd_panel_io_tx_param(s_io, -1, NULL, 0);
    send(0x28, NULL, 0); // DISPOFF
    send(0x10, NULL, 0); // SLPIN
    s_on = false;
}
