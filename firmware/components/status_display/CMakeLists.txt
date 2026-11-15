# AMOLED display for the CO5300 (spec 5.6): the tok/s number (sd_core) or the chat screen (sd_chat,
# Plan 4 Task 14). sd_core.c and sd_chat.c are portable (host_test/); fonts are rodata (flash).
idf_component_register(
    SRCS "status_display.c" "sd_core.c" "font_digits.c" "sd_chat.c" "font_chat.c"
    INCLUDE_DIRS "include"
    PRIV_REQUIRES esp_lcd esp_driver_spi esp_driver_gpio esp_timer heap freertos)
