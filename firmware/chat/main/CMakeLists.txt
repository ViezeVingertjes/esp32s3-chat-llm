idf_component_register(SRCS "chat_main.c" INCLUDE_DIRS "."
    REQUIRES ellm ellm_stage status_display esp_partition esp_timer esp_psram heap freertos esp_hw_support
             esp_driver_usb_serial_jtag)
