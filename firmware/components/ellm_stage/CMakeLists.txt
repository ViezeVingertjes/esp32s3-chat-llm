# Core staging policy (ellm_stage.c: portable, host_test/) and the device boot shared by
# firmware/app and firmware/chat (ellm_boot.c: pack mmap, PSRAM arena + flash spill, autoload).
idf_component_register(
    SRCS "ellm_stage.c" "ellm_boot.c"
    INCLUDE_DIRS "include"
    REQUIRES ellm
    PRIV_REQUIRES esp_partition esp_timer heap esp_rom)
