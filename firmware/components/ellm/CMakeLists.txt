# Shared ELLM runtime (portable C in runtime/) plus ESP32-specific helpers.
# ELLM_HOT (IRAM placement) is defined in runtime/ellm_kernels.h under ESP_PLATFORM.
idf_component_register(
    SRCS "../../../runtime/ellm_kernels.c" "../../../runtime/ellm_pack.c" "../../../runtime/ellm.c"
         "../../../runtime/ellm_tok.c" "../../../runtime/ellm_chat.c"
         "ellm_dual.c"
    INCLUDE_DIRS "../../../runtime" "include"
    REQUIRES freertos)
# Float determinism (global constraints): no multiply-add contraction (GCC/Xtensa fuses into
# madd.s by default), so device float results are bit-identical to the host build.
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
