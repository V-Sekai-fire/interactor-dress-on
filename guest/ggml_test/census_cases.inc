// The Gate 3 census shapes test-backend-ops' own eval list does not reach,
// included into make_test_cases_eval() under GGML_GUEST_EXTRA_EVAL_CASES
// (CMakeLists.txt, ggml_test). One block per op family; the lead merges them.
// Each case runs against the in-guest ggml-cpu, so sizes are cut down where the
// census shape would cost minutes of emulated CPU (the GPU timing of the full
// shapes is probes.cpp's fa_perf).

// FLASH_ATTN_EXT (K8): Pixal3D's DiT attention, D = 128, H = 12, f32 Q/K/V,
// no mask, prec F32 (census_pixal3d.csv; test-backend-ops tests f32 K/V only
// at head sizes 64 and 72, and D = 128 only with f16 K/V).
for (int64_t kv : { 5, 1029 }) { // proj cross-attention (Lk = 5); the odd Lk
    for (int64_t nb : { 64, 912 }) {
        test_cases.emplace_back(new test_flash_attn_ext(128, 128, 12, {1, 1}, kv, nb, false, false, 0.0f, 0.0f,
                                                        GGML_PREC_F32, GGML_TYPE_F32, GGML_TYPE_F32));
    }
}
test_cases.emplace_back(new test_flash_attn_ext(128, 128, 12, {1, 1}, 4096, 16, false, false, 0.0f, 0.0f,
                                                GGML_PREC_F32, GGML_TYPE_F32, GGML_TYPE_F32)); // Lk = 4096
test_cases.emplace_back(new test_flash_attn_ext(128, 128, 12, {1, 1}, 912, 912, false, false, 0.0f, 0.0f,
                                                GGML_PREC_F32, GGML_TYPE_F32, GGML_TYPE_F32)); // Lq = Lk = 912
