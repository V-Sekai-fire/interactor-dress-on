# ── MoGe-3 camera FoV (vendor/moge3) as ggml graph code ────────────────────
# Included by CMakeLists.txt. Weights: tools/models/convert_moge3_to_gguf.py
# -> models/moge-3-vitl/moge3_f32.gguf. Oracle: gates/7-pixal3d/aux-models/
# moge3_ref.py -> moge3_ref_out/ (not in git). Label model; exit 77 without.
set(IDO_MOGE3_GGUF "${IDO_MODELS_DIR}/moge-3-vitl/moge3_f32.gguf" CACHE FILEPATH
    "MoGe-3 GGUF (tools/models/convert_moge3_to_gguf.py)")
set(IDO_MOGE3_REF "${IDO_ROOT}/gates/7-pixal3d/aux-models/moge3_ref_out" CACHE PATH
    "moge3_ref.py output directory")
add_library(moge3 STATIC ${IDO_ROOT}/vendor/moge3/moge3.cpp)
target_include_directories(moge3 PUBLIC ${IDO_ROOT}/vendor/moge3)
target_link_libraries(moge3 PUBLIC ggml)
target_compile_features(moge3 PUBLIC cxx_std_14)
add_executable(test_moge3 ${CMAKE_CURRENT_LIST_DIR}/test_moge3.cpp)
target_link_libraries(test_moge3 PRIVATE moge3)
add_test(NAME moge3_fov COMMAND test_moge3)
set_tests_properties(moge3_fov PROPERTIES LABELS "moge3;model" SKIP_RETURN_CODE 77
  ENVIRONMENT "MOGE3_GGUF=${IDO_MOGE3_GGUF};MOGE3_REF=${IDO_MOGE3_REF}")
