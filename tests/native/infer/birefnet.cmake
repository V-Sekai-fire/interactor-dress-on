# ── BiRefNet_HR-matting (Pixal3D's background removal) as ggml graph code ──
# Included by CMakeLists.txt (uses IDO_ROOT, IDO_MODELS_DIR, IDO_INFER_THREADS).
# The graph runs on ggml-vulkan; the deformable sampler is a ggml-cpu custom
# op (vendor/birefnet/birefnet.h). Weights: tools/models/convert_birefnet_to_gguf.py
# -> models/BiRefNet_HR-matting/birefnet_hr_matting_f16.gguf. Oracle:
# gates/7-pixal3d/aux-models/birefnet_ref.py -> birefnet_ref_out/ (bunny) and
# birefnet_ref_out_s14_partial/ (not in git). Labelled model; exit 77 without.
set(IDO_BIREFNET_GGUF "${IDO_MODELS_DIR}/BiRefNet_HR-matting/birefnet_hr_matting_f16.gguf" CACHE FILEPATH
    "BiRefNet GGUF (tools/models/convert_birefnet_to_gguf.py)")
set(IDO_BIREFNET_REF "${IDO_ROOT}/gates/7-pixal3d/aux-models/birefnet_ref_out" CACHE PATH
    "birefnet_ref.py output for 04_BunnyCake.jpg")
set(IDO_BIREFNET_REF_S14 "${IDO_ROOT}/gates/7-pixal3d/aux-models/birefnet_ref_out_s14_partial" CACHE PATH
    "birefnet_ref.py output for Pixal3D s_14_img.jpg (logits only)")
add_library(birefnet STATIC ${IDO_ROOT}/vendor/birefnet/birefnet.cpp)
target_include_directories(birefnet PUBLIC ${IDO_ROOT}/vendor/birefnet)
target_link_libraries(birefnet PUBLIC ggml Threads::Threads)
target_compile_features(birefnet PUBLIC cxx_std_14)
add_executable(test_birefnet ${CMAKE_CURRENT_LIST_DIR}/test_birefnet.cpp)
target_link_libraries(test_birefnet PRIVATE birefnet)
add_test(NAME birefnet
  COMMAND test_birefnet ${IDO_BIREFNET_GGUF} ${IDO_BIREFNET_REF} ${IDO_BIREFNET_REF_S14} ${IDO_INFER_THREADS})
set_tests_properties(birefnet PROPERTIES LABELS "birefnet;model" SKIP_RETURN_CODE 77)
