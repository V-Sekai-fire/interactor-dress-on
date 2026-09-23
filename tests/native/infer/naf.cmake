# ── NAF (Pixal3D's DINOv3 feature upsampler) as ggml graph code ────────────
# Included by CMakeLists.txt after the trellis2 block (C++14; uses T2,
# IDO_ROOT, IDO_INFER_THREADS). vendor/trellis2/naf.{h,cpp} are local to that
# tree (see its CITATION.cff).
# Weights: tools/models/convert_naf_to_gguf.py -> models/NAF/naf_release_f32.gguf.
# Oracle: gates/7-pixal3d/aux-models/naf_ref.py -> naf_oracle/<config>/*.npy
# (14 GB, not in git). Both tests are labelled model and exit 77 without them.
set(IDO_NAF_GGUF "${IDO_ROOT}/models/NAF/naf_release_f32.gguf" CACHE FILEPATH
    "NAF GGUF (tools/models/convert_naf_to_gguf.py)")
set(IDO_NAF_ORACLE "${IDO_ROOT}/gates/7-pixal3d/aux-models/naf_oracle" CACHE PATH
    "naf_ref.py oracle root (holds shape_512/, shape_1024/, tex_1024/)")
add_library(naf STATIC ${T2}/naf.cpp)
target_include_directories(naf PUBLIC ${T2})
target_link_libraries(naf PUBLIC ggml)
target_compile_features(naf PUBLIC cxx_std_14)
add_executable(test_naf ${CMAKE_CURRENT_LIST_DIR}/test_naf.cpp)
target_link_libraries(test_naf PRIVATE naf)
add_test(NAME naf_shape_512
  COMMAND test_naf ${IDO_NAF_GGUF} ${IDO_NAF_ORACLE} shape_512 ${IDO_INFER_THREADS})
# One S = 1024 encoder serves both (naf.md section 1); tex_1024's 4 GiB output
# is evaluated and compared one block row at a time.
add_test(NAME naf_1024
  COMMAND test_naf ${IDO_NAF_GGUF} ${IDO_NAF_ORACLE} shape_1024,tex_1024 ${IDO_INFER_THREADS})
set_tests_properties(naf_shape_512 naf_1024 PROPERTIES
  LABELS "naf;model" SKIP_RETURN_CODE 77 TIMEOUT 3600)
