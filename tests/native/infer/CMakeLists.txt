# Host-native build of the infer stage's two ggml ports (Cut 7):
#   vendor/trellis2    (pixal3d-ggml: TRELLIS.2 image -> 3D; Pixal3D to come)
#   vendor/skin-tokens (skin-tokens-ggml: SkinTokens / TokenRig auto-rig)
# against ONE ggml, the org's, with ggml-cpu only. This is the host oracle and
# port bench; the guest ELF swaps the backend to ggml-rd (Cut 3) and runs one
# thread. Neither vendor tree's own CMakeLists.txt is used: trellis2's adds a
# bundled or workspace-relative ggml, skin-tokens' copies and patches its own
# (deleted, see vendor/skin-tokens/CITATION.cff). The source lists below are
# theirs, verbatim.
#
#   bash tests/native/infer/build.sh          (configure + build + ctest -LE model)
cmake_minimum_required(VERSION 3.28)  # set_property(TEST ... DIRECTORY)
project(ido_infer_host LANGUAGES C CXX)
# No C++20 modules here; module scanning under an sccache launcher fails
# (clang-scan-deps rejects -E).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

get_filename_component(IDO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# The one ggml. C:/contract-manifest/2-contract/ggml is V-Sekai-fire/ggml
# @04b55bba; once Cut 3 merges, point this at ${IDO_ROOT}/vendor/ggml.
set(IDO_GGML_DIR "C:/contract-manifest/2-contract/ggml" CACHE PATH
    "ggml source tree (V-Sekai-fire/ggml)")
# nlohmann/json 3.11.2 for skin-tokens' GLB I/O (src/glb.cpp only), from the
# org's stable-diffusion-ggml thirdparty/json.hpp.
set(IDO_JSON_HPP "C:/contract-manifest/3-interactor/stable-diffusion-ggml/thirdparty/json.hpp"
    CACHE FILEPATH "nlohmann/json single header (3.11+)")
# CPU threads for host runs (the guest will be 1). trellis2 reads
# TRELLIS2_N_THREADS; skin-tokens takes runtime_options::threads.
set(IDO_INFER_THREADS 8 CACHE STRING "ggml-cpu threads for host runs")

set(T2 "${IDO_ROOT}/vendor/trellis2")
set(ST "${IDO_ROOT}/vendor/skin-tokens")

if(NOT EXISTS "${IDO_GGML_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "IDO_GGML_DIR=${IDO_GGML_DIR} is not a ggml tree")
endif()

# ── ggml: static, CPU only, no OpenMP (the guest has none; no libomp.dll) ──
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${IDO_GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
enable_testing()

# ── trellis2 (C++14, as upstream) ──────────────────────────────────────────
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(trellis2 STATIC
  ${T2}/trellis2.cpp
  ${T2}/trellis2_capi.cpp
  ${T2}/mesh_export.cpp
  ${T2}/third_party/xatlas/xatlas.cpp)
target_include_directories(trellis2 PUBLIC ${T2} ${T2}/stb
  PRIVATE ${T2}/examples ${T2}/third_party/xatlas)
target_link_libraries(trellis2 PUBLIC ggml PRIVATE Threads::Threads)
target_compile_features(trellis2 PUBLIC cxx_std_14)
# Upstream's tests/ and their ctest registration, unmodified. Model tests exit
# 77 (skipped) when TRELLIS2_GGUF_DIR / TRELLIS2_DUMPS hold no assets.
add_subdirectory(${T2}/tests ${CMAKE_BINARY_DIR}/trellis2-tests)
foreach(t marching_cubes pbr_sampling mesh_export background_removal preprocess
          dino ss_flow_forward ss_sample ss_dec slat cascade texture)
  set_property(TEST ${t} DIRECTORY ${T2}/tests APPEND PROPERTY ENVIRONMENT
    "TRELLIS2_N_THREADS=${IDO_INFER_THREADS}")
  set_property(TEST ${t} DIRECTORY ${T2}/tests APPEND PROPERTY LABELS trellis2)
endforeach()

# ── skin-tokens (C++23, as upstream), static: no SKINTOKENS_SHARED ─────────
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(IDO_JSON_INC "${CMAKE_BINARY_DIR}/json-include")
configure_file("${IDO_JSON_HPP}" "${IDO_JSON_INC}/nlohmann/json.hpp" COPYONLY)
add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
target_include_directories(nlohmann_json INTERFACE "${IDO_JSON_INC}")

add_library(skintokens STATIC
  ${ST}/src/backend.cpp
  ${ST}/src/binding.cpp
  ${ST}/src/c_api.cpp
  ${ST}/src/gguf.cpp
  ${ST}/src/glb.cpp
  ${ST}/src/mesh_encoder.cpp
  ${ST}/src/model.cpp
  ${ST}/src/qwen.cpp
  ${ST}/src/retarget.cpp
  ${ST}/src/skinning.cpp
  ${ST}/src/skin_vae.cpp
  ${ST}/src/tokenizer.cpp
  ${ST}/src/weights.cpp)
add_library(skintokens::skintokens ALIAS skintokens)
target_include_directories(skintokens PUBLIC ${ST}/include PRIVATE ${ST}/src)
target_compile_definitions(skintokens PRIVATE SKINTOKENS_BUILD)
# PUBLIC: the static archive's consumers resolve ggml and threads themselves.
target_link_libraries(skintokens PUBLIC ggml Threads::Threads
  PRIVATE nlohmann_json::nlohmann_json)

add_executable(skintokens-cli ${ST}/src/cli.cpp)
target_link_libraries(skintokens-cli PRIVATE skintokens::skintokens)

# Upstream's tests/, unmodified: api, c-api, binding, tokenizer (no models).
add_subdirectory(${ST}/tests ${CMAKE_BINARY_DIR}/skintokens-tests)
foreach(t skintokens-api-test skintokens-c-api-test skintokens-binding-test
          skintokens-tokenizer-test)
  set_property(TEST ${t} DIRECTORY ${ST}/tests APPEND PROPERTY LABELS skintokens)
endforeach()
