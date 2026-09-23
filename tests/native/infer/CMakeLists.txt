# Host-native build of the infer stage's two ggml ports (Cut 7):
#   vendor/trellis2    (pixal3d-ggml: TRELLIS.2 image -> 3D; Pixal3D to come)
#   vendor/skin-tokens (skin-tokens-ggml: SkinTokens / TokenRig auto-rig)
# against ONE ggml, the org's, with ggml-cpu and ggml-vulkan. This is the host
# oracle and port bench; the guest ELF swaps the backend to ggml-rd (Cut 3) and
# runs one thread. ggml-vulkan (the RTX 4090) is the validation backend for the
# ports -- an oracle/dev backend only, never shipped, so AGENTS rule 2 (GPU
# kernels from Lean) does not reach it; ggml-cpu is for tiny graphs or where a
# measured profile shows it much faster.
#
# Every test runs under a hard cap (IDO_TEST_TIMEOUT, 300 s): a timeout is a
# FAIL, not a wait. The cap is applied last to every registered test,
# whatever TIMEOUT it asked for; host runs outside ctest go through run.sh. Neither vendor tree's own CMakeLists.txt is used: trellis2's adds a
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
# Model weights (tools/models/manifest.tsv; gitignored, so a worktree points
# this at the main checkout's). Model tests skip (77) when absent.
set(IDO_MODELS_DIR "${IDO_ROOT}/models" CACHE PATH "models/ directory (SkinTokens-GGUF, ...)")
# CPU threads for host runs (the guest will be 1). trellis2 reads
# TRELLIS2_N_THREADS; skin-tokens takes runtime_options::threads.
set(IDO_INFER_THREADS 8 CACHE STRING "ggml-cpu threads for host runs")
# ggml-vulkan: the host validation backend (not shipped). OFF gives the
# ggml-cpu-only build of the previous commit.
option(IDO_GGML_VULKAN "build ggml-vulkan (host oracle/dev backend only)" ON)
# Hard cap on every test, in seconds (user rule 2026-09-23: never wait tens
# of minutes on ggml-cpu; shrink the problem instead).
set(IDO_TEST_TIMEOUT 300 CACHE STRING "hard per-test timeout (s); a timeout is a FAIL")

set(T2 "${IDO_ROOT}/vendor/trellis2")
set(ST "${IDO_ROOT}/vendor/skin-tokens")

if(NOT EXISTS "${IDO_GGML_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "IDO_GGML_DIR=${IDO_GGML_DIR} is not a ggml tree")
endif()

# ── ggml: static, CPU (+ Vulkan on the host), no OpenMP (the guest has none;
# no libomp.dll) ──
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${IDO_GGML_VULKAN} CACHE BOOL "" FORCE)
if(IDO_GGML_VULKAN AND DEFINED ENV{VULKAN_SDK})
  # SPIRV-HeadersConfig.cmake lives in the SDK's Lib/cmake; vulkan-1.lib (the
  # SDK's import library) links under lld's mingw mode as it is.
  file(TO_CMAKE_PATH "$ENV{VULKAN_SDK}" IDO_VULKAN_SDK)
  list(APPEND CMAKE_PREFIX_PATH "${IDO_VULKAN_SDK}")
endif()
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${IDO_GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
enable_testing()

# ── the hard timeout: capped at the end of configuration, for every test ───
# Deferred so it sees tests added by included files and upstream tests/
# directories after this point; a TIMEOUT above the cap (or none) becomes
# the cap. ctest reports a capped test as "Timeout" = FAIL.
function(ido_cap_test_timeouts)
  foreach(dir "${CMAKE_CURRENT_SOURCE_DIR}" "${T2}/tests" "${ST}/tests")
    get_property(tests DIRECTORY "${dir}" PROPERTY TESTS)
    foreach(t IN LISTS tests)
      get_property(tmo TEST ${t} DIRECTORY "${dir}" PROPERTY TIMEOUT)
      if(NOT tmo OR tmo GREATER IDO_TEST_TIMEOUT)
        set_property(TEST ${t} DIRECTORY "${dir}" PROPERTY TIMEOUT ${IDO_TEST_TIMEOUT})
      endif()
    endforeach()
  endforeach()
endfunction()
cmake_language(DEFER CALL ido_cap_test_timeouts)

# ── ggml-cpu vs the GPU backend on one attention-shaped graph ──────────────
# No model; runs in ctest -LE model. With IDO_GGML_VULKAN a GPU device is
# required (a missing Vulkan device is a FAIL, not a silent CPU pass).
add_executable(ggml-backends-test ggml_backends_test.cpp)
target_link_libraries(ggml-backends-test PRIVATE ggml)
if(IDO_GGML_VULKAN)
  set(_ido_need_gpu 1)
else()
  set(_ido_need_gpu 0)
endif()
add_test(NAME ggml-backends COMMAND ggml-backends-test ${_ido_need_gpu})
set_tests_properties(ggml-backends PROPERTIES LABELS backend)

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
# nlohmann/json 3.11.2 (skin-tokens asks for 3.11; src/glb.cpp only), from
# the org's fork V-Sekai-fire/json, vendored header-only in
# vendor/skin-tokens/third_party/nlohmann-json (see its CITATION.cff).
add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
target_include_directories(nlohmann_json INTERFACE "${ST}/third_party/nlohmann-json")

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

# ── skin-tokens, guest configuration built for the host ────────────────────
# The same sources with SKINTOKENS_GUEST (vendor/skin-tokens/CITATION.cff):
# no ggml_backend_load_all, registry "RD" or an explicit cpu, one thread, the
# bundle only through a bundle_reader. Built here so the guarded code compiles
# and its load contract is tested before the guest ELF exists. (No glb.cpp,
# retarget.cpp, c_api.cpp: file I/O and the C API stay host-only.)
add_library(skintokens_guestcfg STATIC
  ${ST}/src/backend.cpp ${ST}/src/binding.cpp ${ST}/src/gguf.cpp ${ST}/src/mesh_encoder.cpp
  ${ST}/src/model.cpp ${ST}/src/qwen.cpp ${ST}/src/skinning.cpp ${ST}/src/skin_vae.cpp
  ${ST}/src/tokenizer.cpp ${ST}/src/weights.cpp)
target_include_directories(skintokens_guestcfg PUBLIC ${ST}/include PRIVATE ${ST}/src)
target_compile_definitions(skintokens_guestcfg PUBLIC SKINTOKENS_GUEST PRIVATE SKINTOKENS_BUILD)
target_link_libraries(skintokens_guestcfg PUBLIC ggml)

# ── skin-tokens model tests (label model: need SkinTokens-GGUF/F16) ────────
# skintokens-reader: the bundle_reader/upload path vs the unpatched directory
# path on the FoxGirl avatar, greedy; writes the G4b oracle to
# ${CMAKE_BINARY_DIR}/skintokens-oracle (committed copy:
# tests/native/infer/oracle/skintokens-foxgirl).
add_executable(skintokens-reader-test skintokens_reader_test.cpp)
target_compile_definitions(skintokens-reader-test PRIVATE IDO_ROOT_DIR="${IDO_ROOT}")
target_link_libraries(skintokens-reader-test PRIVATE skintokens::skintokens)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/skintokens-oracle")
add_test(NAME skintokens-reader COMMAND skintokens-reader-test
  "${IDO_MODELS_DIR}/SkinTokens-GGUF/F16" "${IDO_ROOT}/project/fixtures/foxgirl/avatar.obj"
  "${CMAKE_BINARY_DIR}/skintokens-oracle")
set_tests_properties(skintokens-reader PROPERTIES LABELS "skintokens;model" SKIP_RETURN_CODE 77
  TIMEOUT 3600 ENVIRONMENT "SKINTOKENS_THREADS=${IDO_INFER_THREADS};SKINTOKENS_PROFILE=1")

# skintokens-guestcfg: the SKINTOKENS_GUEST load contract on the host.
add_executable(skintokens-guestcfg-test skintokens_guestcfg_test.cpp)
target_link_libraries(skintokens-guestcfg-test PRIVATE skintokens_guestcfg)
add_test(NAME skintokens-guestcfg COMMAND skintokens-guestcfg-test "${IDO_MODELS_DIR}/SkinTokens-GGUF/F16")
set_tests_properties(skintokens-guestcfg PROPERTIES LABELS "skintokens;model" SKIP_RETURN_CODE 77)

# ── Stage 7 auxiliary networks (label model) ──
include(naf.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/pixal3d.cmake)
include(moge3.cmake)

# BiRefNet (Pixal3D background removal): ggml-vulkan + a ggml-cpu sampler op
include(${CMAKE_CURRENT_SOURCE_DIR}/birefnet.cmake)
