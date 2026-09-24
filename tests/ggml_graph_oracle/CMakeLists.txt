# tests/ggml_graph_oracle: the HOST-native reference for Gate 3 G3.graph.
#
#   tests/ggml_graph_oracle/build.sh    (llvm-mingw clang, x86_64)
#
# Builds the vendored ggml natively with ggml-cpu and ggml-vulkan and links
# the same graph builders the guest runs (guest/ggml_test/graph_nets.cpp and
# app_graphs/), so the oracle rebuilds each graph from the same seeds and
# compares the guest's ggml-rd outputs (dumped by project/gate_ggml_graph.gd)
# against ggml-vulkan on the GPU (large graphs) or ggml-cpu (small ones).
#
# ggml-vulkan compiles its own GLSL shaders with glslc here. It is an ORACLE
# only: it never ships and never runs in the guest, so AGENTS.md rule 2 (the
# shipped kernels come from Lean) does not apply to it.
cmake_minimum_required(VERSION 3.19)
project(ggml_graph_oracle C CXX)

set(CMAKE_CXX_STANDARD 17)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
# ORACLE_VULKAN=OFF builds the ggml-cpu arm alone (no Vulkan SDK; --ref=cpu
# --check=none only).
option(ORACLE_VULKAN "build the ggml-vulkan reference backend" ON)
set(GGML_VULKAN ${ORACLE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${ROOT}/vendor/ggml ggml EXCLUDE_FROM_ALL)

add_executable(ggml_graph_oracle
	main.cpp
	${ROOT}/guest/ggml_test/graph_nets.cpp
	${ROOT}/guest/ggml_test/app_graphs/app_graphs.cpp
)
target_include_directories(ggml_graph_oracle PRIVATE ${ROOT}/guest/ggml_test)
if(ORACLE_VULKAN)
	target_compile_definitions(ggml_graph_oracle PRIVATE ORACLE_VULKAN=1)
	target_link_libraries(ggml_graph_oracle PRIVATE ggml-vulkan)
endif()
target_link_libraries(ggml_graph_oracle PRIVATE ggml ggml-cpu ggml-base -static)
