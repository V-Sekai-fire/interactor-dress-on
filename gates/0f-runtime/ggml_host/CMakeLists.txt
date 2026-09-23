# Host-native control for Gate 0F probe 15 (see ../README.md).
#
#   cmake -S gates/0f-runtime/ggml_host -B <build dir> -G Ninja \
#     -DCMAKE_C_COMPILER=x86_64-w64-mingw32-clang -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-clang++ \
#     -DGGML_SRC=<a V-Sekai-fire/ggml checkout; default vendor/ggml>
#
# The first run built it from V-Sekai-fire/ggml @04b55bba (ggml_host.txt).
#
# Same ggml options as the guest, except the ISA ones, which are the host's
# defaults with GGML_NATIVE=OFF.
cmake_minimum_required(VERSION 3.16)
project(ggml_host_0f C CXX)

get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(GGML_SRC "${REPO}/vendor/ggml" CACHE PATH "V-Sekai-fire/ggml source tree")
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${GGML_SRC} ggml EXCLUDE_FROM_ALL)

get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_executable(ggml_host main.cpp ${ROOT}/guest/probes/ggml_probe.cpp)
target_include_directories(ggml_host PRIVATE ${ROOT}/guest/probes)
target_link_libraries(ggml_host PRIVATE ggml ggml-cpu ggml-base -static)
