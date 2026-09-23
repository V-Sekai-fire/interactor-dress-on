# Native (host) build of the curvenet stage's libraries: the same
# cmake/curvenet.cmake the guest uses (cassie_core on godot_lite, Geogram,
# PMP, MWT), compiled with the host clang++ (llvm-mingw on Windows), plus a
# smoke executable. The flat control for the guest build: if something fails
# here, the sandbox is not to blame.
#
#   tests/native/curvenet/build.sh
#
# CURVENET_KERNELS_PENDING=ON (the default here until the Lean side lands
# kernels/cassie/cpp/*_emit.cpp) leaves the five dispatch TUs out; build.sh
# lists the symbols that leaves unresolved.
cmake_minimum_required(VERSION 3.16)
project(curvenet_native C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CURVENET_KERNELS_PENDING "Build curvenet without the Lean-emitted Cassie kernels" ON)
include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/curvenet.cmake)

add_executable(curvenet_smoke
	${CMAKE_CURRENT_LIST_DIR}/smoke_main.cpp
	${CMAKE_CURRENT_LIST_DIR}/smoke_geogram.cpp
)
# The prelude only for the TU that uses Godot names; the Geogram TU stays
# prelude-free, like mwt and geogram themselves.
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/smoke_main.cpp PROPERTIES
	COMPILE_OPTIONS "-include;${DRESS_ON_ROOT}/guest/godot_lite/gdl_prelude.h")
target_include_directories(curvenet_smoke PRIVATE "${CASSIE_SRC}")
target_compile_options(curvenet_smoke PRIVATE ${CURVENET_STRICT_FP})
target_link_libraries(curvenet_smoke PRIVATE cassie_core mwt_subset geogram_subset)
