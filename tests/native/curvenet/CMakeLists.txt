# Native (host) build of the curvenet stage's libraries: the same
# cmake/curvenet.cmake the guest uses (cassie_core on godot_lite, Geogram,
# PMP, MWT), compiled with the host clang++ (llvm-mingw on Windows), plus a
# smoke executable. The flat control for the guest build: if something fails
# here, the sandbox is not to blame.
#
#   tests/native/curvenet/build.sh
#
# The five dispatch TUs build from the committed kernels/cassie/cpp/*_emit.cpp
# (kernels/cassie/gen.sh). CURVENET_KERNELS_PENDING=ON leaves them out, and
# build.sh then lists exactly their five symbols as unresolved.
cmake_minimum_required(VERSION 3.16)
project(curvenet_native C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CURVENET_KERNELS_PENDING "Build curvenet without the Lean-emitted Cassie kernels" OFF)
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

# The five Lean-emitted kernels, run through the vendored dispatchers
# (smoke_kernels.cpp); only when they are built.
if(NOT CURVENET_KERNELS_PENDING)
	add_executable(curvenet_kernels_smoke ${CMAKE_CURRENT_LIST_DIR}/smoke_kernels.cpp)
	target_compile_options(curvenet_kernels_smoke PRIVATE ${CURVENET_STRICT_FP})
	target_link_libraries(curvenet_kernels_smoke PRIVATE cassie_kernels)
endif()

# Gate 4's flat control: guest/curvenet/checks.cpp and curvenet_api.cpp (the
# curvenet_core library curvenet.elf links) as a host executable.
if(NOT CURVENET_KERNELS_PENDING)
	add_executable(cassie_checks ${CMAKE_CURRENT_LIST_DIR}/checks_main.cpp)
	target_link_libraries(cassie_checks PRIVATE curvenet_core)
endif()
