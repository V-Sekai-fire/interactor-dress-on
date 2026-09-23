# tests/ggml_rd_kernels: host-native L2 of the ggml-rd kernels (l2.h).
#
#   tests/ggml_rd_kernels/build.sh      (llvm-mingw clang, x86_64; runs both arms)
#
# Links the vendored ggml (native, one thread: the reference), the guest's
# own packers (guest/ggml-rd/rd_pack.cpp and every ops/*.cpp) and every
# kernel's slangc cpp emit (kernels/ggml/cpp, through a runner generated from
# kernels/ggml/kernels.txt). A new op family adds cases/<family>.cpp; the
# globs pick it and its ops/<op>.cpp up.
cmake_minimum_required(VERSION 3.16)
project(ggml_rd_l2 C CXX)

set(CMAKE_CXX_STANDARD 20)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${ROOT}/vendor/ggml ggml EXCLUDE_FROM_ALL)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(HOST_KERNELS ${CMAKE_BINARY_DIR}/host_kernels.cpp)
file(GLOB EMITS CONFIGURE_DEPENDS ${ROOT}/kernels/ggml/cpp/*_emit.cpp)
add_custom_command(
	OUTPUT ${HOST_KERNELS}
	COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_host_kernels.py
		${ROOT}/kernels/ggml/kernels.txt ${HOST_KERNELS}
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_host_kernels.py ${ROOT}/kernels/ggml/kernels.txt ${EMITS}
	COMMENT "host kernel runners from kernels/ggml/kernels.txt"
)

file(GLOB OPS CONFIGURE_DEPENDS ${ROOT}/guest/ggml-rd/ops/*.cpp)
file(GLOB CASES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cases/*.cpp)
add_executable(ggml_rd_l2
	main.cpp
	${CASES}
	${ROOT}/guest/ggml-rd/rd_pack.cpp
	${OPS}
	${HOST_KERNELS}
)
target_include_directories(ggml_rd_l2 PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${ROOT}/guest/ggml-rd
	${ROOT}/kernels/ggml
	${ROOT}/kernels/ggml/cpp
	${ROOT}/guest/avbd/slang-rt
)
target_link_libraries(ggml_rd_l2 PRIVATE ggml ggml-cpu ggml-base -static)
