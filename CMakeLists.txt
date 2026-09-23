cmake_minimum_required(VERSION 3.16)
project(interactor_dress_on C CXX)

# The fiber's context switch is a .S file; clang assembles it, and needs the
# same target triple as C (the toolchain file only sets C and CXX).
if(CMAKE_C_COMPILER_TARGET AND NOT CMAKE_ASM_COMPILER_TARGET)
	set(CMAKE_ASM_COMPILER_TARGET ${CMAKE_C_COMPILER_TARGET})
endif()
enable_language(ASM)

# The guest ELF. It builds against the vendored godot-sandbox guest API and
# nothing else; the riscv64 sysroot comes in through the toolchain file that
# build.sh passes. If this file ever needs another dependency to *link*, the
# independence constraint is being lost -- vendor it under vendor/ instead.

# The helper would otherwise download a generated header from upstream's
# releases. The guest includes <api.hpp> only.
set(DOWNLOAD_RUNTIME_API OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/sandbox-api/cmake sandbox_api)

# The GPU layer is a library, not a member of one program: every ELF that
# touches the GPU links it (composition -- one ELF per stage, each with its own
# heap, rather than one monolith).
add_library(rd_compute STATIC guest/rd_compute.cpp)
target_include_directories(rd_compute PUBLIC guest)
target_link_libraries(rd_compute PUBLIC sandbox_api)

# The AVBD solver, one source two targets (AGENTS.md rule 2): the Lean-emitted
# kernels as slangc cpp (AvbdCpu) and as SPIR-V embedded by kernels/avbd/gen.sh
# into ${CMAKE_BINARY_DIR}/avbd_kernels.inc (AvbdRd). build.sh runs gen.sh
# before configuring so the embedded header exists.
add_library(avbd STATIC
	guest/avbd/avbd_topology.cpp
	guest/avbd/avbd_cpu.cpp
	guest/avbd/avbd_cpu_backward.cpp
	guest/avbd/avbd_rd.cpp
	guest/avbd/avbd_rd_backward.cpp
	guest/avbd/cloth_grid.cpp
)
target_include_directories(avbd PUBLIC guest guest/avbd kernels/avbd/cpp guest/avbd/slang-rt ${CMAKE_BINARY_DIR})
target_link_libraries(avbd PUBLIC rd_compute)

# One ELF per stage (AGENTS.md rule 6), each installed beside the .gd files
# as project/<name>.elf, which is what the host's Sandbox nodes load.
function(add_stage_elf name)
	add_sandbox_program(${name} ${ARGN})
	target_link_libraries(${name} PRIVATE rd_compute -lm -lstdc++)
	add_custom_command(TARGET ${name} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${name}> ${CMAKE_SOURCE_DIR}/project/${name}.elf
		COMMENT "Installing ${name}.elf into project/"
	)
endfunction()

# The curvenet stage (Cut 4): Cassie's pen -> curvenet -> mesh on Geogram,
# PMP and MWT subsets, Eigen-free. cmake/curvenet.cmake defines its libraries.
option(DRESS_ON_WITH_CURVENET "Build the Cassie curvenet stage libraries" ON)
if(DRESS_ON_WITH_CURVENET)
	include(cmake/curvenet.cmake OPTIONAL)
endif()


# Stage 1: the GPU layer's own probes.
add_stage_elf(dress_on guest/main.cpp)

# Stage 2: the drape -- the AVBD solver on both backends, its oracle and bench.
add_stage_elf(drape guest/drape/main.cpp guest/drape/avbd_jobs.cpp guest/jobs.cpp guest/avbd_fixture.cpp
	guest/drape/drape_jobs.cpp guest/drape/drape_scene.cpp guest/drape/primitives.cpp
	guest/drape/lbfgsb.cpp guest/drape/lbfgsb_gate.cpp guest/drape/lbfgsb_jobs.cpp guest/drape/vec_cpu.cpp
	guest/drape/vec_rd.cpp guest/drape/inverse_jobs.cpp)
# The drape driver ports DiffCloth arithmetic that ran on x86-64 without FMA;
# rv64gc has fused multiply-add, which clang would contract into.
set_source_files_properties(guest/drape/drape_jobs.cpp guest/drape/drape_scene.cpp guest/drape/primitives.cpp
	guest/drape/inverse_jobs.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# The L-BFGS-B driver and its CPU kernels are compiled uncontracted too, so the
# guest's cpu path computes what tests/lbfgsb_driver computes on the host.
# vec_rd.cpp includes ${CMAKE_BINARY_DIR}/drape_kernels.inc (kernels/drape/gen.sh).
set_source_files_properties(guest/drape/lbfgsb.cpp guest/drape/lbfgsb_gate.cpp guest/drape/vec_cpu.cpp
	PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_include_directories(drape PRIVATE guest guest/drape)
target_link_libraries(drape PRIVATE avbd)

# Gate 0F: probes.elf, the sandbox runtime probes (gates/0f-runtime). One ELF,
# its own Sandbox, sharing only rd_compute. Its kernels come from
# Lean (lean/ `lake exe emit_probes`) through kernels/probes/gen.sh into
# ${CMAKE_BINARY_DIR}/probes_kernels.inc.
add_stage_elf(probes
	guest/probes/main.cpp
	guest/probes/zfh_probe.cpp
	guest/fiber/fiber.cpp
	guest/fiber/fiber_riscv64.S
)
target_include_directories(probes PRIVATE guest guest/probes ${CMAKE_BINARY_DIR})
# The Zfh control arm: this one TU may emit half-precision instructions.
set_source_files_properties(guest/probes/zfh_probe.cpp PROPERTIES COMPILE_OPTIONS "-march=rv64gc_zfh")

# Probe 15: ggml-cpu in the guest at rv64gc, static, one thread, no OpenMP.
# Cut 3 vendors V-Sekai-fire/ggml at vendor/ggml; until then point GGML_SRC
# at a checkout of it (build.sh passes $GGML_SRC through). The probe TUs
# themselves build at sandbox-api's -march=rv64gcv_zba_zbb_zbs_zbc; ggml's own
# CMake gives its libraries -march=rv64gc.
set(GGML_SRC "${CMAKE_SOURCE_DIR}/vendor/ggml" CACHE PATH "V-Sekai-fire/ggml source tree for probes.elf")
if(EXISTS "${GGML_SRC}/CMakeLists.txt")
	foreach(opt GGML_NATIVE GGML_BACKEND_DL BUILD_SHARED_LIBS GGML_OPENMP GGML_LLAMAFILE
			GGML_RVV GGML_RV_ZFH GGML_RV_ZVFH GGML_RV_ZICBOP GGML_RV_ZIHINTPAUSE GGML_RV_ZVFBFWMA
			GGML_BUILD_TESTS GGML_BUILD_EXAMPLES GGML_CCACHE)
		set(${opt} OFF CACHE BOOL "" FORCE)
	endforeach()
	add_subdirectory(${GGML_SRC} ggml EXCLUDE_FROM_ALL)
	# clang predefines __riscv_v_intrinsic for every riscv64 target, V or not,
	# and ggml-cpu keys its RVV paths on that macro alone; at rv64gc they then
	# fail on vfloat32m8_t. Undefine it so the scalar paths are compiled.
	target_compile_options(ggml-cpu PRIVATE -U__riscv_v_intrinsic)
	target_sources(probes PRIVATE guest/probes/ggml_probe.cpp)
	target_compile_definitions(probes PRIVATE PROBES_WITH_GGML=1)
	target_link_libraries(probes PRIVATE ggml ggml-cpu ggml-base)
else()
	message(STATUS "probes.elf: no ggml at GGML_SRC=${GGML_SRC}; probe 15 reports SKIP")
endif()

# The fit stage (Cut 6): cloth-fit's garment solve (PolyFEM) with the
# Lean SDF sampler. Last, because it sets directory-wide guest numerics for
# the targets it creates. DRESS_ON_WITH_FIT=OFF (build.sh BUILD_FIT=0) skips it.
include(cmake/fit.cmake)
