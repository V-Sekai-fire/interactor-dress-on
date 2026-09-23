cmake_minimum_required(VERSION 3.16)
project(interactor_dress_on C CXX)

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

# Stage 1: the GPU layer's own probes.
add_stage_elf(dress_on guest/main.cpp)

# Stage 2: the drape -- the AVBD solver on both backends, its oracle and bench.
add_stage_elf(drape guest/drape/main.cpp guest/drape/avbd_jobs.cpp guest/jobs.cpp guest/avbd_fixture.cpp
	guest/drape/drape_jobs.cpp guest/drape/drape_scene.cpp guest/drape/primitives.cpp
	guest/drape/lbfgsb.cpp guest/drape/lbfgsb_gate.cpp guest/drape/lbfgsb_jobs.cpp guest/drape/vec_cpu.cpp
	guest/drape/vec_rd.cpp)
# The drape driver ports DiffCloth arithmetic that ran on x86-64 without FMA;
# rv64gc has fused multiply-add, which clang would contract into.
set_source_files_properties(guest/drape/drape_jobs.cpp guest/drape/drape_scene.cpp guest/drape/primitives.cpp
	PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# The L-BFGS-B driver and its CPU kernels are compiled uncontracted too, so the
# guest's cpu path computes what tests/lbfgsb_driver computes on the host.
# vec_rd.cpp includes ${CMAKE_BINARY_DIR}/drape_kernels.inc (kernels/drape/gen.sh).
set_source_files_properties(guest/drape/lbfgsb.cpp guest/drape/lbfgsb_gate.cpp guest/drape/vec_cpu.cpp
	PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_include_directories(drape PRIVATE guest guest/drape)
target_link_libraries(drape PRIVATE avbd)
