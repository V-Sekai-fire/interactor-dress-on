cmake_minimum_required(VERSION 3.16)
project(interactor_dress_on C CXX)

# The fiber's context switch (guest/fiber, Gate 0F) is a .S file; clang
# assembles it, and needs the same target triple as C (the toolchain file
# only sets C and CXX).
if(CMAKE_C_COMPILER_TARGET AND NOT CMAKE_ASM_COMPILER_TARGET)
	set(CMAKE_ASM_COMPILER_TARGET ${CMAKE_C_COMPILER_TARGET})
endif()
enable_language(ASM)

# __FILE__ (ggml's asserts) and debug info name sources relative to the
# checkout, so a committed ELF names neither the machine nor the worktree
# it was built in, and two checkouts build the same strings.
add_compile_options("-ffile-prefix-map=${CMAKE_SOURCE_DIR}/=")

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
	guest/drape/drape_jobs.cpp guest/drape/drape_scene.cpp guest/drape/primitives.cpp guest/drape/body_mesh.cpp
	guest/drape/lbfgsb.cpp guest/drape/lbfgsb_gate.cpp guest/drape/lbfgsb_jobs.cpp guest/drape/vec_cpu.cpp
	guest/drape/vec_rd.cpp guest/drape/inverse_jobs.cpp vendor/sinew-align/sinew_align.c)
# The drape driver ports DiffCloth arithmetic that ran on x86-64 without FMA;
# rv64gc has fused multiply-add, which clang would contract into.
set_source_files_properties(guest/drape/drape_jobs.cpp guest/drape/drape_scene.cpp guest/drape/primitives.cpp
	guest/drape/inverse_jobs.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# The L-BFGS-B driver and its CPU kernels are compiled uncontracted too, so the
# guest's cpu path computes what tests/lbfgsb_driver computes on the host.
# vec_rd.cpp includes ${CMAKE_BINARY_DIR}/drape_kernels.inc (kernels/drape/gen.sh).
set_source_files_properties(guest/drape/lbfgsb.cpp guest/drape/lbfgsb_gate.cpp guest/drape/vec_cpu.cpp
	PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# The org's rotation fitter (vendor/sinew-align, Align.lean's C port) for the
# fit phase's similarity rest update; its unit test runs in the guest as
# drape_sinew_align_test.
target_include_directories(drape PRIVATE guest guest/drape vendor/sinew-align)
target_link_libraries(drape PRIVATE avbd)

# ggml (vendor/ggml, V-Sekai-fire/ggml @04b55bba): static, riscv64 at rv64gc,
# one thread, no dynamic backend loading, no OpenMP, llamafile or vector
# paths, never Zfh (Gate 0F probe 15 and its Zfh control). ggml's own CMake
# gives its libraries -march=rv64gc; the TUs that include its headers build
# at sandbox-api's -march=rv64gcv_zba_zbb_zbs_zbc. Gate 0F's probes.elf and
# Cut 3's ggml-rd both link it.
foreach(opt GGML_NATIVE GGML_BACKEND_DL BUILD_SHARED_LIBS GGML_OPENMP GGML_LLAMAFILE
		GGML_RVV GGML_RV_ZFH GGML_RV_ZVFH GGML_RV_ZICBOP GGML_RV_ZIHINTPAUSE GGML_RV_ZVFBFWMA
		GGML_BUILD_TESTS GGML_BUILD_EXAMPLES GGML_CCACHE)
	set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(vendor/ggml ggml EXCLUDE_FROM_ALL)
# ggml stamps ggml_commit() with `git rev-parse HEAD` of the enclosing repo,
# which here is this one (vendor/ggml is a subtree): the wrong project, and a
# committed ELF that changes with every commit. Stamp the vendored pin.
get_target_property(_ggml_defs ggml-base COMPILE_DEFINITIONS)
list(FILTER _ggml_defs EXCLUDE REGEX "^GGML_COMMIT=")
list(APPEND _ggml_defs "GGML_COMMIT=\"04b55bba\"")
set_target_properties(ggml-base PROPERTIES COMPILE_DEFINITIONS "${_ggml_defs}")
# clang predefines __riscv_v_intrinsic for every riscv64 target, V or not,
# and ggml-cpu keys its RVV paths on that macro alone; at rv64gc they then
# fail on vfloat32m8_t. Undefine it so the scalar paths are compiled (Gate 0F
# finding 5).
target_compile_options(ggml-cpu PRIVATE -U__riscv_v_intrinsic)

# Gate 0F: probes.elf, the sandbox runtime probes (gates/0f-runtime). One ELF,
# its own Sandbox, sharing only rd_compute. Its kernels come from
# Lean (lean/ `lake exe emit_probes`) through kernels/probes/gen.sh into
# ${CMAKE_BINARY_DIR}/probes_kernels.inc. Probe 15 runs ggml-cpu in the guest.
add_stage_elf(probes
	guest/probes/main.cpp
	guest/probes/zfh_probe.cpp
	guest/probes/ggml_probe.cpp
	guest/fiber/fiber.cpp
	guest/fiber/fiber_riscv64.S
)
target_include_directories(probes PRIVATE guest guest/probes ${CMAKE_BINARY_DIR})
# The Zfh control arm: this one TU may emit half-precision instructions.
set_source_files_properties(guest/probes/zfh_probe.cpp PROPERTIES COMPILE_OPTIONS "-march=rv64gc_zfh")
target_compile_definitions(probes PRIVATE PROBES_WITH_GGML=1)
target_link_libraries(probes PRIVATE ggml ggml-cpu ggml-base)

# --- Cut 3: the ggml-rd backend and Gate 3's ggml_test.elf -----------------------

# The pump protocol: a job on a guest fiber, advanced one host call at a time.
add_library(pump STATIC guest/pump/pump.cpp guest/fiber/fiber.cpp guest/fiber/fiber_riscv64.S)
target_include_directories(pump PUBLIC guest)
target_link_libraries(pump PUBLIC sandbox_api)

# ggml-rd: ggml's backend over rd_compute. An OBJECT library, so every
# ops/<op>.cpp is linked and its GGML_RD_OP registrar runs; the glob means an
# op family adds a file, not a line here. The kernels come from Lean through
# kernels/ggml/gen.sh (${CMAKE_BINARY_DIR}/ggml_kernels.inc).
file(GLOB GGML_RD_OPS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/guest/ggml-rd/ops/*.cpp)
add_library(ggml_rd OBJECT
	guest/ggml-rd/ggml-rd.cpp
	guest/ggml-rd/rd_graph.cpp
	guest/ggml-rd/rd_kernels.cpp
	guest/ggml-rd/rd_pack.cpp
	${GGML_RD_OPS}
)
target_include_directories(ggml_rd PUBLIC guest/ggml-rd kernels/ggml ${CMAKE_BINARY_DIR} vendor/ggml/src)
target_link_libraries(ggml_rd PUBLIC ggml ggml-base rd_compute)

# Gate 3 (G3.ops): test-backend-ops in the guest, ggml-rd against ggml-cpu.
# test-backend-ops.cpp is compiled as is (two guarded edits, see
# vendor/ggml/CITATION.cff) with its main renamed; the link gives every CPU
# backend it creates one thread (guest/ggml_test/main.cpp says why).
add_stage_elf(ggml_test
	guest/ggml_test/main.cpp
	guest/ggml_test/probes.cpp
	guest/ggml_test/census.cpp
	guest/ggml_test/probes_rows.cpp
	guest/ggml_test/probe_conv.cpp
	guest/ggml_test/probe_graph.cpp
	guest/ggml_test/graph_nets.cpp
	guest/ggml_test/app_graphs/app_graphs.cpp
	vendor/ggml/tests/test-backend-ops.cpp
)
set_source_files_properties(vendor/ggml/tests/test-backend-ops.cpp PROPERTIES
	COMPILE_DEFINITIONS "main=ggml_test_backend_ops_main;GGML_GUEST_REGISTER_BACKENDS;N_THREADS=1;GGML_GUEST_EXTRA_EVAL_CASES=\"census_cases.inc\"")
target_include_directories(ggml_test PRIVATE guest guest/ggml_test)
target_link_libraries(ggml_test PRIVATE ggml_rd pump ggml ggml-cpu ggml-base)
target_link_options(ggml_test PRIVATE -Wl,--wrap=ggml_backend_init_by_type)

# Gate 6G.1: rd_worker.elf, GPU round trips from a vmcall on a worker Thread
# (gates/6g-polyfem-gpu/g1-rd-worker). rd_compute plus Lean's saxpby and df32
# dot_reduce, from ${CMAKE_BINARY_DIR}/drape_kernels.inc (kernels/drape/gen.sh).
add_stage_elf(rd_worker guest/rd_worker/main.cpp)
target_include_directories(rd_worker PRIVATE guest ${CMAKE_BINARY_DIR})

# Gate 0G: usd_probe.elf, OpenUSD 26.05 reading a stage from bytes (gates/0g-openusd).
include(cmake/usd_probe.cmake)

# The fit stage (Cut 6): cloth-fit's garment solve (PolyFEM) with the
# Lean SDF sampler. Last, because it sets directory-wide guest numerics for
# the targets it creates. DRESS_ON_WITH_FIT=OFF (build.sh BUILD_FIT=0) skips it.
include(cmake/fit.cmake)
