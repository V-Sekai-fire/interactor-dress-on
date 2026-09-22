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
	guest/avbd/avbd_rd.cpp
	guest/avbd/cloth_grid.cpp
)
target_include_directories(avbd PUBLIC guest guest/avbd kernels/avbd/cpp guest/avbd/slang-rt ${CMAKE_BINARY_DIR})
target_link_libraries(avbd PUBLIC rd_compute)

add_sandbox_program(dress_on guest/main.cpp guest/avbd_fixture.cpp)
target_link_libraries(dress_on PRIVATE avbd rd_compute -lm -lstdc++)

# The project loads res://dress_on.elf; keep it beside the .gd files.
add_custom_command(TARGET dress_on POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:dress_on> ${CMAKE_SOURCE_DIR}/project/dress_on.elf
	COMMENT "Installing dress_on.elf into project/"
)
