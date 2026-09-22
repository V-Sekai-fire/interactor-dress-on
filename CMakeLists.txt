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

add_sandbox_program(dress_on guest/main.cpp)
target_link_libraries(dress_on PRIVATE rd_compute -lm -lstdc++)

# The project loads res://dress_on.elf; keep it beside the .gd files.
add_custom_command(TARGET dress_on POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:dress_on> ${CMAKE_SOURCE_DIR}/project/dress_on.elf
	COMMENT "Installing dress_on.elf into project/"
)
