# Included at the end of cloth-fit's top-level CMakeLists (see
# openvdb_dump_hook.cmake), so polyfem and polyfem_static_cxx_runtime exist.
get_filename_component(_ovd_src "${CMAKE_CURRENT_LIST_DIR}/../fit/openvdb_dump.cpp" ABSOLUTE)
if(NOT EXISTS "${_ovd_src}")
  message(FATAL_ERROR "openvdb_dump.cpp missing: ${_ovd_src}")
endif()
add_executable(openvdb_dump "${_ovd_src}")
# polyfem carries openvdb (and igl, Eigen) as PUBLIC link deps.
target_link_libraries(openvdb_dump PRIVATE polyfem)
# -static like PolyFEM_bin: without it llvm-mingw imports libc++.dll and
# libunwind.dll, and the tool runs only with the toolchain on PATH.
if(COMMAND polyfem_static_cxx_runtime)
  polyfem_static_cxx_runtime(openvdb_dump)
elseif(WIN32 AND NOT MSVC)
  target_link_options(openvdb_dump PRIVATE -static)
endif()
