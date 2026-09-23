# Injected with -DCMAKE_PROJECT_PolyFEM_INCLUDE: runs right after
# project(PolyFEM), before any target exists, so the openvdb_dump target is
# added by a call deferred to the end of the top-level directory. The vendored
# sources are not modified and PolyFEM_bin is built exactly as upstream.
set(_ovd_file "${CMAKE_CURRENT_LIST_DIR}/openvdb_dump_target.cmake")
cmake_language(EVAL CODE "cmake_language(DEFER DIRECTORY [[${CMAKE_SOURCE_DIR}]] CALL include [[${_ovd_file}]])")
