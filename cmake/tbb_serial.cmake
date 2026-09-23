# TBB::tbb as guest/fit/tbb_serial: a header-only, one-thread stand-in for the
# oneTBB calls in ipc-toolkit and scalable-ccd (see tbb/tbb_serial.h for the
# list and the semantics). Include this before ipc-toolkit's recipe: every
# onetbb recipe in the tree returns early when TBB::tbb exists, so oneTBB is
# neither fetched nor compiled nor linked.
if(TARGET TBB::tbb)
    get_target_property(_tbb_aliased TBB::tbb ALIASED_TARGET)
    if(NOT _tbb_aliased STREQUAL "tbb_serial")
        message(FATAL_ERROR "tbb_serial.cmake: TBB::tbb already exists (${_tbb_aliased}); include this before any onetbb recipe")
    endif()
    return()
endif()

add_library(tbb_serial INTERFACE)
target_include_directories(tbb_serial SYSTEM INTERFACE "${CMAKE_CURRENT_LIST_DIR}/../guest/fit/tbb_serial")
target_compile_definitions(tbb_serial INTERFACE TBB_SERIAL_STANDIN=1)
add_library(TBB::tbb ALIAS tbb_serial)
message(STATUS "TBB::tbb is the serial stand-in (guest/fit/tbb_serial), not oneTBB")
