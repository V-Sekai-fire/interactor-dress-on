# Gate 0G: usd_probe.elf over OpenUSD 26.05 + oneTBB 2021.12.0 built static
# for riscv64 by gates/0g-openusd/build_usd_rv64.sh into USD_RV64_DIR (default
# C:/b/g0g). Upstream sources, read-only, outside the repo (README). Skipped
# when that build is absent.
set(USD_RV64_DIR "C:/b/g0g" CACHE PATH "gates/0g-openusd/build_usd_rv64.sh output")
set(USD_RV64_SRC "C:/b/usd2605" CACHE PATH "the patched OpenUSD v26.05 checkout")
if(NOT EXISTS "${USD_RV64_DIR}/usd/pxr/usd/usdSkel/libusd_usdSkel.a")
	message(STATUS "usd_probe.elf: no OpenUSD riscv64 build at ${USD_RV64_DIR}; skipped")
	return()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_usd_inc ${CMAKE_BINARY_DIR}/usd_probe_gen/usd_resources.inc)
add_custom_command(OUTPUT ${_usd_inc}
	COMMAND ${CMAKE_COMMAND} -E env USD_SRC=${USD_RV64_SRC}
		${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/gates/0g-openusd/gen_resources.py ${USD_RV64_DIR}/usd ${_usd_inc}
	DEPENDS ${CMAKE_SOURCE_DIR}/gates/0g-openusd/gen_resources.py
	COMMENT "Embedding OpenUSD plugInfo.json + generatedSchema.usda")

add_stage_elf(usd_probe guest/usd_probe/main.cpp guest/usd_probe/usd_probe_core.cpp ${_usd_inc})
target_include_directories(usd_probe PRIVATE guest/usd_probe ${CMAKE_BINARY_DIR}/usd_probe_gen
	${USD_RV64_DIR}/usd/include ${USD_RV64_DIR}/inst/include)
target_compile_definitions(usd_probe PRIVATE PXR_STATIC=1 MFB_PACKAGE_NAME=usdProbe MFB_ALT_PACKAGE_NAME=usdProbe
	TBB_USE_EXCEPTIONS=1)
# OpenUSD 26.05 headers hold unique_ptr<incomplete> members that C++23's
# constexpr unique_ptr destructor rejects; the pxr-facing TU is C++17 as
# OpenUSD itself is built.
set_source_files_properties(guest/usd_probe/usd_probe_core.cpp PROPERTIES COMPILE_OPTIONS "-std=gnu++17")

# Static pxr libraries register TfTypes, file formats and schemas from static
# initialisers, so each goes in whole (as OpenUSD's own static link does).
set(_usd_libs)
foreach(l usdSkel usdShade usdGeom sdr usd pcp kind sdf ar ts plug trace work vt js gf tf arch)
	file(GLOB_RECURSE _a "${USD_RV64_DIR}/usd/pxr/*/libusd_${l}.a")
	if(NOT _a)
		message(FATAL_ERROR "usd_probe.elf: libusd_${l}.a not found under ${USD_RV64_DIR}/usd")
	endif()
	list(APPEND _usd_libs ${_a})
endforeach()
target_link_libraries(usd_probe PRIVATE -Wl,--whole-archive ${_usd_libs} -Wl,--no-whole-archive
	${USD_RV64_DIR}/inst/lib/libtbb.a -lpthread -ldl)
