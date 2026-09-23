# fit.elf: cloth-fit's garment solve (vendor/cloth-fit, PolyFEM + polysolve +
# ipc-toolkit + libigl) in the guest, driven by guest/fit/fit_driver one phase
# per vmcall. Included from the top-level CMakeLists.txt after the other
# stages, because it sets directory-wide compile options (the guest numerics)
# that must reach every target it creates and none of the others.
#
# The same sources, flags and TBB stand-in build natively as fit_native
# (tests/native/fit), the flat control fit.elf is held to.
#
# AGENTS.md rule 3: Eigen lives only here. Rule 2: the SDF spline sampler is
# the Lean emit in kernels/fit/cpp (kernels/fit/gen.sh), compiled into
# polyfem's SdfSpline.cpp.
option(DRESS_ON_WITH_FIT "Build fit.elf (cloth-fit / PolyFEM in the guest; build.sh BUILD_FIT=0 turns it off)" ON)
if(NOT DRESS_ON_WITH_FIT)
    message(STATUS "fit.elf: off (DRESS_ON_WITH_FIT=OFF)")
    return()
endif()

set(FIT_REPO "${CMAKE_CURRENT_LIST_DIR}/..")
get_filename_component(FIT_REPO "${FIT_REPO}" ABSOLUTE)

# --- the org forks, from tools/fit/prepare_forks.sh ------------------------------
# polysolve comes from the patched worktree (embedded specs); the others are
# the forks at cloth-fit's pins. Packages without an org fork come from the
# CPM cache (CPM_SOURCE_CACHE; build.sh points it at a populated one).
set(FIT_FORKS "${FIT_REPO}/.forks" CACHE PATH "tools/fit/prepare_forks.sh checkouts")
foreach(pair "polysolve;polysolve-guest" "ipc-toolkit;ipc-toolkit" "libigl;libigl" "nlohmann_json;json")
    list(GET pair 0 pkg)
    list(GET pair 1 dir)
    if(NOT DEFINED CPM_${pkg}_SOURCE)
        set(CPM_${pkg}_SOURCE "${FIT_FORKS}/${dir}")
    endif()
    if(NOT IS_DIRECTORY "${CPM_${pkg}_SOURCE}")
        message(FATAL_ERROR "fit.elf: ${CPM_${pkg}_SOURCE} is missing; run tools/fit/prepare_forks.sh (or BUILD_FIT=0)")
    endif()
endforeach()

# --- the guest numerics, as fit_native's FIT_NUMERICS=guest -----------------------
# No FMA contraction (riscv64 has fmadd and clang contracts by default), no
# Eigen vectorisation, scalar rv64gc for the whole solver: sandbox-api's
# rv64gcv flags stay on its own targets and main.cpp.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_DISABLE_FIND_PACKAGE_AVX TRUE)
# FIT_MARCH is Gate 6.P's ISA A/B: rv64gc (the default, fit_native's numerics)
# against e.g. rv64gc_zba_zbb_zbs_zbc or rv64gcv. FIT_ELF names the output
# (project/<FIT_ELF>.elf), so an A/B build does not overwrite project/fit.elf.
set(FIT_MARCH "rv64gc" CACHE STRING "-march for every fit.elf solver target")
set(FIT_ELF "fit" CACHE STRING "fit.elf's target and file name (project/<FIT_ELF>.elf)")
add_compile_options(-march=${FIT_MARCH} -mabi=lp64d -ffp-contract=off)
# An unqualified abs(double) is C's int abs under libstdc++ and the double
# overload under llvm-mingw's libc++: native and guest would silently differ
# (it broke CurveCenterTargetForm's bone choice). Never again.
add_compile_options(-Werror=absolute-value)
add_compile_definitions(EIGEN_DONT_VECTORIZE)
set(EIGEN_DONT_VECTORIZE ON CACHE BOOL "" FORCE)

# --- cloth-fit's options ---------------------------------------------------------
set(POLYFEM_THREADING "NONE" CACHE STRING "" FORCE)
set(POLYFEM_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(POLYFEM_WITH_USD OFF CACHE BOOL "" FORCE)
set(POLYFEM_WITH_PARAVIEWO OFF CACHE BOOL "" FORCE)
set(POLYFEM_WITH_CCACHE OFF CACHE BOOL "" FORCE)
set(POLYFEM_WITH_GARMENT ON CACHE BOOL "" FORCE)
set(FIT_KERNELS_PENDING OFF CACHE BOOL "" FORCE)
set(FIT_KERNELS_DIR "${FIT_REPO}/kernels/fit/cpp" CACHE PATH "" FORCE)
set(FIT_SLANG_PRELUDE_DIR "${FIT_REPO}/guest/avbd/slang-rt" CACHE PATH "" FORCE)
set(POLYSOLVE_WITH_CHOLMOD OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_MKL OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_SPECTRA OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_AMGCL OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_ACCELERATE OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_SUPERLU OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_PARDISO OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_HYPRE OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_CUSOLVER OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_EMBEDDED_SPECS ON CACHE BOOL "" FORCE)
# Gate 0F: fesetround is accepted and ignored in the guest, so filib's
# directed-rounding intervals would be silently wrong.
set(IPC_TOOLKIT_WITH_FILIB OFF CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_WITH_CUDA OFF CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(SCALABLE_CCD_WITH_CUDA OFF CACHE BOOL "" FORCE)

# Some CPM packages (predicates, triangle) still ask for CMake < 3.5, which
# CMake 4 refuses; the native build passes the same floor.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# TBB::tbb before ipc-toolkit's recipe runs: the serial stand-in, no oneTBB.
include("${CMAKE_CURRENT_LIST_DIR}/tbb_serial.cmake")

add_subdirectory("${FIT_REPO}/vendor/cloth-fit" cloth-fit EXCLUDE_FROM_ALL)

# garment/CMakeLists.txt adds its sources PUBLIC, which would recompile
# optimize.cpp, GarmentNLProblem.cpp and run_retarget.cpp into every consumer.
set_property(TARGET polyfem PROPERTY INTERFACE_SOURCES "")
# Two known ones stay warnings, both off the solve path: finite-diff's
# compare_gradient log message (abs_diff / abs(x(i))), and polysolve's
# Solver::verify_gradient (runs only with a gradient_fd_strategy other than
# the spec's default "None"). Neither has an edit here: finite-diff has no
# org fork, and the polysolve worktree carries only the embedded-specs patch.
target_compile_options(finitediff_finitediff PRIVATE -Wno-error=absolute-value)
target_compile_options(polysolve PRIVATE -Wno-error=absolute-value)
# Specs come from the embedded copies (utils/SpecPaths.cpp, optimize.cpp).
target_compile_definitions(polyfem PUBLIC POLYFEM_EMBEDDED_SPECS)
# polyfem::warnings makes -Wnon-virtual-dtor an error, and SdfSpline.cpp
# includes Slang's generated C++ prelude (ISlangUnknown, ITexture, ...), which
# trips it. Appended last, so it wins; the warning itself stays on.
set_property(TARGET polyfem_warnings APPEND PROPERTY INTERFACE_COMPILE_OPTIONS -Wno-error=non-virtual-dtor)

# --- the embedded specs --------------------------------------------------------------
# guest/fit/specs is committed and checked against cloth-fit's json-specs and
# polysolve by the native build (resolve_specs --check); here it is embedded.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(FIT_SPECS_DIR "${FIT_REPO}/guest/fit/specs")
set(FIT_SPEC_FILES
    "${FIT_SPECS_DIR}/input-spec.resolved.json"
    "${FIT_SPECS_DIR}/nonlinear-solver-spec.json"
    "${FIT_SPECS_DIR}/linear-solver-spec.json")
set(FIT_EMBED_CPP "${CMAKE_BINARY_DIR}/fit_embedded_specs.cpp")
add_custom_command(
    OUTPUT "${FIT_EMBED_CPP}"
    COMMAND ${Python3_EXECUTABLE} "${FIT_REPO}/tools/fit/embed_files.py" "${FIT_EMBED_CPP}" ${FIT_SPEC_FILES}
    DEPENDS "${FIT_REPO}/tools/fit/embed_files.py" ${FIT_SPEC_FILES}
    COMMENT "Embedding guest/fit/specs"
    VERBATIM)

# The solver side of fit.elf, compiled as fit_native compiles it (C++17, the
# guest numerics above, no sandbox flags): the driver, the specs, and the
# guest-only helpers (SDF dump, probes, I/O counters).
add_library(fit_core STATIC
    "${FIT_REPO}/guest/fit/fit_driver.cpp"
    "${FIT_REPO}/guest/fit/fit_tools.cpp"
    "${FIT_REPO}/guest/fit/fit_probes.cpp"
    "${FIT_EMBED_CPP}")
target_include_directories(fit_core PUBLIC "${FIT_REPO}/guest/fit")
target_link_libraries(fit_core PUBLIC polyfem)

# --- fit.elf --------------------------------------------------------------------------
add_stage_elf(${FIT_ELF} guest/fit/main.cpp)
target_link_libraries(${FIT_ELF} PRIVATE fit_core)
target_compile_options(${FIT_ELF} PRIVATE -ffp-contract=off)
# Every file open in the ELF is counted and refused with EACCES
# (guest/fit/fit_tools.cpp): the guest has no filesystem (Gate 0F probe 3), and
# the driver must not try.
target_link_options(${FIT_ELF} PRIVATE
    "-Wl,--wrap=open,--wrap=open64,--wrap=openat,--wrap=openat64,--wrap=fopen,--wrap=fopen64")
