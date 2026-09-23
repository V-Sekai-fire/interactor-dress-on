# fit_native: the native same-code build of fit.elf's solver (Cut 6).
#
# Compiles the cloth-fit sources fit.elf needs (not PolyFEM_bin's main.cpp or
# run_retarget.cpp), guest/fit/fit_driver.cpp and tools/fit/fit_native.cpp
# with the guest's numerics: POLYFEM_THREADING=NONE, -ffp-contract=off,
# EIGEN_DONT_VECTORIZE, IPC_TOOLKIT_WITH_FILIB=OFF (Gate 0F: fesetround is not
# honoured in the guest), specs embedded (POLYFEM_EMBEDDED_SPECS, polysolve
# with vendor/cloth-fit-patches/polysolve-embedded-specs.patch). Build with
# tests/native/fit/build.sh.
#
# FIT_SDF selects the FitForm:
#   openvdb  FitForm as vendored at FIT_FITFORM_REV (the OpenVDB SDF the
#            upstream oracle samples), extracted from git with its two debug
#            OBJ writes guarded. Native only: an interim until the brick-grid
#            SdfGrid FitForm (working tree) has its Lean kernel.
#   sdfgrid  the working tree's FitForm + SdfGrid + SdfSpline (no OpenVDB).
#            FIT_SDF_SAMPLER picks the spline sampler behind SdfSpline.hpp:
#              kernel     SdfSpline.cpp over the Lean emit (needs FIT_KERNELS_DIR
#                         and FIT_SLANG_PRELUDE_DIR)
#              reference  sdf_spline_reference.cpp, the Gate 6a reference
#                         (gates/6-fit/sdf/spline_ref.h); native only, the
#                         default until the Lean kernel is emitted
#              pending    SdfSpline.cpp with FIT_KERNELS_PENDING (throws)
cmake_minimum_required(VERSION 3.25)

# As cloth-fit's top level: tight-inclusion must not find AVX (it would add
# -mavx2 and vector code paths the guest does not have).
set(CMAKE_DISABLE_FIND_PACKAGE_AVX TRUE)

project(fit_native C CXX)

set(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
get_filename_component(REPO "${REPO}" ABSOLUTE)
set(CF "${REPO}/vendor/cloth-fit")
set(CF_SRC "${CF}/src/polyfem")

set(FIT_SDF "openvdb" CACHE STRING "FitForm SDF backend: openvdb (interim, native only) or sdfgrid")
set_property(CACHE FIT_SDF PROPERTY STRINGS openvdb sdfgrid)
set(FIT_FITFORM_REV "87d591f2" CACHE STRING "Commit whose OpenVDB FitForm FIT_SDF=openvdb builds")
set(FORKS "${REPO}/.forks" CACHE PATH "tools/forks/fetch.sh checkouts")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# FIT_NUMERICS=guest (default): the guest's floating-point contract on every
# target including the deps, and polyfem without threading. =upstream: the
# oracle's numerics instead (vectorised Eigen, contraction as clang defaults,
# POLYFEM_THREADING=TBB held to one thread by fit_native), which separates
# "the driver is not upstream's loop" from "the guest numerics differ".
set(FIT_NUMERICS "guest" CACHE STRING "guest | upstream")
set_property(CACHE FIT_NUMERICS PROPERTY STRINGS guest upstream)
if(FIT_NUMERICS STREQUAL "guest")
    add_compile_options(-ffp-contract=off)
    add_compile_definitions(EIGEN_DONT_VECTORIZE)
    set(EIGEN_DONT_VECTORIZE ON CACHE BOOL "" FORCE)
endif()

# Dependencies through cloth-fit's own recipes; the org forks come in via
# CPM_<pkg>_SOURCE (build.sh), polysolve from the patched worktree.
list(APPEND CMAKE_MODULE_PATH "${CF}/cmake/recipes" "${CF}/cmake/polyfem")
include(polyfem_cpm_cache)

set(POLYSOLVE_WITH_CHOLMOD OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_MKL OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_SPECTRA OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_WITH_AMGCL OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_EMBEDDED_SPECS ON CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_WITH_FILIB OFF CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(POLYSOLVE_BUILD_TESTS OFF CACHE BOOL "" FORCE)

include(eigen)
if(FIT_SDF STREQUAL "openvdb")
    set(USE_BLOSC OFF CACHE BOOL "" FORCE)
    set(USE_ZLIB OFF CACHE BOOL "" FORCE)
    include(openvdb)
endif()
include(polysolve)
include(libigl)
include(finite_diff)
include(ipc_toolkit)
include(spdlog)
include(jse)

# ---------------------------------------------------------------------------
# polyfem_fit: the cloth-fit sources the garment solve uses.
set(POLYFEM_FIT_SOURCES
    ${CF_SRC}/autogen/auto_derivatives1.cpp
    ${CF_SRC}/autogen/auto_derivatives2.cpp
    ${CF_SRC}/autogen/auto_derivatives3.cpp
    ${CF_SRC}/io/MatrixIO.cpp
    ${CF_SRC}/io/OBJReader.cpp
    ${CF_SRC}/io/OBJWriter.cpp
    ${CF_SRC}/mesh/MeshUtils.cpp
    ${CF_SRC}/solver/FullNLProblem.cpp
    ${CF_SRC}/solver/forms/ContactForm.cpp
    ${CF_SRC}/solver/forms/garment_forms/GarmentForm.cpp
    ${CF_SRC}/solver/forms/garment_forms/GarmentALForm.cpp
    ${CF_SRC}/solver/forms/garment_forms/CurveConstraintForm.cpp
    ${CF_SRC}/solver/forms/garment_forms/CurveCenterTargetForm.cpp
    ${CF_SRC}/utils/getRSS.c
    ${CF_SRC}/utils/JSONUtils.cpp
    ${CF_SRC}/utils/Logger.cpp
    ${CF_SRC}/utils/MatrixCache.cpp
    ${CF_SRC}/utils/MatrixUtils.cpp
    ${CF_SRC}/utils/par_for.cpp
    ${CF_SRC}/utils/SpecPaths.cpp
    ${CF_SRC}/utils/StringUtils.cpp
    ${CF_SRC}/garment/optimize.cpp
    ${CF_SRC}/garment/GarmentNLProblem.cpp
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(FIT_SDF STREQUAL "openvdb")
    set(FITFORM_DIR "${CMAKE_BINARY_DIR}/fitform_openvdb")
    set(FITFORM_SUB "${FITFORM_DIR}/polyfem/solver/forms/garment_forms")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/extract_fitform.py"
                "${REPO}" "${FIT_FITFORM_REV}" "${FITFORM_SUB}"
        RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "extract_fitform.py failed")
    endif()
    list(APPEND POLYFEM_FIT_SOURCES "${FITFORM_SUB}/FitForm.cpp")
else()
    set(FIT_SDF_SAMPLER "reference" CACHE STRING "kernel | reference (native only) | pending")
    set_property(CACHE FIT_SDF_SAMPLER PROPERTY STRINGS kernel reference pending)
    list(APPEND POLYFEM_FIT_SOURCES
        ${CF_SRC}/solver/forms/garment_forms/FitForm.cpp
        ${CF_SRC}/solver/forms/garment_forms/SdfGrid.cpp)
    if(FIT_SDF_SAMPLER STREQUAL "reference")
        list(APPEND POLYFEM_FIT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/sdf_spline_reference.cpp")
    else()
        list(APPEND POLYFEM_FIT_SOURCES ${CF_SRC}/solver/forms/garment_forms/SdfSpline.cpp)
    endif()
endif()

add_library(polyfem_fit STATIC ${POLYFEM_FIT_SOURCES})
if(FIT_SDF STREQUAL "openvdb")
    # Ahead of src/ so <polyfem/.../FitForm.hpp> is the extracted OpenVDB one.
    target_include_directories(polyfem_fit BEFORE PUBLIC "${FITFORM_DIR}")
    target_compile_definitions(polyfem_fit PUBLIC FIT_FITFORM_OUT_DIR)
    target_link_libraries(polyfem_fit PUBLIC openvdb)
else()
    target_compile_definitions(polyfem_fit PUBLIC FIT_SDF_GRID)
    set(FIT_KERNELS_DIR "" CACHE PATH "kernels/fit/cpp (sdf_spline_hessian_emit.cpp)")
    set(FIT_SLANG_PRELUDE_DIR "" CACHE PATH "slang-cpp-prelude.h directory")
    if(FIT_SDF_SAMPLER STREQUAL "kernel")
        if(NOT EXISTS "${FIT_KERNELS_DIR}/sdf_spline_hessian_emit.cpp")
            message(FATAL_ERROR "FIT_SDF_SAMPLER=kernel needs FIT_KERNELS_DIR with sdf_spline_hessian_emit.cpp")
        endif()
        target_include_directories(polyfem_fit PRIVATE "${FIT_KERNELS_DIR}" "${FIT_SLANG_PRELUDE_DIR}")
    elseif(FIT_SDF_SAMPLER STREQUAL "pending")
        target_compile_definitions(polyfem_fit PRIVATE FIT_KERNELS_PENDING)
    endif()
endif()
target_include_directories(polyfem_fit PUBLIC "${CF}/src")
target_compile_definitions(polyfem_fit PUBLIC
    EIGEN_STACK_ALLOCATION_LIMIT=0
    POLYFEM_SMALL_N=80
    POLYFEM_BIG_N=1000
    POLYFEM_EMBEDDED_SPECS)
target_link_libraries(polyfem_fit PUBLIC
    Eigen3::Eigen polysolve::polysolve igl::core finitediff::finitediff
    ipc::toolkit spdlog::spdlog jse::jse)
if(FIT_NUMERICS STREQUAL "upstream")
    target_compile_definitions(polyfem_fit PUBLIC POLYFEM_WITH_TBB)
    target_link_libraries(polyfem_fit PUBLIC TBB::tbb)
endif()

# ---------------------------------------------------------------------------
# Embedded specs: committed under guest/fit/specs, regenerated here by
# resolve_specs and checked for staleness on every build.
set(SPECS_DIR "${REPO}/guest/fit/specs")
set(SPEC_FILES
    "${SPECS_DIR}/input-spec.resolved.json"
    "${SPECS_DIR}/nonlinear-solver-spec.json"
    "${SPECS_DIR}/linear-solver-spec.json")

add_executable(resolve_specs "${REPO}/tools/fit/resolve_specs.cpp")
target_link_libraries(resolve_specs PRIVATE jse::jse)
target_link_options(resolve_specs PRIVATE -static)

set(SPECS_OUT "${CMAKE_BINARY_DIR}/specs")
file(MAKE_DIRECTORY "${SPECS_OUT}")
add_custom_target(check_specs ALL
    COMMAND resolve_specs "${CF}/json-specs" "${polysolve_SOURCE_DIR}" "${SPECS_OUT}" --check "${SPECS_DIR}"
    DEPENDS resolve_specs
    COMMENT "Checking guest/fit/specs against vendor/cloth-fit/json-specs and polysolve"
    VERBATIM)

set(EMBED_CPP "${CMAKE_BINARY_DIR}/fit_embedded_specs.cpp")
add_custom_command(
    OUTPUT "${EMBED_CPP}"
    COMMAND ${Python3_EXECUTABLE} "${REPO}/tools/fit/embed_files.py" "${EMBED_CPP}" ${SPEC_FILES}
    DEPENDS "${REPO}/tools/fit/embed_files.py" ${SPEC_FILES}
    COMMENT "Embedding guest/fit/specs"
    VERBATIM)

add_library(fit_driver STATIC "${REPO}/guest/fit/fit_driver.cpp" "${EMBED_CPP}")
target_link_libraries(fit_driver PUBLIC polyfem_fit)
target_include_directories(fit_driver PUBLIC "${REPO}/guest/fit")
add_dependencies(fit_driver check_specs)

# ---------------------------------------------------------------------------
add_executable(fit_native "${REPO}/tools/fit/fit_native.cpp")
target_link_libraries(fit_native PRIVATE fit_driver psapi)
# --io-probe: every file open in the process goes through a counting wrapper.
target_link_options(fit_native PRIVATE -static
    "-Wl,--wrap=fopen,--wrap=_wfopen,--wrap=_open,--wrap=_wopen,--wrap=_sopen_s,--wrap=_wsopen_s,--wrap=CreateFileA,--wrap=CreateFileW")
