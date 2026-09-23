# curvenet.elf's libraries (Cut 4): Cassie's pen -> curvenet -> mesh path on
# Geogram, PMP and MWT subsets. Included from CMakeLists.txt when
# DRESS_ON_WITH_CURVENET is ON.
#
# Eigen-free by construction (AGENTS.md rule 3): no target here names an Eigen
# include directory, PMP builds with PMP_NO_EIGEN, and the Cassie solver is
# the Track 5 Eigen-free one. Eigen survives only in fit.elf.
#
# Every library is EXCLUDE_FROM_ALL until the curvenet ELF links them: the
# kernels/cassie/cpp/*_emit.cpp the dispatchers include are produced by the
# Lean side of Cut 4, and the default build must not wait on them.

include(${CMAKE_CURRENT_LIST_DIR}/cassie_sources.cmake)

# Until the Lean side of Cut 4 lands kernels/cassie/cpp/*_emit.cpp, the five
# dispatch TUs cannot compile. ON leaves them (and cassie_kernels) out so the
# rest of the stage can be compile- and link-checked; the symbols they would
# define stay unresolved and are listed by tests/native/curvenet.
option(CURVENET_KERNELS_PENDING "Build curvenet without the Lean-emitted Cassie kernels" OFF)
include(${CMAKE_CURRENT_LIST_DIR}/godot_lite.cmake)

# Bit-determinism: the beautify chain runs on every peer from the same stroke
# samples, so no fast-math anywhere and strict IEEE evaluation (SCsub's clang
# branch). The warning flag silences targets without full strict-FP support.
set(CURVENET_STRICT_FP -ffp-model=strict -Wno-unsupported-floating-point-opt)

# ---- Geogram -----------------------------------------------------------------
add_library(geogram_subset STATIC EXCLUDE_FROM_ALL ${GEOGRAM_SUBSET_SOURCES})
target_include_directories(geogram_subset PUBLIC
	"${GEOGRAM_SUBSET}"
	# OpenNL's .c files include <OpenNL/nl.h> relative to third_party.
	"${GEOGRAM_SUBSET}/geogram/third_party"
	# geofile.h wants zlib.h; headers only, nothing links zlib.
	"${GEOGRAM_SUBSET}/zlib"
)
target_compile_definitions(geogram_subset PUBLIC _USE_MATH_DEFINES GEOGRAM_VERSION="1.9.9")
target_compile_options(geogram_subset PRIVATE ${CURVENET_STRICT_FP} -w)

# ---- PMP ---------------------------------------------------------------------
add_library(pmp_subset STATIC EXCLUDE_FROM_ALL ${PMP_SUBSET_SOURCES})
target_include_directories(pmp_subset PUBLIC "${PMP_SUBSET}")
# pmp::Point is double (types.h:17), matching Cassie's remesh paths; PUBLIC so
# every TU that sees pmp headers agrees on the layout.
target_compile_definitions(pmp_subset PUBLIC _USE_MATH_DEFINES PMP_SCALAR_TYPE_64=1 PMP_NO_EIGEN)
target_compile_options(pmp_subset PRIVATE ${CURVENET_STRICT_FP} -w)

# ---- MWT ---------------------------------------------------------------------
# DMWT.cpp calls cassie::delaunay_triangulate_2d_raw, so the Geogram-backed
# delaunay_geogram.cpp (Godot-free) builds here, beside its only caller.
add_library(mwt_subset STATIC EXCLUDE_FROM_ALL ${MWT_SUBSET_SOURCES} ${CASSIE_DELAUNAY_SOURCES})
target_include_directories(mwt_subset PUBLIC "${MWT_SUBSET}" PRIVATE "${CASSIE_SRC}")
target_compile_definitions(mwt_subset PUBLIC _USE_MATH_DEFINES)
target_compile_options(mwt_subset PRIVATE ${CURVENET_STRICT_FP} -w)
target_link_libraries(mwt_subset PUBLIC geogram_subset)

# ---- Cassie kernels ------------------------------------------------------------
# Lean -> Slang -> slangc -target cpp (AGENTS.md rule 2). No godot_lite
# prelude: the Slang prelude's Vector<T, N> and gdl's Vector cannot share a TU.
if(CURVENET_KERNELS_PENDING)
	# Headers only: cassie_core still includes the dispatch declarations.
	add_library(cassie_kernels INTERFACE)
	target_include_directories(cassie_kernels INTERFACE "${CASSIE_SRC}/solver/slang_dispatch")
else()
	add_library(cassie_kernels STATIC EXCLUDE_FROM_ALL ${CASSIE_KERNEL_SOURCES})
	target_include_directories(cassie_kernels
		PUBLIC "${CASSIE_SRC}/solver/slang_dispatch"
		PRIVATE "${DRESS_ON_ROOT}/guest/avbd/slang-rt" "${DRESS_ON_ROOT}/kernels/cassie/cpp")
	target_compile_options(cassie_kernels PRIVATE ${CURVENET_STRICT_FP} -Wno-non-virtual-dtor)
endif()

# ---- Cassie ------------------------------------------------------------------
add_library(cassie_core STATIC EXCLUDE_FROM_ALL ${CASSIE_CORE_SOURCES})
target_include_directories(cassie_core PUBLIC
	"${CASSIE_SRC}"
	"${DRESS_ON_ROOT}/guest/godot_lite"
	"${DRESS_ON_ROOT}/vendor/godot-core-subset"
)
target_compile_options(cassie_core PRIVATE
	"SHELL:-include ${DRESS_ON_ROOT}/guest/godot_lite/gdl_prelude.h"
	${CURVENET_STRICT_FP}
)
target_link_libraries(cassie_core PUBLIC
	cassie_kernels pmp_subset mwt_subset geogram_subset
	# guest/godot_lite's own library, once it defines one.
	$<TARGET_NAME_IF_EXISTS:godot_lite>
)

# ---- Compile check -------------------------------------------------------------
# Scratch target until curvenet.elf exists: every curvenet library, built for
# whatever toolchain configured this tree (riscv64 via build.sh's toolchain
# file; the host via tests/native/curvenet). Not part of the default build.
add_custom_target(curvenet_compile_check)
add_dependencies(curvenet_compile_check godot_lite geogram_subset pmp_subset mwt_subset cassie_core)
if(NOT CURVENET_KERNELS_PENDING)
	add_dependencies(curvenet_compile_check cassie_kernels)
endif()
