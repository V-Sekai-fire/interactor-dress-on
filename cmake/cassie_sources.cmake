# Source lists for curvenet.elf's vendored subsets (Cut 4). Absolute paths,
# rooted at the repository. The trees under vendor/ are produced by
# tools/vendor/{cassie,thirdparty}_subset.sh and hold exactly the subset, so
# the third-party lists glob them; the Cassie list is explicit because its
# files split three ways by what they may see.

get_filename_component(DRESS_ON_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(CASSIE_SRC "${DRESS_ON_ROOT}/vendor/cassie/src")
set(GEOGRAM_SUBSET "${DRESS_ON_ROOT}/vendor/geogram-subset")
set(PMP_SUBSET "${DRESS_ON_ROOT}/vendor/pmp-subset")
set(MWT_SUBSET "${DRESS_ON_ROOT}/vendor/mwt")

# Cassie proper: compiled against guest/godot_lite through the force-included
# gdl_prelude.h (namespace gdl), never against api.hpp.
set(CASSIE_CORE_SOURCES
	cassie_beautifier.cpp
	cassie_beautifier_params.cpp
	cassie_remesh.cpp
	cassie_sketcher.cpp
	cassie_stroke_packet.cpp
	cassie_triangulator.cpp
	constraints/cassie_constraint.cpp
	constraints/cassie_intersection_constraint.cpp
	constraints/cassie_intersection_finder.cpp
	constraints/cassie_mirror_plane_constraint.cpp
	constraints/cassie_surface_constraint.cpp
	curves/cassie_curve_fit.cpp
	curves/rdp_simplify.cpp
	polygon_triangulation.cpp
	refine.cpp
	sketch/cassie_curvenet.cpp
	sketch/cassie_curvenet_extractor.cpp
	sketch/cassie_curvenet_knot.cpp
	sketch/cassie_final_stroke.cpp
	sketch/cassie_input_stroke.cpp
	sketch/cassie_polar.cpp
	sketch/cassie_sketch_graph.cpp
	sketch/cassie_surface_manager.cpp
	sketch/cassie_surface_patch.cpp
	solver/cassie_constraint_solver.cpp
	solver/cassie_eigen.cpp
	solver/cassie_pcg.cpp
	solver/fidelity_energy.cpp
	solver/g1_constraint.cpp
	solver/on_surface_energy.cpp
	solver/planarity_constraint.cpp
	solver/position_constraint.cpp
	solver/self_intersection_constraint.cpp
	solver/tangent_constraint.cpp
)
list(TRANSFORM CASSIE_CORE_SOURCES PREPEND "${CASSIE_SRC}/")

# The Lean-emitted kernels' CPU dispatchers. Godot-free: each includes the
# Slang cpp prelude (whose Vector<T, N> would collide with gdl::Vector) and
# one kernels/cassie/cpp/<k>_emit.cpp inside its own namespace.
set(CASSIE_KERNEL_SOURCES
	solver/slang_dispatch/curve_casteljau_dispatch.cpp
	solver/slang_dispatch/curve_generate_bezier_dispatch.cpp
	solver/slang_dispatch/curve_newton_dispatch.cpp
	solver/slang_dispatch/curve_rdp_dispatch.cpp
	solver/slang_dispatch/spmv_dispatch.cpp
)
list(TRANSFORM CASSIE_KERNEL_SOURCES PREPEND "${CASSIE_SRC}/")

# Geogram-backed 2D Delaunay (Godot-free after the rewrite); DMWT.cpp is its
# only caller, so it builds with mwt.
set(CASSIE_DELAUNAY_SOURCES "${CASSIE_SRC}/delaunay_geogram.cpp")

# Geogram: modules/cassie/SCsub's globs (gates/0b-crosscompile) plus
# delaunay_2d.cpp; the vendored tree carries nothing else to compile.
file(GLOB GEOGRAM_SUBSET_SOURCES
	"${GEOGRAM_SUBSET}/geogram/basic/*.cpp"
	"${GEOGRAM_SUBSET}/geogram/numerics/*.cpp"
	"${GEOGRAM_SUBSET}/geogram/mesh/*.cpp"
	"${GEOGRAM_SUBSET}/geogram/delaunay/*.cpp"
	"${GEOGRAM_SUBSET}/geogram/points/*.cpp"
	"${GEOGRAM_SUBSET}/geogram/api/*.cpp"
	"${GEOGRAM_SUBSET}/geogram/bibliography/*.cpp"
	"${GEOGRAM_SUBSET}/geogram/third_party/predicate_generator/*.cpp"
	"${GEOGRAM_SUBSET}/geogram/third_party/numerics/*.cpp"
	"${GEOGRAM_SUBSET}/geogram/third_party/OpenNL/*.c"
)

# PMP core + the remeshing allow-list (no curvature/laplace/numerics/
# smoothing: PMP_NO_EIGEN).
file(GLOB PMP_SUBSET_SOURCES
	"${PMP_SUBSET}/pmp/*.cpp"
	"${PMP_SUBSET}/pmp/algorithms/*.cpp"
)

# MWT (multipolygon_triangulator), DelaunayFaces.cpp included.
file(GLOB MWT_SUBSET_SOURCES "${MWT_SUBSET}/*.cpp")
