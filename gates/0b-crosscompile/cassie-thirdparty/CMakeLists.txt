# Gate 0B: cross-compile the EXACT Geogram / PMP / MWT subset that
# modules/cassie/SCsub compiles, as static libs, for riscv64.
#
# Deliberately not Geogram's own CMake tree: that would try graphics, Lua,
# TetGen (AGPL), Triangle (non-commercial) and parallel Delaunay (OpenMP),
# none of which Cassie uses, and their failures would say nothing about the
# path we need. The globs and exclude lists below are transcribed from SCsub
# lines 186-290 and are the spec of what a Cassie port has to carry.
cmake_minimum_required(VERSION 3.16)
project(cassie_thirdparty C CXX)
set(CMAKE_CXX_STANDARD 17)

set(E "C:/contract-manifest/4-entities/godot/thirdparty")
set(GR "${E}/geogram/src/lib/geogram")

function(glob_without out dir pattern)   # excludes are the remaining args
	file(GLOB all "${dir}/${pattern}")
	set(keep)
	foreach(f ${all})
		get_filename_component(b "${f}" NAME)
		if(NOT b IN_LIST ARGN)
			list(APPEND keep "${f}")
		endif()
	endforeach()
	set(${out} "${keep}" PARENT_SCOPE)
endfunction()

glob_without(geo_basic    "${GR}/basic"    "*.cpp" geofile.cpp android_utils.cpp boolean_expression.cpp)
file(GLOB  geo_numerics   "${GR}/numerics/*.cpp")
set(geo_mesh              "${GR}/mesh/mesh.cpp" "${GR}/mesh/mesh_reorder.cpp")
# delaunay_2d.cpp is un-excluded. SCsub left it out only because Cassie's
# wrapper leaned on Godot's core Delaunay2D; the port swaps that for Geogram's
# own (user: "godot delaunay is terrible"), so its 2D triangulator must
# cross-compile too. TetGen (AGPL), Triangle (non-commercial) and PDEL stay out.
glob_without(geo_delaunay "${GR}/delaunay" "*.cpp" delaunay_tetgen.cpp delaunay_triangle.cpp parallel_delaunay_3d.cpp)
glob_without(geo_points   "${GR}/points"   "*.cpp" co3ne.cpp)
file(GLOB  geo_api        "${GR}/api/*.cpp")
file(GLOB  geo_bib        "${GR}/bibliography/*.cpp")
file(GLOB  geo_predgen    "${GR}/third_party/predicate_generator/*.cpp")
file(GLOB  geo_tpnum      "${GR}/third_party/numerics/*.cpp")
file(GLOB  geo_opennl     "${GR}/third_party/OpenNL/*.c")

add_library(geogram_subset STATIC
	${geo_basic} ${geo_numerics} ${geo_mesh} ${geo_delaunay} ${geo_points}
	${geo_api} ${geo_bib} ${geo_predgen} ${geo_tpnum} ${geo_opennl}
	"${E}/multipolygon_triangulator/DelaunayFaces.cpp")
target_include_directories(geogram_subset PUBLIC
	"${E}/geogram/src/lib" "${GR}/third_party" "${E}/eigen" "${E}/multipolygon_triangulator"
	# geofile.cpp is excluded, but geofile.h is still included transitively and
	# wants zlib.h. SCsub resolves it the same way: headers only, nothing links.
	"${E}/zlib")
target_compile_definitions(geogram_subset PUBLIC _USE_MATH_DEFINES GEOGRAM_VERSION="1.9.9")
target_compile_options(geogram_subset PRIVATE -w)

set(PR "${E}/pmp/src/pmp")
file(GLOB pmp_core "${PR}/*.cpp")
set(pmp_algos)
foreach(a remeshing decimation differential_geometry normals features smoothing utilities
          distance_point_triangle triangulation curvature laplace numerics)
	list(APPEND pmp_algos "${PR}/algorithms/${a}.cpp")
endforeach()
add_library(pmp_subset STATIC ${pmp_core} ${pmp_algos})
target_include_directories(pmp_subset PUBLIC "${E}/pmp/src" "${E}/eigen")
# PMP checks #ifdef PMP_SCALAR_TYPE_64 (types.h:17); SCsub sets exactly this.
target_compile_definitions(pmp_subset PUBLIC _USE_MATH_DEFINES PMP_SCALAR_TYPE_64=1)
target_compile_options(pmp_subset PRIVATE -w)

glob_without(mwt "${E}/multipolygon_triangulator" "*.cpp" DelaunayFaces.cpp)
add_library(mwt_subset STATIC ${mwt} "${CMAKE_CURRENT_LIST_DIR}/shim/delaunay_geogram.cpp")
target_include_directories(mwt_subset PUBLIC "${E}/multipolygon_triangulator" "${E}/eigen"
	"${CMAKE_CURRENT_LIST_DIR}/shim" "${E}/geogram/src/lib")
target_compile_definitions(mwt_subset PUBLIC _USE_MATH_DEFINES GEOGRAM_VERSION="1.9.9")
target_compile_options(mwt_subset PRIVATE -w)
