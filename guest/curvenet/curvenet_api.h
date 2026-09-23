// curvenet_api -- the curvenet stage's whole surface, in std types.
//
// The godot-lite split (plan Cut 4): Cassie's TUs are compiled against
// godot-lite (namespace gdl, force-included prelude) and never see the
// sandbox's api.hpp, whose Variant/String/PackedArray names would collide.
// guest/curvenet/main.cpp includes api.hpp and this header and nothing of
// Cassie's; curvenet_api.cpp and checks.cpp include Cassie and this header
// and nothing of the sandbox's. The same two TUs link into the native
// cassie_checks.exe (tests/native/curvenet), the flat control.
//
// Arrays follow guest/common/mesh_wire.h. Every entry point that returns a
// string catches exceptions and answers "FAIL: ..." instead of unwinding
// into the host.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cn {

// --- state --------------------------------------------------------------------
// Drops every stroke, patch, curvenet and built mesh; keeps params and body.
std::string reset();
// snap_radius (0.03), surface_offset (0.002), target_edge_length (0.02),
// split_closed (1), merge_eps (0.02), mirror (0; 1 = mirror across x=0).
std::string set_param(const std::string &name, double value);
double get_param(const std::string &name);
// The body the pen snaps to; also the sketch context's project_on_patch
// callback. Empty arrays clear it.
std::string set_body(const std::vector<float> &vertices, const std::vector<int32_t> &triangles);

// --- pen ----------------------------------------------------------------------
// A sample within snap_radius of the body lands on it, offset by
// surface_offset along the body normal. pen_begin answers the stroke id (> 0)
// or -1.
int pen_begin(float x, float y, float z, float pressure);
std::string pen_point(int id, float x, float y, float z, float pressure);
// "ok=1 valid=1 closed=1 new_patches=1 patches=1 edges=2 nodes=2 cycles=1"
std::string pen_end(int id);

int patch_count();
std::vector<float> patch_vertices(int i);
std::vector<int32_t> patch_indices(int i);

// --- curvenet -----------------------------------------------------------------
// Sketch graph -> curvenet: one curve per graph edge (cassie_fit_curve over
// the edge polyline), one knot per graph node; bound to the body (else the
// first patch), rest pose updated and orientations computed.
std::string curvenet_build();
// Mesh -> curvenet (CassieCurvenetExtractor).
std::string curvenet_extract(const std::vector<float> &vertices, const std::vector<int32_t> &triangles,
		int target_curve_count, double rdp_error, double fit_error, double curvature_weight);
std::vector<float> curvenet_curves();
std::vector<float> curvenet_knots();

// --- mesh ---------------------------------------------------------------------
// Merge the active patches, weld vertices closer than weld_eps (<= 0: no
// weld), orient every patch away from the body, and when
// target_edge_length > 0 run PMP's uniform remesh with the boundary held as
// a feature. "ok vertices=.. triangles=.. loops=.. components=.. euler=.."
std::string mesh_build(double target_edge_length, double weld_eps);
std::vector<float> mesh_vertices();
std::vector<int32_t> mesh_indices();
std::vector<int32_t> mesh_boundary_loops();
std::vector<int32_t> mesh_patch_ids(); // one per triangle; -1 after a remesh

// --- checks (checks.cpp) ------------------------------------------------------
// One line per check:
//   "PASS <name> ints=a,b,c fsig=<16 hex>/<count> :: <detail>"
// ints are the check's integer outputs, fsig an FNV-1a-64 over the bit
// patterns of its float outputs; the gate compares both, guest vs native.
std::vector<std::string> check_names();
std::string check(const std::string &name);
// Every check, one line each, then "checks: <passed>/<total>".
std::string check_all();

// --- internals shared with checks.cpp -----------------------------------------
struct BuiltMesh {
	std::vector<float> vertices;
	std::vector<int32_t> triangles;
	std::vector<int32_t> patch_ids;
	std::vector<std::vector<int32_t>> loops;
	int components = 0;
	int edges = 0;
	std::string error;
};
// The mesh_build core over explicit parts (each part a vertex/triangle pair).
BuiltMesh build_mesh(const std::vector<std::vector<float>> &part_vertices,
		const std::vector<std::vector<int32_t>> &part_triangles,
		double target_edge_length, double weld_eps);
// Counts written by the last pen_end / curvenet_build, for checks.
struct Counts {
	int edges = 0, nodes = 0, cycles = 0, curves = 0, knots = 0;
};
Counts counts();
// The input samples (after snapping) of every committed stroke, flattened.
std::vector<float> stroke_samples();

} // namespace cn
