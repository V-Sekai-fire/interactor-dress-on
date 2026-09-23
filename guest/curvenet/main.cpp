// curvenet.elf -- the curvenet stage: Cassie's pen -> curvenet -> mesh and
// mesh -> curvenet paths (plan Cut 4).
//
// The godot-lite split: this TU sees the sandbox's api.hpp and
// curvenet_api.h (std types) and nothing of Cassie's; curvenet_api.cpp and
// checks.cpp see Cassie (on godot-lite) and nothing of the sandbox's. So all
// marshalling is here: PackedArray <-> std::vector, String <-> std::string.
// Arrays follow guest/common/mesh_wire.h. Every ADD_API_FUNCTION has a
// no-argument wrapper in project/main.gd (AGENTS.md rule 8).
//
// No GPU here: the stage is CPU-only Cassie, so no RenderingDevice is opened
// and rule 4 does not arise. Single calls are sized to stay inside the
// sandbox's execution_timeout (Gate 4 records the heaviest).

#include <api.hpp>

#include <string>
#include <vector>

#include "curvenet_api.h"

static Variant text(const std::string &s) {
	return Variant(String(s));
}

template <typename T>
static Variant packed(const std::vector<T> &v) {
	return Variant(PackedArray<T>(v));
}

// --- state --------------------------------------------------------------------

static Variant cn_reset() {
	return text(cn::reset());
}

static Variant cn_set_param(String name, double value) {
	return text(cn::set_param(name.utf8(), value));
}

static Variant cn_get_param(String name) {
	return Variant(cn::get_param(name.utf8()));
}

static Variant cn_set_body(PackedArray<float> vertices, PackedArray<int32_t> triangles) {
	return text(cn::set_body(vertices.fetch(), triangles.fetch()));
}

// --- pen ----------------------------------------------------------------------

static Variant pen_begin(double x, double y, double z, double pressure) {
	return Variant(int64_t(cn::pen_begin(float(x), float(y), float(z), float(pressure))));
}

static Variant pen_point(int id, double x, double y, double z, double pressure) {
	return text(cn::pen_point(id, float(x), float(y), float(z), float(pressure)));
}

static Variant pen_end(int id) {
	return text(cn::pen_end(id));
}

// A whole stroke in one call: xyzp holds 4 floats per sample (x, y, z,
// pressure); answers pen_end's line.
static Variant pen_stroke(PackedArray<float> xyzp) {
	const std::vector<float> s = xyzp.fetch();
	if (s.size() < 8 || s.size() % 4 != 0) {
		return text("FAIL: pen_stroke wants 4 floats per sample and at least 2 samples");
	}
	const int id = cn::pen_begin(s[0], s[1], s[2], s[3]);
	if (id < 0) {
		return text("FAIL: pen_begin");
	}
	for (size_t i = 4; i < s.size(); i += 4) {
		cn::pen_point(id, s[i], s[i + 1], s[i + 2], s[i + 3]);
	}
	return text(cn::pen_end(id));
}

static Variant patch_count() {
	return Variant(int64_t(cn::patch_count()));
}

static Variant patch_vertices(int i) {
	return packed(cn::patch_vertices(i));
}

static Variant patch_indices(int i) {
	return packed(cn::patch_indices(i));
}

// --- curvenet -----------------------------------------------------------------

static Variant curvenet_build() {
	return text(cn::curvenet_build());
}

static Variant curvenet_extract(PackedArray<float> vertices, PackedArray<int32_t> triangles, int target,
		double rdp_error, double fit_error, double curvature_weight) {
	return text(cn::curvenet_extract(vertices.fetch(), triangles.fetch(), target, rdp_error, fit_error,
			curvature_weight));
}

static Variant curvenet_curves() {
	return packed(cn::curvenet_curves());
}

static Variant curvenet_knots() {
	return packed(cn::curvenet_knots());
}

// --- mesh ---------------------------------------------------------------------

static Variant mesh_build(double target_edge_length, double weld_eps) {
	return text(cn::mesh_build(target_edge_length, weld_eps));
}

static Variant mesh_vertices() {
	return packed(cn::mesh_vertices());
}

static Variant mesh_indices() {
	return packed(cn::mesh_indices());
}

static Variant mesh_boundary_loops() {
	return packed(cn::mesh_boundary_loops());
}

static Variant mesh_patch_ids() {
	return packed(cn::mesh_patch_ids());
}

// --- checks -------------------------------------------------------------------

static Variant check(String name) {
	return text(cn::check(name.utf8()));
}

static Variant check_all() {
	return text(cn::check_all());
}

static Variant check_names() {
	std::string out;
	for (const std::string &n : cn::check_names()) {
		out += (out.empty() ? "" : " ") + n;
	}
	return text(out);
}

int main() {
	ADD_API_FUNCTION(cn_reset, "String", "", "Drop strokes, patches, curvenet and mesh; keep params and body");
	ADD_API_FUNCTION(cn_set_param, "String", "String name, float value",
			"snap_radius, surface_offset, target_edge_length, split_closed, merge_eps, mirror");
	ADD_API_FUNCTION(cn_get_param, "float", "String name", "A param's value (NaN if unknown)");
	ADD_API_FUNCTION(cn_set_body, "String", "PackedFloat32Array vertices, PackedInt32Array triangles",
			"The body the pen snaps to (mesh_wire; empty clears)");
	ADD_API_FUNCTION(pen_begin, "int", "float x, float y, float z, float pressure", "Start a stroke; its id or -1");
	ADD_API_FUNCTION(pen_point, "String", "int id, float x, float y, float z, float pressure", "Add a sample");
	ADD_API_FUNCTION(pen_end, "String", "int id",
			"Commit: ok valid closed new_patches patches edges nodes cycles");
	ADD_API_FUNCTION(pen_stroke, "String", "PackedFloat32Array xyzp", "A whole stroke, 4 floats per sample");
	ADD_API_FUNCTION(patch_count, "int", "", "Active surface patches");
	ADD_API_FUNCTION(patch_vertices, "PackedFloat32Array", "int i", "Patch i's vertices (mesh_wire)");
	ADD_API_FUNCTION(patch_indices, "PackedInt32Array", "int i", "Patch i's triangles, CCW-outward");
	ADD_API_FUNCTION(curvenet_build, "String", "", "Sketch graph -> curvenet");
	ADD_API_FUNCTION(curvenet_extract, "String",
			"PackedFloat32Array vertices, PackedInt32Array triangles, int target, float rdp_error, float fit_error, float curvature_weight",
			"Mesh -> curvenet");
	ADD_API_FUNCTION(curvenet_curves, "PackedFloat32Array", "", "The curvenet's curves (mesh_wire)");
	ADD_API_FUNCTION(curvenet_knots, "PackedFloat32Array", "", "The curvenet's knots (mesh_wire)");
	ADD_API_FUNCTION(mesh_build, "String", "float target_edge_length, float weld_eps",
			"Merge + weld the patches, optionally PMP-remesh");
	ADD_API_FUNCTION(mesh_vertices, "PackedFloat32Array", "", "The built mesh's vertices");
	ADD_API_FUNCTION(mesh_indices, "PackedInt32Array", "", "The built mesh's triangles");
	ADD_API_FUNCTION(mesh_boundary_loops, "PackedInt32Array", "", "The built mesh's boundary loops");
	ADD_API_FUNCTION(mesh_patch_ids, "PackedInt32Array", "", "Source patch per triangle (-1 after a remesh)");
	ADD_API_FUNCTION(check, "String", "String name", "One Gate 4 check");
	ADD_API_FUNCTION(check_all, "String", "", "Every Gate 4 check");
	ADD_API_FUNCTION(check_names, "String", "", "The Gate 4 check names");
	halt();
}
