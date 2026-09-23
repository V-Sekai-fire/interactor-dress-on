// mesh_wire -- how meshes, curves and knots cross the host boundary.
//
// AGENTS.md rule 6: stages are separate ELFs, and meshes cross between them
// through the host as packed arrays. This header is the guest side of that
// format; project/util/mesh_wire.gd is the host side. std types only, so it
// is usable from any stage without Godot or godot-lite.
//
// Frame and units: body-local Godot frame (+Y up, right-handed), metres.
//
//   vertices  PackedFloat32Array, 3N: x0 y0 z0 x1 y1 z1 ...
//   triangles PackedInt32Array,   3F: counter-clockwise seen from outside
//             (the face normal (b-a)x(c-a) points outward).
//   boundary loops PackedInt32Array:
//             [n, len0, i0 .. i(len0-1), len1, ...]; each loop walks its
//             boundary edges in the direction of the triangle that owns them.
//   curves    PackedFloat32Array (integers stored exactly as floats):
//             [n, then per curve: np, closed, knot0, knot1,
//              np * (pos3, in3, out3)]; in/out are Curve3D control-point
//             handles relative to pos; knot0/knot1 are the knot indices at
//             the curve's start/end (-1 when not bound to a knot).
//   knots     PackedFloat32Array:
//             [n, then per knot: pos3, degree, is_intersection, needs_setup,
//              basis9]; basis9 is the rest-pose Basis, row-major
//             (rows[0].xyz, rows[1].xyz, rows[2].xyz).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mesh_wire {

constexpr int CURVE_HEADER = 4; // np, closed, knot0, knot1
constexpr int CURVE_POINT = 9;  // pos3, in3, out3
constexpr int KNOT_STRIDE = 15; // pos3, degree, is_intersection, needs_setup, basis9

struct Mesh {
	std::vector<float> vertices;    // 3N
	std::vector<int32_t> triangles; // 3F
	size_t vertex_count() const { return vertices.size() / 3; }
	size_t triangle_count() const { return triangles.size() / 3; }
};

// Validates the vertex/triangle pair: sizes are multiples of 3 and every
// index is in range. `err` says what is wrong.
inline bool validate(const std::vector<float> &v, const std::vector<int32_t> &f, std::string &err) {
	if (v.size() % 3 != 0) {
		err = "vertices: size " + std::to_string(v.size()) + " is not a multiple of 3";
		return false;
	}
	if (f.size() % 3 != 0) {
		err = "triangles: size " + std::to_string(f.size()) + " is not a multiple of 3";
		return false;
	}
	const int64_t n = int64_t(v.size() / 3);
	for (size_t i = 0; i < f.size(); ++i) {
		if (f[i] < 0 || f[i] >= n) {
			err = "triangles[" + std::to_string(i) + "] = " + std::to_string(f[i]) + " out of range [0, " + std::to_string(n) + ")";
			return false;
		}
	}
	return true;
}

inline std::vector<int32_t> encode_loops(const std::vector<std::vector<int32_t>> &loops) {
	std::vector<int32_t> out;
	out.push_back(int32_t(loops.size()));
	for (const std::vector<int32_t> &l : loops) {
		out.push_back(int32_t(l.size()));
		out.insert(out.end(), l.begin(), l.end());
	}
	return out;
}

inline bool decode_loops(const std::vector<int32_t> &in, std::vector<std::vector<int32_t>> &loops) {
	loops.clear();
	if (in.empty()) {
		return false;
	}
	size_t at = 1;
	for (int32_t k = 0; k < in[0]; ++k) {
		if (at >= in.size() || in[at] < 0 || at + 1 + size_t(in[at]) > in.size()) {
			return false;
		}
		loops.emplace_back(in.begin() + at + 1, in.begin() + at + 1 + in[at]);
		at += 1 + size_t(in[at]);
	}
	return at == in.size();
}

} // namespace mesh_wire
