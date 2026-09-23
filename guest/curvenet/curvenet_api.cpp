// curvenet_api -- the Cassie side of the curvenet stage (see curvenet_api.h).
// Compiled with the godot-lite prelude, like cassie_core; never includes the
// sandbox's api.hpp.
#include "curvenet_api.h"

#include "cassie_sketcher.h"
#include "curves/cassie_curve_fit.h"
#include "sketch/cassie_curvenet.h"
#include "sketch/cassie_curvenet_extractor.h"
#include "sketch/cassie_final_stroke.h"
#include "sketch/cassie_sketch_graph.h"
#include "sketch/cassie_surface_manager.h"
#include "sketch/cassie_surface_patch.h"

#include "scene/resources/mesh.h"

#include "../common/mesh_wire.h"

#include <pmp/algorithms/remeshing.h>
#include <pmp/surface_mesh.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>

namespace cn {
namespace {

struct Params {
	double snap_radius = 0.03;
	double surface_offset = 0.002;
	double target_edge_length = 0.02;
	double split_closed = 1;
	double merge_eps = 0.02;
	double mirror = 0;
};

struct SketcherDeleter {
	void operator()(CassieSketcher *p) const { memdelete(p); }
};

Params g_p;
std::unique_ptr<CassieSketcher, SketcherDeleter> g_sk;
Ref<CassieSurfacePatch> g_body;
Ref<CassieCurvenet> g_cn;
BuiltMesh g_mesh;
Counts g_counts;

std::string fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char *f, ...) {
	char b[512];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

std::string fail(const std::string &why) {
	return "FAIL: " + why;
}

// The sketcher follows the params; called before every use.
CassieSketcher *sketcher() {
	if (!g_sk) {
		g_sk.reset(memnew(CassieSketcher));
	}
	CassieSketcher *sk = g_sk.get();
	sk->set_split_closed_strokes(g_p.split_closed != 0);
	sk->get_sketch_graph()->set_merge_epsilon(real_t(g_p.merge_eps));
	sk->get_surface_manager()->set_target_edge_length(real_t(g_p.target_edge_length));
	Ref<CassieSketchContext> ctx = sk->get_sketch_context();
	ctx->set_mirror_enabled(g_p.mirror != 0);
	ctx->set_mirror_plane(Plane(Vector3(1, 0, 0), 0));
	ctx->set_project_on_patch_callback(g_body.is_valid() ? g_body->get_callback() : Callable());
	return sk;
}

Ref<CassieSurfacePatch> patch_from_arrays(const std::vector<float> &v, const std::vector<int32_t> &f) {
	PackedVector3Array verts;
	verts.resize(int(v.size() / 3));
	for (size_t i = 0; i < v.size() / 3; ++i) {
		verts.write[int(i)] = Vector3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
	}
	PackedInt32Array idx;
	idx.resize(int(f.size()));
	for (size_t i = 0; i < f.size(); ++i) {
		idx.write[int(i)] = f[i];
	}
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = verts;
	arrays[Mesh::ARRAY_INDEX] = idx;
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	Ref<CassieSurfacePatch> patch;
	patch.instantiate();
	patch->set_mesh(mesh);
	return patch;
}

Vector3 snap(const Vector3 &p) {
	if (g_body.is_null() || g_p.snap_radius <= 0) {
		return p;
	}
	const Dictionary d = g_body->project(p);
	if (!bool(d.get("on_surface", false))) {
		return p;
	}
	if (double(d.get("distance", 1e30)) > g_p.snap_radius) {
		return p;
	}
	const Vector3 q = d.get("projected", p);
	const Vector3 n = d.get("normal", Vector3());
	return q + n * real_t(g_p.surface_offset);
}

std::vector<Ref<CassieSurfacePatch>> active_patches() {
	std::vector<Ref<CassieSurfacePatch>> out;
	if (!g_sk) {
		return out;
	}
	const TypedArray<CassieSurfacePatch> ps = g_sk->get_surface_manager()->get_patches();
	for (int i = 0; i < ps.size(); ++i) {
		Ref<CassieSurfacePatch> p = ps[i];
		if (p.is_valid()) {
			out.push_back(p);
		}
	}
	return out;
}

// +1 keeps a triangle soup's winding, -1 flips it: the area-weighted normal
// must point the way the body's normal does at the point nearest the soup's
// centroid. No body: +1.
int outward_sign(const std::vector<float> &v, const std::vector<int32_t> &f) {
	if (g_body.is_null() || f.empty()) {
		return 1;
	}
	Vector3 c, n;
	for (size_t t = 0; t + 2 < f.size(); t += 3) {
		const Vector3 a(v[3 * f[t]], v[3 * f[t] + 1], v[3 * f[t] + 2]);
		const Vector3 b(v[3 * f[t + 1]], v[3 * f[t + 1] + 1], v[3 * f[t + 1] + 2]);
		const Vector3 d(v[3 * f[t + 2]], v[3 * f[t + 2] + 1], v[3 * f[t + 2] + 2]);
		n += (b - a).cross(d - a);
		c += (a + b + d) / real_t(3);
	}
	c /= real_t(f.size() / 3);
	const Dictionary pr = g_body->project(c);
	if (!bool(pr.get("on_surface", false))) {
		return 1;
	}
	const Vector3 bn = pr.get("normal", Vector3());
	return n.dot(bn) < 0 ? -1 : 1;
}

void patch_arrays(const Ref<CassieSurfacePatch> &p, std::vector<float> &v, std::vector<int32_t> &f) {
	v.clear();
	f.clear();
	for (int i = 0; i < p->get_vertex_count(); ++i) {
		const Vector3 q = p->get_vertex_position(i);
		v.push_back(float(q.x));
		v.push_back(float(q.y));
		v.push_back(float(q.z));
	}
	for (int t = 0; t < p->get_triangle_count(); ++t) {
		const Vector3i tri = p->get_triangle_indices(t);
		if (tri.x < 0) {
			continue;
		}
		f.push_back(tri.x);
		f.push_back(tri.y);
		f.push_back(tri.z);
	}
	if (outward_sign(v, f) < 0) {
		for (size_t t = 0; t + 2 < f.size(); t += 3) {
			std::swap(f[t + 1], f[t + 2]);
		}
	}
}

template <typename F>
std::string guarded(F &&fn) {
	try {
		return fn();
	} catch (const std::exception &e) {
		return fail(std::string("exception: ") + e.what());
	} catch (...) {
		return fail("unknown exception");
	}
}

} // namespace

// --- state ------------------------------------------------------------------------

std::string reset() {
	return guarded([] {
		g_sk.reset();
		g_cn.unref();
		g_mesh = BuiltMesh();
		g_counts = Counts();
		sketcher();
		return std::string("ok");
	});
}

std::string set_param(const std::string &name, double value) {
	return guarded([&] {
		double *slot = nullptr;
		if (name == "snap_radius") {
			slot = &g_p.snap_radius;
		} else if (name == "surface_offset") {
			slot = &g_p.surface_offset;
		} else if (name == "target_edge_length") {
			slot = &g_p.target_edge_length;
		} else if (name == "split_closed") {
			slot = &g_p.split_closed;
		} else if (name == "merge_eps") {
			slot = &g_p.merge_eps;
		} else if (name == "mirror") {
			slot = &g_p.mirror;
		}
		if (slot == nullptr) {
			return fail("unknown param '" + name + "' (snap_radius, surface_offset, target_edge_length, split_closed, merge_eps, mirror)");
		}
		if (!std::isfinite(value)) {
			return fail(name + " must be finite");
		}
		*slot = value;
		sketcher();
		return fmt("ok %s=%.9g", name.c_str(), value);
	});
}

double get_param(const std::string &name) {
	if (name == "snap_radius") {
		return g_p.snap_radius;
	} else if (name == "surface_offset") {
		return g_p.surface_offset;
	} else if (name == "target_edge_length") {
		return g_p.target_edge_length;
	} else if (name == "split_closed") {
		return g_p.split_closed;
	} else if (name == "merge_eps") {
		return g_p.merge_eps;
	} else if (name == "mirror") {
		return g_p.mirror;
	}
	return NAN;
}

std::string set_body(const std::vector<float> &vertices, const std::vector<int32_t> &triangles) {
	return guarded([&] {
		if (vertices.empty() && triangles.empty()) {
			g_body.unref();
			sketcher();
			return std::string("ok body=none");
		}
		std::string err;
		if (!mesh_wire::validate(vertices, triangles, err)) {
			return fail(err);
		}
		g_body = patch_from_arrays(vertices, triangles);
		sketcher();
		return fmt("ok body vertices=%d triangles=%d", g_body->get_vertex_count(), g_body->get_triangle_count());
	});
}

// --- pen -------------------------------------------------------------------------

int pen_begin(float x, float y, float z, float pressure) {
	try {
		return sketcher()->begin_stroke(snap(Vector3(x, y, z)), pressure);
	} catch (...) {
		return -1;
	}
}

std::string pen_point(int id, float x, float y, float z, float pressure) {
	return guarded([&] {
		sketcher()->add_sample(id, snap(Vector3(x, y, z)), pressure);
		return std::string("ok");
	});
}

std::string pen_end(int id) {
	return guarded([&] {
		CassieSketcher *sk = sketcher();
		const Dictionary r = sk->commit_stroke(id);
		const Ref<CassieFinalStroke> fs = r.get("final_stroke", Variant());
		const TypedArray<CassieSurfacePatch> np = r.get("new_patches", Array());
		const Ref<CassieSketchGraph> g = sk->get_sketch_graph();
		g_counts.edges = g->get_edge_count();
		g_counts.nodes = g->get_node_count();
		g_counts.cycles = int(g->find_cycles().size());
		return fmt("ok=%d valid=%d closed=%d new_patches=%d patches=%d edges=%d nodes=%d cycles=%d",
				int(bool(r.get("ok", false))), int(bool(r.get("is_valid", false))),
				int(fs.is_valid() && fs->is_closed_loop()), int(np.size()),
				sk->get_surface_manager()->get_patch_count(), g_counts.edges, g_counts.nodes, g_counts.cycles);
	});
}

int patch_count() {
	return g_sk ? g_sk->get_surface_manager()->get_patch_count() : 0;
}

std::vector<float> patch_vertices(int i) {
	std::vector<float> v;
	std::vector<int32_t> f;
	const std::vector<Ref<CassieSurfacePatch>> ps = active_patches();
	if (i >= 0 && size_t(i) < ps.size()) {
		patch_arrays(ps[size_t(i)], v, f);
	}
	return v;
}

std::vector<int32_t> patch_indices(int i) {
	std::vector<float> v;
	std::vector<int32_t> f;
	const std::vector<Ref<CassieSurfacePatch>> ps = active_patches();
	if (i >= 0 && size_t(i) < ps.size()) {
		patch_arrays(ps[size_t(i)], v, f);
	}
	return f;
}

// --- curvenet --------------------------------------------------------------------

namespace {

std::string curvenet_summary(const char *what) {
	int inter = 0, setup = 0;
	const TypedArray<CassieCurvenetKnot> ks = g_cn->get_knots();
	for (int i = 0; i < ks.size(); ++i) {
		const Ref<CassieCurvenetKnot> k = ks[i];
		inter += k->get_is_intersection() ? 1 : 0;
		setup += k->get_needs_setup() ? 1 : 0;
	}
	g_counts.curves = g_cn->get_curve_count();
	g_counts.knots = g_cn->get_knot_count();
	return fmt("ok %s curves=%d knots=%d intersections=%d needs_setup=%d", what, g_counts.curves, g_counts.knots, inter, setup);
}

} // namespace

std::string curvenet_build() {
	return guarded([] {
		CassieSketcher *sk = sketcher();
		const Ref<CassieSketchGraph> g = sk->get_sketch_graph();
		const Ref<CassieBeautifierParams> bp = sk->get_beautifier_params();

		std::vector<Ref<CassieSketchGraphEdge>> edges;
		const TypedArray<CassieSketchGraphEdge> all_e = g->get_all_edges();
		for (int i = 0; i < all_e.size(); ++i) {
			edges.push_back(all_e[i]);
		}
		std::sort(edges.begin(), edges.end(), [](const Ref<CassieSketchGraphEdge> &a, const Ref<CassieSketchGraphEdge> &b) {
			return a->get_id() < b->get_id();
		});
		std::vector<Ref<CassieSketchGraphNode>> nodes;
		const TypedArray<CassieSketchGraphNode> all_n = g->get_all_nodes();
		for (int i = 0; i < all_n.size(); ++i) {
			nodes.push_back(all_n[i]);
		}
		std::sort(nodes.begin(), nodes.end(), [](const Ref<CassieSketchGraphNode> &a, const Ref<CassieSketchGraphNode> &b) {
			return a->get_id() < b->get_id();
		});

		TypedArray<CassieFinalStroke> curves;
		for (size_t c = 0; c < edges.size(); ++c) {
			const PackedVector3Array pts = edges[c]->get_points();
			if (pts.size() < 2) {
				return fail(fmt("graph edge %d has %d points", edges[c]->get_id(), int(pts.size())));
			}
			Ref<Curve3D> curve;
			if (pts.size() > 2) {
				curve = cassie_fit_curve(pts, bp->get_bezier_fitting_error(), bp->get_rdp_error());
			}
			if (curve.is_null() || curve->get_point_count() < 2) {
				curve = cassie_fit_line(pts[0], pts[pts.size() - 1]);
			}
			Ref<CassieFinalStroke> s;
			s.instantiate();
			s->set_id(int(c));
			s->set_curve(curve, edges[c]->get_node_a_id() == edges[c]->get_node_b_id());
			s->set_input_samples(pts);
			curves.push_back(s);
		}
		Array node_dicts;
		for (const Ref<CassieSketchGraphNode> &n : nodes) {
			PackedInt32Array incident;
			for (size_t c = 0; c < edges.size(); ++c) {
				if (edges[c]->get_node_a_id() == n->get_id()) {
					incident.push_back(int(c));
				}
				if (edges[c]->get_node_b_id() == n->get_id()) {
					incident.push_back(int(c));
				}
			}
			Dictionary d;
			d["id"] = n->get_id();
			d["position"] = n->get_position();
			d["incident_curve_ids"] = incident;
			node_dicts.push_back(d);
		}
		Dictionary data;
		data["curves"] = curves;
		data["nodes"] = node_dicts;

		g_cn.instantiate();
		g_cn->build_from_graph(data);
		Ref<CassieSurfacePatch> bound = g_body;
		if (bound.is_null()) {
			const std::vector<Ref<CassieSurfacePatch>> ps = active_patches();
			if (!ps.empty()) {
				bound = ps[0];
			}
		}
		g_cn->set_bound_patch(bound);
		g_cn->update_rest_pose(bound);
		g_cn->compute_orientations();
		return curvenet_summary("sketch");
	});
}

std::string curvenet_extract(const std::vector<float> &vertices, const std::vector<int32_t> &triangles,
		int target_curve_count, double rdp_error, double fit_error, double curvature_weight) {
	return guarded([&] {
		std::string err;
		if (!mesh_wire::validate(vertices, triangles, err)) {
			return fail(err);
		}
		const Ref<CassieSurfacePatch> patch = patch_from_arrays(vertices, triangles);
		g_cn = CassieCurvenetExtractor::extract(patch, target_curve_count, float(rdp_error), float(fit_error),
				float(curvature_weight));
		if (g_cn.is_null()) {
			return fail("extractor returned no curvenet");
		}
		return curvenet_summary("extract");
	});
}

std::vector<float> curvenet_curves() {
	std::vector<float> out;
	if (g_cn.is_null()) {
		out.push_back(0);
		return out;
	}
	const TypedArray<CassieFinalStroke> cs = g_cn->get_curves();
	out.push_back(float(cs.size()));
	for (int i = 0; i < cs.size(); ++i) {
		const Ref<CassieFinalStroke> s = cs[i];
		const Ref<Curve3D> c = s.is_valid() ? s->get_curve() : Ref<Curve3D>();
		const int np = c.is_valid() ? c->get_point_count() : 0;
		const Vector2i k = g_cn->get_curve_endpoint_knots(i);
		out.push_back(float(np));
		out.push_back(s.is_valid() && s->is_closed_loop() ? 1.0f : 0.0f);
		out.push_back(float(k.x));
		out.push_back(float(k.y));
		for (int j = 0; j < np; ++j) {
			const Vector3 v[3] = { c->get_point_position(j), c->get_point_in(j), c->get_point_out(j) };
			for (const Vector3 &q : v) {
				out.push_back(float(q.x));
				out.push_back(float(q.y));
				out.push_back(float(q.z));
			}
		}
	}
	return out;
}

std::vector<float> curvenet_knots() {
	std::vector<float> out;
	if (g_cn.is_null()) {
		out.push_back(0);
		return out;
	}
	const TypedArray<CassieCurvenetKnot> ks = g_cn->get_knots();
	out.push_back(float(ks.size()));
	for (int i = 0; i < ks.size(); ++i) {
		const Ref<CassieCurvenetKnot> k = ks[i];
		const Transform3D t = k->get_rest_pose_transform();
		out.push_back(float(t.origin.x));
		out.push_back(float(t.origin.y));
		out.push_back(float(t.origin.z));
		out.push_back(float(k->get_projection_pose_tangents().size()));
		out.push_back(k->get_is_intersection() ? 1.0f : 0.0f);
		out.push_back(k->get_needs_setup() ? 1.0f : 0.0f);
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c) {
				out.push_back(float(t.basis.rows[r][c]));
			}
		}
	}
	return out;
}

// --- mesh ------------------------------------------------------------------------

namespace {

// Vertex welding on a uniform grid of cell size eps: each vertex, in order,
// joins the first earlier representative within eps (searching the 27
// neighbouring cells), else becomes one. Deterministic in the input order.
std::vector<int32_t> weld(const std::vector<float> &v, double eps, std::vector<float> &out_v) {
	const size_t n = v.size() / 3;
	std::vector<int32_t> remap(n);
	out_v.clear();
	if (eps <= 0) {
		out_v = v;
		for (size_t i = 0; i < n; ++i) {
			remap[i] = int32_t(i);
		}
		return remap;
	}
	auto cell = [eps](float x) { return int64_t(std::floor(double(x) / eps)); };
	auto key = [](int64_t x, int64_t y, int64_t z) {
		return uint64_t(x * 73856093) ^ uint64_t(y * 19349663) ^ uint64_t(z * 83492791);
	};
	std::unordered_multimap<uint64_t, int32_t> grid;
	const double eps2 = eps * eps;
	for (size_t i = 0; i < n; ++i) {
		const float *p = &v[3 * i];
		const int64_t cx = cell(p[0]), cy = cell(p[1]), cz = cell(p[2]);
		int32_t found = -1;
		for (int dx = -1; dx <= 1 && found < 0; ++dx) {
			for (int dy = -1; dy <= 1 && found < 0; ++dy) {
				for (int dz = -1; dz <= 1 && found < 0; ++dz) {
					auto range = grid.equal_range(key(cx + dx, cy + dy, cz + dz));
					for (auto it = range.first; it != range.second; ++it) {
						const float *q = &out_v[3 * size_t(it->second)];
						const double ex = double(p[0]) - q[0], ey = double(p[1]) - q[1], ez = double(p[2]) - q[2];
						if (ex * ex + ey * ey + ez * ez <= eps2 && (found < 0 || it->second < found)) {
							found = it->second;
						}
					}
				}
			}
		}
		if (found < 0) {
			found = int32_t(out_v.size() / 3);
			out_v.insert(out_v.end(), p, p + 3);
			grid.emplace(key(cx, cy, cz), found);
		}
		remap[i] = found;
	}
	return remap;
}

void topology(BuiltMesh &m) {
	const size_t nv = m.vertices.size() / 3;
	const std::vector<int32_t> &f = m.triangles;
	// Directed edges -> count; an undirected edge is boundary when it has
	// exactly one directed use.
	std::map<std::pair<int32_t, int32_t>, int> directed;
	std::map<std::pair<int32_t, int32_t>, int> undirected;
	for (size_t t = 0; t + 2 < f.size(); t += 3) {
		for (int e = 0; e < 3; ++e) {
			const int32_t a = f[t + e], b = f[t + (e + 1) % 3];
			directed[{ a, b }] += 1;
			undirected[{ std::min(a, b), std::max(a, b) }] += 1;
		}
	}
	m.edges = int(undirected.size());
	std::multimap<int32_t, int32_t> next;
	for (const auto &kv : directed) {
		const int32_t a = kv.first.first, b = kv.first.second;
		if (undirected[{ std::min(a, b), std::max(a, b) }] == 1) {
			next.emplace(a, b);
		}
	}
	m.loops.clear();
	while (!next.empty()) {
		auto it = next.begin();
		const int32_t start = it->first;
		std::vector<int32_t> loop{ start };
		int32_t at = it->second;
		next.erase(it);
		while (at != start) {
			loop.push_back(at);
			auto nx = next.find(at);
			if (nx == next.end()) {
				break; // an open chain (non-manifold boundary): keep what was walked
			}
			at = nx->second;
			next.erase(nx);
		}
		m.loops.push_back(loop);
	}
	// Components: union-find over the vertices the triangles use.
	std::vector<int32_t> parent(nv);
	for (size_t i = 0; i < nv; ++i) {
		parent[i] = int32_t(i);
	}
	std::function<int32_t(int32_t)> find = [&](int32_t x) {
		while (parent[size_t(x)] != x) {
			parent[size_t(x)] = parent[size_t(parent[size_t(x)])];
			x = parent[size_t(x)];
		}
		return x;
	};
	std::vector<bool> used(nv, false);
	for (size_t t = 0; t + 2 < f.size(); t += 3) {
		for (int e = 0; e < 3; ++e) {
			used[size_t(f[t + e])] = true;
			const int32_t a = find(f[t + e]), b = find(f[t + (e + 1) % 3]);
			if (a != b) {
				parent[size_t(std::max(a, b))] = std::min(a, b);
			}
		}
	}
	m.components = 0;
	for (size_t i = 0; i < nv; ++i) {
		if (used[i] && find(int32_t(i)) == int32_t(i)) {
			++m.components;
		}
	}
}

bool remesh(BuiltMesh &m, double target, std::string &err) {
	pmp::SurfaceMesh mesh;
	const size_t nv = m.vertices.size() / 3;
	std::vector<pmp::Vertex> vh(nv);
	for (size_t i = 0; i < nv; ++i) {
		vh[i] = mesh.add_vertex(pmp::Point(m.vertices[3 * i], m.vertices[3 * i + 1], m.vertices[3 * i + 2]));
	}
	for (size_t t = 0; t + 2 < m.triangles.size(); t += 3) {
		mesh.add_triangle(vh[size_t(m.triangles[t])], vh[size_t(m.triangles[t + 1])], vh[size_t(m.triangles[t + 2])]);
	}
	// The boundary is the seam to the next stage: hold it as a feature so the
	// remesh splits it but never moves it off its polyline.
	auto efeature = mesh.edge_property<bool>("e:feature", false);
	auto vfeature = mesh.vertex_property<bool>("v:feature", false);
	for (pmp::Edge e : mesh.edges()) {
		if (mesh.is_boundary(e)) {
			efeature[e] = true;
			vfeature[mesh.vertex(e, 0)] = true;
			vfeature[mesh.vertex(e, 1)] = true;
		}
	}
	try {
		pmp::uniform_remeshing(mesh, pmp::Scalar(target), 10, true);
	} catch (const std::exception &e) {
		err = std::string("uniform_remeshing: ") + e.what();
		return false;
	}
	mesh.garbage_collection();
	m.vertices.clear();
	m.triangles.clear();
	for (pmp::Vertex v : mesh.vertices()) {
		const pmp::Point p = mesh.position(v);
		m.vertices.push_back(float(p[0]));
		m.vertices.push_back(float(p[1]));
		m.vertices.push_back(float(p[2]));
	}
	for (pmp::Face f : mesh.faces()) {
		for (pmp::Vertex v : mesh.vertices(f)) {
			m.triangles.push_back(int32_t(v.idx()));
		}
	}
	m.patch_ids.assign(m.triangles.size() / 3, -1);
	return true;
}

} // namespace

BuiltMesh build_mesh(const std::vector<std::vector<float>> &part_vertices,
		const std::vector<std::vector<int32_t>> &part_triangles,
		double target_edge_length, double weld_eps) {
	BuiltMesh m;
	std::vector<float> soup;
	std::vector<int32_t> tris, ids;
	for (size_t p = 0; p < part_vertices.size() && p < part_triangles.size(); ++p) {
		const int32_t base = int32_t(soup.size() / 3);
		soup.insert(soup.end(), part_vertices[p].begin(), part_vertices[p].end());
		for (int32_t i : part_triangles[p]) {
			tris.push_back(base + i);
		}
		ids.insert(ids.end(), part_triangles[p].size() / 3, int32_t(p));
	}
	std::vector<float> welded;
	const std::vector<int32_t> remap = weld(soup, weld_eps, welded);
	// Remap, drop triangles the weld collapsed, then drop unused vertices.
	std::vector<int32_t> f2, id2;
	for (size_t t = 0; t + 2 < tris.size(); t += 3) {
		const int32_t a = remap[size_t(tris[t])], b = remap[size_t(tris[t + 1])], c = remap[size_t(tris[t + 2])];
		if (a == b || b == c || a == c) {
			continue;
		}
		f2.insert(f2.end(), { a, b, c });
		id2.push_back(ids[t / 3]);
	}
	std::vector<int32_t> compact(welded.size() / 3, -1);
	for (int32_t &i : f2) {
		if (compact[size_t(i)] < 0) {
			compact[size_t(i)] = int32_t(m.vertices.size() / 3);
			m.vertices.insert(m.vertices.end(), &welded[3 * size_t(i)], &welded[3 * size_t(i)] + 3);
		}
		i = compact[size_t(i)];
	}
	m.triangles = std::move(f2);
	m.patch_ids = std::move(id2);
	if (target_edge_length > 0 && !m.triangles.empty()) {
		if (!remesh(m, target_edge_length, m.error)) {
			return m;
		}
	}
	topology(m);
	return m;
}

std::string mesh_build(double target_edge_length, double weld_eps) {
	return guarded([&] {
		std::vector<std::vector<float>> pv;
		std::vector<std::vector<int32_t>> pf;
		for (const Ref<CassieSurfacePatch> &p : active_patches()) {
			pv.emplace_back();
			pf.emplace_back();
			patch_arrays(p, pv.back(), pf.back());
		}
		g_mesh = build_mesh(pv, pf, target_edge_length, weld_eps);
		if (!g_mesh.error.empty()) {
			return fail(g_mesh.error);
		}
		const int nv = int(g_mesh.vertices.size() / 3), nf = int(g_mesh.triangles.size() / 3);
		return fmt("ok patches=%d vertices=%d triangles=%d loops=%d components=%d euler=%d", int(pv.size()), nv, nf,
				int(g_mesh.loops.size()), g_mesh.components, nv - g_mesh.edges + nf);
	});
}

std::vector<float> mesh_vertices() {
	return g_mesh.vertices;
}

std::vector<int32_t> mesh_indices() {
	return g_mesh.triangles;
}

std::vector<int32_t> mesh_boundary_loops() {
	return mesh_wire::encode_loops(g_mesh.loops);
}

std::vector<int32_t> mesh_patch_ids() {
	return g_mesh.patch_ids;
}

Counts counts() {
	return g_counts;
}

std::vector<float> stroke_samples() {
	std::vector<float> out;
	if (!g_sk) {
		return out;
	}
	const TypedArray<CassieFinalStroke> ss = g_sk->get_committed_strokes();
	for (int i = 0; i < ss.size(); ++i) {
		const Ref<CassieFinalStroke> s = ss[i];
		const PackedVector3Array pts = s->get_input_samples();
		for (int j = 0; j < pts.size(); ++j) {
			out.push_back(float(pts[j].x));
			out.push_back(float(pts[j].y));
			out.push_back(float(pts[j].z));
		}
	}
	return out;
}

} // namespace cn
