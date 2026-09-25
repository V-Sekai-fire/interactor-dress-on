// Gate 4's checks. This TU runs unchanged in curvenet.elf (check/check_all)
// and in the native cassie_checks.exe (tests/native/curvenet), the flat
// control: a check that fails in both is Cassie or godot-lite, one that fails
// only in the guest is the sandbox.
//
// Every check has a negative control that must come out differently, so a
// PASS cannot mean "nothing ran". The first five port
// cassie-flow-project/checks/*.gd (contract-manifest 4-entities) one to one;
// the rest are this stage's own.
//
// Output, one line per check (see curvenet_api.h):
//   "PASS <name> ints=a,b,c fsig=<12 hex>/<count> :: <detail>"
// ints: the integer outputs; fsig: SHA-256 (first 12 hex digits) over the
// float32 bit patterns of the float outputs, little-endian, in order. Gate 4 compares both, guest vs native.
//
// Compiled with the godot-lite prelude, like cassie_core.
#include "curvenet_api.h"

#include "cassie_beautifier.h"
#include "cassie_beautifier_params.h"
#include "constraints/cassie_mirror_plane_constraint.h"
#include "delaunay_geogram.h"
#include "sketch/cassie_input_stroke.h"
#include "sketch/cassie_sketch_graph.h"
#include "sketch/cassie_surface_manager.h"
#include "solver/cassie_constraint_solver.h"

#include "../common/mesh_wire.h"
#include "../common/sha256.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <utility>

namespace cn {
namespace {

struct Out {
	bool pass = false;
	std::vector<int64_t> ints;
	std::vector<float> floats;
	std::string detail;
};

std::string fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char *f, ...) {
	char b[1024];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

std::string fsig(const std::vector<float> &v) {
	sha256::Ctx h;
	for (float x : v) {
		uint32_t bits;
		std::memcpy(&bits, &x, 4);
		unsigned char le[4];
		for (int i = 0; i < 4; ++i)
			le[i] = (unsigned char)(bits >> (8 * i));
		h.update(le, 4);
	}
	return h.hex().substr(0, 12);
}

void append(std::vector<float> &dst, const std::vector<float> &src) {
	dst.insert(dst.end(), src.begin(), src.end());
}

void push3(std::vector<float> &dst, const Vector3 &v) {
	dst.push_back(float(v.x));
	dst.push_back(float(v.y));
	dst.push_back(float(v.z));
}

// "key=value" out of a pen_end / mesh_build answer; -1 when absent.
int kv(const std::string &s, const std::string &key) {
	const std::string k = key + "=";
	size_t at = 0;
	while ((at = s.find(k, at)) != std::string::npos) {
		if (at == 0 || s[at - 1] == ' ') {
			return std::atoi(s.c_str() + at + k.size());
		}
		at += k.size();
	}
	return -1;
}

// --- meshes -----------------------------------------------------------------------

// An icosphere of radius r, `subdiv` loop subdivisions of the icosahedron,
// CCW-outward.
void icosphere(float r, int subdiv, std::vector<float> &v, std::vector<int32_t> &f) {
	const double t = (1.0 + std::sqrt(5.0)) / 2.0;
	std::vector<double> p = { -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1,
		-t, 0, -1, -t, 0, 1 };
	f = { 0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3,
		6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1 };
	auto norm = [&](size_t i) {
		const double l = std::sqrt(p[3 * i] * p[3 * i] + p[3 * i + 1] * p[3 * i + 1] + p[3 * i + 2] * p[3 * i + 2]);
		p[3 * i] /= l;
		p[3 * i + 1] /= l;
		p[3 * i + 2] /= l;
	};
	for (size_t i = 0; i < 12; ++i) {
		norm(i);
	}
	for (int s = 0; s < subdiv; ++s) {
		std::map<std::pair<int32_t, int32_t>, int32_t> mid;
		auto midpoint = [&](int32_t a, int32_t b) {
			const std::pair<int32_t, int32_t> k(std::min(a, b), std::max(a, b));
			auto it = mid.find(k);
			if (it != mid.end()) {
				return it->second;
			}
			const int32_t m = int32_t(p.size() / 3);
			for (int c = 0; c < 3; ++c) {
				p.push_back((p[3 * size_t(a) + c] + p[3 * size_t(b) + c]) * 0.5);
			}
			norm(size_t(m));
			mid.emplace(k, m);
			return m;
		};
		std::vector<int32_t> g;
		for (size_t i = 0; i + 2 < f.size(); i += 3) {
			const int32_t a = f[i], b = f[i + 1], c = f[i + 2];
			const int32_t ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
			g.insert(g.end(), { a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca });
		}
		f = g;
	}
	v.resize(p.size());
	for (size_t i = 0; i < p.size(); ++i) {
		v[i] = float(p[i] * r);
	}
}

// A unit cube on shared corners, 12 triangles, CCW-outward.
void cube(std::vector<float> &v, std::vector<int32_t> &f) {
	v = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1 };
	f = { 0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5 };
}

PackedVector3Array segment(const Vector3 &a, const Vector3 &b, int n) {
	PackedVector3Array out;
	for (int i = 0; i <= n; ++i) {
		out.push_back(a.lerp(b, real_t(i) / real_t(n)));
	}
	return out;
}

// --- the cassie-flow checks -------------------------------------------------------

// beautify_determinism.gd: identical CassieInputStroke samples give identical
// Curve3D output; a 0.05 bump on one sample must differ.
Out beautify_determinism() {
	auto run = [](bool perturb) {
		const int N = 32;
		Ref<CassieInputStroke> s;
		s.instantiate();
		for (int i = 0; i < N; ++i) {
			const float t = float(i) / float(N - 1);
			Vector3 p(t * 0.5f, Math::sin(t * float(Math::PI)) * 0.1f, 0.0f);
			if (perturb && i == N / 2) {
				p += Vector3(0.0f, 0.05f, 0.0f);
			}
			s->add_sample(p, t * 0.5f, 0.5f);
		}
		Ref<CassieSketchContext> ctx;
		ctx.instantiate();
		Ref<CassieBeautifierParams> params;
		params.instantiate();
		Ref<CassieBeautifier> b;
		b.instantiate();
		const Dictionary res = b->beautify(s, ctx, params, true, false);
		std::vector<float> sig;
		const Ref<Curve3D> c = res.get("curve", Variant());
		if (c.is_valid()) {
			for (int i = 0; i < c->get_point_count(); ++i) {
				push3(sig, c->get_point_position(i));
				push3(sig, c->get_point_in(i));
				push3(sig, c->get_point_out(i));
			}
		}
		return sig;
	};
	const std::vector<float> a = run(false), b = run(false), c = run(true);
	Out o;
	const bool same = a == b, differs = a != c;
	o.ints = { int64_t(a.size()), same, differs };
	o.floats = a;
	o.pass = !a.empty() && same && differs;
	o.detail = fmt("identical inputs match on %d floats: %s; 0.05-perturbed differs: %s", int(a.size()), same ? "yes" : "NO",
			differs ? "yes (control caught)" : "NO");
	return o;
}

// curvenet_extract.gd: three strokes closing a triangle give >= 1 cycle; the
// open path (one stroke fewer) 0.
Out curvenet_extract_check() {
	const Vector3 a(0, 0, 0), b(1, 0, 0), c(0.5, 1, 0);
	auto cycles = [](const std::vector<std::pair<Vector3, Vector3>> &segs, std::vector<float> *nodes) {
		TypedArray<PackedVector3Array> polys;
		for (const auto &s : segs) {
			PackedVector3Array p;
			p.push_back(s.first);
			p.push_back(s.second);
			polys.push_back(p);
		}
		Ref<CassieSketchGraph> g;
		g.instantiate();
		g->build_from_polylines(polys, real_t(1e-3));
		if (nodes) {
			for (int id = 0; id < 64; ++id) {
				const Ref<CassieSketchGraphNode> n = g->get_node(id);
				if (n.is_valid()) {
					push3(*nodes, n->get_position());
				}
			}
		}
		return int(g->find_cycles().size());
	};
	Out o;
	const int closed = cycles({ { a, b }, { b, c }, { c, a } }, &o.floats);
	const int open = cycles({ { a, b }, { b, c } }, nullptr);
	o.ints = { closed, open };
	o.pass = closed >= 1 && open == 0;
	o.detail = fmt("triangle -> %d cycle(s); open path -> %d (control)", closed, open);
	return o;
}

int triangle_total(const Ref<CassieSketchGraph> &g, std::vector<float> *verts) {
	Ref<CassieSurfaceManager> sm;
	sm.instantiate();
	sm->set_async_triangulation(false);
	sm->set_graph(g);
	sm->update();
	int total = 0;
	const TypedArray<CassieSurfacePatch> ps = sm->get_patches();
	for (int i = 0; i < ps.size(); ++i) {
		const Ref<CassieSurfacePatch> p = ps[i];
		total += p->get_triangle_count();
		if (verts) {
			for (int v = 0; v < p->get_vertex_count(); ++v) {
				push3(*verts, p->get_vertex_position(v));
			}
		}
	}
	return total;
}

// patch_pipeline.gd: a triangle cycle through CassieSurfaceManager gives
// >= 1 patch triangle; a one-edge graph 0.
Out patch_pipeline() {
	auto graph = [](bool closed) {
		TypedArray<PackedVector3Array> polys;
		const Vector3 a(0, 0, 0), b(1, 0, 0), c(0.5, 1, 0);
		PackedVector3Array ab, bc, ca;
		ab.push_back(a);
		ab.push_back(b);
		bc.push_back(b);
		bc.push_back(c);
		ca.push_back(c);
		ca.push_back(a);
		polys.push_back(ab);
		if (closed) {
			polys.push_back(bc);
			polys.push_back(ca);
		}
		Ref<CassieSketchGraph> g;
		g.instantiate();
		g->build_from_polylines(polys, real_t(1e-3));
		return g;
	};
	Out o;
	const int closed = triangle_total(graph(true), &o.floats);
	const int degenerate = triangle_total(graph(false), nullptr);
	o.ints = { closed, degenerate };
	o.pass = closed >= 1 && degenerate == 0;
	o.detail = fmt("cycle -> %d triangles; one-edge graph -> %d (control)", closed, degenerate);
	return o;
}

// crossing_split.gd: three overshooting strokes through add_stroke_intersecting
// split into 9 edges on 9 nodes and close >= 1 cycle; an edge the later
// strokes never touch keeps its id while a crossed one is replaced; plain
// add_stroke (endpoint merge only) closes nothing over 3 edges.
Out crossing_split() {
	const real_t prox = real_t(0.02);
	const PackedVector3Array empty;
	auto overshooting = []() {
		std::vector<PackedVector3Array> s;
		s.push_back(segment(Vector3(-1.2f, 0, 0), Vector3(1.2f, 0, 0), 16));
		s.push_back(segment(Vector3(1.1f, -0.3f, 0), Vector3(-0.2f, 1.8f, 0), 16));
		s.push_back(segment(Vector3(0.2f, 1.8f, 0), Vector3(-1.1f, -0.3f, 0), 16));
		return s;
	};
	Out o;
	Ref<CassieSketchGraph> split;
	split.instantiate();
	for (const PackedVector3Array &s : overshooting()) {
		split->add_stroke_intersecting(s, empty, prox);
	}
	const int cycles = int(split->find_cycles().size());
	const int tris = triangle_total(split, &o.floats);
	const int edges = split->get_edge_count(), nodes = split->get_node_count();

	Ref<CassieSketchGraph> far;
	far.instantiate();
	far->add_stroke_intersecting(segment(Vector3(5, 5, 0), Vector3(6, 5, 0), 4), empty, prox);
	far->add_stroke_intersecting(segment(Vector3(-1, 0, 0), Vector3(1, 0, 0), 4), empty, prox);
	far->add_stroke_intersecting(segment(Vector3(0, -1, 0), Vector3(0, 1, 0), 4), empty, prox);
	const bool kept0 = far->get_edge(0).is_valid(), gone1 = far->get_edge(1).is_null();

	Ref<CassieSketchGraph> plain;
	plain.instantiate();
	for (const PackedVector3Array &s : overshooting()) {
		plain->add_stroke(s, empty);
	}
	const int plain_cycles = int(plain->find_cycles().size());
	const int plain_edges = plain->get_edge_count();

	o.ints = { edges, nodes, cycles, tris, kept0, gone1, plain_cycles, plain_edges };
	o.pass = edges == 9 && nodes == 9 && cycles >= 1 && tris >= 1 && kept0 && gone1 && plain_cycles == 0 && plain_edges == 3;
	o.detail = fmt("%d edges, %d nodes, %d cycle(s), %d triangles; far edge 0 kept %s, crossed edge 1 replaced %s; "
				   "add_stroke -> %d cycles over %d edges (control)",
			edges, nodes, cycles, tris, kept0 ? "yes" : "NO", gone1 ? "yes" : "NO", plain_cycles, plain_edges);
	return o;
}

// constraint_solver.gd: a mirror-plane constraint per anchor pulls a curve
// 0.05 above y=0 to within 5e-3 of it; the raw input fails that tolerance.
Out constraint_solver() {
	const real_t tol = real_t(5e-3), off = real_t(0.05);
	auto make = [off]() {
		Ref<Curve3D> c;
		c.instantiate();
		c->add_point(Vector3(-0.5f, off, 0));
		c->add_point(Vector3(0, off, 0));
		c->add_point(Vector3(0.5f, off, 0));
		return c;
	};
	const Plane plane(Vector3(0, 1, 0), 0);
	auto worst = [&plane](const Ref<Curve3D> &c) {
		real_t w = 0;
		for (int i = 0; i < c->get_point_count(); ++i) {
			w = MAX(w, Math::abs(plane.distance_to(c->get_point_position(i))));
		}
		return w;
	};
	const real_t before = worst(make());
	const Ref<Curve3D> target = make();
	TypedArray<CassieConstraint> cs;
	for (int i = 0; i < target->get_point_count(); ++i) {
		const Vector3 p = target->get_point_position(i);
		Ref<CassieMirrorPlaneConstraint> m;
		m.instantiate();
		m->set_plane_normal(plane.normal);
		m->set_position(Vector3(p.x, 0, p.z));
		cs.push_back(m);
	}
	Ref<CassieSolverParams> params;
	params.instantiate();
	params->set_mu_fidelity(0.0);
	params->set_proximity_threshold(0.2);
	Ref<CassieConstraintSolver> solver;
	solver.instantiate();
	const Dictionary res = solver->solve(target, cs, PackedVector3Array(), params, false);
	const Ref<Curve3D> solved = res.get("curve", Variant());
	Out o;
	const real_t after = solved.is_valid() ? worst(solved) : real_t(1e30);
	if (solved.is_valid()) {
		for (int i = 0; i < solved->get_point_count(); ++i) {
			push3(o.floats, solved->get_point_position(i));
		}
	}
	o.ints = { solved.is_valid() ? solved->get_point_count() : -1, before > tol, after <= tol };
	o.pass = solved.is_valid() && before > tol && after <= tol;
	o.detail = fmt("mirror pins pulled anchors %.5f -> %.5f (tol %.5f); raw input fails the tolerance: %s", double(before),
			double(after), double(tol), before > tol ? "yes (control caught)" : "NO");
	return o;
}

// --- this stage's checks ------------------------------------------------------------

// Draws a stroke through the API: n samples on the circle of latitude `lat`
// (radians) at radius r, sweeping `sweep` radians; the last sample of a full
// sweep lands exactly on the first. Returns pen_end's answer.
std::string pen_circle(float r, float lat, float sweep, int n) {
	const float y = r * std::sin(lat), rho = r * std::cos(lat);
	int id = -1;
	for (int i = 0; i <= n; ++i) {
		const float a = (i == n && sweep >= float(2 * Math::PI)) ? 0.0f : sweep * float(i) / float(n);
		const float x = rho * std::cos(a), z = rho * std::sin(a);
		if (i == 0) {
			id = pen_begin(x, y, z, 0.5f);
		} else {
			pen_point(id, x, y, z, 0.5f);
		}
	}
	return pen_end(id);
}

// pen_sphere: one closed stroke at 30 degrees latitude, drawn 1 cm off an
// r = 0.5 icosphere, snaps onto it (every sample within [0.499, 0.5025] of
// the centre) and gives 1 patch; mesh_build gives 1 boundary loop, with and
// without the PMP remesh; curvenet_build gives 2 curves on 2 knots (the
// sketcher splits a closed stroke into two edges). Controls: a 270 degree
// arc gives 0 patches, and split_closed = 0 gives 0 patches (a self-loop
// edge bounds nothing).
Out pen_sphere() {
	std::vector<float> bv;
	std::vector<int32_t> bf;
	icosphere(0.5f, 3, bv, bf);
	const float lat = float(Math::PI / 6.0), full = float(2 * Math::PI);
	Out o;
	set_param("split_closed", 1);
	set_body(bv, bf);

	reset();
	const std::string e = pen_circle(0.51f, lat, full, 64);
	const int patches = kv(e, "patches");
	const std::vector<float> samples = stroke_samples();
	int snapped = 0;
	for (size_t i = 0; i + 2 < samples.size(); i += 3) {
		const double r = std::sqrt(double(samples[i]) * samples[i] + double(samples[i + 1]) * samples[i + 1] +
				double(samples[i + 2]) * samples[i + 2]);
		snapped += (r >= 0.499 && r <= 0.5025) ? 1 : 0;
	}
	const std::string m0 = mesh_build(0, 1e-5);
	const int loops = kv(m0, "loops"), tris = kv(m0, "triangles");
	append(o.floats, mesh_vertices());
	const std::string m1 = mesh_build(0.02, 1e-5);
	const int loops_remeshed = kv(m1, "loops"), tris_remeshed = kv(m1, "triangles");
	const std::string cb = curvenet_build();
	const int curves = counts().curves, knots = counts().knots;
	append(o.floats, curvenet_curves());
	append(o.floats, curvenet_knots());

	reset();
	const std::string arc = pen_circle(0.51f, lat, float(1.5 * Math::PI), 48);
	const int arc_patches = kv(arc, "patches");

	reset();
	set_param("split_closed", 0);
	const std::string whole = pen_circle(0.51f, lat, full, 64);
	const int whole_patches = kv(whole, "patches");
	set_param("split_closed", 1);

	set_body({}, {});
	reset();

	const int n_samples = int(samples.size() / 3);
	o.ints = { patches, n_samples, snapped, loops, tris, loops_remeshed, tris_remeshed, curves, knots, arc_patches,
		whole_patches };
	o.pass = patches == 1 && n_samples > 0 && snapped == n_samples && loops == 1 && tris > 0 && loops_remeshed == 1 &&
			tris_remeshed > 0 && curves == 2 && knots == 2 && arc_patches == 0 && whole_patches == 0;
	o.detail = fmt("closed stroke -> %d patch, %d/%d samples snapped; mesh %d tris %d loop, remeshed %d tris %d loop; "
				   "curvenet %d curves %d knots; 270-degree arc -> %d patches, split_closed=0 -> %d (controls) [%s | %s | %s]",
			patches, snapped, n_samples, tris, loops, tris_remeshed, loops_remeshed, curves, knots, arc_patches,
			whole_patches, e.c_str(), m1.c_str(), cb.c_str());
	return o;
}

// extractor_cube: a unit cube gives 12 curves on 8 knots of degree >= 3 (its
// 90-degree edges); control: a smooth icosphere(3) gives at most 1 curve
// (no dihedral reaches the extractor's 15-degree floor).
Out extractor_cube() {
	std::vector<float> v;
	std::vector<int32_t> f;
	cube(v, f);
	Out o;
	const std::string r = curvenet_extract(v, f, 200, 1e-3, 1e-2, 0);
	const std::vector<float> curves = curvenet_curves(), knots = curvenet_knots();
	append(o.floats, curves);
	append(o.floats, knots);
	const int nc = int(curves.empty() ? -1 : curves[0]), nk = int(knots.empty() ? -1 : knots[0]);
	int deg3 = 0;
	for (int k = 0; k < nk; ++k) {
		deg3 += knots[1 + size_t(k) * mesh_wire::KNOT_STRIDE + 3] >= 3 ? 1 : 0;
	}
	std::vector<float> iv;
	std::vector<int32_t> iff;
	icosphere(0.5f, 3, iv, iff);
	curvenet_extract(iv, iff, 200, 1e-3, 1e-2, 0);
	const std::vector<float> ic = curvenet_curves();
	const int ico = int(ic.empty() ? -1 : ic[0]);
	o.ints = { nc, nk, deg3, ico };
	o.pass = nc == 12 && nk == 8 && deg3 == 8 && ico <= 1;
	o.detail = fmt("cube -> %d curves, %d knots, %d of degree >= 3; icosphere(3) -> %d curves (control) [%s]", nc, nk, deg3,
			ico, r.c_str());
	return o;
}

// delaunay_small_scale: Geogram BDEL2d through delaunay_triangulate_2d_raw on
// a 1 mm polygon 1 m from the origin (8 hull points on a 0.5 mm circle, 3
// inside) gives 2n - 2 - h triangles; control: collinear points are refused.
Out delaunay_small_scale() {
	std::vector<double> xy;
	const double cx = 1.0, cy = 0.75, r = 0.5e-3;
	const int h = 8;
	for (int i = 0; i < h; ++i) {
		const double a = 2 * Math::PI * i / h;
		xy.push_back(cx + r * std::cos(a));
		xy.push_back(cy + r * std::sin(a));
	}
	const double inner[6] = { 0.1e-3, 0.05e-3, -0.2e-3, 0.1e-3, 0.05e-3, -0.25e-3 };
	for (int i = 0; i < 3; ++i) {
		xy.push_back(cx + inner[2 * i]);
		xy.push_back(cy + inner[2 * i + 1]);
	}
	const int n = int(xy.size() / 2), want = 2 * n - 2 - h;
	int *faces = nullptr, nf = 0;
	const bool ok = cassie::delaunay_triangulate_2d_raw(xy.data(), n, &faces, &nf);
	Out o;
	for (int i = 0; ok && i < 3 * nf; ++i) {
		o.floats.push_back(float(faces[i]));
	}
	delete[] faces;
	std::vector<double> line;
	for (int i = 0; i < 6; ++i) {
		line.push_back(cx + 0.2e-3 * i);
		line.push_back(cy + 0.1e-3 * i);
	}
	int *lf = nullptr, lnf = 0;
	const bool line_ok = cassie::delaunay_triangulate_2d_raw(line.data(), 6, &lf, &lnf);
	delete[] lf;
	o.ints = { ok, nf, want, line_ok };
	o.pass = ok && nf == want && !line_ok;
	o.detail = fmt("1 mm polygon, n=%d h=%d -> %d triangles (2n-2-h = %d); collinear -> %s (control)", n, h, nf, want,
			line_ok ? "accepted" : "refused");
	return o;
}

// mesh_weld: a unit square as two triangles in two parts (six vertices)
// welds into 1 component with 1 boundary loop of 4 and Euler characteristic 1;
// control: weld_eps = 0 leaves 2 components and 2 loops.
Out mesh_weld() {
	const std::vector<std::vector<float>> pv = { { 0, 0, 0, 1, 0, 0, 1, 1, 0 }, { 0, 0, 0, 1, 1, 0, 0, 1, 0 } };
	const std::vector<std::vector<int32_t>> pf = { { 0, 1, 2 }, { 0, 1, 2 } };
	const BuiltMesh w = build_mesh(pv, pf, 0, 1e-6);
	const BuiltMesh u = build_mesh(pv, pf, 0, 0);
	const int wv = int(w.vertices.size() / 3), wf = int(w.triangles.size() / 3);
	const int euler = wv - w.edges + wf;
	const int loop0 = w.loops.empty() ? 0 : int(w.loops[0].size());
	Out o;
	o.floats = w.vertices;
	o.ints = { wv, w.edges, wf, w.components, int(w.loops.size()), loop0, euler, u.components, int(u.loops.size()) };
	o.pass = w.components == 1 && w.loops.size() == 1 && loop0 == 4 && euler == 1 && u.components == 2 && u.loops.size() == 2;
	o.detail = fmt("welded: V=%d E=%d F=%d, %d component, %d loop of %d, Euler %d; weld_eps=0 -> %d components, %d loops (control)",
			wv, w.edges, wf, w.components, int(w.loops.size()), loop0, euler, u.components, int(u.loops.size()));
	return o;
}

// A capped cylinder of radius r around the y axis from y0 to y1, nseg
// segments around and nrow rows up, CCW-outward.
void cylinder(float r, float y0, float y1, int nseg, int nrow, std::vector<float> &v, std::vector<int32_t> &f) {
	v.clear();
	f.clear();
	for (int j = 0; j <= nrow; ++j) {
		const float y = y0 + (y1 - y0) * float(j) / float(nrow);
		for (int i = 0; i < nseg; ++i) {
			const double a = 2 * Math::PI * i / nseg;
			v.insert(v.end(), { float(r * std::cos(a)), y, float(r * std::sin(a)) });
		}
	}
	auto id = [nseg](int i, int j) { return int32_t(j * nseg + i % nseg); };
	for (int j = 0; j < nrow; ++j) {
		for (int i = 0; i < nseg; ++i) {
			const int32_t a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
			f.insert(f.end(), { a, d, b, b, d, c });
		}
	}
	const int32_t bot = int32_t(v.size() / 3), top = bot + 1;
	v.insert(v.end(), { 0, y0, 0, 0, y1, 0 });
	for (int i = 0; i < nseg; ++i) {
		f.insert(f.end(), { bot, id(i, 0), id(i + 1, 0), top, id(i + 1, nrow), id(i, nrow) });
	}
}

// Draws one stroke through the API, sample by sample; pen_end's answer.
std::string pen_polyline(const std::vector<Vector3> &p) {
	const int id = pen_begin(float(p[0].x), float(p[0].y), float(p[0].z), 0.5f);
	for (size_t i = 1; i < p.size(); ++i) {
		pen_point(id, float(p[i].x), float(p[i].y), float(p[i].z), 0.5f);
	}
	return pen_end(id);
}

// The Cut 8 scripted skirt (project/xr/pen_source_scripted.gd): a waist ring
// and a hem ring, each as two half rings (left +x, right -x) from the front
// (+z) to the back (-z), then a front and a back seam from waist to hem,
// ending exactly on the ring points; half_samples 24, seam_samples 12.
struct Skirt {
	std::vector<std::pair<std::string, std::vector<Vector3>>> strokes;
};

Skirt skirt_strokes(float waist_y, float hem_y, float r) {
	auto at = [r](float y, double a) { return Vector3(real_t(r * std::cos(a)), real_t(y), real_t(r * std::sin(a))); };
	auto arc = [&](float y, double a0, double a1, int n) {
		std::vector<Vector3> out;
		for (int i = 0; i <= n; ++i) {
			out.push_back(at(y, a0 + (a1 - a0) * double(i) / double(n)));
		}
		return out;
	};
	auto line = [](const Vector3 &a, const Vector3 &b, int n) {
		std::vector<Vector3> out;
		for (int i = 0; i <= n; ++i) {
			out.push_back(a.lerp(b, real_t(double(i) / double(n))));
		}
		out[size_t(n)] = b;
		return out;
	};
	const double pi = Math::PI;
	Skirt s;
	s.strokes.emplace_back("waist_left", arc(waist_y, pi / 2, -pi / 2, 24));
	s.strokes.emplace_back("waist_right", arc(waist_y, pi / 2, 3 * pi / 2, 24));
	s.strokes.emplace_back("hem_left", arc(hem_y, pi / 2, -pi / 2, 24));
	s.strokes.emplace_back("hem_right", arc(hem_y, pi / 2, 3 * pi / 2, 24));
	s.strokes.emplace_back("seam_front", line(at(waist_y, pi / 2), at(hem_y, pi / 2), 12));
	s.strokes.emplace_back("seam_back", line(at(waist_y, -pi / 2), at(hem_y, -pi / 2), 12));
	return s;
}

// What a patch touches, from its vertices: the waist ring (a vertex within
// 1 cm of its height), the hem ring, the front seam and the back seam (a
// vertex within 5 mm of the x = 0 plane on that side, at least 5 cm from
// both rings, so a cap's knot does not count); side: the sign of its mean x.
struct PatchTouch {
	bool waist = false, hem = false, front = false, back = false, cap = false;
	int side = 0;
	bool panel() const { return waist && hem && front && back; }
};

PatchTouch patch_touch(const std::vector<float> &v, float waist_y, float hem_y) {
	PatchTouch t;
	double mean_x = 0, ymin = 1e30, ymax = -1e30;
	for (size_t i = 0; i + 2 < v.size(); i += 3) {
		const float x = v[i], y = v[i + 1], z = v[i + 2];
		t.waist = t.waist || std::fabs(y - waist_y) < 0.01f;
		t.hem = t.hem || std::fabs(y - hem_y) < 0.01f;
		const bool mid = y > hem_y + 0.05f && y < waist_y - 0.05f;
		t.front = t.front || (mid && z > 0 && std::fabs(x) < 0.005f);
		t.back = t.back || (mid && z < 0 && std::fabs(x) < 0.005f);
		mean_x += x;
		ymin = std::min(ymin, double(y));
		ymax = std::max(ymax, double(y));
	}
	t.side = mean_x > 0 ? 1 : -1;
	t.cap = !v.empty() && ymax - ymin < 0.03;
	return t;
}

// Draws the skirt; `boundary` marks the four half rings as boundary strokes,
// `drop` names a stroke left out. Answers the last pen_end.
std::string draw_skirt(const Skirt &s, bool boundary, const std::string &drop) {
	reset();
	std::string last;
	for (const auto &st : s.strokes) {
		if (st.first == drop) {
			continue;
		}
		set_param("boundary", boundary && st.first.rfind("seam", 0) != 0 ? 1 : 0);
		last = pen_polyline(st.second);
	}
	set_param("boundary", 0);
	return last;
}

// skirt_tube: a skirt is an open tube, and the garment is its two panels,
// each bounded by a half waist, a seam, a half hem and the other seam, never
// the caps over the waist and the hem. The Cut 8 scripted skirt on a capped
// cylinder body (r 0.15 m; rings of r 0.16 at y 0.9 and 0.5, which snap onto
// it), the four half rings drawn as boundary strokes (param boundary = 1):
//   - the stroke ends meet in 4 knots of degree 3 and the curvenet has 6
//     curves (every half ring and seam end merges into a shared knot);
//   - find_cycles sees the 2 panels and the 2 caps; the caps are made only of
//     boundary strokes, so they are openings: 2 cycles, 2 openings, 2 patches,
//     each a panel (it touches both rings and both seams), one per side;
//   - mesh_build welds them into 1 component with 2 boundary loops (one at
//     each ring), Euler characteristic 0, every interior edge shared by two
//     triangles in opposite directions and every triangle facing outward;
//     mesh_patch_ids names both panels;
//   - the PMP remesh (0.02) keeps all that, and every remeshed triangle gets
//     the patch of its own side (it used to get -1).
// Controls: without the back seam, 0 panels (no patch at all); with the rings
// drawn as ordinary strokes, the caps come back (4 patches, 2 of them caps).
Out skirt_tube() {
	const float waist_y = 0.9f, hem_y = 0.5f, ring_r = 0.16f;
	std::vector<float> bv;
	std::vector<int32_t> bf;
	cylinder(0.15f, 0.3f, 1.1f, 64, 16, bv, bf);
	const Skirt s = skirt_strokes(waist_y, hem_y, ring_r);
	Out o;
	set_body(bv, bf);

	const std::string e = draw_skirt(s, true, "");
	const int edges = kv(e, "edges"), nodes = kv(e, "nodes"), cycles = kv(e, "cycles"), openings = kv(e, "openings");
	const int patches = patch_count();
	int panels = 0, sides = 0;
	std::vector<int> side_of(size_t(std::max(patches, 0)), 0);
	for (int i = 0; i < patches; ++i) {
		const PatchTouch t = patch_touch(patch_vertices(i), waist_y, hem_y);
		panels += t.panel() ? 1 : 0;
		sides += t.side;
		side_of[size_t(i)] = t.side;
	}

	const std::string cb = curvenet_build();
	const int curves = counts().curves, knots = counts().knots;
	const std::vector<float> kn = curvenet_knots();
	int deg3 = 0;
	for (int k = 0; k < knots && size_t(1 + (k + 1) * mesh_wire::KNOT_STRIDE) <= kn.size(); ++k) {
		deg3 += kn[1 + size_t(k) * mesh_wire::KNOT_STRIDE + 3] == 3.0f ? 1 : 0;
	}
	append(o.floats, kn);

	// The welded mesh and its topology, orientation and patch ids; the same
	// after the PMP remesh.
	struct Topo {
		int components = -1, loops = -1, euler = 0, loops_at_rings = 0, twice = -1, inward = -1, unassigned = -1,
			wrong_side = -1, id_count = 0;
	};
	auto topo = [&](const std::string &answer) {
		Topo t;
		t.components = kv(answer, "components");
		t.loops = kv(answer, "loops");
		const std::vector<float> v = mesh_vertices();
		const std::vector<int32_t> f = mesh_indices(), ids = mesh_patch_ids();
		std::map<std::pair<int32_t, int32_t>, int> directed;
		std::map<std::pair<int32_t, int32_t>, int> undirected;
		std::map<int32_t, int> id_seen;
		t.twice = 0;
		t.inward = 0;
		t.unassigned = 0;
		t.wrong_side = 0;
		for (size_t k = 0; k + 2 < f.size(); k += 3) {
			Vector3 p[3];
			for (int c = 0; c < 3; ++c) {
				const int32_t a = f[k + c], b = f[k + (c + 1) % 3];
				directed[{ a, b }] += 1;
				undirected[{ std::min(a, b), std::max(a, b) }] += 1;
				p[c] = Vector3(v[3 * size_t(a)], v[3 * size_t(a) + 1], v[3 * size_t(a) + 2]);
			}
			const Vector3 n = (p[1] - p[0]).cross(p[2] - p[0]);
			const Vector3 c = (p[0] + p[1] + p[2]) / real_t(3);
			t.inward += n.dot(Vector3(c.x, 0, c.z)) <= 0 ? 1 : 0;
			const int32_t id = k / 3 < ids.size() ? ids[k / 3] : -1;
			id_seen[id] += 1;
			if (id < 0 || id >= patches) {
				t.unassigned += 1;
			} else if (std::fabs(c.x) > 0.01f && (c.x > 0 ? 1 : -1) != side_of[size_t(id)]) {
				t.wrong_side += 1;
			}
		}
		for (const auto &kv2 : directed) {
			t.twice += kv2.second > 1 ? 1 : 0;
		}
		t.euler = int(v.size() / 3) - int(undirected.size()) + int(f.size() / 3);
		t.id_count = int(id_seen.size());
		std::vector<std::vector<int32_t>> loops;
		mesh_wire::decode_loops(mesh_boundary_loops(), loops);
		bool waist_loop = false, hem_loop = false;
		for (const std::vector<int32_t> &l : loops) {
			bool all_w = true, all_h = true;
			for (int32_t i : l) {
				all_w = all_w && std::fabs(v[3 * size_t(i) + 1] - waist_y) < 0.03f;
				all_h = all_h && std::fabs(v[3 * size_t(i) + 1] - hem_y) < 0.03f;
			}
			waist_loop = waist_loop || all_w;
			hem_loop = hem_loop || all_h;
		}
		t.loops_at_rings = int(waist_loop) + int(hem_loop);
		return t;
	};
	const std::string m0 = mesh_build(0, 1e-5);
	const Topo t0 = topo(m0);
	append(o.floats, mesh_vertices());
	const std::string m1 = mesh_build(0.02, 1e-5);
	const Topo t1 = topo(m1);
	append(o.floats, mesh_vertices());

	// Controls.
	const std::string drop = draw_skirt(s, true, "seam_back");
	const int drop_patches = patch_count();
	int drop_panels = 0;
	for (int i = 0; i < drop_patches; ++i) {
		drop_panels += patch_touch(patch_vertices(i), waist_y, hem_y).panel() ? 1 : 0;
	}
	const std::string plain = draw_skirt(s, false, "");
	const int plain_patches = patch_count();
	int plain_panels = 0, plain_caps = 0;
	for (int i = 0; i < plain_patches; ++i) {
		const PatchTouch t = patch_touch(patch_vertices(i), waist_y, hem_y);
		plain_panels += t.panel() ? 1 : 0;
		plain_caps += t.cap ? 1 : 0;
	}

	set_body({}, {});
	reset();

	o.ints = { nodes, edges, knots, deg3, curves, cycles, openings, patches, panels, sides, t0.components, t0.loops, t0.euler,
		t0.loops_at_rings, t0.twice, t0.inward, t0.id_count, t0.unassigned, t0.wrong_side, t1.components, t1.loops, t1.euler,
		t1.loops_at_rings, t1.twice, t1.inward, t1.id_count, t1.unassigned, t1.wrong_side, drop_patches, drop_panels,
		plain_patches, plain_panels, plain_caps };
	auto good = [](const Topo &t) {
		return t.components == 1 && t.loops == 2 && t.euler == 0 && t.loops_at_rings == 2 && t.twice == 0 && t.inward == 0 &&
				t.id_count == 2 && t.unassigned == 0 && t.wrong_side == 0;
	};
	o.pass = nodes == 4 && edges == 6 && knots == 4 && deg3 == 4 && curves == 6 && cycles == 2 && openings == 2 &&
			patches == 2 && panels == 2 && sides == 0 && good(t0) && good(t1) && drop_panels == 0 && plain_patches == 4 &&
			plain_panels == 2 && plain_caps == 2;
	o.detail = fmt("rings as boundary strokes -> %d knots (%d of degree 3), %d curves, %d cycles + %d openings, %d patches, "
				   "%d panels (one per side: %s); mesh: %d component, %d loops at the rings, Euler %d, %d directed edges used "
				   "twice, %d inward, %d patch ids (%d unassigned, %d on the wrong side); remeshed: %d component, %d loops at "
				   "the rings, Euler %d, %d ids, %d unassigned, %d on the wrong side; back seam dropped -> %d panels (%d "
				   "patches), rings not boundary -> %d patches, %d caps (controls)",
			knots, deg3, curves, cycles, openings, patches, panels, sides == 0 ? "yes" : "NO", t0.components,
			t0.loops_at_rings, t0.euler, t0.twice, t0.inward, t0.id_count, t0.unassigned, t0.wrong_side, t1.components,
			t1.loops_at_rings, t1.euler, t1.id_count, t1.unassigned, t1.wrong_side, drop_panels, drop_patches, plain_patches,
			plain_caps) +
			" [" + e + " | " + m0 + " | " + m1 + " | " + cb + " | " + drop + " | " + plain + "]";
	return o;
}

const std::vector<std::pair<std::string, std::function<Out()>>> &table() {
	static const std::vector<std::pair<std::string, std::function<Out()>>> t = {
		{ "beautify_determinism", beautify_determinism },
		{ "curvenet_extract", curvenet_extract_check },
		{ "patch_pipeline", patch_pipeline },
		{ "crossing_split", crossing_split },
		{ "constraint_solver", constraint_solver },
		{ "pen_sphere", pen_sphere },
		{ "extractor_cube", extractor_cube },
		{ "delaunay_small_scale", delaunay_small_scale },
		{ "mesh_weld", mesh_weld },
		{ "skirt_tube", skirt_tube },
	};
	return t;
}

} // namespace

std::vector<std::string> check_names() {
	std::vector<std::string> out;
	for (const auto &e : table()) {
		out.push_back(e.first);
	}
	return out;
}

std::string check(const std::string &name) {
	for (const auto &e : table()) {
		if (e.first != name) {
			continue;
		}
		Out o;
		try {
			o = e.second();
		} catch (const std::exception &ex) {
			return "FAIL " + name + " ints= fsig=000000000000/0 :: exception: " + ex.what();
		} catch (...) {
			return "FAIL " + name + " ints= fsig=000000000000/0 :: unknown exception";
		}
		std::string ints;
		for (size_t i = 0; i < o.ints.size(); ++i) {
			ints += (i ? "," : "") + std::to_string(o.ints[i]);
		}
		return fmt("%s %s ints=%s fsig=%s/%d :: ", o.pass ? "PASS" : "FAIL", name.c_str(), ints.c_str(),
					   fsig(o.floats).c_str(), int(o.floats.size())) +
				o.detail;
	}
	return "FAIL: unknown check '" + name + "'";
}

std::string check_all() {
	std::string out;
	int pass = 0, total = 0;
	for (const std::string &n : check_names()) {
		const std::string line = check(n);
		pass += line.rfind("PASS ", 0) == 0 ? 1 : 0;
		++total;
		out += line + "\n";
	}
	return out + fmt("checks: %d/%d", pass, total);
}

} // namespace cn
