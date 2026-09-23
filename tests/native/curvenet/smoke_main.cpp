// curvenet smoke: the Cut 4 libraries (cassie_core on godot_lite, Geogram,
// PMP, MWT) compile and link natively, and do something. Each check has a
// negative control that must come out differently, so "passes" cannot mean
// "nothing ran".
//
//   1. Geogram BDEL2d on a unit square: 2 cells. Control: 4 colinear points
//      give 0 cells (Cassie's wrapper reports false).
//   2. CassieSketchGraph::build_from_polylines on three strokes forming a
//      triangle: 3 nodes, 3 edges, >= 1 cycle of 3 edges. Control: the same
//      strokes less one (an open path) give 0 cycles.
//
// Compiled with the godot_lite prelude, like cassie_core.
#include "sketch/cassie_sketch_graph.h"

#include <cstdio>

int smoke_geogram_bdel2d(const double *p_xy, int p_n);
int smoke_cassie_delaunay_2d(const double *p_xy, int p_n);

static int g_fail = 0;

static void check(bool p_ok, const char *p_what) {
	std::printf("%s %s\n", p_ok ? "PASS" : "FAIL", p_what);
	if (!p_ok) {
		++g_fail;
	}
}

static PackedVector3Array segment(const Vector3 &a, const Vector3 &b, int p_samples = 8) {
	PackedVector3Array pts;
	pts.resize(p_samples);
	for (int i = 0; i < p_samples; ++i) {
		pts.write[i] = a.lerp(b, real_t(i) / real_t(p_samples - 1));
	}
	return pts;
}

int main() {
	// 1. Delaunay.
	const double square[8] = { 0, 0, 1, 0, 1, 1, 0, 1 };
	const double line[8] = { 0, 0, 1, 0, 2, 0, 3, 0 };
	const int geo_sq = smoke_geogram_bdel2d(square, 4);
	const int geo_line = smoke_geogram_bdel2d(line, 4);
	const int cas_sq = smoke_cassie_delaunay_2d(square, 4);
	const int cas_line = smoke_cassie_delaunay_2d(line, 4);
	std::printf("geogram BDEL2d: square %d cells, colinear %d cells\n", geo_sq, geo_line);
	std::printf("cassie delaunay_triangulate_2d_raw: square %d faces, colinear %d faces\n", cas_sq, cas_line);
	check(geo_sq == 2, "BDEL2d triangulates a square into 2 cells");
	check(geo_line == 0, "control: BDEL2d leaves 4 colinear points with 0 cells");
	check(cas_sq == 2, "delaunay_triangulate_2d_raw: square -> 2 faces");
	check(cas_line == 0, "control: delaunay_triangulate_2d_raw: colinear -> 0 faces");

	// 2. Sketch graph.
	const Vector3 v0(0, 0, 0), v1(1, 0, 0), v2(0.5, real_t(0.866), 0);
	TypedArray<PackedVector3Array> tri;
	tri.push_back(segment(v0, v1));
	tri.push_back(segment(v1, v2));
	tri.push_back(segment(v2, v0));
	Ref<CassieSketchGraph> g;
	g.instantiate();
	const int edges = g->build_from_polylines(tri, real_t(0.02));
	const Array cycles = g->find_cycles();
	const int first = cycles.size() > 0 ? PackedInt32Array(cycles[0]).size() : 0;
	std::printf("triangle: %d edges added, %d nodes, %d edges, %d cycles, first cycle %d edges\n",
			edges, g->get_node_count(), g->get_edge_count(), int(cycles.size()), first);
	check(g->get_node_count() == 3 && g->get_edge_count() == 3, "triangle: 3 nodes, 3 edges");
	check(cycles.size() >= 1 && first == 3, "triangle: >= 1 cycle, first walks 3 edges");

	TypedArray<PackedVector3Array> open;
	open.push_back(segment(v0, v1));
	open.push_back(segment(v1, v2));
	Ref<CassieSketchGraph> h;
	h.instantiate();
	h->build_from_polylines(open, real_t(0.02));
	const int open_cycles = int(h->find_cycles().size());
	std::printf("open path: %d nodes, %d edges, %d cycles\n", h->get_node_count(), h->get_edge_count(), open_cycles);
	check(open_cycles == 0, "control: an open path of 2 strokes has 0 cycles");

	std::printf("%s (%d failed)\n", g_fail == 0 ? "curvenet smoke: OK" : "curvenet smoke: FAILED", g_fail);
	return g_fail == 0 ? 0 : 1;
}
