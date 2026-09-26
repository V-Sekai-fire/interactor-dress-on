// See graph_api.h. The only TU that sees both std types and CASSIE (with the
// godot-lite prelude).
#include "graph_api.h"

#include "sketch/cassie_sketch_graph.h"

namespace cgraph {

std::string cycles(const float *points, size_t n_points, const int32_t *counts, size_t n_strokes, float merge_epsilon,
		Result &out) {
	size_t total = 0;
	for (size_t k = 0; k < n_strokes; ++k) {
		if (counts[k] < 0)
			return "ERR: stroke " + std::to_string(k) + " has a negative count";
		total += (size_t)counts[k];
	}
	if (total != n_points)
		return "ERR: counts sum " + std::to_string(total) + " != points " + std::to_string(n_points);
	Ref<CassieSketchGraph> graph;
	graph.instantiate();
	if (merge_epsilon > 0.0f)
		graph->set_merge_epsilon(merge_epsilon);
	const PackedVector3Array empty_normals;
	out = Result();
	out.strokes_in = (int)n_strokes;
	size_t at = 0;
	for (size_t k = 0; k < n_strokes; ++k) {
		const int32_t n = counts[k];
		if (n >= 2) {
			PackedVector3Array s;
			s.resize(n);
			for (int32_t i = 0; i < n; ++i) {
				const float *p = points + (at + (size_t)i) * 3;
				s.set(i, Vector3(p[0], p[1], p[2]));
			}
			if (graph->add_stroke(s, empty_normals) >= 0)
				out.strokes_added++;
		}
		at += (size_t)n;
	}
	const Array found = graph->find_cycles();
	for (int i = 0; i < found.size(); ++i) {
		const PackedInt32Array c = found[i];
		out.cycle_sizes.push_back(c.size());
	}
	out.nodes = graph->get_node_count();
	out.edges = graph->get_edge_count();
	return "ok";
}

} // namespace cgraph
