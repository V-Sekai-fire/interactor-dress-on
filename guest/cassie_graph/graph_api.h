// cassie_graph.elf's core: CASSIE's sketch graph (vendor/cassie-graph) on
// raw strokes, as godot-cassie's pipeline bench runs it. std types only, so
// main.cpp (the sandbox API) and a native control can both call it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cgraph {

struct Result {
	int strokes_in = 0; // strokes handed over
	int strokes_added = 0; // add_stroke answered >= 0
	int nodes = 0, edges = 0;
	std::vector<int32_t> cycle_sizes; // edges per find_cycles() entry, in its order
};

// points: xyz per point, strokes back to back; counts: points per stroke.
// Strokes under 2 points are skipped, as the bench does. "ok" or "ERR: ...".
// merge_epsilon <= 0 keeps CASSIE's default (0.02 m).
std::string cycles(const float *points, size_t n_points, const int32_t *counts, size_t n_strokes, float merge_epsilon,
		Result &out);

} // namespace cgraph
