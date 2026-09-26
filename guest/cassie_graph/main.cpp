// cassie_graph.elf -- CASSIE's sketch graph (godot-cassie's CassieSketchGraph,
// the port of CycleDetection.cs) on raw strokes: add_stroke per stroke, then
// find_cycles, as the module's pipeline bench runs a .curves file. The loop's
// expected cycle counts for a sketch come from here. Every ADD_API_FUNCTION has
// a no-argument wrapper in project/main.gd (rule 8).
#include <api.hpp>

#include <string>
#include <vector>

#include "graph_api.h"

static Variant cg_cycles(PackedArray<float> points, PackedArray<int32_t> counts, double merge_epsilon) {
	try {
		const std::vector<float> p = points.fetch();
		const std::vector<int32_t> c = counts.fetch();
		if (p.size() % 3 != 0)
			return Variant(String("FAIL: " + std::to_string(p.size()) + " floats is not whole xyz points"));
		cgraph::Result r;
		const std::string s = cgraph::cycles(p.data(), p.size() / 3, c.data(), c.size(), (float)merge_epsilon, r);
		if (s != "ok")
			return Variant(String(s));
		Dictionary d = Dictionary::Create();
		d["strokes_in"] = Variant(r.strokes_in);
		d["strokes_added"] = Variant(r.strokes_added);
		d["nodes"] = Variant(r.nodes);
		d["edges"] = Variant(r.edges);
		d["cycles"] = Variant((int)r.cycle_sizes.size());
		d["cycle_sizes"] = Variant(PackedArray<int32_t>(r.cycle_sizes.data(), r.cycle_sizes.size()));
		return Variant(d);
	} catch (const std::exception &e) {
		return Variant(String(std::string("FAIL: cg_cycles: ") + e.what()));
	}
}

int main() {
	ADD_API_FUNCTION(cg_cycles, "Dictionary", "PackedFloat32Array points, PackedInt32Array counts, float merge_epsilon",
			"CASSIE's sketch graph on raw strokes: strokes_in, strokes_added, nodes, edges, cycles, cycle_sizes");
	halt();
}
