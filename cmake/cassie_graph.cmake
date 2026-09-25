# cassie_graph.elf: CASSIE's sketch graph from godot-cassie's module
# (vendor/cassie-graph, tools/vendor/cassie_graph_subset.sh) through the same
# godot_lite shim as curvenet. A library of its own, since vendor/cassie holds
# an older CassieSketchGraph under the same name.
set(CASSIE_GRAPH_SRC "${DRESS_ON_ROOT}/vendor/cassie-graph/src")
add_library(cassie_graph_core STATIC EXCLUDE_FROM_ALL
	"${CASSIE_GRAPH_SRC}/sketch/cassie_sketch_graph.cpp"
	"${DRESS_ON_ROOT}/guest/cassie_graph/graph_api.cpp"
)
target_include_directories(cassie_graph_core PUBLIC
	"${CASSIE_GRAPH_SRC}"
	"${DRESS_ON_ROOT}/guest/cassie_graph"
	"${DRESS_ON_ROOT}/guest/godot_lite"
	"${DRESS_ON_ROOT}/vendor/godot-core-subset"
)
target_compile_options(cassie_graph_core PRIVATE
	"SHELL:-include ${DRESS_ON_ROOT}/guest/godot_lite/gdl_prelude.h"
	${CURVENET_STRICT_FP}
)
target_link_libraries(cassie_graph_core PUBLIC $<TARGET_NAME_IF_EXISTS:godot_lite>)

if(COMMAND add_stage_elf)
	add_stage_elf(cassie_graph "${DRESS_ON_ROOT}/guest/cassie_graph/main.cpp")
	target_include_directories(cassie_graph PRIVATE "${DRESS_ON_ROOT}/guest/cassie_graph")
	target_link_libraries(cassie_graph PRIVATE cassie_graph_core)
endif()
