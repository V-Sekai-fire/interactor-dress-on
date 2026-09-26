#!/usr/bin/env bash
# Vendor CASSIE's sketch graph (CassieSketchGraph: add_stroke + find_cycles, the
# port of CASSIE's CycleDetection.cs) from godot-cassie's module into
# vendor/cassie-graph/src, for cassie_graph.elf.
#
#   tools/vendor/cassie_graph_subset.sh
#
# Source: V-Sekai-fire/entities-godot at a4e33d895f (the workspace's
# 4-entities/godot-cassie checkout; override with GODOT_CASSIE),
# modules/cassie/src/sketch/cassie_sketch_graph.{h,cpp}, copied unedited. Its
# engine headers resolve to guest/godot_lite and vendor/godot-core-subset.
# curvenet.elf keeps vendor/cassie (an older revision), so Gate 4 is untouched.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GC="${GODOT_CASSIE:-$ROOT/../../4-entities/godot-cassie}"
PIN="a4e33d895f"
DST="$ROOT/vendor/cassie-graph/src/sketch"

have="$(git -C "$GC" rev-parse --short=10 HEAD)"
if [ "$have" != "$PIN" ] && [ "${ALLOW_OTHER_REV:-0}" != 1 ]; then
	echo "error: $GC is at $have, the graph is pinned to $PIN (ALLOW_OTHER_REV=1 to override)" >&2
	exit 1
fi
mkdir -p "$DST"
for f in cassie_sketch_graph.h cassie_sketch_graph.cpp; do
	git -C "$GC" show "$PIN:modules/cassie/src/sketch/$f" > "$DST/$f"
done
echo "vendored $(wc -l < "$DST/cassie_sketch_graph.cpp") + $(wc -l < "$DST/cassie_sketch_graph.h") lines from $PIN"
