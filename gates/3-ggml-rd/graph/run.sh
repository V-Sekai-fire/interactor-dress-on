#!/usr/bin/env bash
# Gate 3 G3.graph + G3.cost: build the host oracle, then run the gate.
#   gates/3-ggml-rd/graph/run.sh                       every run -> graph/results.txt
#   gates/3-ggml-rd/graph/run.sh runs=graph_dit8        extra user arguments go to the gate
# The guest runs ggml-rd only; tests/ggml_graph_oracle is the reference
# (project/gate_ggml_graph.gd says what runs where). Godot's stdout goes to
# graph/run.log; the verdicts are in the results file, not the log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
ORACLE="$(bash "$ROOT/tests/ggml_graph_oracle/build.sh" | sed -n 's/^oracle: //p')"
[ -x "$ORACLE" ] || { echo "error: no oracle at '$ORACLE'" >&2; exit 1; }
OUT=graph
for a in "$@"; do case "$a" in --out=*) OUT="${a#--out=}" ;; esac; done
mkdir -p "$ROOT/gates/3-ggml-rd/$OUT"
godot --path "$ROOT/project" --script gate_ggml_graph.gd --rendering-driver vulkan --xr-mode off \
	++ --oracle="$(cygpath -m "$ORACLE")" "$@" > "$ROOT/gates/3-ggml-rd/$OUT/run.log" 2>&1 || true
tail -12 "$ROOT/gates/3-ggml-rd/$OUT/results.txt"
grep -q '^RESULT: PASS' "$ROOT/gates/3-ggml-rd/$OUT/results.txt"
