#!/usr/bin/env bash
# The loop's authored skirt at several mesh_build edge lengths, for the fit
# budget: Gate 8's --gate=pen (stops after MESH) writes <out>.mesh.obj, the
# garment fit.elf gets. Usage: gates/6-fit/budget/pen_meshes.sh [edge...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
ROOT="$(cd "$HERE/../../.." && pwd -W 2>/dev/null || pwd)"
for e in "${@:-0.03 0.04 0.05}"; do
  out="$HERE/pen-$e.txt"
  rm -f "$out"
  timeout 300 godot --path "$ROOT/project" --script gate_loop.gd --rendering-driver vulkan --xr-mode off -- \
    --gate=pen --out="$out" --wallclock=120 --allow-fixture=infer,rig --mesh-edge="$e" \
    > "$HERE/pen-$e.log" 2>&1
  echo "edge $e rc $? $(grep -m1 'RESULT' "$out" 2>/dev/null) $(grep -o 'vertices=[0-9]* triangles=[0-9]*' "$out" | head -1)"
done
