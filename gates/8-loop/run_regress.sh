#!/usr/bin/env bash
# Cut 8 split main.gd into stages; the gates that ran before it must still
# pass on this tree. Runs Gate 1 (gate_rd_compute.gd), Gate 2 (gate_avbd.gd),
# Gate 4 (gate_curvenet.gd) and Gate 0F (gate_runtime.gd) and keeps their
# output under gates/8-loop/regress/. Gates 2, 4 and 0F write their own
# results files in gates/2-avbd/, gates/4-curvenet/ and gates/0f-runtime/;
# those are copied here and the
# committed evidence is put back (git checkout), so this run does not
# rewrite another gate's record.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GODOT="${GODOT:-$HOME/scoop/apps/godot/current/godot.console.exe}"
OUT="$HERE/regress"
mkdir -p "$OUT"
cd "$ROOT"

run() { # name script timeout_s
	local t0=$(date +%s)
	timeout "$3" "$GODOT" --path project --script "$2" --rendering-driver vulkan --xr-mode off > "$OUT/$1.log" 2>&1
	local rc=$?
	echo "$1: rc=$rc $(( $(date +%s) - t0 )) s" | tee -a "$OUT/summary.txt"
}

: > "$OUT/summary.txt"
run gate_rd_compute gate_rd_compute.gd 300
grep -a 'RESULT' "$OUT/gate_rd_compute.log" | tee -a "$OUT/summary.txt"
run gate_avbd gate_avbd.gd 700
cp gates/2-avbd/results.txt "$OUT/gate_avbd.results.txt"
tail -n 1 "$OUT/gate_avbd.results.txt" | tee -a "$OUT/summary.txt"
run gate_curvenet gate_curvenet.gd 400
cp gates/4-curvenet/results.txt "$OUT/gate_curvenet.results.txt"
tail -n 1 "$OUT/gate_curvenet.results.txt" | tee -a "$OUT/summary.txt"
run probe_curvenet_wrappers probe_curvenet_wrappers.gd 150
grep -ac "() -> " "$OUT/probe_curvenet_wrappers.log" | sed "s/^/probe_curvenet_wrappers calls: /" | tee -a "$OUT/summary.txt"
run gate_runtime gate_runtime.gd 2000
cp gates/0f-runtime/results.txt "$OUT/gate_runtime.results.txt"
grep -a 'SUMMARY' "$OUT/gate_runtime.results.txt" | tee -a "$OUT/summary.txt"
git checkout -- gates/2-avbd/results.txt gates/4-curvenet/results.txt gates/0f-runtime/results.txt gates/0f-runtime/ggml_host.txt 2>/dev/null
echo "done" >> "$OUT/summary.txt"
