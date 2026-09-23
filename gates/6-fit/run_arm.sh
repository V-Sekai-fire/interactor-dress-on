#!/usr/bin/env bash
# One gate_fit.gd arm in its own Godot process (Gate 6). Arms are independent
# (a fresh fit Sandbox each), so several can run at once; the per-arm log is
# gates/6-fit/runs/<tag>.txt, flushed per line, and Godot's own stdout goes to
# runs/<tag>.godot.log (Godot buffers redirected stdout: read the .txt).
#
#   bash gates/6-fit/run_arm.sh TAG --arm=solve --phases=1 [--mem=N] [--elf=res://fit_zb.elf] [--fitweight0]
#   bash gates/6-fit/run_arm.sh probes --arm=probes
#   bash gates/6-fit/run_arm.sh report --arm=report
#
# GODOT (default: godot on PATH) is Godot 4.7.2; Vulkan, no XR.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="$1"
shift
mkdir -p "$ROOT/gates/6-fit/runs"
rc=0
"${GODOT:-godot}" --path "$ROOT/project" --script gate_fit.gd --rendering-driver vulkan --xr-mode off \
	-- --tag="$TAG" "$@" > "$ROOT/gates/6-fit/runs/$TAG.godot.log" 2>&1 || rc=$?
# 0 = PASS, 1 = FAIL (the arm said so). Anything else is the Godot process
# dying (139: a segfault, as below the heap floor, Gate 6.P); the arm cannot
# write that itself, so it goes into its log here.
if [ "$rc" -gt 1 ]; then
	echo "EXIT $rc: the Godot process died (no RESULT line from the arm)" >> "$ROOT/gates/6-fit/runs/$TAG.txt"
fi
exit "$rc"
