#!/usr/bin/env bash
# Gate U, end to end: the host oracle, the memory ladder (one Godot process a
# rung, since a rung below the floor may kill Godot), the gate, and the rule-8
# probe. Run from the repo root. Every Godot run: --rendering-driver vulkan
# --xr-mode off, launches staggered by a few seconds (AGENTS.md).
#
#   bash gates/U-usd/run.sh              # everything
#   RUNGS="96 128" bash gates/U-usd/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GODOT="${GODOT:-$HOME/scoop/apps/godot/current/godot.console.exe}"
PY="${PY:-$HOME/scoop/apps/python/current/python}"
RUNGS="${RUNGS:-64 96 128 160 192 256 320 384 512}"
cd "$ROOT"

echo "== host oracle"
timeout 300 "$PY" gates/U-usd/host_oracle.py > gates/U-usd/host-oracle.log 2>&1
echo "exit=$?"; head -3 gates/U-usd/host-oracle.log

echo "== ladder"
mkdir -p gates/U-usd/ladder
for r in $RUNGS; do
	timeout 120 "$GODOT" --path project --script gate_usdz.gd --rendering-driver vulkan --xr-mode off -- --rung="$r" \
		> "gates/U-usd/ladder/$r.log" 2>&1
	echo "rung $r: exit=$? $(grep -c '^PASS' "gates/U-usd/ladder/$r.txt" 2>/dev/null || echo 0) PASS $(grep -c '^FAIL' "gates/U-usd/ladder/$r.txt" 2>/dev/null || echo 0) FAIL: $(tail -1 "gates/U-usd/ladder/$r.txt" 2>/dev/null)"
	sleep 3
done

echo "== gate"
timeout 320 "$GODOT" --path project --script gate_usdz.gd --rendering-driver vulkan --xr-mode off > gates/U-usd/run.log 2>&1
echo "exit=$?"; tail -1 gates/U-usd/results.txt
sleep 3

echo "== rule 8"
timeout 150 "$GODOT" --path project --script tests/probe_main_wrappers.gd --rendering-driver vulkan --xr-mode off > gates/U-usd/wrappers.log 2>&1
echo "exit=$?"; tail -1 gates/8-loop/wrappers.txt
cp gates/8-loop/wrappers.txt gates/U-usd/wrappers.txt
