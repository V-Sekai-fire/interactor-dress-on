#!/usr/bin/env bash
# The avbd fit (the pick, gate_fit_avbd.gd --only=f60-i32-s300) of every
# variant, one Godot process after another (staggered: AGENTS.md):
# avbd-<name>.txt / .json / .log and avbd-<name>-ladder-f60-i32-s300.fitted.obj.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
GODOT="${GODOT:-$HOME/scoop/apps/godot/current/godot.console.exe}"
for name in "$@"; do
	t0=$(date +%s)
	timeout 600 "$GODOT" --path "$ROOT/project" --script gate_fit_avbd.gd --rendering-driver vulkan --xr-mode off -- \
		--out="$HERE/avbd-$name.txt" --only=f60-i32-s300 --wallclock=300 --garment="$HERE/v-$name.obj" > "$HERE/avbd-$name.log" 2>&1
	rc=$?
	mv -f "$HERE/ladder-f60-i32-s300.fitted.obj" "$HERE/avbd-$name.fitted.obj" 2>/dev/null
	mv -f "$HERE/ladder-f60-i32-s300.png" "$HERE/avbd-$name.png" 2>/dev/null
	rm -f "$HERE/avbd-$name.authored.obj"
	echo "avbd $name rc=$rc wall_s=$(( $(date +%s) - t0 )) $(grep -o 'DONE fit[^|]*ms/step=[0-9.]*' "$HERE/avbd-$name.txt" | cut -c1-60)" | tee -a "$HERE/avbd-runs.txt"
	sleep 4
done
