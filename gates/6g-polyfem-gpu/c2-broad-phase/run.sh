#!/usr/bin/env bash
# Gate 6g.C2: the CCD broad phase on the GPU (cut 6g-C, stage 2), foxgirl, psd.
#
#   bash gates/6g-polyfem-gpu/c2-broad-phase/run.sh            # all arms
#   bash gates/6g-polyfem-gpu/c2-broad-phase/run.sh audit      # one arm: audit | gpu | cpu
#
# Arms (each its own Godot process, project/gate_fit.gd; logs under runs/):
#   audit  phase 0 with the broad phase on the device and every build audited
#          by ipc-toolkit's build on the CPU (--broad=2): missed, extra, time
#          per build both ways; the Hessian on the device too (--gpu=1)
#   gpu    every phase with both GPU paths on (--gpu=1 --broad=1): convergence,
#          intersections, the garment for the fit-gap check
#   cpu    every phase on the CPU paths (--gpu=0 --broad=0): the flat control
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$ROOT/gates/6g-polyfem-gpu/c2-broad-phase"
mkdir -p "$HERE/runs"
GODOT="${GODOT:-$HOME/scoop/apps/godot/current/godot.console.exe}"
arm() {
	local tag="$1"; shift
	echo "== $tag $(date +%H:%M:%S)"
	"$GODOT" --path "$ROOT/project" --script gate_fit.gd --rendering-driver vulkan --xr-mode off \
		-- --tag="c2-$tag" "$@" > "$HERE/runs/$tag.godot.log" 2>&1
	echo "   exit $?"
	cp "$ROOT/gates/6-fit/runs/c2-$tag.txt" "$HERE/runs/$tag.txt" 2>/dev/null
	cp "$ROOT/gates/6-fit/runs/c2-$tag.garment.f64" "$HERE/runs/$tag.garment.f64" 2>/dev/null
	grep -E "^(ok|FAIL|RESULT|summary)|broad phase:|gpu:" "$HERE/runs/$tag.txt" | cut -c1-220 | tail -24
	sleep 3
}
case "${1:-all}" in
	audit) arm audit --arm=solve --psd --gpu=1 --broad=2 --phases=1 --wall=7200 ;;
	gpu)   arm gpu --arm=solve --psd --gpu=1 --broad=1 --wall=7200 ;;
	cpu)   arm cpu --arm=solve --psd --gpu=0 --broad=0 --wall=7200 ;;
	all)
		arm audit --arm=solve --psd --gpu=1 --broad=2 --phases=1 --wall=7200
		arm gpu --arm=solve --psd --gpu=1 --broad=1 --wall=7200
		arm cpu --arm=solve --psd --gpu=0 --broad=0 --wall=7200
		;;
	*) echo "arm: audit | gpu | cpu | all" >&2; exit 2 ;;
esac
