#!/usr/bin/env bash
# Gate 6g.C1: SimilarityForm's Hessian on the GPU (cut 6g-C, stage 1), foxgirl, psd.
#
#   bash gates/6g-polyfem-gpu/c1-hessian/run.sh            # all four arms, in sequence
#   bash gates/6g-polyfem-gpu/c1-hessian/run.sh check      # one arm: check | gpu | cpu | twin
#
# Arms (each its own Godot process, project/gate_fit.gd; logs under runs/):
#   check  fit_begin -> fit_gpu_check -> phase 0 -> fit_gpu_check: the kernels
#          against the CPU double path (blocks, 1e-9), the cpp twin against
#          the device (1e-12), the flipped-sign control, the assembled pattern
#   gpu    every phase with the Hessian on the device (--gpu=1)
#   cpu    every phase on the CPU path (--gpu=0): the flat control, bitwise
#          the fit before this cut
#   twin   phase 0 with the cpp twin (--gpu=2): the kernels' second target
# Every Godot run: --rendering-driver vulkan --xr-mode off; launches are
# staggered by the sequence itself.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$ROOT/gates/6g-polyfem-gpu/c1-hessian"
mkdir -p "$HERE/runs"
GODOT="${GODOT:-$HOME/scoop/apps/godot/current/godot.console.exe}"
arm() {
	local tag="$1"; shift
	echo "== $tag $(date +%H:%M:%S)"
	"$GODOT" --path "$ROOT/project" --script gate_fit.gd --rendering-driver vulkan --xr-mode off \
		-- --tag="c1-$tag" "$@" > "$HERE/runs/$tag.godot.log" 2>&1
	echo "   exit $?"
	cp "$ROOT/gates/6-fit/runs/c1-$tag.txt" "$HERE/runs/$tag.txt" 2>/dev/null
	grep -E "^(ok|FAIL|RESULT|summary)" "$HERE/runs/$tag.txt" | cut -c1-200 | tail -20
	sleep 3
}
case "${1:-all}" in
	check) arm check --arm=gpucheck --psd --gpu=1 --broad=0 ;;
	gpu)   arm gpu --arm=solve --psd --gpu=1 --broad=0 --wall=7200 ;;
	cpu)   arm cpu --arm=solve --psd --gpu=0 --broad=0 --wall=7200 ;;
	twin)  arm twin --arm=solve --psd --gpu=2 --broad=0 --phases=1 --wall=7200 ;;
	all)
		arm check --arm=gpucheck --psd --gpu=1 --broad=0
		arm gpu --arm=solve --psd --gpu=1 --broad=0 --wall=7200
		arm cpu --arm=solve --psd --gpu=0 --broad=0 --wall=7200
		arm twin --arm=solve --psd --gpu=2 --broad=0 --phases=1 --wall=7200
		;;
	*) echo "arm: check | gpu | cpu | twin | all" >&2; exit 2 ;;
esac
