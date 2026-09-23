#!/usr/bin/env bash
# One fit budget candidate in the guest: Gate 8's flat loop (FIXTURE infer,
# rig), so the fit is the loop's own, then CHECK and DRAPE.
#   gates/6-fit/budget/run_guest.sh NAME [gate_loop args, e.g. --fit-elf=res://fit_prof.elf --fit-force-psd=0]
# Results: gates/6-fit/budget/guest-NAME.{txt,json,png,log,mesh.obj,fitted.obj}.
# One Godot GPU gate at a time on the shared machine; the gate quits on its
# own wall clock (--wallclock, default 3600 s here).
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
ROOT="$(cd "$HERE/../../.." && pwd -W 2>/dev/null || pwd)"
name="$1"; shift
out="$HERE/guest-$name.txt"
rm -f "$out"
timeout 4000 godot --path "$ROOT/project" --script gate_loop.gd --rendering-driver vulkan --xr-mode off -- \
  --gate=loop --out="$out" --wallclock=3600 --allow-fixture=infer,rig "$@" > "$HERE/guest-$name.log" 2>&1
echo "rc $? $(grep -m1 '^RESULT' "$out")"
