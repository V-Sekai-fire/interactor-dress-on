#!/usr/bin/env bash
# The native PolyFEM fit (gates/6-fit's fit_native, the guest's bitwise-equal
# control, ~20x faster) of every variant, one after another, each under
# timeout 900: native-<name>/ (garment_final.obj in the solve frame, the
# normalisation line in native-<name>.log). FIT_NATIVE defaults to the
# SdfGrid kernel build of Gate 6.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_NATIVE="${FIT_NATIVE:-/c/b/fit-native-sdfk/fit_native.exe}"
for name in "$@"; do
	out="$HERE/native-$name"
	mkdir -p "$out"
	t0=$(date +%s)
	timeout 900 "$FIT_NATIVE" --setup "$HERE/setup-$name.json" --root / --out "$out" > "$HERE/native-$name.log" 2>&1
	rc=$?
	echo "native $name rc=$rc wall_s=$(( $(date +%s) - t0 ))" | tee -a "$HERE/native-runs.txt"
done
