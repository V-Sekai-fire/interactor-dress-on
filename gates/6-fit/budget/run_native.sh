#!/usr/bin/env bash
# One fit budget candidate, native (fit_native: fit.elf's code and numerics).
#   gates/6-fit/budget/run_native.sh NAME [fit_native args, e.g. --set /pointer=json ...]
# Setup: gates/6-fit/budget/loop_setup.json (the loop's: FoxGirl fixture body
# and skeleton, the authored skirt from pen_meshes.sh, fit_config.json with
# incremental_steps 1). Full debug log and garment in $BUDGET_OUT/NAME
# (default C:/b/budget); gates/6-fit/budget/NAME.log keeps the phase lines,
# polysolve's timing lines and the final summary.
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
ROOT="$(cd "$HERE/../../.." && pwd -W 2>/dev/null || pwd)"
EXE="${FIT_NATIVE:-C:/b/fit-native-6c/fit_native.exe}"
OUT="${BUDGET_OUT:-C:/b/budget}"
name="$1"; shift
mkdir -p "$OUT/$name"
"$EXE" --root "$ROOT" --setup gates/6-fit/budget/loop_setup.json --out "$OUT/$name" "$@" \
  > "$OUT/$name/run.log" 2>&1
rc=$?
{
  echo "# fit_native $* (rc $rc)"
  grep -E "^(inputs|begin|phase |final|time|sdf grid|normalisation)|timing\]|Finished:|Reached|\[Newton\]\[.*\] (Assembly|Inverting|Line search)|assembly|inverting|checking for intersections|CCD|broad phase" "$OUT/$name/run.log" \
    | sed 's/\x1b\[[0-9;]*m//g' | grep -vE "iter=|pre LS" | head -400
} > "$HERE/$name.log"
grep -E "^(phase |final|time)" "$HERE/$name.log"
