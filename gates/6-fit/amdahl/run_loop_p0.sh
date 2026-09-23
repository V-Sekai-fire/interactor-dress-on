#!/usr/bin/env bash
# The loop skirt's fit phase 0 (the AL phase), native, default solver config.
#
#   bash gates/6-fit/amdahl/run_loop_p0.sh samply    # runs/loop-p0-direct.samply.json.gz + -samply.log
#   bash gates/6-fit/amdahl/run_loop_p0.sh counters  # runs/loop-p0-direct-counters.log
#
# Input: the 932-vertex authored skirt and the loop's setup, which live in the
# parked budget study (branch cut-6c: gates/6-fit/budget/pen-0.03.mesh.obj and
# loop_setup.json); LOOP_ROOT is a checkout of that branch.
# Env: LOOP_ROOT (cut-6c checkout), SAMPLY (samply.exe), FIT_EXE (the fit.elf
# twin, default C:/b/fit-native-tbbs), FIT_COUNTERS (the counters build of
# build_counters.sh, default C:/b/fit-native-amdahl), OUT_ROOT (garment outputs).
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
RUNS="$HERE/runs"
: "${LOOP_ROOT:?set LOOP_ROOT to a checkout of cut-6c}"
SAMPLY="${SAMPLY:-samply}"
EXE="${FIT_EXE:-C:/b/fit-native-tbbs/fit_native.exe}"
CNT="${FIT_COUNTERS:-C:/b/fit-native-amdahl/fit_native.exe}"
OUT="${OUT_ROOT:-C:/b/amdahl-out}"
export PATH="/c/Users/ernest.lee/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64/bin:$PATH"
ARGS=(--root "$LOOP_ROOT" --setup gates/6-fit/budget/loop_setup.json --phases 1)
mkdir -p "$RUNS"
case "${1:-samply}" in
samply)
  LOG="$RUNS/loop-p0-direct-samply.log"
  {
    echo "# $EXE ${ARGS[*]} (samply)  $(date -Iseconds)"
    timeout 600 "$SAMPLY" record --save-only --unstable-presymbolicate \
      -o "$RUNS/loop-p0-direct.samply.json.gz" -- "$EXE" "${ARGS[@]}" --out "$OUT/loop-p0-direct" 2>&1
    echo "# exit $?"
  } > "$LOG" 2>&1 ;;
counters)
  LOG="$RUNS/loop-p0-direct-counters.log"
  {
    echo "# $CNT ${ARGS[*]}  $(date -Iseconds)"
    timeout 600 "$CNT" "${ARGS[@]}" --out "$OUT/loop-p0-counters" 2>&1
    echo "# exit $?"
  } > "$LOG" 2>&1 ;;
*) echo "usage: $0 samply|counters" >&2; exit 2 ;;
esac
grep -aE "^phase|^final|^time|^# exit" "$LOG"
