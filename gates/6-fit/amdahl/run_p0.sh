#!/usr/bin/env bash
# One foxgirl phase-0 run of fit_native, capped, logged to runs/p0-<name>.log.
#
#   bash gates/6-fit/amdahl/run_p0.sh <name> [fit_native args...]
#
# Env: FIT_EXE (default the counters build C:/b/fit-native-amdahl/fit_native.exe),
#      CAP_S (timeout, default 180), OUT_ROOT (garment outputs, default scratch).
# Runs from the repo root (fit_native's default --setup is
# tools/native/foxgirl_oracle.json, relative to it).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
name="$1"; shift
EXE="${FIT_EXE:-/c/b/fit-native-amdahl/fit_native.exe}"
CAP="${CAP_S:-180}"
OUT="${OUT_ROOT:-/c/b/amdahl-out}/$name"
LOG="$HERE/runs/p0-$name.log"
mkdir -p "$HERE/runs" "$(dirname "$OUT")"
export PATH="/c/Users/ernest.lee/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64/bin:$PATH"
cd "$ROOT"
{
  echo "# run_p0.sh $name: $EXE --phases 1 --out $OUT $*  (cap ${CAP}s)  $(date -Iseconds)"
  t0=$(date +%s.%N)
  timeout "$CAP" "$EXE" --phases 1 --out "$OUT" "$@" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  echo "# exit $rc  wall $(python -c "print(round($t1-$t0,2))") s$( [ $rc -eq 124 ] && echo '  TIMEOUT')"
} > "$LOG" 2>&1
tail -1 "$LOG"
