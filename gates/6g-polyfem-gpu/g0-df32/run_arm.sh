#!/usr/bin/env bash
# One capped fit_native run of the Gate 6g.0 scratch build.
#
#   bash run_arm.sh <name> <phases> [ENV=VALUE ...]
#
# Runs from the gate worktree root (fit_native's default --setup is
# tools/native/foxgirl_oracle.json); log -> $LOGS/<name>.log, garment ->
# $OUT/<name>/, g6g report appended to the log, CCD step TSV and the last
# displaced collision surface next to the garment.
set -uo pipefail
name="$1"; phases="$2"; shift 2
EXE="${G6G_EXE:-C:/b/g6g0/fit_native.exe}"
REPO="${G6G_REPO:-C:/Users/ernest.lee/AppData/Local/Temp/claude/C--interactor-dress-on/5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5/scratchpad/ido-6g0}"
LOGS="${LOGS:-C:/b/g6g0-logs}"
OUT="${OUT:-C:/b/g6g0-out}"
CAP="${CAP_S:-300}"
mkdir -p "$LOGS" "$OUT/$name"
export PATH="/c/Users/ernest.lee/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64/bin:$PATH"
cd "$REPO"
{
  echo "# g6g0 $name: $EXE --phases $phases --out $OUT/$name  env: $*  (cap ${CAP}s)  $(date -Iseconds)"
  t0=$(date +%s.%N)
  env G6G_CCD_TSV="$OUT/$name/ccd.tsv" G6G_SURFACE="$OUT/$name/surface.obj" "$@" \
    timeout "$CAP" "$EXE" --phases "$phases" --out "$OUT/$name" ${EXTRA_ARGS:-} 2>&1
  rc=$?
  t1=$(date +%s.%N)
  echo "# exit $rc  wall $(python -c "print(round($t1-$t0,2))") s$( [ $rc -eq 124 ] && echo '  TIMEOUT')"
} > "$LOGS/$name.log" 2>&1
tail -1 "$LOGS/$name.log"
