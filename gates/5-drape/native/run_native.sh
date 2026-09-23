#!/usr/bin/env bash
# Native reference for Gate 5 (plan Cut 5 "Native reference"): the upstream
# DiffCloth sphere demo, AVBD build of cloth-dynamics-standalone, seed 1.
#
#   gates/5-drape/native/run_native.sh            run the demo, then collect
#   FROM=<output dir name> run_native.sh          collect an existing run only
#
# The tool must run from the standalone repo root: it resolves src/assets/ and
# writes output/<experiment>/ relative to it. AVBD_* environment variables
# change its solver config ([avbd-config] line); none may be set.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="${STANDALONE:-C:/cloth-dynamics-standalone}"
EXE="${EXE:-build-win/tool_cloth_dynamics.exe}"

if env | grep -q '^AVBD_'; then
    echo "AVBD_* is set in the environment; unset it first" >&2
    exit 1
fi
cd "$S" || exit 1

if [ -z "${FROM:-}" ]; then
    before="$(ls -1 output)"
    start=$(date +%s)
    timeout 1800 "./$EXE" -demo sphere -seed 1 > "$HERE/stdout.log" 2>&1
    rc=$?
    echo "exit $rc wall_s $(( $(date +%s) - start ))"
    [ $rc -ne 0 ] && exit $rc
    FROM="$(comm -13 <(echo "$before") <(ls -1 output) | grep '^rotating_sphere-' | tail -1)"
fi
D="$S/output/$FROM"
[ -d "$D/iter0" ] || { echo "no $D/iter0" >&2; exit 1; }
echo "collecting $D"

rm -rf "$HERE/iter0" "$HERE/iter1" "$HERE/iter2" "$HERE/iter3"
mkdir -p "$HERE/iter0"
for f in 0 1 10 50 100 350 1-SPHERE; do cp "$D/iter0/$f.obj" "$HERE/iter0/"; done
cp "$D/iter0/param.txt" "$HERE/iter0/"
for i in 1 2 3; do
    [ -f "$D/iter$i/param.txt" ] && mkdir -p "$HERE/iter$i" && cp "$D/iter$i/param.txt" "$HERE/iter$i/"
done
for f in backwardLog.txt forwardLog.txt task_info.txt scene-config.txt iters.txt perf.txt; do
    cp "$D/$f" "$HERE/"
done
echo "$FROM" > "$HERE/source_dir.txt"
