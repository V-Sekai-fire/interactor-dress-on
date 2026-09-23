#!/usr/bin/env bash
# Gate 5 G5's pre-chaos native reference (Cut 5c): the same DiffCloth sphere
# demo as ../run_native.sh (AVBD build of cloth-dynamics-standalone at
# e361584, seed 1), but over STEPS steps (default 60) instead of 350, and with
# every objective evaluation's mu, loss and dL/dmu printed at full precision.
#
#   gates/5-drape/native/short/run_native_short.sh           STEPS=60
#   STEPS=40 gates/5-drape/native/short/run_native_short.sh
#
# The tool has no step-count option: rotatingSphereScene.stepNum is a
# constant, and backwardLog.txt prints dL/dmu with 5 decimals (at 60 steps the
# loss is ~1e-6, so that print would be 0.0000x). So this script does not
# touch the standalone checkout or its build-win/. It copies two of its
# translation units into build/native_short/, patches the copies
#   OptimizationTaskConfigurations.cpp: rotatingSphereScene .stepNum = STEPS
#   OptimizeHelper.cpp: after "Loss is:", one line
#       [g5-native] mu=<%.17g> loss=<%.17g> dL/dmu=<%.17g>
# compiles them with build-win's own flags (ninja -t commands), links them
# with build-win's other objects into build/native_short/tool_sN.exe, and
# runs that from the standalone root (where it finds src/assets and writes
# output/). Everything else is the recorded reference binary's code.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
S="${STANDALONE:-C:/cloth-dynamics-standalone}"
STEPS="${STEPS:-60}"
NINJA="${NINJA:-$HOME/.pixi/bin/ninja.exe}"
B="$ROOT/build/native_short"
OUT="$HERE/s$STEPS"

if env | grep -q '^AVBD_'; then
	echo "AVBD_* is set in the environment; unset it first" >&2
	exit 1
fi
[ "$(git -C "$S" rev-parse HEAD)" = e361584c6e52a1b6f2635b93a8dff80eafc00052 ] || { echo "standalone is not at e361584" >&2; exit 1; }
"$NINJA" -C "$S/build-win" -n tool_cloth_dynamics | grep -q "no work to do" || { echo "build-win is stale" >&2; exit 1; }
mkdir -p "$B" "$OUT"
CMDS="$("$NINJA" -C "$S/build-win" -t commands tool_cloth_dynamics)"

# The compiler path comes back with backslashes, which eval would eat (the
# -D SOURCE_PATH=\"...\" escapes elsewhere on the line must stay).
slashcc() {
	local first="${1%% *}"
	echo "${first//\\//} ${1#* }"
}

compile() { # <tu path under src/code> <patched source>
	local tu="$1" src="$2" line obj
	line="$(echo "$CMDS" | grep -F -- "-c C:/cloth-dynamics-standalone/src/code/$tu" | tail -1)"
	[ -n "$line" ] || { echo "no compile command for $tu" >&2; exit 1; }
	obj="$B/$(basename "$tu").obj"
	# Same flags; the patched copy compiles from build/, so its own directory
	# joins the quote-include path first.
	line="${line/-c C:\/cloth-dynamics-standalone\/src\/code\/$tu/-iquote $S/src/code/$(dirname "$tu") -c $src}"
	line="$(echo "$line" | sed -E "s# -o [^ ]+\.obj # -o $obj #; s# -MD -MT [^ ]+ -MF [^ ]+##")"
	(cd "$S/build-win" && eval "$(slashcc "$line")")
	echo "$obj"
}

sed -E "/rotatingSphereScene = \{/,/\};/ s/\.stepNum = 350,/.stepNum = $STEPS,/" \
	"$S/src/code/optimization/OptimizationTaskConfigurations.cpp" > "$B/OptimizationTaskConfigurations.cpp"
grep -A14 "rotatingSphereScene = {" "$B/OptimizationTaskConfigurations.cpp" | grep -q "\.stepNum = $STEPS," || { echo "stepNum patch did not apply" >&2; exit 1; }
sed -E 's|^(\tLogging::logOk\("Loss is: " \+ d2str\(loss, 8\) \+ "\\n"\);)$|\1\n\tstd::printf("[g5-native] mu=%.17g loss=%.17g dL/dmu=%.17g\\n", x[0], loss, grad[0]);|' \
	"$S/src/code/optimization/OptimizeHelper.cpp" > "$B/OptimizeHelper.cpp"
grep -q "g5-native" "$B/OptimizeHelper.cpp" || { echo "print patch did not apply" >&2; exit 1; }

O1="$(compile optimization/OptimizationTaskConfigurations.cpp "$B/OptimizationTaskConfigurations.cpp")"
O2="$(compile optimization/OptimizeHelper.cpp "$B/OptimizeHelper.cpp")"
LINK="$(echo "$CMDS" | tail -1 | sed -E 's#^.*cmd.exe /C "cd \. && ##; s# && cd \."$##')"
LINK="${LINK//CMakeFiles\/tool_cloth_dynamics.dir\/src\/code\/optimization\/OptimizationTaskConfigurations.cpp.obj/$O1}"
LINK="${LINK//CMakeFiles\/tool_cloth_dynamics.dir\/src\/code\/optimization\/OptimizeHelper.cpp.obj/$O2}"
LINK="$(echo "$LINK" | sed -E "s# -o tool_cloth_dynamics\.exe # -o $B/tool_s$STEPS.exe #; s# -Wl,--out-implib,[^ ]+##")"
(cd "$S/build-win" && eval "$(slashcc "$LINK")")
echo "built $B/tool_s$STEPS.exe"

cd "$S"
before="$(ls -1 output)"
start=$(date +%s)
PATH="$S/build-win:$PATH" timeout 1800 "$B/tool_s$STEPS.exe" -demo sphere -seed 1 > "$OUT/stdout.log" 2>&1 || { echo "exit $?" >&2; exit 1; }
echo "wall_s $(( $(date +%s) - start ))"
FROM="$(comm -13 <(echo "$before") <(ls -1 output) | grep '^rotating_sphere-' | tail -1)"
D="$S/output/$FROM"
for f in backwardLog.txt task_info.txt scene-config.txt iters.txt; do
	cp "$D/$f" "$OUT/"
done
echo "$FROM" > "$OUT/source_dir.txt"
# (the logger's colour escapes precede the line)
grep -ao "\[g5-native\].*" "$OUT/stdout.log" | tee "$OUT/evaluations.txt"
