#!/usr/bin/env bash
# L0 of the ggml-rd kernels: the Lean side.
#   lean-build.log             `lake build Ggml`: every native_decide pin in
#                              lean/Ggml holds (the kernels' text, MUL = ADD with
#                              one operator, the shared helpers, the generated
#                              ggml_rd_params.h), then kernels/ggml/gen.sh in its
#                              default mode: the committed slang/ and params
#                              header are exactly what Lean emits.
#   lean-negative-control.log  the controls, each must FAIL:
#                              (1) a pin with one character changed
#                                  (numthreads 256 -> 128) under native_decide;
#                              (1b) norm_f32's pin with its tree's first
#                                  step 128 -> 127 (the K3/K4 row kernels);
#                              (1c) MulMatTiled's pin with the group-memory row
#                                  padding changed (As[16][65] -> [16][64]);
#                              (2) gen.sh's check against a slang/ with one
#                                  character changed (256u -> 255u), run on a
#                                  copy of kernels/ggml so nothing committed is
#                                  touched.
# The last line of each log is RESULT: PASS or RESULT: FAIL.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LEAN="$ROOT/lean"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
WROOT="$(cygpath -m "$ROOT" 2>/dev/null || echo "$ROOT")"
WTMP="$(cygpath -m "$TMP" 2>/dev/null || echo "$TMP")"
clean() { sed -e "s#$ROOT#<checkout>#g" -e "s#$WROOT#<checkout>#g" -e "s#$TMP#<tmp>#g" -e "s#$WTMP#<tmp>#g"; }

rc=0
{
	echo "# ggml-rd L0, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "== lake build Ggml"
	if ( cd "$LEAN" && lake build Ggml 2>&1 ); then echo "PASS lake build Ggml"; else echo "FAIL lake build Ggml"; rc=1; fi
	echo "== kernels/ggml/gen.sh (check mode: Lean's emission vs the committed files)"
	if BUILD_DIR="$TMP/build" bash "$ROOT/kernels/ggml/gen.sh" 2>&1; then echo "PASS gen.sh check"; else echo "FAIL gen.sh check"; rc=1; fi
	echo "RESULT: $([ $rc = 0 ] && echo PASS || echo FAIL)"
} 2>&1 | clean > "$HERE/lean-build.log"
grep -q "^RESULT: PASS" "$HERE/lean-build.log" || rc=1

nrc=0
{
	echo "# ggml-rd L0 negative controls, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "== (1) native_decide on add_f32's pin with numthreads 256 -> 128 (must be rejected)"
	cat > "$TMP/NegGgml.lean" <<'EOF'
import Ggml.SlangCodegen.Binary
open Ggml.SlangCodegen.Binary

example : LeanSlang.emit addF32 =
    expectedAdd.replace "[numthreads(256, 1, 1)]" "[numthreads(128, 1, 1)]" := by
  native_decide
EOF
	if ( cd "$LEAN" && lake env lean "$TMP/NegGgml.lean" 2>&1 ); then
		echo "FAIL the changed pin was accepted"; nrc=1
	else
		echo "PASS the changed pin is rejected"
	fi
	echo "== (1b) native_decide on norm_f32's pin with the first tree step 128u -> 127u (must be rejected)"
	cat > "$TMP/NegGgmlNorm.lean" <<'EOF2'
import Ggml.SlangCodegen.Norm
open Ggml.SlangCodegen.Norm

example : LeanSlang.emit normF32 =
    expectedNorm.replace "sh[(t + 128u)]" "sh[(t + 127u)]" := by
  native_decide
EOF2
	if ( cd "$LEAN" && lake env lean "$TMP/NegGgmlNorm.lean" 2>&1 ); then
		echo "FAIL the changed norm pin was accepted"; nrc=1
	else
		echo "PASS the changed norm pin is rejected"
	fi
	echo "== (1c) native_decide on mul_mat_tiled_f16_f32's pin with As[16][65] -> As[16][64] (must be rejected)"
	cat > "$TMP/NegGgmlMm.lean" <<'EOF'
import Ggml.SlangCodegen.MulMatTiled
open Ggml.SlangCodegen.MulMatTiled
open Ggml.SlangCodegen.MulMat

example : LeanSlang.emit (shader .f16 .f32) =
    expected.replace "groupshared float As[16][65];" "groupshared float As[16][64];" := by
  native_decide
EOF
	if ( cd "$LEAN" && lake env lean "$TMP/NegGgmlMm.lean" 2>&1 ); then
		echo "FAIL the changed pin was accepted"; nrc=1
	else
		echo "PASS the changed pin is rejected"
	fi
	echo "== (2) gen.sh check against a copy of kernels/ggml whose add_f32.slang says 255u for 256u (must fail)"
	mkdir -p "$TMP/tree/kernels" "$TMP/tree/guest/ggml-rd"
	cp -r "$ROOT/kernels/ggml" "$TMP/tree/kernels/ggml"
	cp "$ROOT/kernels/embed_spv.py" "$TMP/tree/kernels/"
	cp "$ROOT/guest/ggml-rd/ggml_rd_params.h" "$TMP/tree/guest/ggml-rd/"
	sed -i 's/256u/255u/' "$TMP/tree/kernels/ggml/slang/add_f32.slang"
	if CLOTH_LEAN="$LEAN" BUILD_DIR="$TMP/build2" bash "$TMP/tree/kernels/ggml/gen.sh" 2>&1; then
		echo "FAIL the tampered copy passed the check"; nrc=1
	else
		echo "PASS the tampered copy is refused"
	fi
	echo "RESULT: $([ $nrc = 0 ] && echo PASS || echo FAIL)"
} 2>&1 | clean > "$HERE/lean-negative-control.log"
grep -q "^RESULT: PASS" "$HERE/lean-negative-control.log" || rc=1

tail -3 "$HERE/lean-build.log"
tail -3 "$HERE/lean-negative-control.log"
exit $rc
