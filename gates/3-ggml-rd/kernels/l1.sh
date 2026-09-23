#!/usr/bin/env bash
# L1 of the ggml-rd kernels, after kernels/ggml/gen.sh (which already ran
# spirv-val on every kernel):
#   - the fixed-layout check on every kernel in kernels.txt (must pass), and
#     its negative controls (must fail): add_f32 compiled at slangc's default
#     -O1, which drops the unused s2 from the SPIR-V while the reflection
#     JSON still lists it; and kernels/avbd's saxpby, another layout;
#   - every cpp emit compiled for riscv64 by the guest's clang (the x86
#     compile is the L2 harness, tests/ggml_rd_kernels). A kernel with
#     workgroup barriers has no cpp emit (gen.sh); its <k>_serial sibling,
#     which must exist, is compiled in its place.
# Writes l1.log beside this script; the last line is RESULT: PASS or FAIL.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
SLANGC="${SLANGC:-slangc}"
command -v "$SLANGC" >/dev/null 2>&1 || SLANGC="$HOME/scoop/apps/vulkan/current/Bin/slangc"
SYSROOT="${RISCV64_SYSROOT:-/c/contract-manifest/3-interactor/mujoco-sandbox-demo/third_party/riscv64-sysroot}"
CLANG="${RV_CLANG:-$HOME/scoop/apps/llvm/current/bin/clang++}"
TABLE="$ROOT/kernels/ggml/gen_ggml_kernel_table.py"
KERNELS=$(grep -v '^#' "$ROOT/kernels/ggml/kernels.txt" | grep -v '^[[:space:]]*$' | tr '\n' ' ')
TMP="$(mktemp -d)"
rc=0
{
	echo "# ggml-rd L1, $(date -u +%Y-%m-%dT%H:%M:%SZ), $("$SLANGC" -version 2>&1 | head -1)"
	echo "== fixed-layout check on the built kernels (must pass): $KERNELS"
	if python "$TABLE" --spv-dir "$BUILD/spv-ggml" --check-only $KERNELS; then echo "PASS layout"; else echo "FAIL layout"; rc=1; fi

	echo "== negative: add_f32 at slangc's default -O1 with -preserve-params (must fail)"
	( cd "$ROOT/kernels/ggml" && "$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main -preserve-params \
		-reflection-json "$TMP/add_f32_o1.refl.json" -o "$TMP/add_f32_o1.spv" slang/add_f32.slang )
	if python "$TABLE" --spv-dir "$TMP" --check-only add_f32_o1; then echo "FAIL the -O1 build passed the check"; rc=1; else echo "PASS the -O1 build is refused"; fi

	echo "== negative: kernels/avbd saxpby, another layout (must fail)"
	( cd "$ROOT/kernels/avbd" && "$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main -O0 -preserve-params \
		-reflection-json "$TMP/saxpby.refl.json" -o "$TMP/saxpby.spv" slang/saxpby.slang )
	if python "$TABLE" --spv-dir "$TMP" --check-only saxpby; then echo "FAIL saxpby passed the check"; rc=1; else echo "PASS saxpby is refused"; fi

	echo "== cpp emits compiled for riscv64 ($("$CLANG" --version | head -1))"
	for k in $KERNELS; do
		if grep -q 'GroupMemoryBarrierWithGroupSync' "$ROOT/kernels/ggml/slang/$k.slang"; then
			if [ -f "$ROOT/kernels/ggml/cpp/${k}_serial_emit.cpp" ] && [ ! -f "$ROOT/kernels/ggml/cpp/${k}_emit.cpp" ]; then
				echo "SKIP riscv64 $k (workgroup barriers, no cpp emit; ${k}_serial is compiled instead)"
			else
				echo "FAIL $k: workgroup barriers need ${k}_serial's emit and no emit of their own"
				rc=1
			fi
			continue
		fi
		if "$CLANG" --target=riscv64-unknown-linux-gnu --sysroot="$SYSROOT/sysroot" -march=rv64gc -mabi=lp64d \
			-isystem "$SYSROOT/sysroot/include/c++/14" -isystem "$SYSROOT/sysroot/include/c++/14/riscv64-linux-gnu" \
			-std=c++17 -O2 -w -c "$ROOT/kernels/ggml/cpp/${k}_emit.cpp" -o "$TMP/${k}_rv64.o"; then
			echo "PASS riscv64 $k ($(wc -c < "$TMP/${k}_rv64.o") bytes of object)"
		else
			echo "FAIL riscv64 $k"
			rc=1
		fi
	done
	echo "RESULT: $([ $rc = 0 ] && echo PASS || echo FAIL)"
} > "$HERE/l1.log" 2>&1
rm -rf "$TMP"
sed "s#$ROOT#<checkout>#g" "$HERE/l1.log" > "$HERE/l1.log.tmp" && mv "$HERE/l1.log.tmp" "$HERE/l1.log"
cat "$HERE/l1.log"
exit $rc
