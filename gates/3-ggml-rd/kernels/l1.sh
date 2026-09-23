#!/usr/bin/env bash
# L1 of the ggml-rd kernels, after kernels/ggml/gen.sh (which already ran
# spirv-val on every kernel):
#   - the fixed-layout check on every kernel in kernels.txt (must pass), and
#     its negative controls (must fail): add_f32 compiled at slangc's default
#     -O1, which drops the unused s2 from the SPIR-V while the reflection
#     JSON still lists it; and kernels/avbd's saxpby, another layout;
#   - every cpp emit compiled for riscv64 by the guest's clang (the x86
#     compile is the L2 harness, tests/ggml_rd_kernels); a kernel that shares group memory has
#     none (slangc's cpp target rejects it) and is listed as SKIP, its
#     sibling compiled in its place;
#   - the cpp siblings (kernels/ggml/cpp_siblings.txt), two controls that
#     must fail: slangc -target cpp on mul_mat_tiled_f16_f32 (its group
#     barrier, E36107: why the sibling exists), and the table check given a
#     pair whose thread groups differ (the host would run the sibling over
#     the wrong grid).
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
# Kernels with a cpp sibling (kernels/ggml/cpp_siblings.txt) have no cpp emit.
GPU_ONLY=$(grep -v '^#' "$ROOT/kernels/ggml/cpp_siblings.txt" | awk 'NF { print $1 }' | tr '\n' ' ')
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

	echo "== negative: slangc -target cpp on a kernel with a group barrier, mul_mat_tiled_f16_f32 (must fail)"
	if ( cd "$ROOT/kernels/ggml" && "$SLANGC" -target cpp -stage compute -entry main -preserve-params \
		-o "$TMP/tiled_emit.cpp" slang/mul_mat_tiled_f16_f32.slang ) > "$TMP/tiled_cpp.log" 2>&1; then
		echo "FAIL slangc emitted cpp for a kernel with GroupMemoryBarrierWithGroupSync"; rc=1
	else
		echo "PASS refused: $(grep -o 'error.E[0-9]*.: [a-z ]*' "$TMP/tiled_cpp.log" | head -1), at $(grep -o "see using of '[A-Za-z]*'" "$TMP/tiled_cpp.log" | head -1)"
	fi

	echo "== negative: a cpp_siblings pair with different thread groups, mul_mat_tiled_f16_f32 -> mul_mat_serial_vec_f16_f32 (must fail)"
	echo "mul_mat_tiled_f16_f32 mul_mat_serial_vec_f16_f32" > "$TMP/bad_siblings.txt"
	if python "$TABLE" --spv-dir "$BUILD/spv-ggml" --siblings "$TMP/bad_siblings.txt" --out "$TMP/table.inc" $KERNELS; then
		echo "FAIL the mismatched pair passed"; rc=1
	else
		echo "PASS the mismatched pair is refused"
	fi

	echo "== cpp emits compiled for riscv64 ($("$CLANG" --version | head -1))"
	for k in $KERNELS; do
		case " $GPU_ONLY " in *" $k "*) echo "SKIP riscv64 $k (GPU-only: cpp_siblings.txt names its sibling)"; continue ;; esac
		if [ ! -f "$ROOT/kernels/ggml/cpp/${k}_emit.cpp" ] && grep -q '^groupshared \|GroupMemoryBarrierWithGroupSync' "$ROOT/kernels/ggml/slang/$k.slang"; then
			echo "SKIP riscv64 $k (group-shared: no cpp target; ${k}_serial is its cpp sibling)"
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
