#!/usr/bin/env bash
# Native (host) check of godot-lite: the shim (guest/godot_lite) plus the
# vendored Godot core subset, compiled with the host clang++ (llvm-mingw on
# Windows), no Godot, no sandbox.
#
#   1. build and run test_godot_lite (Vector COW, HashMap order, Array /
#      Dictionary aliasing, Ref / Callable / Mesh, Curve3D against Godot's
#      own expectations and a Godot 4.7.2 run: curve3d_ref.gd);
#   2. compile every Cassie core TU in vendor/cassie against the shim the
#      way cmake/curvenet.cmake does (prelude force-included, PMP_NO_EIGEN),
#      so "Cassie compiles almost verbatim" stays a checked claim.
#
#   tests/native/godot_lite/build.sh
#   CXX=clang++ OUT=/tmp/x tests/native/godot_lite/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
CXX="${CXX:-clang++}"
OUT="${OUT:-$ROOT/build/native-godot-lite}"
mkdir -p "$OUT/obj"

CXXFLAGS=(-std=c++17 -O2 -g -Wall -Wno-unused-function -Wno-unused-private-field
	-I "$ROOT/guest/godot_lite" -I "$ROOT/vendor/godot-core-subset"
	-include "$ROOT/guest/godot_lite/gdl_prelude.h")

mapfile -t SRCS < <(
	find "$ROOT/guest/godot_lite" "$ROOT/vendor/godot-core-subset" -name '*.cpp' | sort
	echo "$HERE/test_godot_lite.cpp"
)

OBJS=()
pids=()
for src in "${SRCS[@]}"; do
	rel="${src#"$ROOT"/}"
	obj="$OUT/obj/${rel//\//_}.o"
	OBJS+=("$obj")
	"$CXX" "${CXXFLAGS[@]}" -c "$src" -o "$obj" &
	pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "compile failed" >&2; exit 1; }

"$CXX" "${OBJS[@]}" -o "$OUT/test_godot_lite.exe"
"$OUT/test_godot_lite.exe" "$HERE/curve3d_ref_godot-4.7.2.txt"

# 2. Cassie core against the shim (syntax only: the kernels and third-party
# libraries it calls are other targets).
CASSIE="$ROOT/vendor/cassie/src"
if [ -d "$CASSIE" ]; then
	CASSIE_FLAGS=(-std=c++17 -fsyntax-only -w
		-I "$ROOT/guest/godot_lite" -I "$ROOT/vendor/godot-core-subset"
		-include "$ROOT/guest/godot_lite/gdl_prelude.h"
		-I "$CASSIE" -I "$CASSIE/solver/slang_dispatch"
		-I "$ROOT/vendor/pmp-subset" -I "$ROOT/vendor/mwt"
		-I "$ROOT/vendor/geogram-subset" -I "$ROOT/vendor/geogram-subset/geogram/third_party"
		-D_USE_MATH_DEFINES -DPMP_SCALAR_TYPE_64=1 -DPMP_NO_EIGEN)
	mapfile -t CASSIE_SRCS < <(find "$CASSIE" -name '*.cpp' ! -path '*/slang_dispatch/*' ! -name 'delaunay_geogram.cpp' | sort)
	pids=()
	for src in "${CASSIE_SRCS[@]}"; do
		"$CXX" "${CASSIE_FLAGS[@]}" "$src" &
		pids+=($!)
	done
	fail=0
	for p in "${pids[@]}"; do wait "$p" || fail=1; done
	[ "$fail" -eq 0 ] || { echo "cassie: compile against godot-lite failed" >&2; exit 1; }
	echo "cassie: ${#CASSIE_SRCS[@]} core TUs compile against godot-lite"
fi
