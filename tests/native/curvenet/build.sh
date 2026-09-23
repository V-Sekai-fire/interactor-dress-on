#!/usr/bin/env bash
# Native build + smoke of the curvenet libraries (see CMakeLists.txt here).
#
#   tests/native/curvenet/build.sh
#   CXX=clang++ OUT=/tmp/x tests/native/curvenet/build.sh
#
# 1. configure + build every curvenet library and curvenet_smoke (-k 0, so
#    one bad TU does not hide the rest); report the TU count;
# 2. list the symbols cassie_core references that no library defines -- with
#    the kernels built (the default) there are none; with
#    CURVENET_KERNELS_PENDING=ON exactly the five dispatchers' entry points;
# 3. run curvenet_smoke, and curvenet_kernels_smoke when the kernels are built;
# 4. with the kernels built, run cassie_checks (Gate 4's checks, the flat
#    control for curvenet.elf) into gates/4-curvenet/native-checks.log, which
#    project/gate_curvenet.gd compares the guest against.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="${OUT:-$ROOT/build/native-curvenet}"
PENDING="${CURVENET_KERNELS_PENDING:-OFF}"
NINJA="$(command -v ninja || echo "$HOME/.pixi/bin/ninja.exe")"
NM="$(command -v llvm-nm || echo nm)"
m() { cygpath -m "$1" 2>/dev/null || echo "$1"; }

cmake -S "$(m "$HERE")" -B "$(m "$OUT")" -G Ninja \
	-DCMAKE_MAKE_PROGRAM="$(m "$NINJA")" \
	-DCMAKE_C_COMPILER="${CC:-clang}" -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
	-DCMAKE_BUILD_TYPE=Release -DCURVENET_KERNELS_PENDING="$PENDING" >/dev/null

LIBS=(godot_lite geogram_subset pmp_subset mwt_subset cassie_core)
[ "$PENDING" = ON ] || LIBS+=(cassie_kernels)
cmake --build "$OUT" --target "${LIBS[@]}" -- -k 0
n_obj=$(find "$OUT" -name '*.o' -o -name '*.obj' | grep -v CMakeFiles/curvenet_smoke | wc -l)
echo "curvenet: $n_obj library TUs compiled natively ($PENDING = CURVENET_KERNELS_PENDING)"

# Undefined in cassie_core, defined nowhere in the curvenet libraries or the
# C/C++ runtime (the link below settles the runtime; this names the rest).
defs=$(mktemp); undef=$(mktemp)
for l in "${LIBS[@]}"; do
	a=$(find "$OUT" -name "lib$l.a" | head -1)
	"$NM" -C --defined-only "$a" 2>/dev/null | awk 'NF>=3{ $1=""; $2=""; sub(/^  /,""); print }' >>"$defs"
done
sort -u -o "$defs" "$defs"
# The runtime's share is dropped: C functions (no "::"), std::, the C++ ABI.
"$NM" -C -u "$(find "$OUT" -name libcassie_core.a | head -1)" 2>/dev/null |
	awk 'NF>=2{ $1=""; sub(/^ /,""); print }' | sort -u | comm -23 - "$defs" |
	grep '::' | grep -v -E '^(void |)std::|^(vtable|typeinfo|typeinfo name) for (std|__cxxabiv1)::|^operator (new|delete)' >"$undef" || true
echo "cassie_core: $(wc -l <"$undef") symbols unresolved by the curvenet libraries:"
sed 's/^/  /' "$undef"
# Pending, exactly the five dispatchers' entry points; otherwise none.
extra=$(grep -v '^cassie_slang_dispatch::' "$undef" || true)
n_disp=$(grep -c '^cassie_slang_dispatch::' "$undef" || true)
rm -f "$defs" "$undef"
[ -z "$extra" ] || { echo "error: unresolved beyond the kernel dispatchers" >&2; exit 1; }
if [ "$PENDING" = ON ]; then want=5; else want=0; fi
[ "$n_disp" -eq "$want" ] || { echo "error: $n_disp dispatcher symbols unresolved, want $want" >&2; exit 1; }

cmake --build "$OUT" --target curvenet_smoke
"$OUT/curvenet_smoke"
if [ "$PENDING" != ON ]; then
	cmake --build "$OUT" --target curvenet_kernels_smoke
	"$OUT/curvenet_kernels_smoke"
fi
if [ "$PENDING" != ON ]; then
	cmake --build "$OUT" --target cassie_checks
	LOG="$ROOT/gates/4-curvenet/native-checks.log"
	rc=0
	{
		echo "# cassie_checks (native, $("${CXX:-clang++}" --version | head -1))"
		"$OUT/cassie_checks" 2>&1 || rc=$?
	} >"$LOG"
	cat "$LOG"
	[ "$rc" -eq 0 ] || { echo "error: cassie_checks failed (exit $rc)" >&2; exit 1; }
fi
