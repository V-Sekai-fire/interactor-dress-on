#!/usr/bin/env bash
# Gate 6a: brick-grid SDF + spline sampler vs OpenVDB. Native, CPU only.
#
#   gates/6-fit/sdf/run.sh
#
# Environment (defaults are the desk's oracle of record, tools/native/README.md):
#   G6A_OUT       scratch dir for points, dumps, exe (default C:/b/g6a)
#   ORACLE_RUN    1-thread upstream run dir (default C:/b/cf-up-out1)
#   OPENVDB_DUMP  tools/fit/openvdb_dump built by tools/native (default C:/b/cf-up/openvdb_dump.exe)
#   EIGEN_DIR     Eigen 3.4.0 source (default: the CPM cache entry under C:/b/cpm-native/eigen)
#   LLVM_MINGW    toolchain (default ~/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64)
#
# libigl and nlohmann/json come from the V-Sekai-fire forks in .forks/
# (tools/forks/fetch.sh). The spline kernel is the Lean emit
# kernels/fit/cpp/sdf_spline_hessian_emit.cpp (kernels/fit/gen.sh); if it is
# missing the build defines FIT_KERNELS_PENDING and (a) checks the reference
# sampler alone.
#
# With the kernel it also runs two things for a.kernel:
#   lean_fixtures    the reference and the emit on the two fixtures that
#                    lean/Fit/SdfSplineHessian.lean pins with native_decide
#                    (prints the bits the pins hold; exit 1 if the emit differs)
#   kernel control   the committed Slang with 2/3 written as the float literal
#                    (2.0 / 3.0), compiled by slangc into $OUT/negk: a.kernel
#                    must FAIL (it is never committed; AGENTS.md rule 2)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="${G6A_OUT:-C:/b/g6a}"
RUN="${ORACLE_RUN:-C:/b/cf-up-out1}"
DUMP="${OPENVDB_DUMP:-C:/b/cf-up/openvdb_dump.exe}"
EIGEN_DIR="${EIGEN_DIR:-$(ls -d C:/b/cpm-native/eigen/*/ | head -1)}"
LLVM_MINGW="${LLVM_MINGW:-$HOME/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64}"
CXX="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang++"
CF="$ROOT/vendor/cloth-fit"
AVATAR_FACES="$CF/garment-data/assets/avatars/FoxGirl/avatar.obj"
NOFIT="$CF/garment-data/assets/garments/LCL_Skirt_DressEvening_003/no-fit.txt"
GARMENT="$RUN/step_garment_252.obj"
TARGET="$RUN/target_avatar.obj"
mkdir -p "$OUT"

KDIR="${FIT_KERNELS_DIR:-$ROOT/kernels/fit/cpp}"
KFLAGS=(-DFIT_KERNELS_PENDING)
if [ -f "$KDIR/sdf_spline_hessian_emit.cpp" ]; then
	KFLAGS=(-I "$KDIR" -I "$ROOT/guest/avbd/slang-rt")
fi

echo "== build gate6a (${KFLAGS[*]}) =="
# -O2, no -march: the same x86-64 baseline (no FMA contraction) openvdb_dump was built for.
"$CXX" -std=c++17 -O2 -static -Wno-invalid-offsetof -Wno-deprecated-literal-operator \
	-I "$CF/src" -I "$EIGEN_DIR" -I "$ROOT/.forks/libigl/include" -I "$ROOT/.forks/json/include" \
	"${KFLAGS[@]}" \
	"$HERE/gate6a.cpp" \
	"$CF/src/polyfem/solver/forms/garment_forms/SdfGrid.cpp" \
	"$CF/src/polyfem/solver/forms/garment_forms/SdfSpline.cpp" \
	-o "$OUT/gate6a.exe"

GATE_CXX=("$CXX" -std=c++17 -O2 -static -Wno-invalid-offsetof -Wno-deprecated-literal-operator
	-I "$CF/src" -I "$EIGEN_DIR" -I "$ROOT/.forks/libigl/include" -I "$ROOT/.forks/json/include")
KERNEL=0
[ "${KFLAGS[0]}" = -I ] && KERNEL=1
if [ "$KERNEL" = 1 ]; then
	echo "== lean_fixtures =="
	"${GATE_CXX[@]}" "${KFLAGS[@]}" "$HERE/lean_fixtures.cpp" \
		"$CF/src/polyfem/solver/forms/garment_forms/SdfSpline.cpp" -o "$OUT/lean_fixtures.exe"
	"$OUT/lean_fixtures.exe"

	echo "== kernel control: 2/3 as a float literal =="
	SLANGC="${SLANGC:-slangc}"
	command -v "$SLANGC" >/dev/null 2>&1 || SLANGC="$HOME/scoop/apps/vulkan/current/Bin/slangc"
	mkdir -p "$OUT/negk/slang" "$OUT/negk/cpp"
	sed 's|(double(2) / double(3))|(2.0 / 3.0)|' "$ROOT/kernels/fit/slang/sdf_spline_hessian.slang" \
		> "$OUT/negk/slang/sdf_spline_hessian.slang"
	grep -q '(2.0 / 3.0)' "$OUT/negk/slang/sdf_spline_hessian.slang"
	( cd "$OUT/negk" && "$SLANGC" -target cpp -stage compute -entry main \
		-o cpp/sdf_spline_hessian_emit.cpp slang/sdf_spline_hessian.slang 2>/dev/null )
	grep -o '0\.6666[0-9]*' "$OUT/negk/cpp/sdf_spline_hessian_emit.cpp"
	"${GATE_CXX[@]}" -I "$OUT/negk/cpp" -I "$ROOT/guest/avbd/slang-rt" \
		"$HERE/gate6a.cpp" \
		"$CF/src/polyfem/solver/forms/garment_forms/SdfGrid.cpp" \
		"$CF/src/polyfem/solver/forms/garment_forms/SdfSpline.cpp" \
		-o "$OUT/gate6a_negk.exe"
fi

for pop in fit all; do
	extra=()
	[ "$pop" = all ] && extra=(--all)
	echo "== points ($pop) =="
	python "$HERE/make_points.py" "$GARMENT" "$NOFIT" "$OUT/points_$pop.txt" "${extra[@]}"
	echo "== openvdb_dump ($pop) =="
	"$DUMP" "$TARGET" --faces "$AVATAR_FACES" --voxel 0.01 --points "$OUT/points_$pop.txt" --out "$OUT/vdb_$pop.jsonl"
	sha256sum "$OUT/points_$pop.txt" "$OUT/vdb_$pop.jsonl"
done

rc=0
for pop in fit all; do
	echo "== gate6a ($pop) =="
	"$OUT/gate6a.exe" "$TARGET" "$AVATAR_FACES" "$OUT/vdb_$pop.jsonl" --voxel 0.01 --label "$pop" || rc=1
done
if [ "$KERNEL" = 1 ]; then
	for pop in fit all; do
		echo "== kernel control ($pop): a.kernel must FAIL =="
		if "$OUT/gate6a_negk.exe" "$TARGET" "$AVATAR_FACES" "$OUT/vdb_$pop.jsonl" --voxel 0.01 --label "$pop-negk" \
			| grep -E 'a\.kernel|SUMMARY' | tee "$OUT/negk_$pop.txt"; then :; fi
		if grep -q '^FAIL .*a\.kernel' "$OUT/negk_$pop.txt"; then
			echo "PASS [$pop] kernel control: the float-literal emit fails a.kernel"
		else
			echo "FAIL [$pop] kernel control: the float-literal emit passed a.kernel"
			rc=1
		fi
	done
fi
exit $rc
