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
# (tools/forks/fetch.sh). The spline kernel is compiled in only when
# kernels/fit/cpp/sdf_spline_hessian_emit.cpp exists; until then the build
# defines FIT_KERNELS_PENDING and (a) checks the reference sampler alone.
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
exit $rc
