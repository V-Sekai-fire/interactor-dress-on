#!/usr/bin/env bash
# Vendor the Geogram / PMP / MWT subsets curvenet.elf compiles (Cut 4).
#
#   tools/vendor/thirdparty_subset.sh              # copy + apply local adaptations
#   THIRDPARTY_NO_PATCH=1 tools/vendor/thirdparty_subset.sh   # pristine copy
#
# Source: V-Sekai-fire/entities-godot at c165a519d2, thirdparty/ (checkout at
# C:/contract-manifest/4-entities/godot; override with ENTITIES_GODOT).
#
#   vendor/geogram-subset  the sources gates/0b-crosscompile/cassie-thirdparty
#                          globs (modules/cassie/SCsub's list, plus
#                          delaunay/delaunay_2d.cpp) and the headers they
#                          include, transitively; zlib headers only (geofile.h
#                          wants zlib.h, nothing links it).
#   vendor/pmp-subset      pmp core + remeshing, decimation,
#                          differential_geometry, normals, features, utilities,
#                          distance_point_triangle, triangulation and their
#                          headers. Not curvature/laplace/numerics/smoothing:
#                          those are the Eigen users, and curvenet.elf is
#                          Eigen-free (PMP_NO_EIGEN, see patches/pmp.patch).
#   vendor/mwt             multipolygon_triangulator, all 18 files.
#
# Adaptations: tools/vendor/patches/{geogram,pmp,mwt}.patch, listed in each
# CITATION.cff. Fails if any prebuilt kernel lands under vendor/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# Python on Windows cannot open /c/... paths; hand it C:/... ones.
mpath() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
EG="$(mpath "${ENTITIES_GODOT:-C:/contract-manifest/4-entities/godot}")"
PIN="c165a519d2"
TP="$EG/thirdparty"
PY="$(command -v python3 || command -v python)"

have="$(git -C "$EG" rev-parse --short=10 HEAD)"
if [ "$have" != "$PIN" ] && [ "${ALLOW_OTHER_REV:-0}" != 1 ]; then
	echo "error: $EG is at $have, the subsets are pinned to $PIN (ALLOW_OTHER_REV=1 to override)" >&2
	exit 1
fi

# wipe DIR: empty a vendored tree but keep its hand-written CITATION.cff.
wipe() {
	mkdir -p "$1"
	find "$1" -mindepth 1 -maxdepth 1 ! -name CITATION.cff -exec rm -rf {} +
}

# glob_without DIR PATTERN [EXCLUDED-BASENAME ...]
glob_without() {
	local dir="$1" pat="$2"
	shift 2
	local f b x skip
	for f in "$dir"/$pat; do
		[ -f "$f" ] || continue
		b="${f##*/}"
		skip=0
		for x in "$@"; do [ "$b" = "$x" ] && skip=1; done
		[ "$skip" = 1 ] || echo "$f"
	done
}

# ---- Geogram ---------------------------------------------------------------
GR="$TP/geogram/src/lib/geogram"
GEO_DST="$ROOT/vendor/geogram-subset"
mapfile -t GEO_SRC < <(
	glob_without "$GR/basic" '*.cpp' geofile.cpp android_utils.cpp boolean_expression.cpp
	glob_without "$GR/numerics" '*.cpp'
	echo "$GR/mesh/mesh.cpp"
	echo "$GR/mesh/mesh_reorder.cpp"
	glob_without "$GR/delaunay" '*.cpp' delaunay_tetgen.cpp delaunay_triangle.cpp parallel_delaunay_3d.cpp
	glob_without "$GR/points" '*.cpp' co3ne.cpp
	glob_without "$GR/api" '*.cpp'
	glob_without "$GR/bibliography" '*.cpp'
	glob_without "$GR/third_party/predicate_generator" '*.cpp'
	glob_without "$GR/third_party/numerics" '*.cpp'
	glob_without "$GR/third_party/OpenNL" '*.c'
)
# DelaunayFaces.cpp (mwt) is the other Geogram client; its headers come too.
mapfile -t GEO_ALL < <("$PY" "$HERE/include_closure.py" \
	-I "$TP/geogram/src/lib" -I "$GR/third_party" -I "$TP/zlib" \
	"${GEO_SRC[@]}" "$TP/multipolygon_triangulator/DelaunayFaces.cpp" | tr -d '\r')

wipe "$GEO_DST"
for f in "${GEO_ALL[@]}"; do
	case "$f" in
	"$GR"/*) rel="geogram/${f#"$GR"/}" ;;
	"$TP/zlib"/*) rel="zlib/${f#"$TP/zlib"/}" ;;
	*) continue ;; # mwt's own headers are vendored with mwt
	esac
	mkdir -p "$GEO_DST/${rel%/*}"
	cp "$f" "$GEO_DST/$rel"
done
cp "$TP/geogram/LICENSE" "$GEO_DST/LICENSE"
cp "$TP/zlib/LICENSE" "$GEO_DST/zlib/LICENSE" 2>/dev/null || true

# ---- PMP -------------------------------------------------------------------
PR="$TP/pmp/src/pmp"
PMP_DST="$ROOT/vendor/pmp-subset"
PMP_SRC=("$PR/surface_mesh.cpp")
for a in remeshing decimation differential_geometry normals features utilities \
	distance_point_triangle triangulation; do
	PMP_SRC+=("$PR/algorithms/$a.cpp" "$PR/algorithms/$a.h")
done
mapfile -t PMP_ALL < <("$PY" "$HERE/include_closure.py" -I "$TP/pmp/src" "${PMP_SRC[@]}" | tr -d '\r')
wipe "$PMP_DST"
for f in "${PMP_ALL[@]}"; do
	case "$f" in
	"$PR"/algorithms/curvature.h) continue ;; # its only include sits behind PMP_NO_EIGEN
	"$PR"/*) rel="pmp/${f#"$PR"/}" ;;
	*) continue ;;
	esac
	mkdir -p "$PMP_DST/${rel%/*}"
	cp "$f" "$PMP_DST/$rel"
done
cp "$TP/pmp/LICENSE.txt" "$PMP_DST/LICENSE.txt"

# ---- MWT -------------------------------------------------------------------
MW="$TP/multipolygon_triangulator"
MWT_DST="$ROOT/vendor/mwt"
wipe "$MWT_DST"
cp "$MW"/*.h "$MW"/*.cpp "$MW/LICENSE" "$MWT_DST/"
n_mwt="$(find "$MWT_DST" -type f ! -name CITATION.cff | wc -l)"
if [ "$n_mwt" -ne 18 ]; then
	echo "error: vendor/mwt holds $n_mwt files, expected 18" >&2
	exit 1
fi

# ---- adaptations -----------------------------------------------------------
if [ "${THIRDPARTY_NO_PATCH:-0}" != 1 ]; then
	for p in geogram pmp mwt; do
		git -C "$ROOT" apply --whitespace=nowarn "$HERE/patches/$p.patch"
		# git apply writes CRLF under core.autocrlf; keep the files LF as
		# the source has them.
		sed -n 's|^+++ b/||p' "$HERE/patches/$p.patch" | while read -r f; do
			sed -i 's/\r$//' "$ROOT/$f"
		done
	done
fi

# curvenet.elf is Eigen-free (AGENTS.md rule 3): no live Eigen include may
# survive in a subset outside a PMP_NO_EIGEN guard.
live="$(grep -rnE '^\s*#\s*include\s*<Eigen/' "$GEO_DST" "$PMP_DST" "$MWT_DST" || true)"
if [ -n "$live" ] && [ "${THIRDPARTY_NO_PATCH:-0}" != 1 ]; then
	for hit in $(echo "$live" | cut -d: -f1 | sort -u); do
		if ! grep -q 'PMP_NO_EIGEN' "$hit"; then
			echo "error: unguarded Eigen include in $hit" >&2
			exit 1
		fi
	done
fi

bash "$HERE/check_no_prebuilt.sh" "$ROOT"
for d in "$GEO_DST" "$PMP_DST" "$MWT_DST"; do
	echo "$(basename "$d"): $(find "$d" -type f | wc -l) files, $(du -sk "$d" | cut -f1) KiB"
done
