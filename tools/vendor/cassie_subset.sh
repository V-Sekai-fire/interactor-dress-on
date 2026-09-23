#!/usr/bin/env bash
# Vendor the Cassie subset curvenet.elf compiles (Cut 4) into vendor/cassie/src.
#
#   tools/vendor/cassie_subset.sh            # copy + apply the local adaptations
#   CASSIE_NO_PATCH=1 tools/vendor/cassie_subset.sh   # pristine copy only
#
# Source: V-Sekai-fire/entities-godot at c165a519d2, modules/cassie/src
# (checkout at C:/contract-manifest/4-entities/godot; override with
# ENTITIES_GODOT). The file list is explicit: pen -> sketcher -> beautifier ->
# constraints/curves/solver -> sketch graph -> curvenet/knot/extractor/polar
# -> triangulator/remesh/refine. Left out on purpose: cassie_path_3d,
# cassie_surface, intrinsic_triangulation, polygon_triangulation_godot,
# profile_mover, edge_collider, quad_mesh, the GPU dispatchers
# (cassie_mas_gpu, cassie_slang_gpu, saxpby_dispatch), lean_ffi,
# register_types, doc_classes and tests.
#
# The adaptations live in tools/vendor/patches/cassie.patch and are listed in
# vendor/cassie/CITATION.cff. The script fails if any prebuilt kernel (*.spv,
# *.cpu.cpp) lands under vendor/ (AGENTS.md rule 2).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
EG="${ENTITIES_GODOT:-C:/contract-manifest/4-entities/godot}"
PIN="c165a519d2"
SRC="$EG/modules/cassie/src"
DST="$ROOT/vendor/cassie/src"

have="$(git -C "$EG" rev-parse --short=10 HEAD)"
if [ "$have" != "$PIN" ] && [ "${ALLOW_OTHER_REV:-0}" != 1 ]; then
	echo "error: $EG is at $have, the subset is pinned to $PIN (ALLOW_OTHER_REV=1 to override)" >&2
	exit 1
fi

FILES=(
	cassie_beautifier.cpp cassie_beautifier.h
	cassie_beautifier_params.cpp cassie_beautifier_params.h
	cassie_remesh.cpp cassie_remesh.h
	cassie_sketcher.cpp cassie_sketcher.h
	cassie_stroke_packet.cpp cassie_stroke_packet.h
	cassie_triangulator.cpp cassie_triangulator.h
	constraints/cassie_constraint.cpp constraints/cassie_constraint.h
	constraints/cassie_intersection_constraint.cpp constraints/cassie_intersection_constraint.h
	constraints/cassie_intersection_finder.cpp constraints/cassie_intersection_finder.h
	constraints/cassie_mirror_plane_constraint.cpp constraints/cassie_mirror_plane_constraint.h
	constraints/cassie_surface_constraint.cpp constraints/cassie_surface_constraint.h
	curves/cassie_curve_fit.cpp curves/cassie_curve_fit.h
	curves/rdp_simplify.cpp curves/rdp_simplify.h
	delaunay_geogram.cpp delaunay_geogram.h
	polygon_triangulation.cpp polygon_triangulation.h
	refine.cpp refine.h
	sketch/cassie_curvenet.cpp sketch/cassie_curvenet.h
	sketch/cassie_curvenet_extractor.cpp sketch/cassie_curvenet_extractor.h
	sketch/cassie_curvenet_knot.cpp sketch/cassie_curvenet_knot.h
	sketch/cassie_final_stroke.cpp sketch/cassie_final_stroke.h
	sketch/cassie_input_stroke.cpp sketch/cassie_input_stroke.h
	sketch/cassie_polar.cpp sketch/cassie_polar.h
	sketch/cassie_sketch_graph.cpp sketch/cassie_sketch_graph.h
	sketch/cassie_surface_manager.cpp sketch/cassie_surface_manager.h
	sketch/cassie_surface_patch.cpp sketch/cassie_surface_patch.h
	solver/cassie_constraint_solver.cpp solver/cassie_constraint_solver.h
	solver/cassie_eigen.cpp solver/cassie_eigen.h
	solver/cassie_pcg.cpp solver/cassie_pcg.h
	solver/dense_matrix.h
	solver/fidelity_energy.cpp solver/fidelity_energy.h
	solver/g1_constraint.cpp solver/g1_constraint.h
	solver/hard_constraint.h
	solver/on_surface_energy.cpp solver/on_surface_energy.h
	solver/planarity_constraint.cpp solver/planarity_constraint.h
	solver/position_constraint.cpp solver/position_constraint.h
	solver/self_intersection_constraint.cpp solver/self_intersection_constraint.h
	solver/soft_constraint.h
	solver/tangent_constraint.cpp solver/tangent_constraint.h
	solver/slang_dispatch/curve_casteljau_dispatch.cpp solver/slang_dispatch/curve_casteljau_dispatch.h
	solver/slang_dispatch/curve_generate_bezier_dispatch.cpp solver/slang_dispatch/curve_generate_bezier_dispatch.h
	solver/slang_dispatch/curve_newton_dispatch.cpp solver/slang_dispatch/curve_newton_dispatch.h
	solver/slang_dispatch/curve_rdp_dispatch.cpp solver/slang_dispatch/curve_rdp_dispatch.h
	solver/slang_dispatch/spmv_dispatch.cpp solver/slang_dispatch/spmv_dispatch.h
)
if [ "${#FILES[@]}" -ne 83 ]; then
	echo "error: the Cassie subset is 83 files, the list has ${#FILES[@]}" >&2
	exit 1
fi

rm -rf "$DST"
for f in "${FILES[@]}"; do
	mkdir -p "$DST/$(dirname "$f")"
	cp "$SRC/$f" "$DST/$f"
done
# Cassie's own licence (the module is MIT, as Godot) travels with it.
cp "$EG/LICENSE.txt" "$ROOT/vendor/cassie/LICENSE.txt"
cp "$EG/modules/cassie/REFERENCES.bib" "$ROOT/vendor/cassie/REFERENCES.bib"

if [ "${CASSIE_NO_PATCH:-0}" != 1 ]; then
	git -C "$ROOT" apply --whitespace=nowarn "$HERE/patches/cassie.patch"
fi
# git apply writes CRLF under core.autocrlf; keep the tree uniformly LF.
find "$ROOT/vendor/cassie" -type f -exec sed -i 's/\r$//' {} +

n="$(find "$DST" -type f | wc -l)"
if [ "$n" -ne 83 ]; then
	echo "error: vendor/cassie/src holds $n files, expected 83" >&2
	exit 1
fi
bash "$HERE/check_no_prebuilt.sh" "$ROOT"
echo "cassie_subset: $n files, $(du -sk "$DST" | cut -f1) KiB in vendor/cassie/src"
