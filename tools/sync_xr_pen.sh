#!/usr/bin/env bash
# Copy xr-grid's pen (addons/procedural_3d_grid: SketchTool, SimpleSketch,
# hand.gd, the grid) from vendor/xr-grid into project/addons/, byte for byte,
# and write its CITATION.cff there. Gate 8 checks the copy with
#   tools/sync_xr_pen.sh --check
# which is diff -r against vendor/xr-grid, CITATION.cff excepted; it must be
# empty. The pen bridge (project/xr/pen_bridge.gd) reads SketchTool's state;
# nothing in the copy is edited, so strokes.gd (whose just_pressed is always
# false) is copied but not used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/vendor/xr-grid/addons/procedural_3d_grid"
DST="$ROOT/project/addons/procedural_3d_grid"

if [ "${1:-}" = "--check" ]; then
	if diff -r -x CITATION.cff "$SRC" "$DST"; then
		echo "sync_xr_pen: project/addons/procedural_3d_grid == vendor/xr-grid (diff -r empty)"
		exit 0
	fi
	echo "sync_xr_pen: FAIL, the copy differs from vendor/xr-grid"
	exit 1
fi

rm -rf "$DST"
mkdir -p "$(dirname "$DST")"
cp -r "$SRC" "$DST"
XR_COMMIT="$(git -C "$ROOT" log -1 --format=%H -- vendor/xr-grid)"
cat > "$DST/CITATION.cff" <<EOF
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "procedural_3d_grid, xr-grid's meshing pen (copied into project/addons)"
abstract: >-
  addons/procedural_3d_grid from vendor/xr-grid (transport-xr-grid, the
  org's fork of V-Sekai.xr-grid; see vendor/xr-grid/CITATION.cff), copied
  byte for byte by tools/sync_xr_pen.sh so project/xr_main.tscn can use its
  SketchTool, SimpleSketch and hand.gd. No local adaptations: Gate 8 checks
  diff -r against vendor/xr-grid is empty (this file excepted). The pen
  bridge (project/xr/pen_bridge.gd) reads SketchTool.active and the tool's
  position and forwards pen_begin / pen_point / pen_end to curvenet.elf,
  which leaves strokes.gd (its just_pressed is always false) unused.
  Copied from this repo's commit ${XR_COMMIT} of vendor/xr-grid.
authors:
  - family-names: Lee
    given-names: "K. S. Ernest (iFire)"
  - alias: celyk
repository-code: "https://github.com/V-Sekai-fire/transport-xr-grid"
commit: ${XR_COMMIT}
license: MIT
EOF
echo "sync_xr_pen: copied $(find "$DST" -type f | wc -l) files into project/addons/procedural_3d_grid"
