#!/usr/bin/env bash
# The Gate 0F probe kernels, from Lean to SPIR-V embedded in probes.elf.
#
#   kernels/probes/gen.sh              # emit from Lean, then spirv + embed
#   kernels/probes/gen.sh --no-emit    # use the committed slang/ (no Lean)
#
# Two kernels, both emitted by Lean, none hand-written (AGENTS.md rule 2):
#  - saxpby (dst = a*x + b*y), cloth-dynamics/lean `emit_shaders`
#    (Cloth.SlangCodegen.Saxpby): the "+1" dispatch of the fiber and
#    references_max probes (x = ones, y = dst, a = b = 1) and the whole-buffer
#    writer of the large-buffer probe;
#  - half_load (o[i] = float(w[i]) * 2), contract-lean-slang emit-fp's
#    TestFp fixture: the f16 storage read. emit_shaders does not emit it, so
#    the committed slang/ copy is used (see kernels.txt for its provenance).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEAN="${CLOTH_LEAN:-/c/cloth-dynamics-standalone/lean}"
BUILD="${BUILD_DIR:-$ROOT/build}"
SLANGC="${SLANGC:-slangc}"
command -v "$SLANGC" >/dev/null 2>&1 || SLANGC="$HOME/scoop/apps/vulkan/current/Bin/slangc"

KERNELS=$(grep -v '^#' "$HERE/kernels.txt" | tr '\n' ' ')

if [ "${1:-}" != "--no-emit" ]; then
	EMIT="$LEAN/.lake/build/bin/emit_shaders"
	TMP="$(mktemp -d)"
	if [ -x "$EMIT" ] || [ -x "$EMIT.exe" ]; then
		"$EMIT" "$TMP" >/dev/null
	else
		( cd "$LEAN" && lake exe emit_shaders "$TMP" >/dev/null )
	fi
	for k in $KERNELS; do
		if [ -f "$TMP/$k.slang" ]; then cp "$TMP/$k.slang" "$HERE/slang/$k.slang"; fi
	done
fi

mkdir -p "$BUILD/spv-probes"
for k in $KERNELS; do
	"$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main \
		-o "$BUILD/spv-probes/$k.spv" "$HERE/slang/$k.slang"
done
python "$HERE/../embed_spv.py" "$BUILD/spv-probes" "$BUILD/probes_kernels.inc"
