#!/usr/bin/env bash
# fit.elf's kernels, from Lean (the kernels/avbd/gen.sh shape).
#
#   kernels/fit/gen.sh                 # emit from Lean, then cpp + spirv validation
#   kernels/fit/gen.sh --no-emit       # use the committed slang/ (no lake)
#   kernels/fit/gen.sh --from <dir>    # use .slang already emitted into <dir>
#
#   Lean (lean/, `lake exe emit_fit`)                      ->  slang/<k>.slang  (committed)
#     slangc -target cpp    ->  cpp/<k>_emit.cpp                                 (committed)
#     slangc -target spirv  ->  <build>/spv-fit/<k>.spv                          (validation only)
#
# The cpp emit is what fit.elf and fit_native compile (garment_forms/SdfSpline.cpp
# includes it; FIT_KERNELS_DIR points here). fit.elf has no GPU path yet, so
# the SPIR-V is only compiled, to prove the same Slang is a valid GPU kernel
# (double needs SPIR-V Float64); nothing embeds it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEAN="${CLOTH_LEAN:-$ROOT/lean}"
BUILD="${BUILD_DIR:-$ROOT/build}"
SPV="$BUILD/spv-fit"
SLANGC="${SLANGC:-slangc}"
command -v "$SLANGC" >/dev/null 2>&1 || SLANGC="$HOME/scoop/apps/vulkan/current/Bin/slangc"

MODE=emit
FROM=""
case "${1:-}" in
	--no-emit) MODE=none ;;
	--from) MODE=from; FROM="$2" ;;
	"") ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
esac

KERNELS=$(grep -v '^#' "$HERE/kernels.txt" | awk 'NF {print $1}' | tr '\n' ' ')

if [ "$MODE" = emit ]; then
	command -v lake >/dev/null 2>&1 || { echo "error: lake not on PATH (or pass --no-emit)" >&2; exit 1; }
	FROM="$(mktemp -d)"
	echo "== emitting Slang from Lean at $LEAN =="
	( cd "$LEAN" && lake exe emit_fit "$FROM" >/dev/null )
fi
if [ -n "$FROM" ]; then
	mkdir -p "$HERE/slang"
	for k in $KERNELS; do
		cp "$FROM/$k.slang" "$HERE/slang/$k.slang"
	done
	echo "== $(echo $KERNELS | wc -w) kernels into slang/ =="
fi

mkdir -p "$HERE/cpp" "$SPV"
echo "== slangc -target cpp =="
# Relative paths from $HERE: slangc writes the input path into a #line
# directive, and an absolute one would make the committed cpp depend on
# where the checkout (or worktree) lives.
for k in $KERNELS; do
	( cd "$HERE" && "$SLANGC" -target cpp -stage compute -entry main -o "cpp/${k}_emit.cpp" "slang/$k.slang" )
done
echo "== slangc -target spirv (validation only) =="
for k in $KERNELS; do
	"$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main \
		-o "$SPV/$k.spv" "$HERE/slang/$k.slang"
	echo "  $k.spv $(wc -c < "$SPV/$k.spv") bytes"
done
