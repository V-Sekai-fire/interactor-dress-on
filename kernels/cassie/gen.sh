#!/usr/bin/env bash
# The curvenet stage's Cassie kernels, from Lean to both targets.
#
#   kernels/cassie/gen.sh                 # emit from Lean, then cpp + spirv
#   kernels/cassie/gen.sh --no-emit       # use the committed slang/ (no lake)
#   kernels/cassie/gen.sh --from <dir>    # use .slang already emitted into <dir>
#
# One source, two targets (AGENTS.md rule 2), as kernels/avbd/gen.sh:
#   Lean (lean/, `lake exe emit_cassie` for curve_*,
#         `lake exe emit_shaders` for spmv_df32)    ->  slang/<k>.slang       (committed)
#     slangc -target cpp    ->  cpp/<k>_emit.cpp                               (committed)
#     slangc -target spirv  ->  <build>/spv_cassie/<k>.spv, spirv-val         (build artefact)
# The cpp emits are the in-guest CPU path: the dispatchers under
# vendor/cassie/src/solver/slang_dispatch include them. The SPIR-V is only
# validated for now; nothing dispatches it yet.
#
# Cassie.kernels pairs a GPU and a CPU module per kernel. When they emit
# different text (a groupshared kernel, which slangc -target cpp rejects with
# E36107, paired with a serial sibling) emit_cassie also writes
# <k>.cpu.slang, and the cpp target is lowered from that instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEAN="${CLOTH_LEAN:-$ROOT/lean}"
BUILD="${BUILD_DIR:-$ROOT/build}"
SLANGC="${SLANGC:-slangc}"
command -v "$SLANGC" >/dev/null 2>&1 || SLANGC="$HOME/scoop/apps/vulkan/current/Bin/slangc"
SPIRV_VAL="${SPIRV_VAL:-spirv-val}"
command -v "$SPIRV_VAL" >/dev/null 2>&1 || SPIRV_VAL="$HOME/scoop/apps/vulkan/current/Bin/spirv-val"

MODE=emit
FROM=""
case "${1:-}" in
	--no-emit) MODE=none ;;
	--from) MODE=from; FROM="$2" ;;
	"") ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
esac

KERNELS=$(grep -v '^#' "$HERE/kernels.txt" | tr '\n' ' ')

if [ "$MODE" = emit ]; then
	command -v lake >/dev/null 2>&1 || { echo "error: lake not on PATH (or pass --no-emit)" >&2; exit 1; }
	FROM="$(mktemp -d)"
	trap 'rm -rf "$FROM"' EXIT
	echo "== emitting Slang from Lean at $LEAN =="
	# emit_shaders first, emit_cassie second: a Cassie kernel wins a name clash.
	( cd "$LEAN" && lake exe emit_shaders "$FROM" >/dev/null && lake exe emit_cassie "$FROM" >/dev/null )
fi
if [ -n "$FROM" ]; then
	mkdir -p "$HERE/slang"
	for k in $KERNELS; do
		cp "$FROM/$k.slang" "$HERE/slang/$k.slang"
		if [ -f "$FROM/$k.cpu.slang" ]; then
			cp "$FROM/$k.cpu.slang" "$HERE/slang/$k.cpu.slang"
		else
			rm -f "$HERE/slang/$k.cpu.slang"
		fi
	done
	echo "== $(echo $KERNELS | wc -w) kernels into slang/ =="
fi

mkdir -p "$HERE/cpp" "$BUILD/spv_cassie"
echo "== slangc -target cpp =="
# Relative paths from $HERE: slangc writes the input path into a #line
# directive, and an absolute one would make the committed cpp name the
# checkout (gates/lean README).
for k in $KERNELS; do
	src="slang/$k.slang"
	[ -f "$HERE/slang/$k.cpu.slang" ] && src="slang/$k.cpu.slang"
	( cd "$HERE" && "$SLANGC" -target cpp -stage compute -entry main -o "cpp/${k}_emit.cpp" "$src" )
done
echo "== slangc -target spirv + spirv-val =="
for k in $KERNELS; do
	"$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main \
		-o "$BUILD/spv_cassie/$k.spv" "$HERE/slang/$k.slang"
	"$SPIRV_VAL" "$BUILD/spv_cassie/$k.spv"
	echo "  $k.spv valid ($(wc -c < "$BUILD/spv_cassie/$k.spv") bytes)"
done
