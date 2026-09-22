#!/usr/bin/env bash
# The AVBD kernels, from Lean to both targets.
#
#   kernels/avbd/gen.sh                 # emit from Lean, then cpp + spirv + table + embed
#   kernels/avbd/gen.sh --no-emit       # use the committed slang/ (no lake)
#   kernels/avbd/gen.sh --from <dir>    # use .slang already emitted into <dir>
#
# One source, two targets, exactly as the org's guest-avbd does it:
#   Lean (cloth-dynamics/lean, `lake exe emit_shaders`)  ->  slang/<k>.slang   (committed)
#     slangc -target cpp    ->  cpp/<k>_emit.cpp                                (committed)
#     slangc -target spirv  ->  <build>/spv/<k>.spv + .refl.json               (build artefact)
#       gen_avbd_kernel_table.py  ->  AvbdKernelTable.inc                       (committed)
#       ../embed_spv.py           ->  <build>/avbd_kernels.inc                  (build artefact)
# The cpp emits are the in-guest CPU path (AvbdCpu); the SPIR-V is the GPU
# path (AvbdRd over rd_compute). Both come from the same emitted Slang, so
# neither can drift from Lean or from each other.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEAN="${CLOTH_LEAN:-/c/cloth-dynamics-standalone/lean}"
BUILD="${BUILD_DIR:-$ROOT/build}"
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

KERNELS=$(grep -v '^#' "$HERE/kernels.txt" | tr '\n' ' ')

if [ "$MODE" = emit ]; then
	command -v lake >/dev/null 2>&1 || { echo "error: lake not on PATH (or pass --no-emit)" >&2; exit 1; }
	FROM="$(mktemp -d)"
	echo "== emitting Slang from Lean at $LEAN =="
	( cd "$LEAN" && lake exe emit_shaders "$FROM" >/dev/null )
fi
if [ -n "$FROM" ]; then
	mkdir -p "$HERE/slang"
	for k in $KERNELS; do
		cp "$FROM/$k.slang" "$HERE/slang/$k.slang"
	done
	echo "== $(echo $KERNELS | wc -w) kernels into slang/ =="
fi

mkdir -p "$HERE/cpp" "$BUILD/spv"
echo "== slangc -target cpp =="
for k in $KERNELS; do
	"$SLANGC" -target cpp -stage compute -entry main -o "$HERE/cpp/${k}_emit.cpp" "$HERE/slang/$k.slang"
done
echo "== slangc -target spirv (+ reflection) =="
for k in $KERNELS; do
	"$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main \
		-reflection-json "$BUILD/spv/$k.refl.json" -o "$BUILD/spv/$k.spv" "$HERE/slang/$k.slang"
done
echo "== binding table =="
python "$HERE/gen_avbd_kernel_table.py" --build-dir "$BUILD/spv" --out "$HERE/AvbdKernelTable.inc" $KERNELS
echo "== embedding SPIR-V =="
python "$HERE/../embed_spv.py" "$BUILD/spv" "$BUILD/avbd_kernels.inc"
