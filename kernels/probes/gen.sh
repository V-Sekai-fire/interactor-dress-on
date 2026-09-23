#!/usr/bin/env bash
# The Gate 0F probe kernels, from Lean to SPIR-V embedded in probes.elf.
#
#   kernels/probes/gen.sh              # emit from Lean, FAIL if it differs from slang/, then spirv + embed
#   kernels/probes/gen.sh --update     # emit from Lean into slang/ (review the diff), then spirv + embed
#   kernels/probes/gen.sh --no-emit    # use the committed slang/ (no lake)
#
# Every kernel is emitted by lean/ `lake exe emit_probes` (AGENTS.md rule 2):
#  - saxpby       Cloth.SlangCodegen.Saxpby (the AVBD kernel): the "+1"
#                 dispatch of the fiber and references_max probes and the
#                 whole-buffer writer of the large-buffer probe;
#  - half_load    Probes.HalfLoad: the f16 storage read;
#  - probe_add, probe_scale, probe_acc   Probes.Set0: one set-0 layout
#                 (b0 params, b1-b3 sources, b4 destination) for the shared
#                 uniform-set and in-place probes.
# saxpby and half_load compile as the AVBD kernels do. Each probe_* kernel
# compiles three ways, because slangc (2026.13.1) keeps unused parameters
# only when -preserve-params is given AND the optimiser is off:
#   <k>.spv          -O0 -preserve-params   all five bindings (the layout Cut 3 wants)
#   <k>_stripped.spv -O0                    unused sources dropped (the control)
#   <k>_o1pp.spv     -preserve-params       default -O1: dropped again; the flag alone is not enough
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEAN="${CLOTH_LEAN:-$ROOT/lean}"
BUILD="${BUILD_DIR:-$ROOT/build}"
SLANGC="${SLANGC:-slangc}"
command -v "$SLANGC" >/dev/null 2>&1 || SLANGC="$HOME/scoop/apps/vulkan/current/Bin/slangc"

MODE=check
case "${1:-}" in
	--no-emit) MODE=none ;;
	--update) MODE=update ;;
	"") ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
esac

KERNELS=$(grep -v '^#' "$HERE/kernels.txt" | tr '\n' ' ')

if [ "$MODE" != none ]; then
	command -v lake >/dev/null 2>&1 || { echo "error: lake not on PATH (or pass --no-emit)" >&2; exit 1; }
	TMP="$(mktemp -d)"
	echo "== emitting the probe kernels from Lean at $LEAN =="
	( cd "$LEAN" && lake exe emit_probes "$TMP" >/dev/null )
	DIFFS=0
	for k in $KERNELS; do
		[ -f "$TMP/$k.slang" ] || { echo "error: emit_probes did not write $k.slang" >&2; exit 1; }
		if [ "$MODE" = update ]; then
			cp "$TMP/$k.slang" "$HERE/slang/$k.slang"
		elif ! cmp -s <(tr -d '\r' < "$TMP/$k.slang") <(tr -d '\r' < "$HERE/slang/$k.slang" 2>/dev/null); then
			echo "DIFF: Lean emits a different slang/$k.slang:" >&2
			diff <(tr -d '\r' < "$HERE/slang/$k.slang" 2>/dev/null) <(tr -d '\r' < "$TMP/$k.slang") >&2 || true
			DIFFS=$((DIFFS + 1))
		fi
	done
	rm -rf "$TMP"
	if [ "$DIFFS" -gt 0 ]; then
		echo "error: $DIFFS committed probe kernel(s) differ from the Lean emission; rerun with --update and review" >&2
		exit 1
	fi
	echo "== $(echo $KERNELS | wc -w) kernels $([ "$MODE" = update ] && echo written || echo 'match Lean') =="
fi

rm -rf "$BUILD/spv-probes"
mkdir -p "$BUILD/spv-probes"
spv() { # <out name> <kernel> [slangc flags...]
	local out="$1" k="$2"
	shift 2
	( cd "$HERE" && "$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main "$@" \
		-o "$BUILD/spv-probes/$out.spv" "slang/$k.slang" )
}
for k in $KERNELS; do
	case "$k" in
	probe_*)
		spv "$k" "$k" -O0 -preserve-params
		spv "${k}_stripped" "$k" -O0
		spv "${k}_o1pp" "$k" -preserve-params ;;
	*)
		spv "$k" "$k" ;;
	esac
done
python "$HERE/../embed_spv.py" "$BUILD/spv-probes" "$BUILD/probes_kernels.inc"
