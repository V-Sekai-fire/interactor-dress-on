#!/usr/bin/env bash
# The ggml-rd kernels, from Lean to both targets.
#
#   kernels/ggml/gen.sh              # emit from Lean, FAIL if it differs from slang/ (or the
#                                    # params header), then cpp + spirv + check + table + embed
#   kernels/ggml/gen.sh --update     # emit from Lean into slang/ and the params header
#                                    # (review the diff), then the same
#   kernels/ggml/gen.sh --no-emit    # use the committed slang/ (no lake); what build.sh runs
#
# One source, two targets (AGENTS.md rule 2):
#   Lean (lean/Ggml, `lake exe emit_ggml`)                  ->  slang/<k>.slang            (committed)
#                                                           ->  guest/ggml-rd/ggml_rd_params.h (committed)
#     slangc -target cpp                                    ->  cpp/<k>_emit.cpp           (committed;
#                                                               the host test harness tests/ggml_rd_kernels
#                                                               runs them against ggml-cpu)
#     slangc -target spirv -O0 -preserve-params             ->  <build>/spv-ggml/<k>.spv + .refl.json
#       spirv-val                                           ->  (fails the run on any error)
#       gen_ggml_kernel_table.py (fixed-layout check)       ->  GgmlKernelTable.inc        (committed)
#       ../embed_spv.py --namespace ggml_kernels            ->  <build>/ggml_kernels.inc   (build artefact)
# controls.txt names control kernels (off the layout on purpose, for gates):
# emitted and compiled the same way, the layout check must REFUSE each, and
# they are embedded apart            ->  <build>/ggml_controls.inc  (namespace ggml_controls)
#
# -O0 -preserve-params: slangc 2026.13.1 keeps a binding the kernel does
# not touch only with both (Gate 0F; at -O1 it drops it from the SPIR-V and
# still lists it in the reflection JSON). Every kernel must bind the same
# six descriptors, so the table step reads the SPIR-V and refuses a kernel
# that differs. The driver compiles SPIR-V with its own optimiser.
#
# cpp_siblings.txt pairs a kernel that shares group memory with its serial
# sibling (slangc -target cpp rejects the barrier, E36107): no cpp is
# emitted for the kernel (the sibling's cpp stands in for it on the host),
# and the table step checks each pair.
#
# Kernels are the names in kernels.txt, in order: a kernel's id (params
# word 0) is its line index. Relative slangc paths (as kernels/avbd/gen.sh)
# keep the committed cpp independent of where the checkout lives.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEAN="${CLOTH_LEAN:-$ROOT/lean}"
BUILD="${BUILD_DIR:-$ROOT/build}"
SLANGC="${SLANGC:-slangc}"
command -v "$SLANGC" >/dev/null 2>&1 || SLANGC="$HOME/scoop/apps/vulkan/current/Bin/slangc"
SPIRV_VAL="${SPIRV_VAL:-spirv-val}"
command -v "$SPIRV_VAL" >/dev/null 2>&1 || SPIRV_VAL="$HOME/scoop/apps/vulkan/current/Bin/spirv-val"
PY="${PYTHON:-python}"

MODE=check
case "${1:-}" in
	--no-emit) MODE=none ;;
	--update) MODE=update ;;
	"") ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
esac

KERNELS=$(grep -v '^#' "$HERE/kernels.txt" | grep -v '^[[:space:]]*$' | tr '\n' ' ')
CONTROLS=$(grep -v '^#' "$HERE/controls.txt" | grep -v '^[[:space:]]*$' | tr '\n' ' ')
SIBLINGS="$HERE/cpp_siblings.txt"
GPU_ONLY=$(grep -v '^#' "$SIBLINGS" | awk 'NF { print $1 }' | tr '\n' ' ')
gpu_only() { case " $GPU_ONLY " in *" $1 "*) return 0 ;; esac; return 1; }

if [ "$MODE" != none ]; then
	command -v lake >/dev/null 2>&1 || { echo "error: lake not on PATH (or pass --no-emit)" >&2; exit 1; }
	FROM="$(mktemp -d)"
	echo "== emitting Slang from Lean at $LEAN =="
	( cd "$LEAN" && lake exe emit_ggml "$FROM" "$FROM/ggml_rd_params.h" >/dev/null )
	mkdir -p "$HERE/slang"
	DIFFS=0
	# same <emitted> <committed>: equal up to CR (a Windows checkout has CRLF).
	same() { cmp -s <(tr -d '\r' < "$1") <(tr -d '\r' < "$2" 2>/dev/null); }
	for k in $KERNELS $CONTROLS; do
		[ -f "$FROM/$k.slang" ] || { echo "error: emit_ggml wrote no $k.slang (kernels.txt or controls.txt names a kernel Lean does not emit)" >&2; exit 1; }
		if [ "$MODE" = update ]; then
			cp "$FROM/$k.slang" "$HERE/slang/$k.slang"
		elif ! same "$FROM/$k.slang" "$HERE/slang/$k.slang"; then
			echo "DIFF: Lean emits a different slang/$k.slang:" >&2
			diff <(tr -d '\r' < "$HERE/slang/$k.slang" 2>/dev/null) <(tr -d '\r' < "$FROM/$k.slang") >&2 || true
			DIFFS=$((DIFFS + 1))
		fi
	done
	HDR="$ROOT/guest/ggml-rd/ggml_rd_params.h"
	if [ "$MODE" = update ]; then
		cp "$FROM/ggml_rd_params.h" "$HDR"
	elif ! same "$FROM/ggml_rd_params.h" "$HDR"; then
		echo "DIFF: Lean emits a different guest/ggml-rd/ggml_rd_params.h" >&2
		DIFFS=$((DIFFS + 1))
	fi
	rm -rf "$FROM"
	if [ "$DIFFS" -gt 0 ]; then
		echo "error: $DIFFS committed file(s) differ from the Lean emission; rerun with --update and review" >&2
		exit 1
	fi
	echo "== $(echo $KERNELS | wc -w) kernels, $(echo $CONTROLS | wc -w) control(s) and the params header $([ "$MODE" = update ] && echo written || echo 'match Lean') =="
fi

mkdir -p "$HERE/cpp"
rm -rf "$BUILD/spv-ggml" "$BUILD/spv-ggml-controls"
mkdir -p "$BUILD/spv-ggml" "$BUILD/spv-ggml-controls"
echo "== slangc -target cpp =="
for k in $KERNELS; do
	# slangc's cpp target rejects GroupMemoryBarrierWithGroupSync (E36107), so
	# a kernel that shares group memory has no cpp emit. Its host stand-in is
	# either its cpp_siblings.txt sibling (same thread group and grid) or its
	# `<k>_serial` sibling in kernels.txt (tests/ggml_rd_kernels/gen_host_kernels.py).
	if gpu_only "$k"; then
		rm -f "$HERE/cpp/${k}_emit.cpp" # its cpp_siblings.txt sibling runs on the host
		continue
	fi
	if grep -q '^groupshared \|GroupMemoryBarrierWithGroupSync' "$HERE/slang/$k.slang"; then
		case " $KERNELS " in
			*" ${k}_serial "*) rm -f "$HERE/cpp/${k}_emit.cpp"; echo "$k: group-shared, no cpp (the host runs ${k}_serial)"; continue ;;
			*) echo "error: $k shares group memory and has neither a cpp_siblings.txt pair nor a ${k}_serial in kernels.txt" >&2; exit 1 ;;
		esac
	fi
	( cd "$HERE" && "$SLANGC" -target cpp -stage compute -entry main -preserve-params \
		-o "cpp/${k}_emit.cpp" "slang/$k.slang" 2>&1 | grep -v "has been renamed to 'main_0'" || true )
	[ -s "$HERE/cpp/${k}_emit.cpp" ] || { echo "error: no cpp for $k" >&2; exit 1; }
done
echo "== slangc -target spirv -O0 -preserve-params (+ reflection), spirv-val =="
for k in $KERNELS; do
	( cd "$HERE" && "$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main -O0 -preserve-params \
		-reflection-json "$BUILD/spv-ggml/$k.refl.json" -o "$BUILD/spv-ggml/$k.spv" "slang/$k.slang" )
	"$SPIRV_VAL" --target-env vulkan1.2 "$BUILD/spv-ggml/$k.spv"
done
echo "== fixed-layout check + kernel table =="
"$PY" "$HERE/gen_ggml_kernel_table.py" --spv-dir "$BUILD/spv-ggml" --siblings "$SIBLINGS" \
	--out "$HERE/GgmlKernelTable.inc" $KERNELS
echo "== embedding SPIR-V =="
# The reflection JSON sits beside each .spv; embed_spv takes only *.spv.
"$PY" "$HERE/../embed_spv.py" --namespace ggml_kernels "$BUILD/spv-ggml" "$BUILD/ggml_kernels.inc"
echo "== controls: compiled alike, the layout check must refuse each =="
for k in $CONTROLS; do
	( cd "$HERE" && "$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main -O0 -preserve-params \
		-reflection-json "$BUILD/spv-ggml-controls/$k.refl.json" -o "$BUILD/spv-ggml-controls/$k.spv" "slang/$k.slang" )
	"$SPIRV_VAL" --target-env vulkan1.2 "$BUILD/spv-ggml-controls/$k.spv"
	if "$PY" "$HERE/gen_ggml_kernel_table.py" --spv-dir "$BUILD/spv-ggml-controls" --check-only "$k" 2>/dev/null; then
		echo "error: control $k passed the layout check" >&2
		exit 1
	fi
	echo "control $k: refused by the layout check, as it must be"
done
"$PY" "$HERE/../embed_spv.py" --namespace ggml_controls "$BUILD/spv-ggml-controls" "$BUILD/ggml_controls.inc"
