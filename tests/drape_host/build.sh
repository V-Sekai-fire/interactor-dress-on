#!/usr/bin/env bash
# Build the host-native drape harness (see drape_host.cpp). SAN=1 adds
# AddressSanitizer and UBSan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$ROOT/build/drape_host}"
mkdir -p "$OUT"
CXX="${CXX:-clang++}"
FLAGS="-std=c++20 -O1 -g -ffp-contract=off"
if [ "${SAN:-0}" = 1 ]; then FLAGS="$FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer"; fi
"$CXX" $FLAGS \
	-I"$ROOT/guest/drape" -I"$ROOT/guest" -I"$ROOT/guest/avbd" -I"$ROOT/kernels/avbd/cpp" -I"$ROOT/guest/avbd/slang-rt" \
	"$HERE/drape_host.cpp" "$ROOT/guest/jobs.cpp" "$ROOT/guest/drape/drape_scene.cpp" "$ROOT/guest/drape/primitives.cpp" \
	"$ROOT/guest/drape/body_mesh.cpp" \
	"$ROOT/guest/avbd/avbd_cpu.cpp" "$ROOT/guest/avbd/avbd_cpu_backward.cpp" "$ROOT/guest/avbd/avbd_topology.cpp" \
	-o "$OUT/drape_host"
echo "$OUT/drape_host"
