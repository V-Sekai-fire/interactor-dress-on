#!/usr/bin/env bash
# The curvenet libraries with the Lean-emitted Cassie kernels, for riscv64:
# configure a guest tree (the org sysroot's toolchain.cmake, as build.sh)
# with CURVENET_KERNELS_PENDING=OFF, build curvenet_compile_check (every
# curvenet library, cassie_kernels included), then list what cassie_core
# needs from cassie_slang_dispatch:: and what cassie_kernels defines there.
#
#   gates/4-curvenet/riscv64-kernels.sh > gates/4-curvenet/riscv64-kernels.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SYSROOT="${RISCV64_SYSROOT:-/c/contract-manifest/3-interactor/mujoco-sandbox-demo/third_party/riscv64-sysroot}"
OUT="${OUT:-$ROOT/build/rv64-curvenet}"
NINJA="$(command -v ninja || echo "$HOME/.pixi/bin/ninja.exe")"
LLVM="$HOME/scoop/apps/llvm/current/bin"
export PATH="$LLVM:$PATH"
NM="$LLVM/llvm-nm"
m() { cygpath -m "$1" 2>/dev/null || echo "$1"; }

cmake -S "$(m "$ROOT")" -B "$(m "$OUT")" -G Ninja \
	-DCMAKE_MAKE_PROGRAM="$(m "$NINJA")" \
	-DCMAKE_TOOLCHAIN_FILE="$(m "$SYSROOT/toolchain.cmake")" \
	-DCMAKE_BUILD_TYPE=Release -DCURVENET_KERNELS_PENDING=OFF >/dev/null
cmake --build "$OUT" --target curvenet_compile_check 2>&1 | grep -v '^\[' || true

lib() { find "$OUT" -name "lib$1.a" | head -1; }
echo "cassie_kernels: $(basename "$(lib cassie_kernels)"), $("$NM" "$(lib cassie_kernels)" | grep -c ' [TtWw] ') text symbols, arch: $(file -b "$(find "$OUT" -path '*cassie_kernels.dir*' -name '*.o*' | head -1)" | cut -d, -f1-2)"
need=$(mktemp); have=$(mktemp)
"$NM" -C -u "$(lib cassie_core)" | awk '{ $1=""; sub(/^ /,""); print }' | grep '^cassie_slang_dispatch::' | sort -u >"$need"
"$NM" -C --defined-only "$(lib cassie_kernels)" | awk 'NF>=3{ $1=""; $2=""; sub(/^  /,""); print }' | grep '^cassie_slang_dispatch::' | sort -u >"$have"
echo "cassie_core needs $(wc -l <"$need") cassie_slang_dispatch symbols:"
sed 's/^/  /' "$need"
echo "cassie_kernels defines $(wc -l <"$have"):"
sed 's/^/  /' "$have"
unres=$(comm -23 "$need" "$have" | wc -l)
echo "unresolved cassie_slang_dispatch symbols: $unres"
# Every emitted kernel body is in the archive, in its own namespace.
for k in curve_casteljau curve_generate_bezier curve_newton curve_rdp spmv; do
	echo "  cassie_slang_$k::main_0_Thread: $("$NM" -C --defined-only "$(lib cassie_kernels)" | grep -c "cassie_slang_$k::main_0_Thread")"
done
echo "Eigen:: symbols in cassie_kernels + cassie_core: $( { "$NM" -C "$(lib cassie_kernels)"; "$NM" -C "$(lib cassie_core)"; } | grep -c 'Eigen::' || true)"
rm -f "$need" "$have"
[ "$unres" -eq 0 ]
