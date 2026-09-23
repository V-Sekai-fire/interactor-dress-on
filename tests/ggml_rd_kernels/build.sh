#!/usr/bin/env bash
# Build and run the host-native L2 of the ggml-rd kernels (l2.h), both arms:
#   the kernels vs ggml-cpu         -> gates/3-ggml-rd/kernels/l2.log          (must PASS)
#   --control=swap-nb               -> gates/3-ggml-rd/kernels/l2-control.log  (must PASS:
#                                      every case whose addresses the swap changes FAILs)
# Uses llvm-mingw clang (x86_64), like gates/0f-runtime/ggml_host. The build
# directory defaults outside the checkout (C:/b/...): long worktree paths
# plus ggml's CMake tree exceed Windows' path limit. It is named after the
# checkout, so op-family worktrees each get their own (L2_BUILD_DIR overrides).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MINGW="$(ls -d "$HOME"/llvm-mingw/llvm-mingw-*-ucrt-x86_64/bin | sort | tail -1)"
export PATH="$MINGW:$PATH"
NINJA="$(command -v ninja || echo "$HOME/.pixi/bin/ninja.exe")"
B="${L2_BUILD_DIR:-C:/b/ggml-rd-l2-$(printf %s "$ROOT" | md5sum | cut -c1-8)}"
OUT="$ROOT/gates/3-ggml-rd/kernels"
mkdir -p "$OUT"
if [ ! -f "$B/build.ninja" ]; then
	cmake -S "$(cygpath -m "$HERE")" -B "$B" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
		-DCMAKE_C_COMPILER=x86_64-w64-mingw32-clang -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-clang++ \
		-DCMAKE_BUILD_TYPE=Release
fi
cmake --build "$B" --target ggml_rd_l2
rc=0
"$B/ggml_rd_l2.exe" > "$OUT/l2.log" 2>&1 || rc=1
"$B/ggml_rd_l2.exe" --control=swap-nb > "$OUT/l2-control.log" 2>&1 || rc=1
tail -3 "$OUT/l2.log"
tail -3 "$OUT/l2-control.log"
exit $rc
