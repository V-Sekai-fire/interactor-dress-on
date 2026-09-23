#!/usr/bin/env bash
# Build the host-native G3.graph oracle (CMakeLists.txt says what it is):
#   tests/ggml_graph_oracle/build.sh          -> $B/ggml_graph_oracle.exe
# llvm-mingw clang (x86_64), like tests/ggml_rd_kernels. The build directory
# is outside the checkout (C:/b/...): ggml-vulkan's shader tree plus a long
# worktree path exceed Windows' path limit (ORACLE_BUILD_DIR overrides).
# Needs the Vulkan SDK (VULKAN_SDK: headers, vulkan-1.lib, glslc).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MINGW="$(ls -d "$HOME"/llvm-mingw/llvm-mingw-*-ucrt-x86_64/bin | sort | tail -1)"
export PATH="$MINGW:$PATH"
NINJA="$(command -v ninja || echo "$HOME/.pixi/bin/ninja.exe")"
B="${ORACLE_BUILD_DIR:-C:/b/ggml-graph-oracle-$(printf %s "$ROOT" | md5sum | cut -c1-8)}"
if [ ! -f "$B/build.ninja" ]; then
	cmake -S "$(cygpath -m "$HERE")" -B "$B" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
		-DCMAKE_C_COMPILER=x86_64-w64-mingw32-clang -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-clang++ \
		-DCMAKE_BUILD_TYPE=Release
fi
cmake --build "$B" --target "${1:-ggml_graph_oracle}"
echo "oracle: $B/ggml_graph_oracle.exe"
