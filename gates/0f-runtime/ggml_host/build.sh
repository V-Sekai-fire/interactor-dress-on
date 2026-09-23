#!/usr/bin/env bash
# Build and run the host-native ggml control for probe 15.
#   GGML_SRC=<V-Sekai-fire/ggml checkout> gates/0f-runtime/ggml_host/build.sh
#     -> ../ggml_host.txt, read by gate_runtime.gd
# GGML_SRC defaults to vendor/ggml (Cut 3); HOST_BUILD_DIR to build-ggml-host/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINGW="$(ls -d "$HOME"/llvm-mingw/llvm-mingw-*-ucrt-x86_64/bin | sort | tail -1)"
export PATH="$MINGW:$PATH"
NINJA="$(command -v ninja || echo "$HOME/.pixi/bin/ninja.exe")"
B="${HOST_BUILD_DIR:-$HERE/../../../build-ggml-host}"
if [ ! -f "$B/build.ninja" ]; then
	cmake -S "$(cygpath -m "$HERE")" -B "$B" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
		-DCMAKE_C_COMPILER=x86_64-w64-mingw32-clang -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-clang++ \
		-DCMAKE_BUILD_TYPE=Release ${GGML_SRC:+-DGGML_SRC="$GGML_SRC"}
fi
cmake --build "$B" --target ggml_host
"$B/ggml_host.exe" 256 | tee "$HERE/../ggml_host.txt"
