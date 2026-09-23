#!/usr/bin/env bash
# Build the infer stage's host oracle (tests/native/infer/CMakeLists.txt) with
# llvm-mingw: vendor/trellis2 + vendor/skin-tokens over the org ggml, ggml-cpu
# only, then run the no-model tests (ctest -LE model).
#
#   bash tests/native/infer/build.sh [targets...]
#
# Environment (all optional):
#   INFER_BUILD    build dir (default C:/b/infer-host)
#   IDO_GGML_DIR   ggml tree (default C:/contract-manifest/2-contract/ggml
#                  = V-Sekai-fire/ggml @04b55bba; vendor/ggml after Cut 3)
#   IDO_JSON_HPP   nlohmann/json 3.11+ single header (default: the org's
#                  stable-diffusion-ggml thirdparty/json.hpp)
#   LLVM_MINGW     toolchain (default ~/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64)
#   NINJA          ninja.exe (default: on PATH, else a pixi/rattler env's)
#   INFER_JOBS     ninja -j (default 8: the desk is shared)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
BUILD="${INFER_BUILD:-C:/b/infer-host}"
LLVM_MINGW="${LLVM_MINGW:-$(cygpath -m "$HOME")/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64}"
export PATH="$(cygpath -u "$LLVM_MINGW")/bin:$PATH"
JOBS="${INFER_JOBS:-8}"

if [ -z "${NINJA:-}" ]; then
  NINJA="$(command -v ninja 2>/dev/null || true)"
  if [ -z "$NINJA" ]; then
    NINJA="$(ls "$LOCALAPPDATA"/rattler/cache/envs/*/envs/default/Library/bin/ninja.exe 2>/dev/null | head -1 || true)"
  fi
fi
: "${NINJA:?no ninja on PATH or in a rattler env; set NINJA}"

EXTRA=()
[ -n "${IDO_GGML_DIR:-}" ] && EXTRA+=("-DIDO_GGML_DIR=$IDO_GGML_DIR")
[ -n "${IDO_JSON_HPP:-}" ] && EXTRA+=("-DIDO_JSON_HPP=$IDO_JSON_HPP")

if [ ! -f "$BUILD/build.ninja" ] || [ ${#EXTRA[@]} -gt 0 ]; then
  cmake -S "$HERE" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM="$(cygpath -m "$NINJA")" \
    -DCMAKE_C_COMPILER="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang.exe" \
    -DCMAKE_CXX_COMPILER="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang++.exe" \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    "${EXTRA[@]}"
fi

cmake --build "$BUILD" -j "$JOBS" ${1:+--target "$@"}
if [ $# -eq 0 ]; then
  ctest --test-dir "$BUILD" -LE model --output-on-failure
fi
