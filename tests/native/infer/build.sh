#!/usr/bin/env bash
# Build the infer stage's host oracle (tests/native/infer/CMakeLists.txt) with
# llvm-mingw: vendor/trellis2 + vendor/skin-tokens over the org ggml, ggml-cpu
# and ggml-vulkan, then run the no-model tests (ctest -LE model). Every test
# is capped at IDO_TEST_TIMEOUT (300 s) and the whole ctest at that times the
# test count; a timeout is a FAIL. Single host runs: run.sh (same cap).
#
#   bash tests/native/infer/build.sh [targets...]
#
# Environment (all optional):
#   INFER_BUILD    build dir (default C:/b/infer-host)
#   IDO_GGML_DIR   ggml tree (default C:/contract-manifest/2-contract/ggml
#                  = V-Sekai-fire/ggml @04b55bba; vendor/ggml after Cut 3)
#   IDO_MODELS_DIR models/ with SkinTokens-GGUF/F16 (default: <repo>/models;
#                  a worktree points it at the main checkout's)
#   LLVM_MINGW     toolchain (default ~/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64)
#   NINJA          ninja.exe (default: on PATH, else a pixi/rattler env's)
#   INFER_JOBS     ninja -j (default 8: the desk is shared)
#   IDO_GGML_VULKAN  ON|OFF (default ON; needs VULKAN_SDK with glslc)
#   IDO_TEST_TIMEOUT hard per-test cap in seconds (default 300)
#   CTEST_ARGS     extra ctest arguments (e.g. "-L naf")
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
# ggml-vulkan builds vulkan-shaders-gen as an ExternalProject, a nested
# cmake that sees only the environment: it needs ninja on PATH and CC/CXX.
export PATH="$(dirname "$(cygpath -u "$NINJA")"):$PATH"
export CC="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang.exe"
export CXX="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang++.exe"

EXTRA=()
[ -n "${IDO_GGML_DIR:-}" ] && EXTRA+=("-DIDO_GGML_DIR=$IDO_GGML_DIR")
[ -n "${IDO_GGML_VULKAN:-}" ] && EXTRA+=("-DIDO_GGML_VULKAN=$IDO_GGML_VULKAN")
[ -n "${IDO_TEST_TIMEOUT:-}" ] && EXTRA+=("-DIDO_TEST_TIMEOUT=$IDO_TEST_TIMEOUT")
[ -n "${IDO_MODELS_DIR:-}" ] && EXTRA+=("-DIDO_MODELS_DIR=$IDO_MODELS_DIR")

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
  CAP="${IDO_TEST_TIMEOUT:-300}"
  # --timeout covers any test the configure-time cap missed; the outer
  # timeout bounds the run (a hung ctest itself is a FAIL too).
  N="$(ctest --test-dir "$BUILD" -N -LE model ${CTEST_ARGS:-} | sed -n 's/^Total Tests: //p')"
  timeout -k 10 $(( CAP * (${N:-1} + 1) )) \
    ctest --test-dir "$BUILD" -LE model --timeout "$CAP" --output-on-failure ${CTEST_ARGS:-}
fi
