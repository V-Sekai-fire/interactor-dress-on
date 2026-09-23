#!/usr/bin/env bash
# Build fit_native (tests/native/fit/CMakeLists.txt) with llvm-mingw under the
# tools/native pixi env (cmake, ninja, python; boost/tbb headers for the
# interim OpenVDB FitForm).
#
#   pixi run --manifest-path tools/native/pixi.toml bash tests/native/fit/build.sh [targets...]
#
# Environment (all optional):
#   FIT_BUILD         build dir (default C:/b/fit-native)
#   FIT_SDF           openvdb (default, interim) | sdfgrid
#   FIT_NUMERICS      guest (default) | upstream (the oracle's numerics, a control)
#   CPM_SOURCE_CACHE  CPM cache for packages without an org fork (default C:/b/cpm-native)
#   LLVM_MINGW        toolchain (default ~/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64)
#   FIT_JOBS          ninja -j (default 8: the desk is shared)
#
# The org forks (tools/forks.tsv) come from .forks/<name> via CPM_<pkg>_SOURCE;
# polysolve from .forks/polysolve-guest, the pin plus
# vendor/cloth-fit-patches/polysolve-embedded-specs.patch
# (tools/fit/prepare_forks.sh).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
ROOT="$(cd "$HERE/../../.." && pwd -W 2>/dev/null || pwd)"
BUILD="${FIT_BUILD:-C:/b/fit-native}"
FORKS="${FORKS_DIR:-$ROOT/.forks}"
FIT_SDF="${FIT_SDF:-openvdb}"
LLVM_MINGW="${LLVM_MINGW:-$(cygpath -m "$HOME")/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64}"
export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-C:/b/cpm-native}"
: "${CONDA_PREFIX:?run under pixi run --manifest-path tools/native/pixi.toml}"
PREFIX="$(cygpath -m "$CONDA_PREFIX")"
export PATH="$(cygpath -u "$LLVM_MINGW")/bin:$PATH"

FORKS_DIR="$FORKS" bash "$ROOT/tools/fit/prepare_forks.sh"

if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -S "$HERE" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DFIT_SDF="$FIT_SDF" \
    -DFIT_NUMERICS="${FIT_NUMERICS:-guest}" \
    -DCMAKE_PREFIX_PATH="$PREFIX/Library" \
    -DPython3_EXECUTABLE="$(cygpath -m "$(command -v python)")" \
    -DCMAKE_C_COMPILER="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang.exe" \
    -DCMAKE_CXX_COMPILER="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang++.exe" \
    -DLIBM_LIBRARY="$LLVM_MINGW/x86_64-w64-mingw32/lib/libm.a" \
    -DCPM_SOURCE_CACHE="$CPM_SOURCE_CACHE" \
    -DCPM_polysolve_SOURCE="$FORKS/polysolve-guest" \
    -DCPM_ipc-toolkit_SOURCE="$FORKS/ipc-toolkit" \
    -DCPM_openvdb_SOURCE="$FORKS/openvdb" \
    -DCPM_libigl_SOURCE="$FORKS/libigl" \
    -DCPM_nlohmann_json_SOURCE="$FORKS/json"
fi

cmake --build "$BUILD" --target "${@:-fit_native}" -- -j "${FIT_JOBS:-8}"
