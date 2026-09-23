#!/usr/bin/env bash
# Build upstream cloth-fit's PolyFEM_bin natively from vendor/cloth-fit
# (llvm-mingw, pixi deps), the Cut 6 oracle, plus tools/fit/openvdb_dump
# against the same configure (added by openvdb_dump_hook.cmake as a deferred
# target; PolyFEM_bin and the vendored sources are untouched).
#
# Run from anywhere:
#   pixi run --manifest-path tools/native/pixi.toml bash tools/native/build_upstream.sh [targets...]
# Default targets: PolyFEM_bin openvdb_dump.
#
# Environment (all optional):
#   CF_BUILD           build dir (default <repo>/build-native; from a long
#                      checkout path use a short one such as C:/b/cf-nat)
#   CPM_SOURCE_CACHE   CPM cache for the packages not overridden below
#                      (default <repo>/.cpm-cache-native)
#   LLVM_MINGW         toolchain (default ~/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64)
#   CF_CONFIGURE_ONLY  1: stop after the CMake configure
#
# The five packages with V-Sekai-fire forks (tools/forks.tsv) come from
# .forks/<name> via CPM_<pkg>_SOURCE, fetched first by tools/forks/fetch.sh at
# the pinned revs; CPM never contacts their upstream URLs. The rest still
# resolve from upstream through CPM (listed in README.md).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
ROOT="$(cd "$HERE/../.." && pwd -W 2>/dev/null || pwd)"
SRC="$ROOT/vendor/cloth-fit"
BUILD="${CF_BUILD:-$ROOT/build-native}"
FORKS="${FORKS_DIR:-$ROOT/.forks}"
LLVM_MINGW="${LLVM_MINGW:-$(cygpath -m "$HOME")/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64}"
export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-$ROOT/.cpm-cache-native}"
: "${CONDA_PREFIX:?run under pixi run --manifest-path tools/native/pixi.toml}"
PREFIX="$(cygpath -m "$CONDA_PREFIX")"

# oneTBB calls windres by bare name; llvm-mingw ships it.
export PATH="$(cygpath -u "$LLVM_MINGW")/bin:$PATH"

FORKS_DIR="$FORKS" bash "$ROOT/tools/forks/fetch.sh" cloth-fit

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(PolyFEM_bin openvdb_dump)

if [ -f "$BUILD/CMakeCache.txt" ]; then
  home="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")"
  [ "$(cygpath -m "$home")" = "$(cygpath -m "$SRC")" ] \
    || { echo "build_upstream.sh: $BUILD was configured from $home, not $SRC" >&2; exit 1; }
fi

if [ ! -f "$BUILD/build.ninja" ]; then
  # CPM package names as the recipes spell them: gh:<owner>/<repo> gives the
  # repo name; libigl and nlohmann_json pass NAME explicitly.
  cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DPOLYSOLVE_WITH_CHOLMOD=OFF \
    -DPOLYSOLVE_WITH_MKL=OFF \
    -DUSE_BLOSC=OFF \
    -DUSE_ZLIB=OFF \
    -DCMAKE_PREFIX_PATH="$PREFIX/Library" \
    -DCMAKE_C_COMPILER="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang.exe" \
    -DCMAKE_CXX_COMPILER="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang++.exe" \
    -DLIBM_LIBRARY="$LLVM_MINGW/x86_64-w64-mingw32/lib/libm.a" \
    -DCPM_SOURCE_CACHE="$CPM_SOURCE_CACHE" \
    -DCPM_polysolve_SOURCE="$FORKS/polysolve" \
    -DCPM_ipc-toolkit_SOURCE="$FORKS/ipc-toolkit" \
    -DCPM_openvdb_SOURCE="$FORKS/openvdb" \
    -DCPM_libigl_SOURCE="$FORKS/libigl" \
    -DCPM_nlohmann_json_SOURCE="$FORKS/json" \
    -DCMAKE_PROJECT_PolyFEM_INCLUDE="$HERE/openvdb_dump_hook.cmake"
fi

[ "${CF_CONFIGURE_ONLY:-0}" = 1 ] && exit 0
cmake --build "$BUILD" --target "${TARGETS[@]}" -- -k 0
