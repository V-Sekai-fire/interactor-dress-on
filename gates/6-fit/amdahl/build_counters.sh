#!/usr/bin/env bash
# Build fit_native with the [amdahl] counters (make_counters_src.py) in a
# scratch build dir. Same configure as tests/native/fit/build.sh for the
# fit.elf twin (FIT_SDF=sdfgrid, kernel sampler, guest numerics, serial TBB),
# except polysolve and ipc-toolkit come from the instrumented copies.
#
#   bash gates/6-fit/amdahl/build_counters.sh
#
# Env: AMDAHL_SRC (default C:/b/amdahl-src), AMDAHL_BUILD (C:/b/fit-native-amdahl),
#      PIXI_ENV (the cloth-fit-native rattler env), FIT_JOBS (12).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
ROOT="$(cd "$HERE/../../.." && pwd -W 2>/dev/null || pwd)"
SRC="${AMDAHL_SRC:-C:/b/amdahl-src}"
BUILD="${AMDAHL_BUILD:-C:/b/fit-native-amdahl}"
ENV="${PIXI_ENV:-C:/Users/ernest.lee/AppData/Local/rattler/cache/envs/cloth-fit-native-11201012261886535436/envs/default}"
LLVM_MINGW="${LLVM_MINGW:-C:/Users/ernest.lee/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64}"
FORKS="$ROOT/.forks"
export PATH="$(cygpath -u "$ENV/Library/bin"):$(cygpath -u "$ENV"):$(cygpath -u "$LLVM_MINGW")/bin:$PATH"
export CPM_SOURCE_CACHE=C:/b/cpm-native

if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -S "$ROOT/tests/native/fit" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DFIT_SDF=sdfgrid -DFIT_SDF_SAMPLER=kernel -DFIT_NUMERICS=guest -DFIT_TBB=serial \
    -DCMAKE_PREFIX_PATH="$ENV/Library" \
    -DPython3_EXECUTABLE="$ENV/python.exe" \
    -DCMAKE_C_COMPILER="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang.exe" \
    -DCMAKE_CXX_COMPILER="$LLVM_MINGW/bin/x86_64-w64-mingw32-clang++.exe" \
    -DLIBM_LIBRARY="$LLVM_MINGW/x86_64-w64-mingw32/lib/libm.a" \
    -DCPM_SOURCE_CACHE="$CPM_SOURCE_CACHE" \
    -DCPM_polysolve_SOURCE="$SRC/polysolve" \
    -DCPM_ipc-toolkit_SOURCE="$SRC/ipc-toolkit" \
    -DCPM_openvdb_SOURCE="$FORKS/openvdb" \
    -DCPM_libigl_SOURCE="$FORKS/libigl" \
    -DCPM_nlohmann_json_SOURCE="$FORKS/json"
fi
cmake --build "$BUILD" --target fit_native -- -j "${FIT_JOBS:-12}"
