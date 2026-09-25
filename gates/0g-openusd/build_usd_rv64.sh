#!/usr/bin/env bash
# Gate 0G: cross-compile oneTBB 2021.12.0 and OpenUSD 26.05 (read-only
# upstream sources outside the repo, see README) for the godot-sandbox guest:
# riscv64, static, no Python, no imaging -- only what reading a stage needs.
#
#   USD_SRC=C:/b/usd2605 TBB_SRC=C:/b/oneTBB-2021.12.0 OUT=C:/b/g0g ./build_usd_rv64.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSROOT="${RISCV64_SYSROOT:-$HERE/../../../../5-repository/riscv64-sysroot}"
USD_SRC="${USD_SRC:-C:/b/usd2605}"
TBB_SRC="${TBB_SRC:-C:/b/oneTBB-2021.12.0}"
OUT="${OUT:-C:/b/g0g}"
NINJA="$(command -v ninja || echo "$HOME/.pixi/bin/ninja")"
# cygpath exists only on the Windows desk; elsewhere the path is already CMake's spelling.
mpath() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
TC="$(mpath "$SYSROOT/toolchain.cmake")"
# The guest libraries build at plain rv64gc (as ggml-cpu does in probes.elf).
FLAGS="-march=rv64gc -U__riscv_v_intrinsic"
# A command-line CMAKE_CXX_FLAGS replaces the toolchain's CMAKE_CXX_FLAGS_INIT, so
# its libstdc++ -isystem paths are repeated here.
RVS="$(mpath "$SYSROOT/sysroot")"
CXXFLAGS_RV="-isystem $RVS/include/c++/14 -isystem $RVS/include/c++/14/riscv64-linux-gnu $FLAGS"

if [ ! -f "$OUT/inst/lib/libtbb.a" ]; then
	cmake -S "$TBB_SRC" -B "$OUT/tbb" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
		-DCMAKE_TOOLCHAIN_FILE="$TC" -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$CXXFLAGS_RV" \
		-DBUILD_SHARED_LIBS=OFF -DTBB_TEST=OFF -DTBB_STRICT=OFF -DTBBMALLOC_BUILD=OFF \
		-DTBB_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$OUT/inst"
	cmake --build "$OUT/tbb" -- -j 16
	cmake --install "$OUT/tbb"
fi

if [ ! -f "$OUT/usd/build.ninja" ]; then
	cmake -S "$USD_SRC" -B "$OUT/usd" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
		-DCMAKE_TOOLCHAIN_FILE="$TC" -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$CXXFLAGS_RV" \
		-DCMAKE_FIND_ROOT_PATH="$OUT/inst" -DTBB_DIR="$OUT/inst/lib/cmake/TBB" \
		-DCMAKE_INSTALL_PREFIX="$OUT/usdinst" \
		-DBUILD_SHARED_LIBS=OFF -DPXR_BUILD_MONOLITHIC=OFF \
		-DPXR_ENABLE_PYTHON_SUPPORT=OFF -DPXR_BUILD_IMAGING=OFF -DPXR_BUILD_USD_IMAGING=OFF \
		-DPXR_BUILD_USDVIEW=OFF -DPXR_BUILD_TESTS=OFF -DPXR_BUILD_EXAMPLES=OFF \
		-DPXR_BUILD_TUTORIALS=OFF -DPXR_BUILD_USD_TOOLS=OFF -DPXR_BUILD_HTML_DOCUMENTATION=OFF \
		-DPXR_BUILD_PYTHON_DOCUMENTATION=OFF -DPXR_ENABLE_GL_SUPPORT=OFF -DPXR_ENABLE_VULKAN_SUPPORT=OFF \
		-DPXR_ENABLE_MATERIALX_SUPPORT=OFF -DPXR_ENABLE_OPENVDB_SUPPORT=OFF -DPXR_ENABLE_PTEX_SUPPORT=OFF \
		-DPXR_ENABLE_OSL_SUPPORT=OFF -DPXR_BUILD_ALEMBIC_PLUGIN=OFF -DPXR_BUILD_DRACO_PLUGIN=OFF \
		-DPXR_BUILD_EXEC=OFF -DPXR_BUILD_USD_VALIDATION=OFF -DPXR_ENABLE_PRECOMPILED_HEADERS=OFF \
		-DCMAKE_POLICY_VERSION_MINIMUM=3.5
fi
# Only the libraries a stage read needs (and what they pull in).
cmake --build "$OUT/usd" --target usdSkel usdShade usdGeom kind -- -j "${JOBS:-16}" -k 0
