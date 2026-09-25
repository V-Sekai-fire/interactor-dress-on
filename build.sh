#!/usr/bin/env bash
# Build the guest ELFs (one per stage: dress_on, drape, curvenet, fit, usd; Gate
# 0F's probes; Gate 3's ggml_test) for the RISC-V sandbox and drop them into
# project/.
#
#   ./build.sh                # configure (once) + build
#   RISCV64_SYSROOT=... ./build.sh
#   GGML_EMIT=1 ./build.sh    # also check kernels/ggml against a fresh Lean emission
#   BUILD_FIT=0 ./build.sh    # skip fit.elf (cloth-fit / PolyFEM, the long part)
#   BUILD_DIR=C:/b/ido6-zb FIT_MARCH=rv64gc_zba_zbb_zbs_zbc FIT_ELF=fit_zb BUILD_TARGETS=fit_zb ./build.sh
#                             # Gate 6.P's ISA A/B: project/fit_zb.elf, the solver at another -march
#
# fit.elf needs the org forks (tools/fit/prepare_forks.sh, run here) and the
# CPM packages cloth-fit pulls without a fork (CPM_SOURCE_CACHE, default
# C:/b/cpm-native, the cache the native fit build fills). project/fit.elf is
# not committed; its sha256 goes into the gate logs.
#
# Needs: cmake, ninja, a clang++ with a riscv64 target (auto-located if the
# bare clang++ is mingw-only), and the riscv64 glibc sysroot from the org's
# mujoco demo (third_party/riscv64-sysroot: toolchain.cmake + sysroot/). The
# sysroot is 150 MB and is not vendored here; point RISCV64_SYSROOT at it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSROOT="${RISCV64_SYSROOT:-/c/contract-manifest/3-interactor/mujoco-sandbox-demo/third_party/riscv64-sysroot}"
BUILD="${BUILD_DIR:-$HERE/build}"

if [ ! -f "$SYSROOT/toolchain.cmake" ]; then
	echo "error: no toolchain.cmake under RISCV64_SYSROOT=$SYSROOT" >&2
	exit 1
fi

# The toolchain file invokes a bare clang++, so a riscv64-capable one must
# resolve first on PATH.
has_riscv() { "$1" --print-targets 2>/dev/null | grep -qi riscv64; }
if ! { command -v clang++ >/dev/null 2>&1 && has_riscv clang++; }; then
	FOUND=""
	for c in "$HOME/scoop/apps/llvm/current/bin/clang++" "/c/Program Files/LLVM/bin/clang++"; do
		if [ -x "$c" ] && has_riscv "$c"; then FOUND="$c"; break; fi
	done
	if [ -z "$FOUND" ]; then
		echo "error: no clang++ with a riscv64 target found (a mingw-only clang will not do)" >&2
		exit 1
	fi
	export PATH="$(dirname "$FOUND"):$PATH"
fi

NINJA="$(command -v ninja || true)"
[ -n "$NINJA" ] || NINJA="$HOME/.pixi/bin/ninja.exe"
[ -x "$NINJA" ] || { echo "error: ninja not found" >&2; exit 1; }

# No V extension: rv64gc plus the bit-manipulation extensions. Ubuntu clang
# 18's vector code traps on the Linux addon (AGENTS.md facts), and the ELF
# does not need it.
# CMake wants the toolchain path in its own spelling.
TOOLCHAIN="$(cygpath -m "$SYSROOT/toolchain.cmake" 2>/dev/null || echo "$SYSROOT/toolchain.cmake")"

# The AVBD kernels: Lean -> Slang (committed) -> cpp (committed) + SPIR-V
# embedded into the build dir. --no-emit skips lake; set AVBD_EMIT=1 to
# regenerate the Slang from lean/ (a subtree of cloth-dynamics/lean).
if [ "${AVBD_EMIT:-0}" = 1 ]; then
	BUILD_DIR="$BUILD" bash "$HERE/kernels/avbd/gen.sh"
else
	BUILD_DIR="$BUILD" bash "$HERE/kernels/avbd/gen.sh" --no-emit
fi

# The Gate 0F probe kernels, same pattern; PROBES_EMIT=1 re-emits them from
# lean/ and fails if they differ from the committed kernels/probes/slang/.
if [ "${PROBES_EMIT:-0}" = 1 ]; then
	BUILD_DIR="$BUILD" bash "$HERE/kernels/probes/gen.sh"
else
	BUILD_DIR="$BUILD" bash "$HERE/kernels/probes/gen.sh" --no-emit
fi

# The drape (L-BFGS-B) kernels, the same way: committed Slang and cpp, SPIR-V
# embedded into $BUILD/drape_kernels.inc. DRAPE_EMIT=1 re-emits from lean/.
if [ "${DRAPE_EMIT:-0}" = 1 ]; then
	BUILD_DIR="$BUILD" bash "$HERE/kernels/drape/gen.sh"
else
	BUILD_DIR="$BUILD" bash "$HERE/kernels/drape/gen.sh" --no-emit
fi

# The ggml-rd kernels (Cut 3), same pattern: GGML_EMIT=1 re-emits them from
# lean/ and fails if they differ from the committed kernels/ggml/slang/ (an
# op family writes them with kernels/ggml/gen.sh --update).
if [ "${GGML_EMIT:-0}" = 1 ]; then
	BUILD_DIR="$BUILD" bash "$HERE/kernels/ggml/gen.sh"
else
	BUILD_DIR="$BUILD" bash "$HERE/kernels/ggml/gen.sh" --no-emit
fi

# A Linux slangc writes the cpp emits with an absolute include of its prelude
# where the reference inlines it; put the inline form back (byte-identical to
# the committed emits, so nothing churns).
python3 "$HERE/tools/inline_prelude.py" "$HERE"
BUILD_FIT="${BUILD_FIT:-1}"
if [ "$BUILD_FIT" = 0 ]; then WITH_FIT=OFF; else WITH_FIT=ON; fi
if [ "$WITH_FIT" = ON ]; then
	bash "$HERE/tools/fit/prepare_forks.sh"
	export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-C:/b/cpm-native}"
fi

if [ ! -f "$BUILD/build.ninja" ]; then
	cmake -S "$HERE" -B "$BUILD" -G Ninja \
		-DCMAKE_MAKE_PROGRAM="$NINJA" \
		-DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
		-DCMAKE_BUILD_TYPE=Release \
		-DSANDBOX_RISCV_EXT_V=OFF \
		-DDRESS_ON_WITH_FIT="$WITH_FIT" \
		${FIT_MARCH:+-DFIT_MARCH="$FIT_MARCH"} ${FIT_ELF:+-DFIT_ELF="$FIT_ELF"}
elif ! grep -q "^DRESS_ON_WITH_FIT:BOOL=$WITH_FIT\$" "$BUILD/CMakeCache.txt"; then
	cmake -B "$BUILD" -DDRESS_ON_WITH_FIT="$WITH_FIT"
fi
# BUILD_TARGETS (space-separated) limits the build, e.g. to one A/B fit ELF.
# shellcheck disable=SC2086
cmake --build "$BUILD" ${BUILD_TARGETS:+--target $BUILD_TARGETS} -- -j "${BUILD_JOBS:-8}"
ls -la "$HERE/project/dress_on.elf" "$HERE/project/drape.elf" "$HERE/project/curvenet.elf" "$HERE/project/probes.elf" \
	"$HERE/project/ggml_test.elf" "$HERE/project/rd_worker.elf"
ls -la "$HERE/project/usd.elf" 2>/dev/null && sha256sum "$HERE/project/usd.elf" || echo "usd.elf: not built (no OpenUSD riscv64 build at USD_RV64_DIR)"
if [ "$WITH_FIT" = ON ]; then
	ls -la "$HERE/project/${FIT_ELF:-fit}.elf"
	sha256sum "$HERE/project/${FIT_ELF:-fit}.elf"
fi
