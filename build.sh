#!/usr/bin/env bash
# Build the guest ELFs (one per stage: dress_on, drape, curvenet; and Gate
# 0F's probes) for the RISC-V sandbox and drop them into project/.
#
#   ./build.sh                # configure (once) + build
#   RISCV64_SYSROOT=... ./build.sh
#   GGML_SRC=<V-Sekai-fire/ggml checkout> ./build.sh   # probes.elf's probe 15, until vendor/ggml
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

if [ ! -f "$BUILD/build.ninja" ]; then
	cmake -S "$HERE" -B "$BUILD" -G Ninja \
		-DCMAKE_MAKE_PROGRAM="$NINJA" \
		-DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
		-DCMAKE_BUILD_TYPE=Release \
		${GGML_SRC:+-DGGML_SRC="$GGML_SRC"}
fi
cmake --build "$BUILD"
ls -la "$HERE/project/dress_on.elf" "$HERE/project/drape.elf" "$HERE/project/curvenet.elf" "$HERE/project/probes.elf"
