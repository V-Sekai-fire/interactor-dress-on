#!/bin/sh
# Build and run the L2 test of the Lean-emitted L-BFGS-B kernels (host-native).
#
#   tests/drape_kernels/build.sh      build, run the fixtures, run the control
#
# Compiles kernels/drape/cpp/*_emit.cpp (committed; regenerate with
# kernels/drape/gen.sh) into one host executable with the slang prelude from
# guest/avbd/slang-rt, then runs it on gates/5-drape/oracle/components twice:
# as is (every fixture must pass) and with theta corrupted (every fixture
# must fail). Both logs go to gates/5-drape/kernels/. When a riscv64 sysroot
# is found (RISCV64_SYSROOT, as for build.sh), the same TU is also compiled
# for riscv64 (object only): the guest target sees the emits the host ran.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUTDIR="$ROOT/build/drape_kernels_test"
LOGS="$ROOT/gates/5-drape/kernels"
CXX=${CXX:-$(ls -d ~/llvm-mingw/llvm-mingw-*-ucrt-x86_64/bin 2>/dev/null | tail -1)/x86_64-w64-mingw32-clang++}
mkdir -p "$OUTDIR" "$LOGS"
"$CXX" -O2 -std=c++17 -static -Wno-everything \
  -I"$ROOT/guest/avbd/slang-rt" -I"$ROOT/kernels/drape/cpp" \
  "$HERE/test.cpp" -o "$OUTDIR/drape_kernels_test.exe"
rc=0
SYSROOT="${RISCV64_SYSROOT:-/c/contract-manifest/3-interactor/mujoco-sandbox-demo/third_party/riscv64-sysroot}/sysroot"
RVCXX=${RVCXX:-$HOME/scoop/apps/llvm/current/bin/clang++}
if [ -d "$SYSROOT" ] && [ -x "$RVCXX" ]; then
  "$RVCXX" --target=riscv64-unknown-linux-gnu --sysroot="$SYSROOT"     -isystem "$SYSROOT/include/c++/14" -isystem "$SYSROOT/include/c++/14/riscv64-linux-gnu"     -O2 -std=c++17 -Wno-everything -c     -I"$ROOT/guest/avbd/slang-rt" -I"$ROOT/kernels/drape/cpp"     "$HERE/test.cpp" -o "$OUTDIR/drape_kernels_test.riscv64.o"
  echo "riscv64: compiled ($(wc -c < "$OUTDIR/drape_kernels_test.riscv64.o") bytes)"
else
  echo "riscv64: skipped (no sysroot at $SYSROOT)"
fi
"$OUTDIR/drape_kernels_test.exe" "$ROOT/gates/5-drape/oracle/components" > "$LOGS/l2_fixtures.log" || rc=1
"$OUTDIR/drape_kernels_test.exe" "$ROOT/gates/5-drape/oracle/components" --corrupt-theta > "$LOGS/l2_control.log" || rc=1
tail -1 "$LOGS/l2_fixtures.log"
tail -1 "$LOGS/l2_control.log"
exit $rc
