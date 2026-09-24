#!/bin/sh
# Build and run the L2 test of the Lean-emitted ANNY kernels (host-native).
#
#   tests/anny_kernels/build.sh      build, run the test, run the control
#
# Compiles kernels/anny/cpp/*_emit.cpp (committed; regenerate with
# kernels/anny/gen.sh) into one host executable with the slang prelude from
# guest/avbd/slang-rt (-ffp-contract=off, as the guest builds the emits),
# then runs it twice: as is (every forward and gradient check must pass) and
# with the backward fed dverts x 1.25 (every gradient group must fail). Both
# logs go to gates/anny/kernels/.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUTDIR="$ROOT/build/anny_kernels_test"
LOGS="$ROOT/gates/anny/kernels"
CXX=${CXX:-c++}
mkdir -p "$OUTDIR" "$LOGS"
"$CXX" -O2 -std=c++17 -ffp-contract=off -Wno-everything \
  -I"$ROOT/guest/avbd/slang-rt" -I"$ROOT/kernels/anny/cpp" \
  "$HERE/test.cpp" -o "$OUTDIR/anny_kernels_test"
rc=0
"$OUTDIR/anny_kernels_test" > "$LOGS/test.log" 2>&1 || rc=1
"$OUTDIR/anny_kernels_test" --control > "$LOGS/control.log" 2>&1 || rc=1
cat "$LOGS/test.log" "$LOGS/control.log"
exit $rc
