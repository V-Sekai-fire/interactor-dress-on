#!/bin/sh
# Build and run the G3 oracle (host-native): LBFGSpp on the inverse_min
# objective compiled from the guest's own sources (AvbdCpu +
# guest/drape/inverse_min.h). Writes gates/5-drape/oracle/inverse_min/.
#
#   tests/inverse_min_oracle/build.sh
#
# LBFGSpp and Eigen come from the clone tests/lbfgsb_oracle/build.sh makes
# (V-Sekai-fire/interactor-aria-lbfgspp @ 10086b6b, .deps/, gitignored); run
# that once first. Neither library reaches a guest ELF.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
ARIA="$ROOT/tests/lbfgsb_oracle/.deps/interactor-aria-lbfgspp"
if [ ! -d "$ARIA/thirdparty/LBFGSpp/include" ]; then
  echo "no LBFGSpp clone at $ARIA: run tests/lbfgsb_oracle/build.sh first" >&2
  exit 1
fi
OUTDIR="$ROOT/build/inverse_min_oracle"
OUT=${OUT:-$ROOT/gates/5-drape/oracle/inverse_min}
mkdir -p "$OUTDIR" "$OUT"
CXX=${CXX:-$(ls -d ~/llvm-mingw/llvm-mingw-*-ucrt-x86_64/bin 2>/dev/null | tail -1)/x86_64-w64-mingw32-clang++}
# -ffp-contract=off: the guest builds AvbdCpu without contraction too.
"$CXX" -O2 -std=c++20 -static -ffp-contract=off \
  -I"$ARIA/thirdparty/LBFGSpp/include" -I"$ARIA/thirdparty/eigen" \
  -I"$ROOT/guest/drape" -I"$ROOT/guest" -I"$ROOT/guest/avbd" -I"$ROOT/kernels/avbd/cpp" -I"$ROOT/guest/avbd/slang-rt" \
  "$HERE/oracle.cpp" "$ROOT/guest/avbd/avbd_cpu.cpp" "$ROOT/guest/avbd/avbd_cpu_backward.cpp" \
  "$ROOT/guest/avbd/avbd_topology.cpp" -o "$OUTDIR/inverse_min_oracle.exe"
"$OUTDIR/inverse_min_oracle.exe" "$OUT" | tee "$OUT/oracle.log"
