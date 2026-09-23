#!/usr/bin/env bash
# Build the org's rotation fitter for the host: its unit test (AlignTest.lean's
# oracle) and sinew_align_cli (gates/6d-fit-avbd/ladder_eval.py's rotation
# fits go through it). CC defaults to clang (llvm-mingw on Windows).
#
#   bash vendor/sinew-align/build.sh        # -> build/sinew-align/{test_sinew_align,sinew_align_cli}.exe, runs the test
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$ROOT/build/sinew-align}"
CC="${CC:-clang}"
mkdir -p "$OUT"
"$CC" -O2 -std=c11 -o "$OUT/test_sinew_align.exe" "$HERE/test_sinew_align.c" "$HERE/sinew_align.c" -lm
"$CC" -O2 -std=c11 -o "$OUT/sinew_align_cli.exe" "$HERE/sinew_align_cli.c" "$HERE/sinew_align.c" -lm
"$OUT/test_sinew_align.exe"
echo "$OUT/sinew_align_cli.exe"
