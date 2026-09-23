#!/usr/bin/env bash
# Gate 2 flat control (plan Cut A step A4): the native Vulkan AVBD tests from
# cloth-dynamics-standalone, run on OUR Lean-emitted SPIR-V (build/spv), not on
# the standalone repo's own tests/slang_validate/build. If the guest disagrees
# with these logs, the kernels are ruled out and the guest driver is at fault.
#
# The native tests load <dir>/<kernel>.spv by name (AvbdSolverVk.cpp
# loadKernels) and bind with their compiled-in AvbdKernelTable.inc;
# kernels/avbd/AvbdKernelTable.inc matches it kernel-for-kernel (24 kernels,
# identical bindings, only row order differs), so the binaries are valid on
# our SPIR-V without a rebuild.
#
# Usage: gates/2-avbd/run_native.sh
#   NATIVE_BIN  dir holding test_avbd_{solver,gradcheck,stategrad}.exe
#   SPV_DIR     SPIR-V dir (default: <repo>/build/spv)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NATIVE_BIN="${NATIVE_BIN:-C:/cloth-dynamics-standalone/src/code/slang_solver/build}"
SPV_DIR="${SPV_DIR:-$ROOT/build/spv}"
# Hand the native exes a Windows path (MSYS would convert it anyway; this
# keeps the log header honest about what they were given).
command -v cygpath >/dev/null 2>&1 && SPV_DIR="$(cygpath -m "$SPV_DIR")"

rc_all=0
for t in solver gradcheck stategrad; do
    exe="$NATIVE_BIN/test_avbd_$t.exe"
    log="$HERE/native_$t.log"
    if [ ! -x "$exe" ]; then
        echo "missing $exe" | tee "$log"
        rc_all=1
        continue
    fi
    {
        echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) $exe $SPV_DIR"
        echo "# exe sha256 $(sha256sum "$exe" | cut -d' ' -f1)"
        echo "# standalone HEAD $(git -C C:/cloth-dynamics-standalone rev-parse --short HEAD 2>/dev/null)"
        echo "# spv set sha256 $(cat "$SPV_DIR"/*.spv | sha256sum | cut -d' ' -f1) ($(ls "$SPV_DIR"/*.spv | wc -l) files)"
    } > "$log"
    "$exe" "$SPV_DIR" >> "$log" 2>&1
    rc=$?
    echo "# exit $rc" >> "$log"
    echo "test_avbd_$t exit=$rc -> $log"
    [ $rc -ne 0 ] && rc_all=1
done
exit $rc_all
