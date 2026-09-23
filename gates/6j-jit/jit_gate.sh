#!/bin/bash
# Gate 6J: the godot-sandbox JIT build (libriscv asmjit, RISCV_ASMJIT=ON; built locally from the fork's main 5b0c4a0 (llvm-mingw clang, -include cstdlib))
# in place of the interpreter DLL. Bitwise gates first (1, 4), then Gate 8 with psd for the time.
# The interpreter DLL is restored at the end unless KEEP_JIT=1; every Godot run --xr-mode off.
set -u
R=/c/interactor-dress-on
G="$HOME/scoop/apps/godot/current/godot.console.exe"
J=/c/b/gs-jit-win/libgodot-riscv.dll
DLL=$R/project/addons/godot_sandbox/bin/libgodot_riscv.windows.template_release.x86_64.dll
O="$TMP/jit"; rm -rf "$O"; mkdir -p "$O"
cd "$R" || exit 1
cp "$DLL" "$O/interp.dll"
restore() { cp "$O/interp.dll" "$DLL" && echo "== interpreter DLL restored"; }
[ "${KEEP_JIT:-0}" = 1 ] || trap restore EXIT
cp "$J" "$DLL" && echo "== JIT DLL in: $(sha256sum "$DLL" | cut -c1-16)"
timeout 600 "$G" --path project --headless --xr-mode off --import > "$O/import.log" 2>&1; echo "== import rc=$?"
T0=$(date +%s); timeout 300 "$G" --path project --script gate_rd_compute.gd --rendering-driver vulkan --xr-mode off > "$O/g1.log" 2>&1; echo "== Gate 1 rc=$? $(( $(date +%s)-T0 )) s $(grep -m1 -o 'Using Device.*' "$O/g1.log")"
T0=$(date +%s); timeout 900 "$G" --path project --script gate_curvenet.gd --rendering-driver vulkan --xr-mode off > "$O/g4.log" 2>&1; echo "== Gate 4 rc=$? $(( $(date +%s)-T0 )) s $(grep -m1 '^RESULT' "$O/g4.log") | $(grep -c '^PASS' "$O/g4.log") PASS $(grep -c '^FAIL' "$O/g4.log") FAIL"
grep -m3 -i 'jit\|binary transl\|asmjit\|translat' "$O/g1.log" "$O/g4.log" | cut -c1-120
T0=$(date +%s); timeout 3000 "$G" --path project --script gate_loop.gd --rendering-driver vulkan --xr-mode off -- --gate=loop --out="$(cygpath -m "$O/flat-jit.txt")" --wallclock=2800 --allow-fixture=infer,rig > "$O/loop.log" 2>&1
echo "== Gate 8 (JIT, psd) rc=$? $(( $(date +%s)-T0 )) s"; grep -E '^STATE (AUTHOR|MESH|FIT_BEGIN|FIT_RUN|CHECK|DRAPE_COLLECT)|^RESULT|^wall' "$O/flat-jit.txt" | cut -c1-150
git -C "$R" checkout -q -- gates project/addons/godot_sandbox/*.import 2>/dev/null
echo DONE
