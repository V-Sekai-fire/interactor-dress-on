#!/bin/bash
# Build the godot-sandbox addon (JIT: RISCV_ASMJIT=ON) for Windows x86_64 from the org fork's synced main, locally.
set -u
G=C:/Users/ernest.lee/AppData/Local/Temp/claude/C--interactor-dress-on/5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5/scratchpad/prs/godot-sandbox
LLVM=C:/Users/ernest.lee/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64
export PATH="$LLVM/bin:$HOME/.pixi/bin:$PATH"
cd "$G" || exit 1
git checkout -q main && echo "== fork main $(git log --oneline -1 | cut -c1-80)"
echo "== submodules already present"
B=C:/b/gs-jit-win; rm -rf "$B"
T0=$(date +%s)
cmake -S "$G" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DRISCV_ASMJIT=ON -DENABLE_SAFEGDSCRIPT=ON \
	-DCMAKE_C_COMPILER="$LLVM/bin/x86_64-w64-mingw32-clang.exe" -DCMAKE_CXX_COMPILER="$LLVM/bin/x86_64-w64-mingw32-clang++.exe" \
	-DCMAKE_RC_COMPILER="$LLVM/bin/x86_64-w64-mingw32-windres.exe" -DCMAKE_CXX_FLAGS="-include cstdlib" > "$B.configure.log" 2>&1
echo "== configure rc=$? $(( $(date +%s)-T0 )) s"; grep -m3 -i 'error' "$B.configure.log" | cut -c1-160
T0=$(date +%s); cmake --build "$B" --parallel 12 > "$B.build.log" 2>&1; echo "== build rc=$? $(( $(date +%s)-T0 )) s errors=$(grep -c ' error:' "$B.build.log")"
grep -m3 ' error:' "$B.build.log" | cut -c1-200
find "$B" -maxdepth 2 -name 'libgodot_riscv*.dll' -exec ls -la {} \; | awk '{print $5, $9}'
