# Gate 6J — the godot-sandbox JIT build in place of the interpreter (parked, 2026-09-23)

**Result: FAIL (parity).** Gate 4 loses bitwise parity with native on the JIT
DLL (11 PASS / 14 FAIL: 8 of the 10 curvenet checks return `<null>` from the
vmcall, "unparsable (guest false, native true)"), so the JIT build cannot
replace the interpreter. Gate 1 passes on it and the loop's early states run
~1.3x faster (AUTHOR 1203 vs 1562 ms, MESH 618 vs 803 ms, FIT_BEGIN 2361 vs
3025 ms against `gates/8-loop/flat-psd.txt`), which is the reason to come back.
The interpreter DLL is what `project/addons/godot_sandbox/bin` ships; nothing
on main depends on this gate. **Parked by the user** ("park the jit/native
translation code"); the cut order in the plan continues without it.

## What was built

- `build_jit_win.sh`: V-Sekai-fire/godot-sandbox `main` @5b0c4a0 (= upstream
  libriscv/godot-sandbox b43ed16 + the Docker gating), `RISCV_ASMJIT=ON`,
  `ENABLE_SAFEGDSCRIPT=ON`, Release, llvm-mingw clang 20260826 (godot-cpp needs
  `-DCMAKE_CXX_FLAGS="-include cstdlib"` under clang: `realloc`/`free`
  undeclared otherwise). 0 errors. Output `C:/b/gs-jit-win/libgodot-riscv.dll`
  (9,311,232 bytes, sha256 0ed4ddbd…); the shipped interpreter DLL is
  9,268,224 bytes, sha256 cb67d058…. Not vendored.
- `jit_gate.sh`: copies the JIT DLL over the addon's, runs `--import`, Gate 1
  (`gate_rd_compute.gd`), Gate 4 (`gate_curvenet.gd`), then Gate 8 with psd for
  the time; restores the interpreter DLL on exit. Every run `--xr-mode off`.

## Logs (evidence)

| file | verdict |
|---|---|
| `import.log` | import OK on the JIT DLL |
| `g1.log` | Gate 1 `RESULT: PASS` (RTX 4090; the `barrier=false` rows are the gate's own no-barrier control and are expected WRONG) |
| `g4.log` | Gate 4 `RESULT: FAIL`, 11 PASS / 14 FAIL (see below) |
| `loop.log`, `flat-jit.txt` | Gate 8 killed by hand during FIT_RUN at Newton iteration 22 (the user parked the run); no wall time |

Gate 4 detail: `beautify_determinism` and `curvenet_extract` match native
(ints and float signatures identical); from `patch_pipeline` on (heap
681,552 B after the pipeline build) every check's vmcall returns `<null>`:
`patch_pipeline`, `crossing_split`, `constraint_solver`, `pen_sphere`,
`extractor_cube`, `delaunay_small_scale`, `mesh_weld`, `skirt_tube`. The
derived checks follow (2/10 identical verdicts, 2/10 float signatures, 8/10
byte-identical repeats, and the three error-path controls `<null>` instead of
their expected error strings). The rule-8 and body-mesh checks (no guest
execution) pass. The interpreter DLL passes all of Gate 4
(`gates/4-curvenet/run.log`: 10/10, 10/10).

## What the gate does not separate (the missing flat control)

The tested DLL differs from the shipped one in **two** ways: the upstream
sync (b43ed16 vs the vendored build's older base) and `RISCV_ASMJIT=ON`. The
speedup and the failure are attributed to neither yet. The control is the same
commit built with `RISCV_ASMJIT=OFF` (same compiler, same flags), which was
not run.

Also open: upstream's CI artifacts (`godot-jit-windows` / `godot-jit-linux`,
gcc builds, run 35896014062 on the fork) were not gated; a clang miscompile of
asmjit's emitter or of godot-cpp under `-include cstdlib` is as plausible as a
JIT bug.

Native binary translation (the `bintr-<hash>` path) is not a route with this
addon: the build loads prebuilt translations only (no emitter), and with
lookup enabled and no library present it segfaults on Windows
(`project/project.godot`, `binary_translation/enabled=false`).

## Resume steps

1. Build the control: `build_jit_win.sh` with `-DRISCV_ASMJIT=OFF` → run
   `jit_gate.sh` (Gates 1 and 4 only). If Gate 4 passes there, the JIT is the
   fault; if it fails too, the upstream sync or the clang build is.
2. Gate the CI gcc artifact (`godot-jit-windows` from the fork's run) the same
   way before any local-compiler hunt.
3. On parity (Gate 4 10/10 + 10/10, Gate 1 PASS, Gate 0F probe 17 for the
   memalign patch after re-vendoring `vendor/sandbox-api/docker/api/native.cpp`
   from upstream's helper): vendor the JIT builds under
   `project/addons/godot_sandbox/bin` with a CITATION.cff naming the CI run,
   keep the interpreter builds under `bin/interp/` as the flat control, rebuild
   the ELFs, rerun Gate 8 for the cycle time.
