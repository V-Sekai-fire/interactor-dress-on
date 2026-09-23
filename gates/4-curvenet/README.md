# Gate 4-curvenet: the Cassie kernels come from lean/, match the module's, and link

**Result: PASS.** CASSIE's four editing-pipeline kernels (`lean/Cassie`,
copied from entities-godot c165a519d2 and renamespaced) plus DiffCloth's
`SpmvDf32` build on this tree's LeanSlang `emit-fp` / Lean 4.30.0 with no
source change. `kernels/cassie/gen.sh` lowers them to the five
`kernels/cassie/cpp/*_emit.cpp` the vendored dispatchers include. Those are
line-for-line the prebuilt `thirdparty/avbd/*.cpu.cpp` that entities-godot
ships, and they compile, link and run natively and for riscv64.

| check | log | result |
|---|---|---|
| `lake build Cassie emit_cassie` | — | 28 jobs, exit 0, 5 s; the six copied `native_decide` examples (two each in CurveGenerateBezier, CurveNewton, CurveRdp) and the three new ones in `Cassie.lean` hold |
| `kernels/cassie/gen.sh` (emit), run twice | `gen.log` | 5 kernels to slang/, cpp and SPIR-V; 5/5 `spirv-val` clean (2056–6904 bytes); the second run leaves `git diff -- kernels/cassie` empty |
| `kernel-parity.sh`: each emit against `modules/cassie/thirdparty/avbd/<k>.cpu.cpp` | `kernel-parity.log` | 5 kernels, 755 lines compared, **0 differing** |
| negative control: one `3.0f` changed to `3.5f` in a copy of the curve_generate_bezier emit | `kernel-parity.log` | 2 differing lines (the one line, both sides) |
| `CURVENET_KERNELS_PENDING=OFF tests/native/curvenet/build.sh` (llvm-mingw, clean) | `native-kernels.log` | 141 TUs (136 + the 5 dispatchers), 0 unresolved, curvenet_smoke 7/7 and curvenet_kernels_smoke 10/10 PASS |
| `riscv64-kernels.sh` (org sysroot, `curvenet_compile_check`, PENDING=OFF) | `riscv64-kernels.log` | 0 warnings; cassie_core needs 5 `cassie_slang_dispatch::` symbols, cassie_kernels defines those 5, **0 unresolved**; 5 `main_0_Thread`s, one per kernel namespace; 0 `Eigen::` |

## What the parity diff strips, and why

Both sides lose the Slang prelude, `#line` directives, CRs and blank lines.
The reference's post-processing (the relative prelude `#include`, the
`SLANG_PRELUDE_EXPORT` neutering and the `namespace cassie_slang_<k>` wrap
that entities-godot's `avbd-codegen` adds) is stripped too. Current slangc
(2026.13.1) inlines the 4.2k-line prelude, guarded by `SLANG_CPP_PRELUDE_H`,
where the reference `#include`d it. The dispatchers include the prelude first
and so skip the inlined copy (the `guest/avbd/avbd_cpu.cpp` pattern). The
task expected some slangc-version naming drift. There is none: the struct,
field, temporary (`_S<n>`) and entry names match, so the dispatchers needed
no change beyond the spmv file name (`spmv_df32_emit.cpp`, recorded in
`vendor/cassie/CITATION.cff` and in `tools/vendor/patches/cassie.patch`).

## Found on the way (not fixed: bug-for-bug with the module)

- **Two epsilons print as zero.** LeanSlang's `litFloat` prints six decimal
  places. `1.0e-12` (curve_generate_bezier's singular-determinant guard) and
  `1.0e-9` (curve_newton's `|den|` guard) therefore emit as `0.000000`, and
  the guards read `abs(x) < 0.0f`, which is never true. The reference
  `.cpu.cpp` has the same `< 0.0f`, so this is how the module ships, not
  something the emit-fp bump introduced. The hand-written C++ these kernels
  replaced used the real epsilons. A fix belongs upstream (a `litFloat` that
  round-trips, or `litHalf`/`cast`-style exact literals) and then here as a
  re-pin.
- **slangc warns E41035 on curve_rdp's local stack.** The warning is
  "possibly uninitialized". The stack is only read below `stack_top`, which
  every push has written, so it is a false positive.

## Pins

entities-godot c165a519d2836f3ded600948abb9d8f799cd4c5f (Lean sources and
the reference `.cpu.cpp`); LeanSlang contract-lean-slang `emit-fp` e0e96da;
Lean 4.30.0; slangc 2026.13.1-1-g84792eb15 (scoop Vulkan SDK).
