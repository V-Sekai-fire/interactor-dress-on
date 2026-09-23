# Gate 6 (driver): fit_driver natively on foxgirl_skirt

`guest/fit/fit_driver.{h,cpp}` is cloth-fit's retarget loop
(`vendor/cloth-fit/src/polyfem/main.cpp`, `garment/run_retarget.cpp:150-326`)
with no file I/O, sliced into one phase per call (phase `2*substep + k`,
k=0 AL solve, k=1 reduced solve). `tools/fit/fit_native.cpp` runs it natively
(`tests/native/fit`, llvm-mingw 20260826, `C:/b/fit-native`). The SDF is the
interim one: FitForm with OpenVDB as vendored at 87d591f2
(`tests/native/fit/extract_fitform.py`, `FIT_SDF=openvdb`), which is what the
oracle samples; the working tree's SdfGrid FitForm replaces it once its Lean
kernel is emitted.

## Result

**Pass on port fidelity; the guest numerics are a different sample of a
chaotic solve.**

| run | inputs | Newton (per phase) | final energy | intersections | io_attempts | CPU s | peak WS MB |
|---|---|---|---|---|---|---|---|
| upstream 1 thread (`C:/b/cf-up-out1`) | f64 | 248 (100/42/39/67) | 0.0014974 (as the control) | none | writes every step | 242 | 940 |
| fit_native, `FIT_NUMERICS=upstream` (control) | f64 | 248 (100/42/39/67) | 0.0014974 | none | 0 | 194 | 938 |
| fit_native, guest numerics | f64 | 182 (24/49/50/59) | 0.0013918 | none | 0 | 182 | 1369* |
| fit_native, guest numerics | f32 | 178 (41/35/21/81) | 0.0013171 | none | 0 | 125 | 912 |

*before the per-substep objects were released ahead of the next substep's
SDF (two OpenVDB grids alive at once); the f32 row is after the fix, and its
result is bitwise the same as before it.

Distances of the final garment (solve frame, voxel h = 0.01), from
`compare.log` (`fit_native --compare`; Hausdorff is vertex-sampled, point to
surface, both directions):

| A vs B | per-vertex max | mean | Hausdorff |
|---|---|---|---|
| control (upstream numerics) vs upstream 1t | **0** | 0 | **0** (bitwise) |
| guest f64 vs upstream 1t | 8.16 | 4.23 | 5.95 |
| guest f32 vs upstream 1t | 11.01 | 5.68 | 7.38 |
| guest f64 vs upstream 16t / 16t repeat | 3.89 / 4.65 | 1.89 / 1.55 | 2.70 / 3.42 |
| guest f32 vs upstream 16t / 16t repeat | 1.36 / 7.87 | 0.51 / 2.70 | 0.98 / 4.77 |
| guest f64 vs guest f32 | 3.41 | 1.54 | |
| upstream 1t vs 16t / 16t repeat | 11.63 / 4.93 | 5.99 / 3.11 | 8.24 / 4.74 |
| upstream 16t vs 16t repeat | 8.42 | 2.98 | 5.00 |
| guest f32, run 1 vs run 2 vs run 3 | 0 | 0 | bitwise |

- **The driver is upstream's loop.** Built with upstream's numerics
  (`FIT_NUMERICS=upstream`: vectorised Eigen, default contraction,
  `POLYFEM_THREADING=TBB` held to one thread) and fed the same f64 OBJ data,
  fit_native reproduces `step_garment_252.obj` bitwise: same 248 Newton
  iterations, same 253 post_steps, same per-iteration energies in the log.
  That is the flat control: whatever else differs is numerics, not the port.
- **Guest numerics diverge at the first Newton step.** With
  `-ffp-contract=off`, `EIGEN_DONT_VECTORIZE` and `POLYFEM_THREADING=NONE`
  (f64 inputs), iteration 0 has the same f = 53.6213, |grad f| = 214.485 and
  Δx·∇f = -107.243 as upstream, but |Δx| = 19.53 against 209.967: the Newton
  system has a direction the gradient does not see, and its component is set
  by rounding. From there the trajectory is another one; it ends
  intersection-free at a lower energy, 5.95 voxels (Hausdorff) from the
  1-thread oracle, which is inside upstream's own spread between 1 and 16
  threads (4.74-8.24). The plan's "native vs upstream Hausdorff ≤ 1 voxel"
  holds for the upstream-numerics build (0) and does not hold for the guest
  numerics against any single upstream run; upstream is not reproducible
  across thread counts at that tolerance either.
- **Guest numerics are deterministic**: three f32 runs are bitwise identical,
  so the guest-vs-native gate (≤ 1e-6) has a fixed target.
- **Float32 inputs** (the wire) move the guest-numerics result 3.41 voxels at
  most from the f64-input run.
- **No file I/O in the driver.** fit_native links `--wrap` for
  fopen/_wfopen/_open/_wopen/_sopen_s/_wsopen_s/CreateFileA/CreateFileW. From
  `FitDriver::begin` through the last phase: 0 opens in every run. Positive
  control: the harness's own 7 opens before `begin` (setup JSON, 2 meshes,
  2 skeletons, no-fit list twice) are all seen, with their paths.
- **Intersection check**: `ipc::my_has_intersections` on the final state
  reports none in every run. Control: one garment vertex moved 80% of the way
  to the avatar centroid (0.073-0.074 in solve units) is reported INTERSECTS.
- **Preview ring**: the newest snapshot equals the final garment exactly
  (seq 181 / 252).
- **SdfGrid FitForm (426c385d), `FIT_SDF=sdfgrid`**: builds (no OpenVDB,
  `C:/b/fit-native-sdf`). Phase 0 (fit form off) matches the OpenVDB guest
  f32 run exactly (41 Newton, energy 0.00033678666696946268); phase 1 stops
  with the sampler's "SDF spline kernel pending" error, reported through
  `PhaseStats::error`, 0 file opens. The foxgirl numbers above move to it
  when the Lean kernel lands.

## Reproduce

```sh
FIT_BUILD=C:/b/fit-native pixi run --manifest-path tools/native/pixi.toml bash tests/native/fit/build.sh
FIT_BUILD=C:/b/fit-native-up FIT_NUMERICS=upstream pixi run --manifest-path tools/native/pixi.toml bash tests/native/fit/build.sh
C:/b/fit-native/fit_native.exe --out C:/b/fit-out-f32 --io-probe --control-push            # from the repo root
C:/b/fit-native/fit_native.exe --out C:/b/fit-out-f64 --io-probe --control-push --no-round
C:/b/fit-native-up/fit_native.exe --out C:/b/fit-out-up64 --io-probe --control-push --no-round
C:/b/fit-native/fit_native.exe --compare C:/b/fit-out-f32/garment_final.obj C:/b/cf-up-out1/step_garment_252.obj
```

Runs were at below-normal priority on a shared desk; wall seconds vary with
load, CPU seconds are the process's own. Logs: `run-*.log` (fit_native
stdout, log level debug as foxgirl_oracle.json sets it), `phases-*.tsv`,
`compare.log`.
