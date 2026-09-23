# Gate 6 — fit.elf on foxgirl_skirt (native, same code)

Cut 6 replaces two things in cloth-fit's garment solve. The OpenVDB avatar SDF
becomes a brick-grid SDF (`SdfGrid`, sampled through `SdfSpline`). The file-driven
`run_retarget` becomes an I/O-free driver that runs one phase per call
(`guest/fit/fit_driver`). This page puts the two together. It compares
`fit_native`, the native build of the code fit.elf will run, against upstream
`PolyFEM_bin`, and it proposes the Gate 6 criterion from these measurements.
The sub-gates have their own pages:

- `sdf/` is Gate 6a: SdfGrid and the spline sampler against OpenVDB.
- `driver/` is the driver port. It is bitwise equal to upstream when built
  with upstream's numerics, using the OpenVDB FitForm.
- `foxgirl/` holds this page's logs: the SdfGrid FitForm end to end.

Nothing on this page ran in the guest, in Godot, or on the GPU.

## Result

- **Port: pass.** The driver is upstream's loop: with upstream numerics and
  OpenVDB it reproduces the 1-thread oracle bitwise (`driver/`).
- **SDF: pass.** Gate 6a passes 9/9 on both point sets, and its control fails
  as it should.
- **Native SdfGrid vs upstream: pass under the criterion proposed below.** The
  plan's "Hausdorff ≤ 1 voxel" cannot be met. The solve is chaotic, and
  upstream misses that bound against itself by 5–8 voxels.
- **Sampler: provisional.** This run uses the Gate 6a **reference** sampler.
  The Lean kernel (`lean/Fit/SdfSplineHessian.lean`) has not been emitted yet.

## Gate 6a (from `sdf/README.md`)

The points are FitForm's samples on the final garment of the 1-thread oracle.
The OpenVDB values come from `openvdb_dump`.

| check | `fit` set: 21,015 samples | `all` set: 78,300 samples |
|---|---|---|
| reference sampler vs OpenVDB `sampleHessian`, on OpenVDB's own stencils | 0 ULP in all 13 outputs | 0 ULP |
| index map (p/h, floor, uvw) vs `worldToIndex` | bitwise | bitwise |
| value within 0.25 voxel, near-surface samples | 100% (p99 0.027, max 0.058 voxel) | 100% (max 0.084) |
| sign agreement | 100% | 100% |
| gradient within 10° | 100% (p95 0.81°, max 3.0°) | 100% (max 5.1°) |
| control: grid shifted +0.5 voxel in x | **fails**: value 58.6%, sign 99.41% | **fails**: 56.0%, 99.35% |

## Foxgirl: native SdfGrid vs upstream

**Setup:**

- **Build:** `fit_native`, built as `FIT_SDF=sdfgrid FIT_SDF_SAMPLER=reference`
  in `C:/b/fit-native-sdf`.
- **Guest numerics:** `-ffp-contract=off`, `EIGEN_DONT_VECTORIZE`,
  `POLYFEM_THREADING=NONE`, no filib.
- **Inputs:** foxgirl_oracle.json at voxel h = 0.01. All distances are in voxels,
  in the solve frame.
- **"vs 1t"** means against `C:/b/cf-up-out1/step_garment_252.obj`.
- **Hausdorff** is vertex-sampled, point to surface, in both directions.
- **Fit gap** is the unsigned distance from the 677 fit vertices (the vertices
  not in `no-fit.txt`) to the final avatar (`fit_gap.py`). At the start it is
  8.36 mean.

| run | Newton (per phase) | final energy (vs 1t) | vs 1t max / mean / Hausdorff | fit gap mean / p95 | intersections | io in driver |
|---|---|---|---|---|---|---|
| **upstream 1t** (oracle) | 248 (100/42/39/67) | 0.0014974 | 0 | 1.847 / 3.921 | none | — |
| upstream 16t | 250 (50/53/45/102) | 0.0012964 (−13.4%) | 11.63 / 5.99 / 8.24 | 1.776 / 3.872 | none | — |
| upstream 16t repeat | 185 (50/21/34/80) | 0.0014293 (−4.6%) | 4.93 / 3.11 / 4.74 | 1.769 / 3.826 | none | — |
| **native SdfGrid, f64 inputs** | 177 (24/36/50/67) | 0.0014288 (−4.6%) | **4.75 / 2.77 / 4.52** | 1.786 / 3.910 | none | 0 |
| **native SdfGrid, f32 inputs** (the wire) | 196 (41/37/11/107) | 0.0013414 (−10.4%) | **10.29 / 5.12 / 6.70** | 1.794 / 3.925 | none | 0 |
| native OpenVDB, f64 (`driver/`) | 182 (24/49/50/59) | 0.0013918 (−7.1%) | 8.16 / 4.23 / 5.95 | 1.797 / 4.005 | none | 0 |
| native OpenVDB, f32 (`driver/`) | 178 (41/35/21/81) | 0.0013171 (−12.0%) | 11.01 / 5.68 / 7.38 | 1.795 / 3.956 | none | 0 |
| SdfGrid, **upstream numerics**, f64 (control) | 260 (100/35/35/90) | 0.0024248 (**+61.9%**) | 11.40 / 3.44 / 6.03 | 1.850 / 3.945 | none | 0 |
| **control: SdfGrid f64, `fit_weight=0`** | 87 (24/14/28/21) | 8.6e-5 (no fit term) | 9.34 / 4.91 / 6.18 | **4.699 / 7.709** | none | 0 |

**Upstream's own band.** These are the numbers any native-vs-upstream bound has
to allow for:

| pair | max | mean | Hausdorff |
|---|---|---|---|
| 16t vs 16t repeat | 8.42 | 2.98 | 5.00 |
| 1t vs 16t / 16t repeat | 11.63 / 4.93 | 5.99 / 3.11 | 8.24 / 4.74 |

- Final energy across the three upstream runs: 0.0012964 – 0.0014974.
- Fit gap mean: 1.769 – 1.847.

What the runs show:

- **The SDF swap changes the path, not the fit.** Phase 0 has the fit form
  off, and every run has the same phase 0 as its numerics twin. With upstream
  numerics that is 100 Newton steps and energy 0.0041895, as in the oracle.
  - **Phase 1 is where the runs part.** The same numerics give 35 Newton steps
    and 0.0055708 on SdfGrid, against 42 and 0.0058845 on OpenVDB. The value
    difference is at most 0.058 voxel (Gate 6a), and that is enough to move
    the trajectory.
  - **SdfGrid vs OpenVDB under guest numerics:** 5.39 / 1.73 / 4.01 for f64
    and 1.41 / 0.64 / 0.80 for f32. The SDF moves the result by less than
    upstream's 16t-vs-16t spread.
- **The native same-code result is inside upstream's band.**
  - **f64 inputs:** closer to the 1-thread oracle than either 16-thread run is
    (Hausdorff 4.52 vs 4.74 / 8.24), and 0.76 from the 16t repeat.
  - **f32 inputs:** Hausdorff 6.70 to 1t, and 1.68 / 4.47 to the two 16t runs.
  - **Fit gap:** equal to upstream's within 0.08 voxel (mean) and 0.09 (p95)
    in every run.
- **Geometric distance alone does not discriminate.** With the fit term off,
  the garment lands only 6.18 voxels (Hausdorff) from the oracle. That is
  inside upstream's 1t-vs-16t spread of 8.24. What catches it is the fit gap:
  4.70 mean against 1.77–1.85.
- **The final energy has more than one basin.** The upstream-numerics SdfGrid
  control ends 62% above the oracle. Yet its fit gap (1.850) and its
  intersection check are as good as upstream's, so the difference must sit
  in the other terms (similarity, curvature, twist, contact); not yet
  decomposed.
  - The four guest-numerics runs stay within −12% … −5% of the oracle.
  - This control is not a shipped configuration. It is one sample showing
    that a different basin is reachable. Decomposing its energy per form is
    open.
- **Deterministic.** Two f64 SdfGrid runs, at different load and from two
  builds of the harness, are bitwise identical (`garment_final.f64`).
- **No file I/O and no intersections.**
  - `io_attempts` is 0 inside the driver in every run.
  - The harness's own 7 opens before `begin` are counted, which shows the
    counter works.
  - `ipc::my_has_intersections` reports none in every run.
  - **Push control:** one vertex pushed 0.074–0.103 toward the avatar
    centroid is reported INTERSECTS in every run.
  - The newest preview snapshot equals the final garment in every run.

### 6.P: time and heap (native, single thread, below-normal priority)

| run | CPU s | peak working set | SDF grid |
|---|---|---|---|
| SdfGrid guest f64 | 103.0 | 214.5 MB | 207 bricks (105,984 voxels), 0.82 MB, 105,984 distance + 105,984 winding queries, 0.57 s fill |
| SdfGrid guest f32 | 107.6 | 177.8 MB | 223 bricks, 0.88 MB, 0.56 s fill |
| OpenVDB guest f32 (`driver/`) | 124.6 | 911.9 MB | 46.5M active voxels, 414 MB |
| upstream 1t `PolyFEM_bin` | 242 | 940 MB | as above, plus debug OBJ writes every step |

- The grid holds under 1 MB, where OpenVDB builds 414 MB.
- The peak working set falls by about 700 MB. Of what is left, the SDF is
  under 1 MB of bricks plus the avatar's AABB and winding-number BVH.
- One f64 run took 163 CPU s while a `-j8` build ran next to it. The result
  was bitwise the same as the 103 s run, so CPU seconds here depend on load.

## Proposed Gate 6 criterion

**Native vs upstream** (same inputs, compared with the 1-thread oracle). This
replaces the plan's "Hausdorff ≤ 1 voxel", which upstream itself misses by 5–8
voxels. It passes only if all of the following hold:

1. **Intersection-free.** `ipc::my_has_intersections` reports none on the final
   state, and the push control reports INTERSECTS.
2. **No file I/O.** `io_attempts == 0` from `begin` to the last phase, and the
   harness opens before `begin` are counted.
3. **The fit holds.** The fit gap is within 0.25 voxel of the oracle's mean
   (1.847) and within 0.5 voxel of its p95 (3.921).
   - **Measured:** every run is within 0.08 (mean) and 0.09 (p95).
   - **Control:** `fit_weight=0` misses by 2.85 (mean) and 3.79 (p95). This
     is the check that does the discriminating.
4. **The energy is in upstream's basin.** The final energy lies within
   upstream's 1t/16t band, widened by 10% on each side:
   [0.9 × 0.0012964, 1.1 × 0.0014974] = **[0.001167, 0.001647]**.
   - **Measured:** the guest-numerics runs give 0.0013171 – 0.0014288.
   - **Fails:** `fit_weight=0` (8.6e-5) and the upstream-numerics SdfGrid
     control (0.0024248). The second failure is intended: a different basin is
     a different garment, so it should be flagged.
5. **Coarse geometric bound.** Hausdorff to the oracle ≤ 8.24 voxels and
   per-vertex max ≤ 11.63 voxels, which is upstream's own worst 1t-vs-16t
   pair.
   - This catches gross failures only; `fit_weight=0` passes it (6.18).
   - **Measured:** 4.52 / 4.75 for f64 and 6.70 / 10.29 for f32.

Native SdfGrid passes all five with f64 inputs and with f32 inputs.

**Guest vs native same code** (the same inputs, the same SDF, the same
sampler, the same numerics). The tight bound stays:

- final vertices within **≤ 1e-6** in solve units, and the final energy within
  1e-6 relative;
- Newton iterations per phase within **±2**.

Native guest numerics are bitwise reproducible, so a fixed target exists.

- **The risk is libm.** The guest's `pow`/`sqrt`/`exp` can round differently
  from ucrt's.
- **Why it matters:** the solve is chaotic, so one ULP in phase 1 is enough
  to leave the bound.
- **How to read it:** the gate logs the first iteration where f or |Δx|
  differ. Phase 0 has no fit term and should match exactly.
- **If it fails on libm alone,** the guest falls back to the native-vs-upstream
  criterion above. The log says so; the bound is not loosened silently.

## Open

- **The Lean sampler.** Rerun this page with `FIT_SDF_SAMPLER=kernel` once
  `lean/Fit/SdfSplineHessian.lean` is emitted.
  - **Expected:** bitwise the same as the reference. Gate 6a already requires
    0 ULP, and a hand-written stand-in through the same call site was 0 ULP
    (`sdf/standin.log`).
  - **Literals:** the emit needs double literals.
- **Energy per form.** Split the final energy by form for the
  upstream-numerics SdfGrid control, to say which term holds the +62%.
- **The same run in the guest.** fit.elf in the sandbox is Cut 6's next step:
  guest vs native, the heap ladder, and the worker-thread vmcall.

## Reproduce

Run from the repo root under `pixi run --manifest-path tools/native/pixi.toml`.

```sh
FIT_BUILD=C:/b/fit-native-sdf FIT_SDF=sdfgrid bash tests/native/fit/build.sh          # FIT_SDF_SAMPLER=reference is the default
FIT_BUILD=C:/b/fit-native-up-sdf FIT_SDF=sdfgrid FIT_NUMERICS=upstream bash tests/native/fit/build.sh
C:/b/fit-native-sdf/fit_native.exe --out C:/b/fit-out-sdf64 --io-probe --control-push --no-round
C:/b/fit-native-sdf/fit_native.exe --out C:/b/fit-out-sdf32 --io-probe --control-push
C:/b/fit-native-up-sdf/fit_native.exe --out C:/b/fit-out-upsdf64 --io-probe --control-push --no-round
C:/b/fit-native-sdf/fit_native.exe --out C:/b/fit-out-sdf64-fw0 --io-probe --control-push --no-round --set /fit_weight=0
C:/b/fit-native-sdf/fit_native.exe --compare C:/b/fit-out-sdf64/garment_final.obj C:/b/cf-up-out1/step_garment_252.obj
python gates/6-fit/fit_gap.py C:/b/cf-up-out1/step_avatar_252.obj \
  vendor/cloth-fit/garment-data/assets/garments/LCL_Skirt_DressEvening_003/no-fit.txt 0.01 \
  C:/b/fit-out-sdf64/garment_final.obj C:/b/cf-up-out1/step_garment_252.obj
```

**Evidence in `foxgirl/`:**

| file | what it holds |
|---|---|
| `run-sdf-*.log` | stdout of each run, debug level |
| `phases-sdf-*.tsv` | per-phase results |
| `compare.log` | every distance on this page, and the bitwise repeat |
| `fit_gap.log` | the fit gap of each run |
| `process.log` | CPU and wall time of each process |

**Upstream numbers** come from `C:/b/cf-up-out{1,16,16b}.log`. Per-phase Newton
counts are the `[Backtracking] iters=` lines between solver starts. The
`driver/` page holds the OpenVDB-FitForm runs.
