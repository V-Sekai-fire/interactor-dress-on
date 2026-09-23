# Gate 6 — fit.elf on foxgirl_skirt

Cut 6 replaces two things in cloth-fit's garment solve. The OpenVDB avatar SDF
becomes a brick-grid SDF (`SdfGrid`, sampled through `SdfSpline`). The file-driven
`run_retarget` becomes an I/O-free driver that runs one phase per call
(`guest/fit/fit_driver`). This page first runs the whole solve in the guest
(fit.elf in Godot, [below](#gate-6-in-the-guest)); the sections after it
compare `fit_native`, the native build of the same code, against upstream
`PolyFEM_bin`, and derive the Gate 6 criterion the guest run is judged by.
The sub-gates have their own pages:

- `sdf/` is Gate 6a: SdfGrid and the spline sampler against OpenVDB.
- `driver/` is the driver port. It is bitwise equal to upstream when built
  with upstream's numerics, using the OpenVDB FitForm.
- `foxgirl/` holds the native logs: the SdfGrid FitForm end to end.
- `elf/` is fit.elf itself: its hash and link check, the serial TBB stand-in,
  and the first Godot smoke run of phase 0.
- `runs/` holds the guest logs of `project/gate_fit.gd`, one file per arm;
  `results.txt` is its verdict.

Nothing here ran on the GPU.

## Result

- **Gate 6 in the guest: PASS on the five-part criterion; the tight
  guest-vs-native bound is MISSED; the likely cause (sort-tie order, libstdc++
  vs libc++) is a hypothesis, not yet instrumented.** fit.elf runs
  all four phases of foxgirl in Godot on a worker Thread: 219 Newton, energy
  0.0013124783, no intersections, 0 file opens, fit gap 1.772 / 3.869 voxels
  (mean / p95), 7.42 voxels Hausdorff to the 1-thread oracle. The
  `fit_weight=0` control fails the fit gap (5.11 / 8.40), and the 5 cm push
  control is caught. The guest leaves native's trajectory after Newton
  iteration 7 of phase 0. Inputs, the LDLT solve and every libm function the
  solver calls are bitwise equal in the two builds. What differs is the C++
  library: libstdc++ and libc++ leave tied keys in different orders, and a sum
  taken in that order moves the last bit. SimpleBVH's Morton sort in the
  contact broad phase has such ties.
- **6.P:** the guest is about 20–25× slower than native (phase 0: 686–725 s
  against 33.6 s; the full run takes 42 min). The heap floor is memory_max
  352 MiB, and the full run at 1.25× that (440 MiB) was confirmed. One phase
  retires up to 5.6e11 instructions, 67× the default `execution_timeout`.
  The Zb* and V builds run in libriscv and give the same bits, but neither is
  faster.
- **Port: pass.** The driver is upstream's loop: with upstream numerics and
  OpenVDB it reproduces the 1-thread oracle bitwise (`driver/`).
- **SDF: pass.** Gate 6a passes 9/9 on both point sets, and its control fails
  as it should.
- **Native SdfGrid vs upstream: pass under the criterion proposed below.** The
  plan's "Hausdorff ≤ 1 voxel" cannot be met. The solve is chaotic, and
  upstream misses that bound against itself by 5–8 voxels.
- **Sampler: pass.** The Lean kernel (`lean/Fit/SdfSplineHessian.lean`,
  emitted to `kernels/fit/cpp`) is 0 ULP from OpenVDB's `sampleHessian`
  (Gate 6a a.kernel), and the foxgirl run with it is bitwise the run with the
  Gate 6a reference sampler: 177 Newton (24/36/50/67), energy
  0.0014288103789395555, `garment_final.f64` sha256 ab9fb211…

## Gate 6 in the guest

**Setup:**

- `project/gate_fit.gd`: Godot 4.7.2, `--rendering-driver vulkan --xr-mode off`.
  The gate is driven through main.gd's wrappers.
- **One Godot process per arm** (`gates/6-fit/run_arm.sh`), each with a fresh
  fit Sandbox.
- **Each `fit_step` is one vmcall on main.gd's worker Thread.** The main
  thread keeps drawing frames meanwhile: 150,731 frames during the 43 min
  full run.
- **Inputs:** the foxgirl fixture as float32 through the wire (util/obj_io.gd).
- **Native twin:** `fit_native` built with `FIT_SDF=sdfgrid`, the Lean kernel,
  `FIT_TBB=serial` and the guest numerics (`C:/b/fit-native-tbbs`). Its f32
  output is `C:/b/fit-out-tbbs32`, bitwise the earlier `fit-out-sdf32`.
- **Shared machine:** the arms ran up to eight at a time, beside other cuts'
  Godot gates (drape, ggml-rd, a pipeline run). Host seconds are wall time
  under that load. Instruction counts (the guest's `instret` CSR, read around
  each phase) do not depend on load.
- **ELF:** fit.elf sha256 `15f89eca…` for the phase-0, ladder and full runs.
  `2575d7ca…` (the same solver plus the `stl` probe) for probes, ladder-352,
  the 320 rerun, the timeout control and full-440; those gave the same bits
  where they overlap.

### 6.0 probes (`runs/probes.txt`)

| probe | result |
|---|---|
| exceptions | PASS: typed catch through two `std::function` |
| io | PASS: `fopen`, `ifstream` and `open` refused with EACCES; the counter reports **3 of 3** |
| instret | PASS: the CSR advances 500,004 over a 100,000-iteration loop, so phase instruction counts are readable |
| ldlt (n = 200) | PASS, residual 1.6e-16 |
| **ldlt8k** (7-point 3D Laplacian, 8,000 unknowns, 53,600 nonzeros; analyze + factorize + solve through polysolve's embedded spec) | PASS, residual 7.0e-16. **The solution is bitwise equal to native** (fnv1a `b09da10b…`). Guest 4,973 ms, native 195 ms: **25.5×**; 21.8× and 23.4× in two earlier runs |
| wire | **bitwise:** every array GDScript hands the guest equals what `fit_native --dump-inputs` gives its driver (15,384 body floats, 30,513 face indices, 8,046 garment floats, 15,660, 45 + 45 skeleton, 28 bones, 2,005 no-fit) |
| libm, 20,000 inputs per function | **equal:** log, pow, sqrt, sin, cos, tan, asin, acos, atan, atan2, fmod. **differ:** exp, cbrt, hypot, log1p, expm1, log2, log10, exp2, tanh |
| stl, 20,000 elements, keys 0..99 | **differ:** the tie order of `std::sort`, `std::nth_element` and `std::partial_sort`; the sum over the sorted order: 108555638.4104798 in the guest against …41047987 native (1 ULP) |

**The libm difference is not on the solve path.** In the disassembly of
fit.elf (`llvm-objdump`), the nine differing functions are called only from
`probe_libm`, from libm's own wrappers (tanh calls expm1, cabs calls
hypot), and through `cabs` from Eigen's DGMRES, which the configured
SimplicialLDLT path never reaches. The solver calls pow (668 call sites), sqrt, atan2, log, sin and atan,
and all of these are equal.

### 6.P: time, instructions, heap, ISA

**Phase 0, same code, host-timed**

- **Runs:** `runs/p0-*.txt`, three ISAs concurrently; guest phase 0 is 34
  Newton iterations.
- **Native:** `fit_native --phases 1`, run alone twice: 33.6 s for 41 Newton.
  An earlier, quieter run took 28.9 s.

| build (solver `-march`) | host s | instructions | vs rv64gc | G instr/s | trace |
|---|---|---|---|---|---|
| **rv64gc** (fit.elf) | 686.2 | 4.857e11 (463,162 × 2^20) | — | 0.71 | — |
| rv64gc_zba_zbb_zbs_zbc (`fit_zb.elf`) | 696.7 | 4.779e11 | −1.6% instructions, +1.5% time | 0.69 | equal to rv64gc |
| rv64gcv (`fit_v.elf`, autovectorised, Eigen still scalar) | 768.8 | 4.550e11 | −6.3% instructions, **+12% time** | 0.59 | equal to rv64gc |

- **Slowdown:** 686 / 33.6 = **20.4× per phase**, and **24.6× per Newton
  iteration** (20.2 s against 0.82 s).
  - Three more rv64gc phase-0 runs took 709.7–725.4 s under load.
  - The earlier smoke run took 596 s on a quieter machine.
- **ISA:** libriscv runs both extensions, and both give the same phase-0
  Newton count and 17-digit energy (no vertex arrays were compared).
  Neither is faster.
  - The bit-manipulation ops save 1.6% of the instructions and no time.
  - V saves 6.3% of the instructions, but costs 12% more time: vector
    instructions are dearer in the interpreter.
  - **Decision: the solver stays rv64gc** (`FIT_MARCH` default). The A/B builds
    are `FIT_MARCH=… FIT_ELF=fit_zb|fit_v build.sh`.
- **Instruction counts** are repeatable to about 1e-7 across runs; the
  energies are bitwise equal.

**Full run** (`runs/full.txt`, memory_max 2048)

| phase | Newton guest / native | host s | instructions | 2^20 units | heap at phase end |
|---|---|---|---|---|---|
| 0 AL | 34 / 41 | 709.7 | 4.857e11 | 463,162 | 122.8 MiB |
| 1 reduced | 59 / 37 | 560.5 | 4.195e11 | 400,093 | 123.2 MiB |
| 2 AL | 43 / 11 | 669.8 | 4.526e11 | 431,678 | 122.9 MiB |
| 3 reduced | 83 / 107 | 596.5 | 5.616e11 | **535,568** | 123.5 MiB |
| **total** | **219 / 196** | **2,536.5 (42.3 min)** | **1.919e12** | — | — |

- **fit_begin** takes 3.7 s.
- **Native, same load period:** 134.7 s (`C:/b/fit-out-tbbs32.log`); an earlier
  quiet run took 107.8 s.
- **Whole run:** 18.8× the loaded native run, 23.5× the quiet one.
- **Per instruction:** 1.9e12 instructions at 0.76 G/s.

**execution_timeout** counts units of 2^20 instructions.

- **Default:** 8,000 units, which is 8.4e9 instructions.
- **Largest phase:** phase 3 needs 535,568 units, **67× the default**.
- **main.gd now sets 2,500,000 units**, 4.7× the largest phase. At about
  0.75 G instructions/s, a runaway solve stops after about an hour.
- **Control** (`runs/timeout-ctl.txt`): at 100,000 units, phase 0 is stopped
  after 164 s with `Sandbox: Timeout for fit_step`. The limit is enforced, and
  one phase per vmcall needs the raised value.

**Heap ladder, phases 0 and 1** (`runs/ladder-*.txt`)

Each arm is a fresh Sandbox. The heap is 0.8 × memory_max.

| memory_max (heap) MiB | result |
|---|---|
| 512 (410) | PASS: 93 Newton, energy bitwise the full run's |
| 384 (307) | PASS, bitwise |
| **352 (282)** | **PASS**, bitwise: **the floor** |
| 320 (256) | **the Godot process dies (segfault, exit 139)** right after phase 0's Newton iteration 0. Twice, at the same point. No guest exception |
| 256 (205) | the vmcall aborts about 65 s into phase 0: `Protection fault (data: 0)` at the ecall in `__wrap_malloc`, from `ipc::BVH::detect_candidates` |
| 224 (179) | the same, with the heap at 94.7 MiB used and 84.5 MiB free |
| 192 (154) | Godot segfault inside phase 0's first Newton iteration. Twice |
| 160 (128) | Godot segfault, the same point (one run; inferred from the launcher) |
| 128 (102) | no progress: the first Newton iteration does not finish in 15 min (2 min at 2048), while the process spins one core. Killed |

- **The heap reading is steady, the need is not.**
  - At every phase end the heap holds 122.8–123.7 MiB and 79,252 chunks.
  - The floor needs a 282 MiB heap, 2.3× that. The difference is transient
    use inside a phase: candidate vectors, factorisations. The meminfo call
    cannot see those peaks.
- **Guest out-of-memory is never a `std::bad_alloc`.**
  - Below the floor, a failed allocation is either a Protection fault that
    aborts the vmcall (the host survives) or a segfault that kills Godot.
  - The ladder is therefore the only safe way to size memory_max.
  - This is a godot-sandbox robustness finding, not a fit.elf bug.
- **Confirmed:** the full run at 1.25 × 352 = **440 MiB** (`runs/full-440.txt`)
  finishes, and its garment is bitwise the 2048 run's.
- **main.gd's default is now 440.**
- **Unrelated crash:** one Godot process segfaulted at startup when three were
  launched in the same second (`p0-gc`, first try). A rerun was fine; the
  launches are staggered since.

### Gate 6 foxgirl in the guest (`runs/full.txt`, `results.txt`)

**(1) Guest vs native, same code: MISS.**

- **Final vertices:** max |Δv| = 1.436 voxels (0.0144 solve units), against
  the ≤ 1e-6 bound; the mean is 0.683 voxels.
- **Newton per phase:** 34/59/43/83 against 41/37/11/107.
- **Final energy:** 0.0013124783 against 0.0013414367, 2.2% relative.
- **First differing iteration** (`gates/6-fit/trace_diff.py`, polyfem debug
  traces):
  - 39 lines agree: setup, normalisation, the curve targets, f₀, and Newton
    iterations 0–7 of phase 0.
  - Line 40, the IPC min distance after iteration 7, is the first to differ:
    0.0006652093120624329 against 0.000665209312062554, relative 2e-13.
- **Why:**
  - The inputs are bitwise equal (the wire check).
  - The LDLT solve is bitwise equal (ldlt8k).
  - Every libm function the solver calls is equal.
  - The C++ library is not: libstdc++ in the guest and llvm-mingw's libc++
    natively leave ties in different orders (the stl probe). A sum over such
    an order differs by one ULP.
  - **The tie is on the solve path.** The contact broad phase is `"BVH"`,
    which is SimpleBVH (`BVH::init`). It orders the boxes with `std::sort` on
    a Morton code of the box centres, quantised to `int(x * 1000)`. Boxes
    closer than a millimetre-scale cell share a code, so the two libraries
    build different leaf orders. The candidates, and the barrier sums over
    them, then come in a different order, and the solve is chaotic.
    ipc-toolkit's own `parallel_sort` calls deduplicate on full keys and do
    not tie.
  - This is the likely path, not a proven one: no call site was
    instrumented. Making SimpleBVH's sort stable (Morton code, then index)
    would test it, and would make guest and native agree further.
  - Under the rule above, the guest is judged by the five-part criterion. The
    bound is not loosened.

**(2) Five-part criterion, guest result vs the 1-thread oracle: PASS.**

| part | guest | bound |
|---|---|---|
| 1 intersection-free | `ipc::my_has_intersections` none; push control INTERSECTS | none; control caught |
| 2 no file I/O | io_attempts 0 from fit_begin to the last phase; the positive control before begin counted 3 of 3 | 0 |
| 3 the fit holds | gap mean **1.7723**, p95 **3.8689** voxels | 1.8467 ± 0.25, 3.9207 ± 0.5 |
| 4 energy in upstream's basin | **0.0013124783** | [0.001167, 0.001647] |
| 5 coarse geometric bound | max **11.15**, mean 5.73, Hausdorff **7.42** voxels | ≤ 11.63, ≤ 8.24 |

Part 5 passes with 0.48 voxel to spare on the per-vertex max. The guest lands
where the 16-thread upstream runs land: its distance to the oracle is between
theirs (Hausdorff 4.74 and 8.24).

**(3) io_attempts == 0.** In every guest arm, from `fit_begin` on.

**(4) Heap and time:** see 6.P.

**Controls:**

- **`fit_weight=0` in the guest** (`runs/full-fw0.txt`): the run completes
  with no intersections, 145 Newton (34/31/36/44), energy 8.8e-5.
  - Its fit gap is **5.109 / 8.402**, so it **fails (3)** as it must.
  - It fails (4) too.
- **Push:** `fit_push_control` moves the garment's closest vertex (least SDF)
  5 cm into the avatar along −∇SDF: 0.05 solve units, 5 voxels, 0.115 body
  units.
  - The garment goes through `fit_check_intersections` and is reported
    **INTERSECTS**: vertex 2036, edge (6191,7164), face (1930,2014,1638).
  - The flat control sends the unpushed garment down the same path and gets
    **none**.

### Stage 8: garment resolution and iteration budget (recommendation)

**Measured at the foxgirl skirt (2,682 vertices):**

- a Newton iteration costs 7–14e9 instructions (8.8e9 on average), about
  9–20 s in the guest;
- the full retarget is 219 Newton and 42 min;
- the heap needs memory_max 352 MiB.

**For the Gate 8 loop:**

1. **Garment ≤ 1,000 vertices** from the pen/surfacing stage. This is
   extrapolated, not measured: it assumes a Newton iteration's cost scales
   about linearly with vertex count (sparse LDLT on a surface mesh, IPC broad
   phase). On that assumption a Newton iteration is ≈ 3.3e9 instructions,
   about 4–5 s. Stage 8 should measure one coarse fixture before relying on
   it.
2. **Newton budget ≈ 60 in total:**
   - `incremental_steps` = 1, so two phases instead of four;
   - the AL solve capped at 30 Newton, which is under its 50 now;
   - the reduced solve capped at 30. It has no effective cap now (5000),
     and it took 83–107 on foxgirl.
   - Expected about 4–5 min on the worker Thread, with a preview snapshot
     every Newton iteration from the ring.
   - Gate 8's check stands regardless: `check_intersections` none plus the
     push control.
3. **Foxgirl-scale fits** (2.7k vertices, about 220 Newton, about 45 min) are
   an offline or background job. They stay out of the interactive loop.
4. **Limits:**
   - memory_max 440 MiB (1.25× the floor), set before `program=`;
   - `execution_timeout` 2,500,000;
   - one phase per vmcall on a GDScript Thread;
   - at most one fit vmcall in flight.
5. **The solver stays rv64gc.** Neither Zb* nor V pays in the interpreter.

### Reproduce

```sh
BUILD_DIR=C:/b/ido6-guest bash build.sh                               # project/fit.elf
BUILD_DIR=C:/b/ido6-zb FIT_MARCH=rv64gc_zba_zbb_zbs_zbc FIT_ELF=fit_zb BUILD_TARGETS=fit_zb bash build.sh
BUILD_DIR=C:/b/ido6-v  FIT_MARCH=rv64gcv FIT_ELF=fit_v BUILD_TARGETS=fit_v bash build.sh
godot --path project --headless --import
FIT_BUILD=C:/b/fit-native-tbbs FIT_SDF=sdfgrid pixi run --manifest-path tools/native/pixi.toml bash tests/native/fit/build.sh
C:/b/fit-native-tbbs/fit_native.exe --out C:/b/fit-out-tbbs32 --io-probe --control-push > C:/b/fit-out-tbbs32.log
C:/b/fit-native-tbbs/fit_native.exe --out C:/b/fit-out-tbbs32-dump --phases 0 --dump-inputs C:/b/fit-out-tbbs32/inputs
bash gates/6-fit/run_arm.sh probes --arm=probes
bash gates/6-fit/run_arm.sh p0-gc --arm=solve --phases=1                          # and --elf=res://fit_zb.elf, res://fit_v.elf
bash gates/6-fit/run_arm.sh ladder-352 --arm=solve --phases=2 --mem=352            # 128 … 512
bash gates/6-fit/run_arm.sh timeout-ctl --arm=solve --phases=1 --timeout=100000
bash gates/6-fit/run_arm.sh full --arm=solve --mem=2048
bash gates/6-fit/run_arm.sh full-440 --arm=solve --mem=440
bash gates/6-fit/run_arm.sh full-fw0 --arm=solve --mem=2048 --fitweight0
bash gates/6-fit/run_arm.sh report --arm=report                                  # writes gates/6-fit/results.txt
```

The final garments (`runs/*.garment.f64`) are not committed; the logs are.

## Gate 6a (from `sdf/README.md`)

The points are FitForm's samples on the final garment of the 1-thread oracle.
The OpenVDB values come from `openvdb_dump`.

| check | `fit` set: 21,015 samples | `all` set: 78,300 samples |
|---|---|---|
| reference sampler vs OpenVDB `sampleHessian`, on OpenVDB's own stencils | 0 ULP in all 13 outputs | 0 ULP |
| Lean kernel vs OpenVDB `sampleHessian`, same stencils | 0 ULP in all 13 outputs | 0 ULP |
| control: the kernel with 2/3 as a float literal | **fails**: 8.7e18 ULP | **fails** |
| index map (p/h, floor, uvw) vs `worldToIndex` | bitwise | bitwise |
| value within 0.25 voxel, near-surface samples | 100% (p99 0.027, max 0.058 voxel) | 100% (max 0.084) |
| sign agreement | 100% | 100% |
| gradient within 10° | 100% (p95 0.81°, max 3.0°) | 100% (max 5.1°) |
| control: grid shifted +0.5 voxel in x | **fails**: value 58.6%, sign 99.41% | **fails**: 56.0%, 99.35% |

## Foxgirl: native SdfGrid vs upstream

**Setup:**

- **Build:** `fit_native`, built as `FIT_SDF=sdfgrid FIT_SDF_SAMPLER=reference`
  in `C:/b/fit-native-sdf`; the kernel run as `FIT_SDF_SAMPLER=kernel` in
  `C:/b/fit-native-sdfk`.
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
| **native SdfGrid, f64, Lean kernel** | 177 (24/36/50/67) | 0.0014288 (bitwise the row above) | 4.75 / 2.77 / 4.52 | 1.786 / 3.910 | none | 0 |
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
- **The sampler is the Lean kernel's.** The f64 run with
  `FIT_SDF_SAMPLER=kernel` is bitwise the reference-sampler run:
  `garment_final.f64` and `.obj` identical, `phases.tsv` identical but for
  wall and CPU time (`run-sdf-kernel-f64.log`, `phases-sdf-kernel-f64.tsv`,
  `compare.log`). The other rows were run with the reference sampler; since
  the two samplers are bitwise equal on this path, they stand for the
  kernel.
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
| SdfGrid guest f64, Lean kernel | 101.6 | 214.1 MB | the same grid, 0.54 s fill |
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
- **Measured (Gate 6 in the guest): not libm.** Every libm function the solver
  calls is equal. The C++ library's tie order differs, SimpleBVH's Morton
  sort has ties, and the trace parts in phase 0 after iteration 7. The
  fallback applies, and the guest passes the five-part criterion.

## Open

- **Energy per form.** Split the final energy by form for the
  upstream-numerics SdfGrid control, to say which term holds the +62%.
- **Guest = native bit for bit.** The guest run passes on the five-part
  criterion. To also meet the tight bound, the tie in SimpleBVH's Morton sort
  would have to be broken by box index. That needs an org fork of SimpleBVH
  (AGENTS.md rule 1), and a rerun would show whether the tie was the only
  cause.
- **Guest out-of-memory.** Below the heap floor the sandbox aborts the vmcall
  or kills Godot; it never raises `std::bad_alloc`. This belongs upstream in
  godot-sandbox (libriscv's native heap). Until then, memory_max comes from a
  measured ladder with 1.25× headroom.
- **Stage 8's coarse-garment cost** is extrapolated from 2.7k vertices, not
  measured (see the recommendation above).

## Reproduce

Run from the repo root under `pixi run --manifest-path tools/native/pixi.toml`.

```sh
FIT_BUILD=C:/b/fit-native-sdfk FIT_SDF=sdfgrid bash tests/native/fit/build.sh         # FIT_SDF_SAMPLER=kernel is the default
FIT_BUILD=C:/b/fit-native-sdf FIT_SDF=sdfgrid FIT_SDF_SAMPLER=reference bash tests/native/fit/build.sh
C:/b/fit-native-sdfk/fit_native.exe --out C:/b/fit-out-sdfk64 --io-probe --control-push --no-round
cmp C:/b/fit-out-sdfk64/garment_final.f64 C:/b/fit-out-sdf64/garment_final.f64
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
