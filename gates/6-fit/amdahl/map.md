Not written to `gates/6-fit/amdahl/map.md`: the harness says report files are returned rather than written, so the content is below and no files were added to the repo. I made no timed runs and did not touch the desk loop job. The only thing I created is a mesh-topology script in the scratchpad.

# Gate 6 Amdahl map: where fit.elf's Newton iteration goes, and what could run on the GPU

## Where ggml-rd is

- **Branch:** `cut-3`, local and `origin/cut-3` at `f1a4babbb`. It is 32 commits ahead of `main` and is **not merged** into `main` or `gate-0h`, so `guest/ggml-rd` is absent from this checkout.
- **Worktree:** `C:/Users/ernest.lee/AppData/Local/Temp/claude/C--interactor-dress-on/5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5/scratchpad/ido-3`.
- **Code:**
  - `guest/ggml-rd/`: `ggml-rd.cpp`, `rd_graph.cpp`, `rd_pack.cpp`, `rd_kernels.cpp`, and `ops/{binary,concat,conv3d,cpy,flash_attn_ext,get_rows,im2col,mul_mat,norm,repeat,rope,soft_max,unary}.cpp`.
  - Also `lean/Ggml`, `kernels/ggml` and `gates/3-ggml-rd` (G3.ops passes on 22 census ops; G3.graph passes).
- **Tensor types it accepts:** F32, F16, BF16 and I32 only (grep of `ops/*.cpp`). There is no F64, no sparse op, no scatter-add, no sort, no eigensolver and no factorization.

## Direct answer: ggml-rd or Lean kernels over rd_compute?

**ggml-rd is not the right vehicle for any of PolyFEM's hot work.** The right vehicle is Lean→Slang kernels on `rd_compute` (rule 2), with each GPU round trip batched into one compute list in the `AvbdRd::run` style.

Why not ggml-rd:
- PolyFEM's hot work is irregular:
  - per-element gathers into sympy-generated 12×12 Hessians;
  - 12×12 eigen-projections;
  - sparse assembly;
  - box-overlap candidate enumeration;
  - interval CCD;
  - sparse LDLT.
- None of that is `mul_mat`, `soft_max`, `rope` and the like.
- The contact parts need fp64, which ggml-rd does not have.
- The only dense-looking pieces are tiny: FitForm's per-face 9×9 (a sum of 15 PᵢᵀhᵢPᵢ) and CurveCenterTargetForm's per-loop outer-product block. ggml's host cost in the guest (about 5 µs per node, plus the app's graph build and gallocr; G3.cost) would exceed the work itself.

What to reuse from ggml-rd is its plumbing, not its op library:
- the fiber pump (WAIT_GPU / COOP / UPLOAD / READ);
- `rdc::Device::buffer_get_into` (≤ 16 MiB staging);
- permanent-RID pipeline and uniform-set caching;
- barrier elision.

## What was measured (the split below is not an estimate)

**Guest profile.**
- **Source:** `fit_prof.elf` (cut-6c, `guest/fit/fit_clock.cpp` wraps `clock_gettime` to `rdinstret`, so 1 "s" = 1e9 instructions). Log: `cut-6c:gates/6-fit/budget/guest-psd-prof.log`, lines 963–969.
- **Case:** the loop's 932-vertex skirt with `force_psd_projection` (the parked study's psd arm), phase 1 (reduced solve), 100 Newton, 3.86e11 instructions, 345 s host.
- **Native twin:** the same arm natively, `C:/b/budget/psd/run.log`, 103 Newton, 31.4 s.

| per Newton, phase 1, 932 skirt | guest instr | share | native s | native share |
|---|---|---|---|---|
| Hessian assembly (`compute_hessian`: all forms, PSD, sparse sums, PᵀHP) | 1.19e9 | 31% | 0.107 | 34% |
| broad phase (`line_search_begin`: swept BVH candidates) | 1.12e9 | 29% | 0.077 | 24% |
| constraint-set updates inside line-search trials (collision set from cached candidates + SDF resample) | 0.445e9 | 12% | 0.049 | 16% |
| linear solve (AMD analyze + LDLT factorize + solve, every iteration) | 0.368e9 | 10% | 0.035 | 11% |
| narrow-phase CCD (Tight Inclusion) | 0.251e9 | 7% | 0.016 | 5% |
| SDF brick fill: the phase's first `solution_changed`, 20.9e9 once, amortised | 0.209e9 | 5% | 0.020 | 6% |
| f, ∇f, NaN checks, trial energies | ~0.17e9 | ~4% | ~0.012 | ~4% |
| **sum of timers** (solver wall 3.61e9: timers overlap by about 5%) | **3.82e9** | | | |

- **The split is the algorithm's, not the interpreter's.** Native has the same shape; the guest/native ratio is about 8–14× per component in this run.
- **The linear solve is 10%, not the bottleneck.** Assembly and the broad phase together are 60%.
- **Not measured:**
  - phase 0 (the AL solve: 50 Newton, 2.4e11, 4.8e9 per Newton; its timing line never printed);
  - foxgirl per component (8.8e9 per Newton on average, Gate 6);
  - the loop's default config on gate-0h (no forced PSD: 253 Newton, 1393 s);
  - how assembly splits across forms;
  - contact candidate and collision counts (logged nowhere).
- **Line-search trials per Newton** (my count from the logs' collision-free/descent step ratios, no runs): mean **1.99** on the 932 skirt (150 line searches; CCD limited the step in 87) and **2.82** on the foxgirl full run (219; CCD-limited in 170).

## Sizes

Topology is from the meshes (my scratch script). The avatar is the foxgirl body for both garments.

| | 932 skirt (loop) | foxgirl skirt |
|---|---|---|
| unknowns (reduced / AL) | 2,796 / 2,797 | 8,046 / 8,047 |
| complete space the forms evaluate in, 3·(avatar + garment) | 18,180 | 23,430 |
| collision mesh V / E / F | 6,060 / 17,876 / 11,899 | 7,810 / 23,118 / 15,391 |
| avatar's share of broad-phase query boxes | 85% | 66% |
| similarity hinge evaluations (12×12 + PSD) | 5,048 | 15,516 |
| boundary loops | 64, 72 | 44, 100 |
| CurveCenterTarget dense blocks (entries) | 83,520 | 107,424 |
| fit faces × 15 samples | 1,728 → 25,920 | 1,401 → 21,015 |
| SDF voxels filled | ≈343,552 (native, same arm) | 105,984 (207 bricks) |
| Hessian nnz excluding contact (stencil estimate) | ≈1.8e5 | ≈3.9e5 |
| contacts / candidates | not logged | not logged |

## Per computation

In the headings below, "(x% measured)" is the guest share from the profile table.

### 1. Hessian assembly (31% measured)

**Call path.**
- `Newton::solve_sparse_linear_system` → `GarmentNLProblem::hessian` → each form's `second_derivative`, as an 18,180-wide sparse matrix, summed.
- Then `P^T H P` and a triplet rebuild to drop row/col 0.

**Forms and what each costs.**
- **SimilarityForm:** autogen `similarity_hessian` (`vendor/cloth-fit/src/polyfem/autogen/auto_derivatives2.cpp:3320`) plus `ipc::project_to_psd`, which is a 12×12 `SelfAdjointEigenSolver`. This runs per hinge, serially.
- **ContactForm:** ipc `potential.tpp`, per collision: distance Hessian plus PSD.
- **FitForm:** 9×9 per face from 15 SDF samples, no PSD.
- **Curve forms:** curvature is 9×9 plus PSD; torsion is 12×12 through `DScalar2` autodiff plus PSD; curve target is 4×4 `DScalar2`; center target is a dense (3N+1)² block per loop.
- **PointPenalty (AL only):** identity on 15,384 avatar degrees of freedom.

**(a) Parallel structure.**
- Per element and independent: 5,048 / 15,516 hinges, ~130 curvature and ~130 torsion stencils, 1,728 / 1,401 fit faces, and an unknown number of contacts.
- Then a sparse sum: a segmented reduction into about 1.8e5 / 3.9e5 nonzeros.

**(b) Precision.**
- *Non-contact forms:* O(1) geometry, so float would give a usable Newton direction (energy and line search stay fp64), but a different trajectory.
- *Contact: fp64 required.*
  - d² runs from about 1e-7 to 4e-6 (dhat = 0.002 in solve units) and is computed from O(1) coordinates. Float loses most of it to cancellation.
  - PSD-projecting a 12×12 whose eigenvalues span the barrier's stiffness (κ = 1e8) down to O(1): float's ε·‖H‖ swamps the tangential eigenvalues.
- *fp64 is affordable at these sizes:*
  - 5k–20k 12×12 eigensolves come to about 1e8–1e9 fp64 flop.
  - That is about 0.1–1 ms at a 4090's ~1.3 TFLOP/s fp64 (1/64 rate).
  - The 1/64 rate is not the constraint here; dispatch latency and transfers are.

**(c) GPU home: Lean→Slang on rd_compute.** Porting it needs:
- `similarity_hessian` and ipc-toolkit's autogen distance Hessians written as Lean expressions (rule 2: no hand kernels);
- a Lean 12×12 Jacobi eigensolver;
- deterministic gather assembly into fixed CSR slots. The existing template is `lean/Cloth/Avbd/AdjacencyKwise` (vertex → incident-constraint CSR with roles, K = 1/3/4) plus `lean/Cloth/SlangCodegen/VbdGather*` (fixed-order gathers). It would have to be extended from diagonal blocks to block pairs. ggml-rd has no fit here.

CPU-side, unmeasured:
- In the reduced phase, `P^T H P` over the 18,180-wide space is an exact extraction of the garment block. Each entry has a single ×1.0 term, so the extraction is bitwise the same.
- Rerunning `fit_prof.elf` at log level `trace` would print the per-form split. The `POLYFEM_SCOPED_TIMER` calls ("similarity hessian", "barrier hessian", "fit hessian") already exist and log at trace.

**(d) Data movement.**
- *Upload:*
  - garment positions: 22 KB / 64 KB per upload;
  - avatar once per phase: 123 KB;
  - collision list: about 20 B each;
  - the CSR pattern whenever contacts change it.
- *Readback:* CSR values only, about 1.5 MB / 3.2 MB plus contact entries, one staging copy.
- **Round trips added: 1 per Newton.**
- The CPU keeps the pattern union, AMD and LDLT.

**(e) Determinism.**
- The driver forms FMAs that slangc's cpp build does not (AGENTS), so guest+GPU ≠ native from the first GPU Hessian.
- Run-to-run determinism holds only with fixed-order gathers and no float atomics.
- GPU-to-GPU bitwise agreement is plausible: the drape is bit-identical on the 3090 and the 4090.

### 2. Broad phase (29% measured)

**Call path.**
- `LineSearch::line_search` → `ContactForm::line_search_begin` → `Candidates::build(x0, x1, dhat/2, BVH)`.
- That builds three SimpleBVH trees (vertices, edges, faces) over the **whole** collision mesh and queries EE (triangular) and FV.
- `can_collide` (`guest/fit/fit_driver.cpp:346-351`) then rejects avatar–avatar pairs.

**(a) Parallel structure.**
- Per query box, independent; the output is a variable-length list.
- 85% (932) / 66% (foxgirl) of query boxes are avatar primitives. Every avatar–avatar pair they return is thrown away through a `std::function` call.
- Brute-forcing only the garment-involved box tests: about 7e7 (932) / 2.5e8 (foxgirl).

**(b) Precision.** Float is enough if boxes are rounded outward (`nextafter`). A superset of candidates is safe, because the distance filter and CCD make the decision.

**(c) GPU home.**
- *CPU first (unmeasured):*
  - Query only garment primitives, without the triangular cut, emitting pairs as (min, max). Avatar–garment EE pairs are currently found only from the avatar-side queries.
  - Build the avatar tree once per phase: the avatar is static in the reduced phase and affine in α in the AL phase.
- *GPU:* a Lean kernel that brute-forces garment boxes against all boxes, as a count pass, an exclusive scan and a write pass. The output order is fixed by index, with no atomics, no Morton sort and no BVH.
- `lean/Cloth/SlangCodegen/SelfCollisionScan` has the brute-force shape but cannot be reused as is: it is float, not conservative, and drops pairs past K.
- Not ggml-rd.

**(d) Data movement.** Upload x0 and x1 (2 × 22 KB / 64 KB); read back the candidate list at about 8 B per pair (count unknown). **Round trips added: 1 per Newton.**

**(e) Determinism.** A canonical candidate order removes the dependence on SimpleBVH's Morton tie order, which is Gate 6's hypothesis for guest ≠ native. Sorting on the CPU path in the org ipc-toolkit fork would test that hypothesis cheaply.

### 3. Collision set and energy per line-search trial (12%, shared with #4)

**Call path.** `Backtracking::compute_descent_step_size` → `solution_changed(new_x)` → `Collisions::build(candidates_)` (per candidate: distance type, d² < dhat²).

**(a) Parallel structure.** Per candidate, plus a compaction. There are 1.99 / 2.82 trials per Newton.

**(b) Precision.** fp64: the classification and d² sit at 1e-7 to 4e-6.

**(c) GPU home.** A Lean kernel with one thread per candidate, keeping the list on the GPU for #1. It only pays if #2 is on the GPU, since the candidates live wherever the broad phase ran.

**(d) Data movement.** Per trial: upload x (22 / 64 KB), read back the barrier energy (a fixed-order fp64 reduction) or the compact list. **Round trips added: 1 per trial, about 2–3 per Newton.**

**(e) Determinism.** An order-preserving compaction keeps it deterministic.

### 4. SDF: resample per trial, and brick fill once per phase (fill: 5% amortised, 20.9e9 per phase)

**Call path.**
- `FitForm::solution_changed` (`vendor/cloth-fit/src/polyfem/solver/forms/garment_forms/FitForm.cpp`) gathers a 4³ stencil per sample from lazily filled 8³ bricks under a mutex, then runs the Lean kernel (`kernels/fit/cpp/sdf_spline_hessian_emit.cpp`) in double.
- A brick fill per voxel is `igl::fast_winding_number` (float) plus an AABB squared distance (double).
- FitForm is disabled in the AL phase, so the first reduced-phase `solution_changed` pays for the fill. That is the 20.9e9 "constraint_set_update" timer. This is an inference; natively the same arm fills 3.4e5 voxels in 3.19 s.

**(a) Parallel structure.** Per sample: 25,920 / 21,015. Per voxel: ≈3.4e5 / 1.06e5 voxels against 10,171 avatar triangles.

**(b) Precision.**
- The resample kernel is double and 0 ULP from OpenVDB (Gate 6a); in fp64 it is about 4e7 flop, negligible.
- Float would pass the 0.25-voxel criteria but would not be bitwise.
- For the fill, a float distance is enough (the winding number is already float).

**(c) GPU home.**
- Resample: the existing `lean/Fit/SdfSplineHessian.lean`. `kernels/fit/gen.sh` already validates its SPIR-V, which requires Float64. Bricks stay resident; missing bricks are flagged and filled, then the kernel is re-dispatched.
- Fill: a new Lean kernel doing brute-force distance plus winding number. It depends only on the avatar, so it can be dispatched at `fit_begin` and read before phase 1, fully overlapping phase 0.

**(d) Data movement.**
- Per trial: upload 22 KB; read back about 207 KB of values. Keep g and h on the GPU for #1, otherwise 2.7 MB.
- Fill: about 2.7 MB once per phase.
- Round trips: the resample shares #3's; the fill adds 0 per Newton.

**(e) Determinism.** GPU double ≠ CPU double bitwise (FMA contraction). A brute-force winding number differs from the fast-winding-number approximation near the sign boundary; Gate 6a's sign check covers that.

### 5. Narrow-phase CCD (7% measured)

**Call path.** `Candidates::compute_collision_free_stepsize` runs Tight Inclusion per candidate (tolerance 1e-3, 200 iterations, and an unlimited retry below `CCD_SMALL_TOI`) with a shared, shrinking `tmax`.

**(a) Parallel structure.** Per candidate, but each is a data-dependent interval branch-and-bound; the reduction is an exact min.

**(b) Precision. fp64 required.**
- Tight Inclusion's conservative filters are derived for double.
- The float build (`TIGHT_INCLUSION_WITH_DOUBLE_PRECISION=OFF`) has a filter comparable to the 1e-4 minimum separation, which means more zero-TOI fallbacks.
- CCD limits 58–78% of steps, so this is on the convergence path.

**(c) GPU home.** It stays on the CPU. It is 7% of the time, and a Lean port of Tight Inclusion in fp64 would be the largest kernel of the lot. scalable-ccd's CUDA narrow phase (double) is the reference design if it is ever needed.

**(d) Data movement.** Read back 1 double. **Round trips added: +1 per Newton, or 0 if fused into #2's compute list.**

**(e) Determinism.** The CPU result depends on candidate order through the shrinking `tmax`. A GPU version needs a fixed `tmax` per pass so the min is order-free.

### 6. Sparse LDLT (10% measured)

**Call path.**
- `analyze_pattern` (AMD plus symbolic) runs **every** iteration, then factorize and solve.
- The residual is then checked against polysolve's `residual_tolerance` 1e-5 × the characteristic length.
- The solver is `Eigen::SimplicialLDLT`.

**(a) Parallel structure.** Sequential along the elimination tree. n = 2,796 / 8,046, with two dense loop blocks (192² and 216² / 132² and 300²).

**(b) Precision.** fp64: the barrier is 1e8 stiff and the residual test is at 1e-5.

**(c) GPU home. It stays on the CPU.**
- A GPU sparse direct solver at n ≤ 8k is latency-bound: hundreds of etree levels means hundreds of barrier-separated dispatches.
- GPU PCG over the existing `Spmv`, `SpmvDf32`, `DotReduce`, `Saxpby`, `SaxpbyIndirectDf32`, `CGAlpha` and `CGBeta` kernels would be an algorithm change (inexact Newton). The 1e-5 residual test would reject its directions and cascade into Projected and then Regularized Newton. Gate the CG iteration count first.
- CPU fix: skip `analyze_pattern` when the pattern has not changed.

**(d) Data movement.** None while it stays on the CPU.

**(e) Determinism.** Bitwise equal between guest and native (Gate 6's ldlt8k probe).

### 7. Energy, gradient and line-search control (about 4%)

These stay on the CPU in fp64.
- Backtracking accepts on `new_energy < old_energy`, and on ‖g‖ comparisons below 1e-4.
- The solve stops at ‖g‖ < 0.01, starting from about 413.
- These decisions hinge on the last bits near convergence. Moving them would save at most 4%.

## Amdahl bound (932 skirt, phase 1, per Newton; zero-cost GPU assumed)

| moved to GPU | CPU left, per Newton | bound |
|---|---|---|
| assembly + broad phase + trial updates | ~1.06e9 | ≤ 3.6× |
| + SDF fill (overlapped with phase 0) | ~0.85e9 | ≤ 4.5× |
| + CCD in fp64 | ~0.60e9 | ≤ 6.4× |
| floor: LDLT + f/∇f/line-search logic | ~0.54e9 | ≤ 7× |

**Round-trip cost.**
- Each of the 4–5 round trips per Newton costs at least one pump frame under rule 4.
- At the loop's own 1,674 frames/s (flat, 1,025,338 frames in 612 s), that is about 3 ms per Newton.
- At 60 Hz it is about 70–85 ms; at 90 Hz XR, about 45–55 ms.
- Against about 1 s of CPU still left per Newton, that is ≤ 8% even at 60 Hz. Frame latency is not the constraint; the CPU residue is.

**Plumbing that does not exist yet.**
- `fit_step` is one blocking vmcall per phase on a GDScript worker Thread.
- The GPU path needs the fit job on a guest fiber that yields WAIT_GPU/READ, pumped on that worker thread with its own local RenderingDevice (submit on one tick, sync on the next).
- ggml-rd's pump is for main-thread `_process`. CPU segments here last about a second, so they cannot run on the main thread.

## Determinism: two findings from reading the code

1. **Candidate order drives the barrier sums and the CCD result.**
   - Candidates come out in BVH query order, which depends on Morton ties.
   - Collisions inherit that order, and so do the barrier gradient and Hessian sums and `setFromTriplets`.
   - CCD's shrinking `tmax` also makes its TOI order-dependent.
   - A canonical sort, which any GPU broad phase needs anyway, also removes the CPU dependence. That is a direct test of Gate 6's tie hypothesis.
2. **`fesetround` is honoured natively and ignored in the guest** (Gate 0F probe 2).
   - `polysolve/nonlinear/line_search/LineSearch.cpp` rounds `step_size *= max_step_size` under `FE_DOWNWARD`, with no `FENV_ACCESS` pragma. ipc-toolkit's `AABB::conservative_inflation` rounds down and up, and does have the pragma.
   - Line searches where the non-CCD factor (nan-free step × limiter) is not a power of two and CCD limits the step: 11 on the loop skirt (from Newton 0: 0.15916978658261366 × toi) and 12 on foxgirl (from Newton 1 of phase 0).
   - Foxgirl's trace still printed identical 17-digit step sizes through Newton 7. So either those products rounded the same, or clang moved the multiply outside the rounding window (the `LineSearch.cpp` site has no pragma). Unresolved.
   - A native control should `--wrap=fesetround` to a no-op before the remaining divergence is blamed on sort ties.
   - Vulkan has no per-operation directed rounding either, so any GPU inflation must use `nextafter` explicitly.

## Gates before building (rule 7)

1. **fp64 on RenderingDevice.** Does Godot's Vulkan device enable `shaderFloat64`, on the 4090/3090, on dzn (the desk loop worker) and on the RunPod GPU? One double dispatch, with a float control. It is unverified: `gates/0f-runtime` and `gates/1-rd-compute` have no fp64 probe, and `kernels/fit/gen.sh` validates the SPIR-V but never dispatches it.
2. **Assembly per form.** `fit_prof.elf` at log level `trace` (the scoped timers exist), plus instret around `ipc::project_to_psd`.
3. **Contact counts.** Candidates and collisions per Newton: a counter in the fork.
4. **CPU fixes, measured on `fit_prof.elf`:**
   - garment-only broad-phase queries;
   - the canonical candidate sort, then rerun `gates/6-fit/trace_diff.py`;
   - a `P^T H P` shortcut in the reduced phase;
   - reusing `analyze_pattern`;
   - `fesetround` neutralised in `fit_native`.
5. **Worker-thread pump.** A local RenderingDevice plus the fiber pump for fit.elf, and the frames per round trip in flat and XR.

**Evidence** (all existing, read-only):
- `C:\Users\ernest.lee\AppData\Local\Temp\claude\C--interactor-dress-on\5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5\scratchpad\ido-6c\gates\6-fit\budget\guest-psd-prof.log`
- `C:\b\budget\psd\run.log`
- `C:\interactor-dress-on\gates\6-fit\results.txt`
- `C:\interactor-dress-on\gates\6-fit\runs\full.godot.log`

**Scratch scripts:**
- `C:\Users\ernest.lee\AppData\Local\Temp\claude\C--interactor-dress-on\5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5\scratchpad\amdahl_sizes.py`
- `C:\Users\ernest.lee\AppData\Local\Temp\claude\C--interactor-dress-on\5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5\scratchpad\ls_trials.py`
- `C:\Users\ernest.lee\AppData\Local\Temp\claude\C--interactor-dress-on\5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5\scratchpad\ls_round.py`