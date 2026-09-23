# Gate 5: drape.elf, DiffCloth's Simulation and L-BFGS-B in the guest

**Result: FAIL, on two of nine criteria.** G1, G3, G4, G6, G7, G8 and G9
pass. G2 fails on one of 20 LBFGSpp traces. G5 fails in two places: dL/dμ at
native's starting μ is 5.02% off (the limit is 5%), and the loss at μ 0.01
prints 1.650 where native prints 1.652. For each failure a flat control
measures how much the problem itself moves under perturbations far smaller
than the tolerance, and in each case that movement already exceeds the
tolerance (see G2 and G5).

The run: [`results.txt`](results.txt) (verdicts, tables and every job's log)
and [`run.log`](run.log). Date 2026-09-23, Godot 4.7.2, RTX 4090, 6.5 min of
wall time:

```
godot --path project --script gate_drape.gd --rendering-driver vulkan --xr-mode off --gpu-index 0 > ../gates/5-drape/run.log 2>&1
```

Godot's `--gpu-index` order changed between boots. Index 0 was the 4090 on
this one; `results.txt` names the adapter in its first line. `-- only=G3,G7`
runs a subset into `results_partial.txt`, and `-- quick` shortens every job.
The gate is frame-driven: vsync is off, there is one `drape_job_tick` or
`drape_tick` per frame, and the wall clock quits in every branch.

| | what | result |
|---|---|---|
| G1 | L-BFGS-B components against LBFGSpp, cpu and rd | **PASS** 20/20 on both. The control (θ×1.25) fails 20/20. Worst rd error: xcp 9.0e-5, drt0 8.6e-5 (limit 1e-4) |
| G2 | 20 whole-problem traces against LBFGSpp | **FAIL** 19/20 on both. rosen_n2_m10_dc: f is 2.1e-5 off (limit 1.05e-6), iterations are 19/19, and the final sets are identical |
| G3 | inverse_min in the guest | **PASS** on both. k_tri recovers to 2.0 ± 1.2e-6 and (k_bend, density) to (1.5, 1.25) ± 4.1e-6, in LBFGSpp's 6 and 13 iterations, with the parameters 1.3e-7 from LBFGSpp's. The zero-gradient arms stay 1.5 and 1.25 away |
| G4 | the sphere demo's forward against the native frames | **PASS**. Faces are identical. Frame 1: 1.2e-10, frame 10: 4.9e-6, frame 50: 5.0e-6 (the OBJ print), frame 100: 4.2e-3, frame 350: 0.10. The μ 0.3 control is further away at every frame from 50 on |
| G5 | native-mode dL/dμ against backwardLog | **FAIL**. At μ 0.01: 0.43%, pass. At μ₀: 5.02%, fail. The loss at 0.01 prints 1.650 against 1.652, fail |
| G6 | unrolled mode against central FD, single colour | **PASS**. Worst errors: 9.5e-5 on the panel and 0.043 on the plane. The table picks unrolled as the default |
| G7 | our L-BFGS-B reproduces the native μ sequence | **PASS**. Fed backwardLog's values, it gives 0.539770 → 0.010000 → 0.375146 on cpu and rd. Live, it gives 0.539770 → 0.010000 → 0.374976. The recompute modes reach 0.300046 over 60 steps and stay at μ₀ over 350 |
| G8 | rule 4 and no Eigen | **PASS**. same_frame_syncs is 0 of 49,624, and the probe raises it. llvm-nm finds 0 Eigen and 0 LBFGSpp symbols. rd_close frees every slot |
| G9 | crossovers and `auto` | **PASS**. L-BFGS-B: cpu is cheaper up to n = 1e5. Drape: at 90 fps rd is cheaper from 196 vertices, and `auto` is set to rd from 160 |

## What this cut added (Task I, the integration)

- **`project/gate_drape.gd`**, rewritten as the single Gate 5 run over G1–G9.
  It reuses Task B's and Task D's jobs, and its main.gd-wrapper checks cover
  inverse_min and the job list (rule 8). Tasks D and B keep their own
  directories ([`drape/`](drape/) and [`lbfgsb/`](lbfgsb/)), which are marked
  as superseded.
- **G3, `inverse_min`.** The objective is `guest/drape/inverse_min.h`, a port
  of `test_avbd_inverse_min.cpp`, and the job is `guest/drape/inverse_jobs.cpp`.
  The oracle is `tests/inverse_min_oracle/`: LBFGSpp on the same objective,
  compiled for the host from the guest's AvbdCpu sources, written to
  [`oracle/inverse_min/`](oracle/inverse_min/).
- **G7**
  - `lbfgsb_replay` runs our driver on backwardLog's table.
  - The gate also runs the live optimizes through `drape_queue_optimize`.
- **G9, `lbfgsb_bench`.** A tridiagonal box QP for n from 10 to 1e5. The drape
  bench runs twice, with uncapped frames and at 90 fps.
- **The trace diagnostic.** [`trace/`](trace/) holds
  `project/probe_drape_trace.gd`, `compare_steps.py`, `loss_attribution.py`,
  `run.sh` and their logs. `sphere_forward stats=all` prints every step's
  statistics the way upstream computes them: in float, with the mean
  accumulated in float.

Fixes found by the integration, each described under its gate:

1. **The native run's μ is 0.5397701956236457, not 0.539770** (G4, G5).
   It is now `kNativeSphereMu0` and the default of `sphere_forward` and
   `sphere_backward`.
2. **Two L-BFGS-B guards used FLT_EPSILON where LBFGSpp uses DBL_EPSILON as
   an absolute threshold** (G7). The Lean kernels were fixed, then re-emitted
   and re-pinned.
3. **Trial points are clamped to the box** (G7).
4. **`drape_open(auto)` is now rd from 160 vertices, down from 256** (G9).

## G1 and G2: the L-BFGS-B kernels and driver against LBFGSpp

These are Task B's jobs over Task K's kernels, unchanged apart from fixes 2
and 3; the component fixtures and the traces give the same numbers as before
those fixes. Full detail is in [`lbfgsb/README.md`](lbfgsb/README.md) and
[`kernels/README.md`](kernels/README.md).

Worst relative error over the 20 component fixtures (limit 1e-4; drt, dg and
step_max 5e-4):

| backend | M | Mv | xcp | drt0 | vecc | step_max0 | drt | dg |
|---|---|---|---|---|---|---|---|---|
| cpu | 5.3e-7 | 4.9e-7 | 6.6e-6 | 6.4e-6 | 2.7e-6 | 1.1e-6 | 3.0e-7 | 6.9e-8 |
| rd | 3.8e-7 | 4.8e-7 | 9.0e-5 | 8.6e-5 | 3.7e-5 | 1.5e-5 | 2.9e-7 | 1.4e-7 |

G2's failing trace is rosen_n2_m10_dc. It is Rosenbrock with n = 2 against an
upper bound of 0.6, using DiffCloth's parameters. It stops on the delta test
(Δf < 1e-3) partway down the valley, so where it stops depends on the path.
- Ours ends at f 0.0506256 (cpu) and 0.0506258 (rd), against LBFGSpp's
  0.0506045. That is 2.1e-5 against a band of 1.05e-6.
- The run matches LBFGSpp on iterations (19), evaluations (25) and the final
  sets. The first iterate that differs by more than 1e-6 is iterate 5.

The flat control is LBFGSpp itself in double, on the same trace
(`tests/lbfgsb_oracle/sensitivity.{cpp,sh}`, [`lbfgsb/sensitivity.log`](lbfgsb/sensitivity.log)).
- Rounding the inputs to float32 moves its f by 2.5e-6.
- Rounding the inputs and also evaluating f and g at float32 x, with g
  handed back as float32 (the interface the guest's objective sees), moves
  it by 4.8e-6.

Both are outside the band, and this is the only trace where either control
leaves it. The float32 vectors inside our driver take it the rest of the way.
With LBFGSpp's default tolerance (the `_tight` trace) the same problem
matches to 6.9e-9.

## G3: inverse_min

cloth-dynamics' smallest end-to-end inverse design has 4 vertices, 4 chained
steps and a synthetic target made by the same solver at the truth. The
parameters are driven by the in-guest L-BFGS-B (LBFGSpp's defaults, m 10), on
AvbdCpu with the cpu vectors and on AvbdRd with the rd vectors. Each run is
compared with LBFGSpp on the host-compiled AvbdCpu objective
([`oracle/inverse_min/`](oracle/inverse_min/)).

| case | backend | recovered | err vs truth (≤ 0.05) | iterations ours / LBFGSpp (±2) | evaluations | \|x − x_LBFGSpp\| (≤ 1e-3) | zero-gradient arm (> 0.1) |
|---|---|---|---|---|---|---|---|
| k_tri: 0.5 → 2 | cpu | 1.99999881 | 1.2e-6 | 6 / 6 | 7 / 7 | 1.3e-7 | 1.5 |
| k_tri | rd | 1.99999881 | 1.2e-6 | 6 / 6 | 7 / 7 | 1.3e-7 | 1.5 |
| (k_bend, density): (0.4, 2.5) → (1.5, 1.25) | cpu | (1.50000405, 1.24999917) | 4.1e-6 | 13 / 13 | 14 / 14 | 1.3e-7 | 1.25 |
| (k_bend, density) | rd | (1.50000405, 1.24999928) | 4.1e-6 | 13 / 13 | 14 / 14 | 1.3e-7 | 1.25 |

- Our targets agree with the host's to 2.4e-7.
- Upstream's own backtracking gradient descent, run on the same objective,
  reaches errors of 0 and 6e-7 (`oracle/inverse_min/oracle.log`).

## G4: the forward against the native sphere demo, at native's exact μ

Faces are identical, in order. Max |dx| against native `iter0`:

| frame | 0 | 1 | 10 | 50 | 100 | 350 |
|---|---|---|---|---|---|---|
| rd, 350 steps | 4.2e-22 | 1.2e-10 | 4.9e-6 | 5.0e-6 | 4.2e-3 | 0.102 |
| cpu, 100 steps | 4.2e-22 | 1.2e-10 | 4.9e-6 | 6.4e-6 | 0.082 | — |
| μ 0.3 (control) | | | | 6.8e-3 | 0.144 | 0.184 |
| limit | | 1e-5 | 1e-3 | | | |

All 350 steps are finite on rd, and all 100 on cpu.

### Native's μ

The native run does not start at the printed 0.539770.
`OptimizeHelper::getRandomParam` calls `srand(1)`, spends the first `rand()`
(41) on a logged "seed", and draws the parameter with `VecXd::setRandom()`:
Eigen's −1 + 2·rand()/RAND_MAX on the second `rand()` (18467), mapped onto
[0.01, 0.95]. That gives μ₀ = **0.5397701956236457**.

With μ₀ in place of the printed value, frame 50 falls from 6.4e-6 to 5.0e-6,
which is the OBJ files' 6-digit print, and frame 350 from 0.122 to 0.102.

### Where the sphere demo leaves native ([`trace/`](trace/))

`compare_steps.py` compares every printed statistic of every step
(|Δx|_max, |Δx|_mean, pred_max, drift_max and its vertex, all computed in
float as upstream computes them) and every self-collision count against the
matching pass of `native/stdout.log`. It also compares every frame against
the native OBJ export, beyond the export's print resolution.

| run | first step with any printed difference | the cause | first frame beyond the print (> 1e-6) |
|---|---|---|---|
| rd, μ₀ (iter0) | 71 | 5 self-collision pushes where native has 6 | 72 |
| rd, μ 0.3 (the target pass) | 81 | 1 push where native has 0 | — (native never exported it) |
| rd, μ 0.01 (iter1) | 120 | a push count | 121 |
| rd, printed 0.539770 | 12 | a statistic's last digit | 37 |
| cpu, μ₀ | 5 | the cpu backend's float order | 23 |

- On rd at the right μ, drape.elf reproduces native digit for digit through
  the self-collision onset at step 69, where both runs apply 6 pushes, and
  on to step 70.
- Each run parts at a single pair on the self-collision threshold. From
  there the difference grows chaotically, to 4e-3 by frame 100.
- We could not locate the remaining float-level difference.
  - The AVBD SPIR-V is byte-identical to native's, except for the `lane <
    count` guards on the padded colour tails.
  - The host arithmetic has been checked against upstream's Vec3f and double
    code.
- cpu against rd is the same code in another float order, and parts from
  step 5.

## G5: native-mode dL/dμ against `native/backwardLog.txt`

The target is our own μ 0.3 run of 350 steps, because native never wrote its
target. The loss is MATCH_TRAJECTORY with K = 20.

| μ | ours (rd) | backwardLog | rel | verdict | loss ours / native | cpu backend (same port) |
|---|---|---|---|---|---|---|
| μ₀ = 0.5397701956 | 0.0121091 | 0.01153 | 5.02% | **FAIL** (limit 5%) | 0.00141610 / 0.00132918 | 0.01117, 7.8% from rd |
| 0.01 | −50.674986 | −50.45588 | 0.43% | pass | 1.64972503 / 1.65198194: prints **1.650** vs 1.652, **FAIL** | −51.09 |
| 0.375146 (info) | 0.0074085 | 0.00781 | 5.1% | — | 0.00060653 / 0.00047714 | 0.005832 |

**The flat control on the problem itself.** The same rd port at μ within
2e-7 of μ₀:

| μ | dL/dμ | loss |
|---|---|---|
| 0.5397701956236457 (μ₀) | 0.0121091 | 0.00141610 |
| 0.5397701859474182 (float32 μ₀, what the optimizer evaluates) | 0.0126385 | 0.0014073 |
| 0.539770 (printed) | 0.012444 | 0.0014124 |
| 0.5397702 | 0.0130591 | 0.0015085 |

- Moving μ by 7e-9 moves dL/dμ by 4.4% and the loss by 0.6%. Moving it by
  2e-7 moves the loss by 6.5%.
- Across these four points dL/dμ spans 7.5% of its mean, and backwardLog's
  0.01153 sits 5.0% below that span.
- In native mode, dL/dμ comes from the friction events of the last 20 steps,
  and upstream's doubled carry weights the last step 2^19 times over the
  20th-last. It is therefore a function of frames 331–350, which are past
  the chaotic onset in both our run and native's.
- A 5% tolerance at μ₀ is tighter than the sphere demo's own conditioning.
  At μ 0.01, where the gradient is large and robust, we agree to 0.43%.

**The loss at μ 0.01** (`trace/loss_attribution.log`):
- native's own iter1 frames against our μ 0.3 target give 1.65180863,
  within 1e-4 of native's 1.65198194;
- our μ 0.01 frames against the same target give 1.64972504.

So our target is right, and the gap comes from our μ 0.01 trajectory, which
parts from native's at step 120 (a push count) and is 0.27 away by frame 350.

## G6: the backward modes against central finite differences

The scenes are an 8x8 1 m panel pinned at two corners, and the same panel
unpinned on a tilted plane (contact, projection and friction on every step).
Both run 20 steps. The table gives the maximum relative error over μ, kTri
and density; the FD step is relative 1e-3.

| scene | unrolled | step | native |
|---|---|---|---|
| panel, colours off | 9.5e-5 | 0.98 | 1.0 |
| panel, colours on | 1.8e-4 | 0.98 | 1.0 |
| plane, colours off | 0.043 | 0.94 | 1.0 |
| plane, colours on | 0.58 | 0.90 | 1.0 |

- **Unrolled has the smallest worst case, so it is the default mode.**
- The criterion (single colour, 5e-2) passes on both scenes.
- On the plane, the FD itself moves 4% between eps and 10·eps: contact makes
  the loss only piecewise smooth.
- Multi-colour Gauss-Seidel is exact on the panel. On the plane it is not,
  because the backward treats each iteration as Jacobi. Use single colour
  when stiffness gradients have to be exact.

## G7: the native μ sequence, and recovering μ = 0.3

**Replay** (`lbfgsb_replay`). The objective is backwardLog's table:
- (0.00132918, 0.01153) at μ₀;
- (1.65198194, −50.45588) at 0.01;
- (0.00047714, 0.00781) at any other μ.

The run uses DiffCloth's parameters (m 10, delta 1e-3, max_linesearch 20)
and bounds [0.01, 0.95]. Our driver evaluates 0.539770 → 0.010000 →
**0.375146** (0.3751455), on cpu and on rd. That is 1 iteration and 3
evaluations, stopped by the delta test, which is native's sequence at its
printed precision.

backwardLog prints dL/dμ at μ₀ to 4 digits, and the second trial depends on
that value. Over g₀ ∈ [0.011525, 0.011535] the trial ranges over
[0.3751453, 0.3751457], and that range contains native's 0.375146.

**Live** (`drape_queue_optimize`, rd, 350 steps, native mode, from float32
μ₀). It evaluates 0.539770 → 0.010000 → 0.374976 and stops after 1 iteration
on the delta test. The structure is native's; the third μ is 1.7e-4 away,
because our f and g at μ₀ carry G5's spread.

**Recompute modes, final μ against the ground truth 0.3** (information; up to
8 iterations):

| mode | 350 steps | 60 steps (before the onset at step 69) |
|---|---|---|
| unrolled | stays at μ₀ (err 0.24). Line-search failure after 12 evaluations: a step of 3.6e-7 raises the loss 10% | **0.300046** (err 4.6e-5), 6 iterations, 9 evaluations |
| step | stays at μ₀ (err 0.24). Converged: \|g\| 3.2e-6 is under epsilon 1e-5 | **0.300525** (err 5.2e-4), 6 iterations, 28 evaluations |

- Over 350 steps the loss is rough in μ at the 1e-7 scale (G5), so no local
  gradient can drive it.
- Upstream's own run "succeeds" at 0.375146 only because the delta test
  stops it after the first line search.
- Over a horizon that ends before the self-collision onset, the unrolled
  gradient recovers the ground truth to 4.6e-5. That run uses tolerances
  scaled to the loss (epsilon and delta 1e-12): its loss starts at 2e-6.

**The fixes the live runs found:**

1. **The ε guards.**
   - LBFGSpp guards the Cauchy step's fpp and the subspace fallbacks' g·d
     with `numeric_limits<double>::epsilon()`, used as an absolute
     threshold. The kernels had FLT_EPSILON instead.
   - On the sphere demo |g| is 1.2e-5, so fpp = g·g is 1.4e-10, which is
     below FLT_EPSILON. The guard fired and shrank the Cauchy step 8000
     times, to under half an ulp of μ. So xcp = x, and the run failed with
     "the moving direction does not decrease the objective".
   - Both guards now use `dblEps` (2^-52, exact in float32) in
     `lean/Drape/SlangCodegen/{Dsl,LbCauchy,LbSubspace}.lean`.
   - Re-emitted and re-pinned: `lake build` passes (91 jobs), and
     `pin.py --check` reports none stale. The G1 fixtures are unchanged
     ([`kernels/README.md`](kernels/README.md)).
2. **The box clamp.**
   - In float32, xp + step·d at step_max landed one ulp below the bound
     (μ 0.00999999 against lb 0.01).
   - The trial point is now clamped to the box (`box_project` after the
     trial's `saxpby`). This is a no-op for interior coordinates, and G1 and
     G2 are unchanged.

## G8: rule 4 and rules 2/3

- `rd_rule4` counts 0 same-frame syncs out of 49,624 syncs over every job
  and every optimize, and `rd_rule4_probe` raises the count 0 → 1.
- `llvm-nm -C project/drape.elf` (run by the gate) lists 9,364 symbols:
  0 match Eigen and 0 match LBFGSpp.
- `rd_close` gives `CLOSED device=freed permanent_slots=0`.

## G9: crossovers, and `auto`

**L-BFGS-B, ms per iteration** (a tridiagonal box QP run for 10 iterations;
the host wall time of one job divided by its iterations):

| n | 10 | 100 | 1e3 | 1e4 | 1e5 |
|---|---|---|---|---|---|
| cpu | 0.37 | 0.26 | 1.35 | 12.1 | 185 |
| rd | 9.1 | 5.0 | 12.7 | 84 | 1368 |

There is no crossover up to 1e5, so `drape_queue_optimize`'s `vec=auto` and
the lbfgsb jobs' `auto` stay on cpu. The rd cost is phases times frames
(three submits per iteration), plus the readback of x and the upload of g at
every evaluation. At n = 10 the run converges after 9 iterations and its
line search stops.

**The drape step, ms per step** (the sphere-demo scene at n x n: solve,
contact and self-collision; 20 frame-driven steps, host-timed):

| vertices | 16 | 36 | 64 | 100 | 144 | 196 | 256 | 400 | 576 | 1024 | 2304 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu, uncapped | 1.6 | 3.9 | 8.1 | 12.5 | 19.5 | 25.4 | 35.7 | 54.6 | 89.4 | 151 | 368 | 692 |
| rd, uncapped | 7.1 | 6.7 | 7.0 | 7.3 | 7.3 | 7.4 | 7.8 | 8.2 | 9.1 | 10.3 | 15.7 | 22.3 |
| cpu, 90 fps | 7.8 | 8.3 | 8.4 | 12.2 | 18.9 | 24.8 | 35.6 | 53.4 | 83.4 | 147 | 370 | 668 |
| rd, 90 fps | 21.7 | 22.2 | 22.2 | 22.2 | 22.2 | 22.2 | 22.2 | 22.2 | 22.3 | 22.9 | 25.5 | 27.3 |

rd's cost is about two frames per step (solve, then contact and scan; a
third frame when pushes need pass 2), so where rd overtakes cpu depends on
the frame rate:
- with frames uncapped (~3.5 ms each), between 36 and 64 vertices;
- at 90 fps, the OpenXR display rate this project deploys at (AGENTS.md
  rule 9), between 144 (cpu 18.9 against rd 22.2) and 196 (24.8 against
  22.2). The interpolated crossover is 160.

`kDrapeAutoRdVerts` is now **160** (it was 256, from gates/2-avbd's substep
bench). The gate checks that the threshold sits inside the measured 90 fps
bracket, and it does. Under 60 Hz vsync the crossover should be higher still
(not measured; roughly 300 if rd stays at two frames a step): the threshold is
a property of the frame rate.

## Limits, and what is not claimed

- **The sphere demo is chaotic past its self-collision onset.** Any
  comparison with native after step ~70 (frames 100 and 350, the 350-step
  losses and gradients) holds only to the spread of the flat control above.
  Only the parts before the onset are exact.
- **The recompute modes are exact only for single-colour Gauss-Seidel.**
  With colours they drift (G6). Native mode is upstream's parity oracle, not
  a gradient; it is 80–100% off against FD.
- **The in-guest L-BFGS-B is float32 vectors with double scalars.** A
  delta-stopped descent through a narrow valley can end 2e-5 from LBFGSpp's
  f (G2).
- **G9's timings are frame-driven**, and they shift with system load
  between runs: the uncapped rd floor was 16 ms per step in Task D's run and
  7 ms here. The L-BFGS-B cpu numbers at n ≤ 1e3 are within a frame and
  noisy.
- The remaining float-level difference from native, which makes the
  step-71 pair go the other way, is not located (see the trace section).

## Files

- `results.txt`, `run.log`: this run.
- `trace/`: the step-by-step comparison with native (`run.sh` and its
  logs). It needs the native output directory
  (`native/source_dir.txt`, under `C:/cloth-dynamics-standalone/output` or
  `$NATIVE_OUT`).
- `native/`: the reference run (its README).
- `oracle/`: the LBFGSpp fixtures and traces, and `inverse_min/` (its
  README).
- `kernels/`, `lbfgsb/`, `drape/`: Tasks K, B and D's own gates and
  evidence.
- The sources:
  - `guest/drape/` (the drape, the L-BFGS-B driver, the jobs, `inverse_min.h`);
  - `lean/Drape/` and `kernels/drape/` (the kernels);
  - `tests/{drape_kernels,lbfgsb_driver,lbfgsb_oracle,inverse_min_oracle,drape_host}`.
