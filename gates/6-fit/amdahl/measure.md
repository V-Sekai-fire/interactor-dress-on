# Amdahl measure: where native fit time goes (foxgirl, phase 0)

These are measurements only. The code is `fit_native`, the native twin of fit.elf: `FIT_SDF=sdfgrid`, the Lean kernel sampler, guest numerics and serial TBB. The input is foxgirl, with `--phases 1` (phase 0, the AL phase). The machine is the desk (AMD Ryzen 7 3800X, 16 logical CPUs), on 2026-09-23 between 08:12 and 08:22.

**Load during every run.** The desk loop job was running the whole time: Docker `dress-on-loop:wsl-test`, one guest thread running fit.elf. A `pixal3d-runpod:dev` container was also up. Nothing waited for either of them.

- `p0-direct` ran alone. So did the two samply runs.
- The five counter runs (`direct-counters` and the four iterative runs) ran at the same time as each other. Their wall times are therefore loaded times. For example, `direct-counters` took 43.6 s, against 35.7 s when run alone.

## Tools and builds

**samply 0.13.1.** This is the prebuilt `samply-x86_64-pc-windows-msvc.zip`. Its sha256 `8fed74da…` matches the release's `.sha256`. It was unpacked into the session scratchpad and not installed.

- **Command:** `samply record --save-only --unstable-presymbolicate -o <runs/…samply.json.gz> -- fit_native.exe --phases 1 …`, at the default 1 kHz.
- **Elevation:** samply ran xperf (ETW) and the recording finished. This shell is not elevated (`net session` fails).
- **Symbols:** the sidecar symbolicated fit_native.exe from its COFF symbol table, so neither a PDB nor a rebuild was needed.
  - Inlined code is charged to its nearest non-inlined caller.
  - Some frames have no name (`fun_…`): static functions in fit_native and internals of system DLLs and the kernel. All of them sit under named callers. Every sample in the phase window matched a bucket rule; 0 were unmatched.
- **Profiler overhead:** phase 0 took 38.72 s under samply, against 35.69 s without it (+8.5%).

**The builds**

- **`C:/b/fit-native-tbbs/fit_native.exe`:** the fit.elf twin from Gate 6, unmodified. It was used for `p0-direct` and both samply profiles.
- **`C:/b/fit-native-amdahl/fit_native.exe`:** the counters build.
  - **Source:** the main checkout at 9cb2ff134 (read only).
  - **Build:** `gates/6-fit/amdahl/build_counters.sh`, the same configure as the tbbs build.
  - **What differs:** polysolve and ipc-toolkit come from scratch copies (`C:/b/amdahl-src`, made by `make_counters_src.py`). The copies only print `[amdahl]` lines: n, nnz(H), each linear solve's `get_info` (iterations, error), the candidate counts of each `Candidates::build`, and the active constraints of each `Collisions::build`.
- **Checks:**
  - The counters build reproduces the tbbs build bitwise: 41 Newton, energy 0.00033678666696946268, no intersections.
  - For CG: the counters build and the tbbs build under samply give the same energy, 0.023054751201408683.

## 1. Phase-0 time breakdown (samply, inclusive, one bucket per sample)

**Reducer:** `reduce_samply.py`.

- **Rule:** each sample goes to the bucket of the deepest frame on its stack that matches a rule.
- **Window:** the samples under `fit::FitDriver::step`. Fractions are shares of that window.
- **"of which alloc/free":** the share of a bucket's samples whose stack holds malloc, free, new, delete, `Rtl*Heap` or `Nt*VirtualMemory`. It is an overlay, not a separate bucket.
- **Raw tables:** `runs/p0-direct.buckets.md`, `runs/p0-cg-jacobi.buckets.md`.

### Case A: direct solver (the configured `Eigen::SimplicialLDLT`), 41 Newton, `runs/p0-direct.samply.json.gz`

| bucket | s | fraction | ms / Newton | of which alloc/free |
|---|---|---|---|---|
| gradient | 0.24 | 0.6% | 5.8 | 2% |
| Hessian: SimilarityForm | 4.37 | 11.2% | 106.6 | 10% |
| Hessian: CurveTorsionForm | 0.16 | 0.4% | 3.8 | 4% |
| Hessian: ContactForm (IPC barrier) | 0.08 | 0.2% | 2.0 | 17% |
| Hessian: CurveCurvatureForm | 0.04 | 0.1% | 1.0 | 19% |
| Hessian: PointPenaltyForm (AL) | 0.02 | 0.1% | 0.6 | 4% |
| Hessian: CurveTargetForm | 0.01 | 0.0% | 0.3 | 23% |
| Hessian: FitForm (SDF) / friction | 0 | 0% | 0 | — |
| Hessian: sum of forms, P^T H P, copies | 2.52 | 6.5% | 61.5 | 19% |
| linear: analyze (AMD + symbolic) | 0.91 | 2.3% | 22.2 | 11% |
| linear: factorize (LDLT) | 4.56 | 11.7% | 111.3 | 0% |
| linear: solve | 0.13 | 0.3% | 3.3 | 1% |
| linear: other (H·Δx residual) | 0.04 | 0.1% | 0.9 | 47% |
| broad phase: CCD candidates (x0→x1) | 8.92 | 22.9% | 217.7 | 12% |
| broad phase: constraint set (dhat) | 0.22 | 0.6% | 5.4 | 11% |
| CCD / max step size (narrow phase) | 13.86 | 35.6% | 337.9 | 21% |
| line search: energy evals | 0.44 | 1.1% | 10.8 | 26% |
| line search: other | 0.04 | 0.1% | 1.0 | 7% |
| constraint-set update (Collisions::build) | 1.97 | 5.1% | 48.1 | 0% |
| energy (Newton loop) | 0.21 | 0.5% | 5.0 | 20% |
| other (Newton loop, AL outer loop, driver) | 0.17 | 0.4% | 4.3 | — |
| **phase-0 window** | **38.94** | **100%** | **950** | **13.7% (5.33 s)** |
| outside the window (inputs, `begin`, exit) | 0.77 | — | — | — |

| group | s | fraction |
|---|---|---|
| CCD / max step size | 13.86 | 35.6% |
| broad phase (both) | 9.14 | 23.5% |
| Hessian assembly (all forms + sum) | 7.21 | 18.5% |
| linear solve (analyze + factorize + solve) | 5.65 | 14.5% |
| constraint-set update | 1.97 | 5.1% |
| line search energy + other | 0.48 | 1.2% |
| gradient + energy + other | 0.62 | 1.6% |
| allocation overlay (across buckets) | 5.33 | 13.7% |

**Inside the largest buckets** (the same profile; `samply_top.py`, callee splits)

- **CCD** (13.82 s in `compute_collision_free_stepsize`):
  - edge-edge `ccd` 8.05 s and face-vertex `ccd` 3.40 s. Both are Tight-Inclusion `ticcd::CCD`.
  - the loop itself 1.27 s, shared_mutex lock and unlock 0.60 s.
- **Broad phase for CCD** (8.92 s):
  - `detect_collision_candidates` 8.09 s: edge-edge 6.65 s and face-vertex 1.64 s. `SimpleBVH::box_search_recursive` alone (self time) is 4.60 s.
  - `BVH::build` 0.79 s.
- **SimilarityForm Hessian** (4.37 s): `set_from_triplets` 1.77 s, memcpy 0.73 s, `project_to_psd` (12×12) 0.61 s, `autogen::similarity_hessian` 0.40 s, self time 0.43 s.
- **Hessian overhead** (2.52 s): the P^T H P sparse product 1.19 s and sparse sums 1.11 s.
- **Factorize** (4.56 s): `SimplicialLDLT::factorize_preordered` self time is 4.39 s.
- **Analyze** (0.91 s): the AMD ordering is 0.79 s of it.

**Check against polysolve's own timers.** These come from `runs/p0-direct.log`, the unprofiled run: 34.20 s for 41 Newton.

- **update_direction 13.50 s:**
  - Hessian assembly 8.10 s (SparseNewton 6.01 s, ProjectedNewton 2.09 s).
  - linear solve 5.39 s (4.49 s and 0.90 s).
- **line_search 20.40 s:**
  - narrow_phase_ccd 11.00 s and broad_phase_ccd 6.87 s.
  - classical_line_search 2.02 s.
  - constraint_set_update 1.84 s.
- **f 0.09 s, grad_f 0.20 s.**

### Case B: `Eigen::ConjugateGradient` + `DiagonalPreconditioner` (Jacobi), 13 Newton, `runs/p0-cg-jacobi.samply.json.gz`

| group | s | fraction | ms / Newton |
|---|---|---|---|
| Hessian assembly | 4.72 | 42.2% | 363 |
| of which SimilarityForm | 3.24 | 29.0% | 249 |
| of which sum of forms, P^T H P | 1.30 | 11.6% | 100 |
| broad phase (both) | 3.43 | 30.7% | 264 |
| linear (CG solve 2.51 s; factorize, meaning the preconditioner, 0.01 s) | 2.53 | 22.7% | 195 |
| CCD / max step size | 0.09 | 0.8% | 7 |
| constraint-set update | 0.01 | 0.1% | 1 |
| line search energy + other | 0.11 | 1.0% | 8 |
| gradient + energy + other | 0.28 | 2.5% | 21 |
| **phase-0 window** | **11.17** | **100%** | **859** |
| allocation overlay | 0.96 | 8.6% | — |

**Case C: the Stage 8 skirt was skipped.** Its phase-0 input cannot be reproduced from anything on disk.

- **The pre-fit mesh is not saved.** The pre-fit 932-vertex garment is curvenet.elf's output inside a Godot run. `gates/8-loop` keeps only its counts (`pen.json` has `mesh.vertices: 932`, no arrays) and the post-fit OBJs (`*.fitted.obj`).
- **Other inputs are made in GDScript.** The fit's setup is built in `project/stages/pipeline.gd`: the similarity from the garment skeleton to the body skeleton, and `incremental_steps` 1.
- **Dumping them means editing tracked code.** It would take changes under `project/`, which this task may not touch.

## Problem sizes (phase 0, direct; `runs/p0-direct-counters.log`)

| quantity | value |
|---|---|
| garment / avatar | 2682 v, 5220 f / 5128 v, 10171 f |
| unknowns n (full-size AL problem) | 8047 = 2682 × 3 + 1 (GarmentNLProblem `full_size` = `reduced_size` + 1) |
| nnz(H) per Newton | 295,417 – 295,765 (median 295,669, ~36.7 per row); 42 solves (36 SparseNewton + 6 ProjectedNewton) |
| CCD candidate builds (`Candidates::build` x0→x1) | 44 in 41 Newton |
| CCD candidates per build (ee + fv; vv = ev = 0) | min 3,111 / median 55,116 / max 3,304,228 / mean 368,291; total 16.2 M |
| the 5 largest builds (builds 3–7 of 44: the early Newton steps, ‖Δx‖ up to 522) | 66% of all CCD candidates |
| CCD candidates per build, series | 3111, 3732, 3304228, 2402958, 1686325, 2314123, 1066989, 786927, 361331, 470030, 167112, 69209, 96683, 116785, 36317, 47856, 17309, 156538, 105038, 40628, 144760, 57213, 32419, 86467, 937273, 53019, 28974, 18569, 41647, 199346, 330358, 620694, 19369, 18321, 22547, 15350, 16422, 13672, 42435, 134432, 41567, 23839, 47276, 5598 |
| constraint-set candidates (static, dhat) | 1 build: 3,111 (ee 2,522, fv 589) |
| active constraints per `Collisions::build` | 72 builds: min 0 / median 74 / max 153 / mean 73 (ev 15, ee 45, fv 12, vv 0 on average) |
| SDF (FitForm) queries in phase 0 | 0 (sdf grid: bricks 0, distance queries 0) |
| peak working set | 174.4 MB |

## 2. Iterative-solver trial (phase 0, 3-minute cap; `runs/p0-<solver>-jacobi.log`)

**Settings:** spec `solver.linear.solver` = the solver and `solver.linear.precond` = `Eigen::DiagonalPreconditioner`. polysolve's defaults were kept: `max_iter` 1000 and `tolerance` 1e-12. Newton's residual check (1e-5) is unchanged.

**AMGCL was not run:** the build has `POLYSOLVE_WITH_AMGCL=OFF`, so it is not compiled.

**How to read the table:**

- **Time:** the phase-0 wall time from the concurrent counter runs. The samply run in brackets ran alone.
- **"rejected":** the number of solves Newton threw out because the residual was above 1e-5. After each rejection polysolve falls back: Newton → ProjectedNewton → RegularizedNewton (reg 1, 1e4, …) → GradientDescent.

| linear solver | exit | Newton | phase-0 wall s | linear solves (rejected) | linear iterations per solve: min / median / max, mean; solves at the 1000 cap | final energy (× direct) | ‖∇f‖ | intersections |
|---|---|---|---|---|---|---|---|---|
| SimplicialLDLT (direct, baseline) | ok | 41 | 35.69 alone; 43.59 concurrent | 42 (0) | direct | 0.00033678666696946268 (1×) | 0.310932 | none |
| ConjugateGradient + Jacobi | ok, 15.0 s | 13 | 14.29 (11.09 alone) | 20 (7) | Newton 3× 1000; Projected 3× 1000; Reg(1) 9: 235 / 616 / 1000; Reg(1e4) 5: 17 / 32 / 68. All: mean 602, 7 at the cap, 12,045 total | 0.023054751201408683 (68.5×) | 0.629955 | none |
| BiCGSTAB + Jacobi | ok, 19.6 s | 13 | 18.81 | 20 (7) | Newton 3× 1000; Projected 3× 1000; Reg(1) 9: 185 / 842 / 1000; Reg(1e4) 5: 10 / 24 / 47. All: mean 629, 8 at the cap, 12,586 total | 0.023054751202681859 (68.5×) | 0.629955 | none |
| MINRES + Jacobi | ok, 18.1 s | 13 | 17.18 | 23 (11) | Newton 3× 1000; Projected 3× 1000; Reg(1) 9: 228 / 763 / 1000; Reg(1e4) 6: 17 / 37.5 / 1000; Reg(1e8) and Reg(1e12) residual NaN → GradientDescent. All: mean 671, 11 at the cap | 0.026497182568297358 (78.7×) | 0.217557 | none |
| GMRES + Jacobi | ok, 28.2 s | 22 | 27.32 | 34 (12) | Newton 4× 1000; Projected 4× 1000; Reg(1) 7: 355 / 1000 / 1000; Reg(1e4) 19: 11 / 16 / 71. All: mean 434, 14 at the cap | 0.11722900497194011 (348×) | 0.780796 | none |

**What the iterative runs share**

- **Stopping point:** every run stopped phase 0 the same way the direct run did. The AL minimize ended with "Gradient vector norm too small" (the AL inner tolerance is ‖∇f‖ < 1), after one minimize. The final energy is 68–348× the direct run's.
- **Unregularised Hessian:** no iterative solver reached the 1e-12 tolerance on it. Every SparseNewton and ProjectedNewton solve ran the full 1000 iterations, and Newton rejected every one of those directions.
- **Candidates:** the CCD candidates stayed small, at most 8,618 for CG against 3.3 M for direct. So narrow-phase CCD fell from 11.0 s to 0.11 s (polysolve timer, CG). Broad-phase CCD fell from 6.87 s to 2.82 s. That is 156 ms per build for direct (44 builds) and 176 ms for CG (16 builds), so the cost per build barely changed.

**Linear solve time** (polysolve `linear_solve` timer; it includes the preconditioner and CG setup)

| solver | linear solve time | total linear iterations |
|---|---|---|
| CG | 3.18 s | 12,045 |
| BiCGSTAB | 7.34 s | 12,586 |
| direct LDLT | 5.39 s | 42 solves |

## Files

**Logs** (`gates/6-fit/amdahl/runs/`)

- `p0-direct.log`: unprofiled baseline.
- `p0-direct-samply.log` and `p0-direct.samply.json.gz` with `.json.syms.json`: the direct profile.
- `p0-direct-counters.log`: counters, and the bitwise check.
- `p0-cg-jacobi.log`, `p0-bicgstab-jacobi.log`, `p0-minres-jacobi.log`, `p0-gmres-jacobi.log`: the iterative runs.
- `p0-cg-jacobi-samply.log` and `p0-cg-jacobi.samply.json.gz` with `.json.syms.json`: the CG profile.
- `p0-*.buckets.md`: reducer output. `summary.txt`: `parse_runs.py` output.

**Scripts** (`gates/6-fit/amdahl/`)

- `samply_top.py`: top functions of a profile.
- `reduce_samply.py`: the buckets.
- `parse_runs.py`: the log summaries.
- `run_p0.sh`: one capped phase-0 run.
- `make_counters_src.py` and `build_counters.sh`: the counters build.
