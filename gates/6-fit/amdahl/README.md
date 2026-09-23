# Gate 6 Amdahl: how much of fit.elf the GPU can take

**Result (2026-09-23):**

- **Ceiling.** If every parallel part of the loop's fit ran on the GPU at no cost, the fit would be at most **6.6–7.7×** faster. The linear solve and the Newton/line-search control stay serial, and they are 13–15% of the guest's time.
- **Modelled.** With per-round-trip latency (rule 4), dispatch cost and the CPU work left behind included, the model gives **5.1–6.0×**. It gives **2.8–3.0×** if narrow-phase CCD stays on the CPU. Moving CCD means porting Tight Inclusion to fp64, the largest port of the lot.
- **The first lever is a solver setting, not the GPU.** `force_psd_projection` is already measured in the guest: the loop's fit goes from **1393 s to 613 s (2.27×)** and stays inside the quality guard. That beats the first GPU stage (broad phase alone, 1.6×) and nearly matches the first two together (2.5×).
- **ggml-rd.** No part of the fit belongs in ggml-rd. The GPU parts are Lean→Slang kernels on `rd_compute`. From ggml-rd the fit takes only the pump and buffer plumbing.

## Where ggml-rd is

- **Branch.** Only on **`cut-3`**, at `f1a4babbb` ("Gate 3 G3.graph on host oracles…"), both locally and as `origin/cut-3`. It is **not merged** into `main` or `gate-0h`: it has 32 commits that `gate-0h` lacks, and `merge-cut-3` still points at `e36a152f7`.
- **Worktree.** `C:/Users/ernest.lee/AppData/Local/Temp/claude/C--interactor-dress-on/5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5/scratchpad/ido-3`.
- **Code.**
  - `guest/ggml-rd/`: `ggml-rd.cpp`, `rd_graph.cpp`, `rd_pack.cpp`, `rd_kernels.cpp`, and `ops/{binary,concat,conv3d,cpy,flash_attn_ext,get_rows,im2col,mul_mat,norm,repeat,rope,soft_max,unary}.cpp`.
  - Also `guest/pump/` and `guest/ggml_test/`. Neither exists on `gate-0h`; `guest/fiber/` does.
  - `lean/Ggml`, `kernels/ggml` and `gates/3-ggml-rd`.
  - The per-kernel sub-branches `cut-3-k1k5` through `cut-3-k8`.
- **What it accepts.** Tensor types F32, F16, BF16 and I32 only. There is no F64, no sparse op, no scatter-add, no sort, no eigensolver and no factorisation (the map's grep of `ops/*.cpp`).

## Question

- What share of fit.elf's guest time could run on the GPU through `RenderingDevice`?
- What would the whole fit gain once round trips, dispatch cost and fp64 are counted?
- Which parts, if any, belong in ggml-rd rather than Lean kernels on `rd_compute`, or on the CPU?
- What wins are there that do not need the GPU?

## How

**1. Four measured profiles.** Each part of a profile is one bucket per sample or per timer, so the parts add up to 100%.

| profile | what | source |
|---|---|---|
| **loop p0 (new)** | The loop's 932-vertex skirt, phase 0 (the AL phase), default config, native, samply. `fit_native` is the fit.elf twin (`C:/b/fit-native-tbbs`). | `runs/loop-p0-direct.*` (this README) |
| loop p1 | The same skirt, phase 1, **in the guest**, with instruction-counted timers. This is the `force_psd_projection` arm (cut-6c `fit_prof.elf`). | the map; `cut-6c:gates/6-fit/budget/guest-psd-prof.log` |
| foxgirl p0 | foxgirl skirt (2682 vertices), phase 0, native, samply | `measure.md`, case A |
| loop p0, psd | The loop skirt, phase 0, psd setting, native polysolve timers | `C:/b/budget/solo-psd/run.log` |

measure.md skipped the loop skirt ("case C") because it thought the pre-fit mesh was lost. It is on disk in the parked budget study: `cut-6c:gates/6-fit/budget/pen-0.03.mesh.obj` (932 vertices) and `loop_setup.json`. `run_loop_p0.sh` runs it.

The new run reproduces the budget study's native baseline bit for bit: 100 Newton, energy `0.13549319149746247` (`C:/b/budget/solo-base/run.log`). So does the counters build.

**2. Time base.** Everything is in guest seconds.

- **Phase split.** Gate 8's loop fit spends 1013 s in phase 0 (144 Newton) and 380 s in phase 1 (109 Newton). Phase 0 is therefore **72.7%** of the 1393 s (`gates/8-loop/README.md`).
- **Blend.** The loop-wide shares weight loop p0 by 72.7% and loop p1 by 27.3%.

**3. Cost model** (`amdahl_model.py`, output in `runs/amdahl_model.txt`). Per Newton iteration, an offloaded part *i* with guest time T_i becomes

> r_i·T_i + k_i·(t_rt + g)

and the fit's speedup is S = T_before / T_after, summed over both phases and all Newton iterations. This is Amdahl's law with an effective s_i = T_i / (r_i·T_i + k_i·(t_rt + g)).

| term | meaning | value |
|---|---|---|
| r_i | share of the part that stays on the CPU | per part, in the cost table below |
| k_i | round trips per Newton | per part, in the cost table below |
| t_rt | one pump round trip (rule 4: at least one pump tick) | two cases, below |
| g | GPU execution and transfer allowance per offload | **2 ms** |

**Round-trip cases (t_rt):**

- **Fast, 0.6 ms.** The loop's main thread ran **1,674 frames/s** during the guest fit (`guest-psd-prof.txt`: 1,025,338 frames in 612.6 s).
- **60 Hz, 16.7 ms.** VR measured about 66 frames/s in Gate 8 (about 15 ms).

**What the 2 ms allowance (g) covers:**

- **Dispatches.** About 8 µs each: 7.97–8.41 µs (`gates/3-ggml-rd` G3.cost) and 6–12 µs from the guest (`gates/1-rd-compute`).
- **Readback of 1 MB or less.** Uploads run at 1.9–3.4 GB/s (`gates/0f-runtime` probe 12). Readback speed is not measured.
- **Compute of 1 ms or less.** The map's fp64 estimate for 5k–20k 12×12 eigensolves on a 4090.

## Results

### 1. Where the time goes (% of each phase)

| part | loop p0, native (new) | loop p1, guest (map) | foxgirl p0, native (measure.md) | **loop blend, guest** |
|---|---|---|---|---|
| broad phase (`Candidates::build`) | **43.2** | 29.9 | 23.5 | **39.5** |
| narrow-phase CCD / max step size | 20.9 | 6.7 | **35.6** | 17.0 |
| Hessian assembly | 18.3 | **31.7** | 18.5 | 22.0 |
| linear solve (AMD + LDLT) | 11.5 | 9.8 | 14.5 | 11.0 |
| constraint set / trial updates | 2.3 | 11.9 | 5.1 | 4.9 |
| SDF brick fill (once per phase) | 0 | 5.6 | 0 | 1.5 |
| f, ∇f, line-search energies, driver | 3.9 | 4.4 | 2.8 | 4.0 |

Loop p1 is the map's timers renormalised to the sum of the disjoint timers, 3.748e9 instructions per Newton. The map's 3.82e9 double-counts the line-search sub-timers.

**Loop p0 in detail** (`runs/loop-p0-direct.buckets.md`, `runs/loop-p0-direct-counters.summary.txt`):

- **Time.** 31.10 s over 100 Newton, 311 ms per Newton natively.
- **Newton caps.** Both AL minimizes stop at their **50-iteration cap** ("Reached iteration limit (limit=50)").
- **Broad phase:**
  - `SimpleBVH::box_search_recursive` self time alone is 22.1% of the phase.
  - The avatar–avatar rejection filter costs 6.4%. That is `can_collide` at `guest/fit/fit_driver.cpp:346-351`, its `std::function` wrapper and ipc's `can_*_collide`.
- **Hessian.** SimilarityForm 10.0%; sums of forms, PᵀHP and copies 6.1%. ContactForm is 0.6% of the phase, 3.2% of assembly.
- **Linear solve.** Factorize 8.7%, analyze 2.3%.
- **Allocation.** 12.7% of samples have an allocator frame (an overlay across buckets).

**Problem sizes (loop p0):**

| quantity | value |
|---|---|
| unknowns n | 2797 |
| nnz(H) | 106,639–107,563 |
| linear solves | 102 (91 SparseNewton, 11 ProjectedNewton) |
| CCD candidates per build | 104 builds: median 40,705, mean 72,100, max 527,210; 7.5 M in total |
| active collisions per build | median 97, max 124 |
| SDF queries | 0 |

The shape depends on the phase and the garment:

- **Loop skirt, phase 0:** broad-phase bound.
- **Foxgirl, phase 0:** CCD bound. Its five largest builds, from long early steps, hold 66% of 16.2 M candidates (measure.md).
- **Loop skirt, phase 1:** assembly and broad phase lead, and the trial updates grow to 12% because FitForm resamples the SDF on every trial.

### 2. Guest against native

| | guest s / Newton | native s / Newton | ratio | sources |
|---|---|---|---|---|
| loop phase 0, default | 7.03 (1013/144) | 0.333 (33.29/100) | **21×** | Gate 8; `solo-base` |
| loop phase 1, default | 3.49 (380/109) | 0.295 (46.96/159) | **11.8×** | Gate 8; `solo-base` |
| loop phase 1, psd | 3.45 | 0.295 | 11.7× | `guest-psd-prof.log`; `solo-psd` |

- **Trajectories differ.** Guest and native do not take the same path (144 against 100 Newton and 109 against 159), so these are per-Newton ratios.
- **Phase 0 agrees with Gate 6** (20–25×, foxgirl phase 0).
- **Per component (phase 1, psd; guest instructions at 0.893 ns each against `solo-psd`):**

  | component | guest / native |
  |---|---|
  | narrow CCD | 15.6× |
  | broad phase | 13.6× |
  | assembly | 10.5× |
  | linear | 10.1× |
  | constraint set | 9.0× |

  The broad phase and CCD are relatively dearer in the guest.
- **Weighting.** Weighting loop p0 by these ratios moves the loop blend to 41.4% broad phase, 20.3% CCD, 19.5% assembly and 9.2% linear. Every ceiling below is given for both the unweighted and the weighted blend.

### 3. What each part costs on the GPU (model inputs and the effective s_i)

| part | GPU home | precision | round trips / Newton | CPU residue | effective s (fast / 60 Hz) |
|---|---|---|---|---|---|
| broad phase | Lean kernel on `rd_compute`: garment boxes × all boxes, count → scan → write, pairs in index order, no atomics | fp32, boxes rounded outward (a superset is safe) | 1 | 1%: list → `ipc::Candidates`; mean 0.58 MB, max 4.2 MB at 8 B/pair | 80–92 / 36–62 |
| assembly (non-contact forms) | Lean kernels: `similarity_hessian`, curve forms, a 12×12 Jacobi PSD, fixed-slot CSR gather | fp32 usable (changes the trajectory); fp64 if the gate passes | 1 | contact Hessian on the CPU in fp64 (3.2% of assembly) + 1%; readback 107k values, 0.86 MB fp64 | 22–23 / 17–18 |
| trial updates (collision set, SDF resample) | Lean kernel per candidate; the existing `lean/Fit/SdfSplineHessian.lean` | **fp64 required** (d² is 1e-7 to 4e-6) | 1.03 (loop p0 counters) / 1.99 (map) | 5%: accept decisions stay on the CPU | 15–16 / 6–7 |
| SDF brick fill | Lean kernel dispatched at `fit_begin`, overlapping phase 0 | fp32 (the winding number is already float) | 0 | 0 | off the critical path |
| narrow CCD | Lean port of Tight Inclusion, fused into the broad phase's list | **fp64 required** | 0 extra | fixed `tmax` per pass | **10 assumed**, not measured |
| linear solve, control | CPU | fp64 | none | all | 1 |

**Latency is not the constraint.**

- A Newton iteration is 3.5–7 s of guest time.
- At 60 Hz, the 4–5 round trips per Newton cost 75–95 ms.
- Across the whole fit, the fast pump and a 60 Hz pump differ by at most 6% (next table).

**Only the trial updates feel the frame.** They are small (2–12%) and pay one trip per trial, so at 60 Hz their s falls to 6–7.

### 4. The ceiling (loop's default config, whole fit, guest)

| moved to the GPU | zero cost | model, fast pump | model, 60 Hz | fit time |
|---|---|---|---|---|
| broad phase | 1.65× (1.71×) | 1.64× | 1.63× | 848–852 s |
| + assembly | 2.60× (2.56×) | 2.50× | 2.47× | 556–564 s |
| + trial updates (fp64) + SDF fill | 3.12× (3.01×) | 2.96× | 2.87× | 471–485 s |
| + narrow CCD (fp64, s = 10) | **6.64× (7.73×)** | **5.40× (6.02×)** | **5.12× (5.68×)** | 231–272 s |

- **Notation.** Numbers in brackets use the ratio-weighted blend.
- **Serial remainder.** Linear solve plus control is 15.0% unweighted and 12.7% weighted.
- **Sensitivity to the CCD speedup.** Everything moved at 60 Hz gives 4.25× at s_ccd = 3, 5.12× at s_ccd = 10 and 5.43× at s_ccd = 30.
- **Answer to (1).** About 7× with everything free, about 5–6× modelled, and about 3× if CCD stays on the CPU.

### 5. Staged plan

Stages are cumulative. They are computed on the psd configuration's profile, because stage A changes it: phase 0 is loop p0-psd (native) and phase 1 is loop p1 (guest); 268 s + 345 s in the guest.

| stage | what | basis | on the psd config | **vs today's 1393 s** | fit |
|---|---|---|---|---|---|
| A | `force_psd_projection` in both solves | **measured, guest** | 1× | **2.27×** | 613 s |
| B | CPU fixes (below): assembly −50%, broad phase −60%, linear −10% | estimated from the samply shares | 1.60× | 3.63× | 383 s |
| C | GPU broad phase + assembly (no fp64 needed) | model | 2.82–2.88× | 6.40–6.55× | 213–218 s |
| D | fp64 gate passes: trial updates on the GPU, SDF fill at `fit_begin` | model | 3.83–4.05× | 8.70–9.21× | 151–160 s |
| E | narrow CCD in fp64 (s = 10 assumed) | model | 5.75–6.28× | 13.1–14.3× | 98–106 s |
| floor | every parallel part free after B | bound | 7.14× | 16.2× | 86 s |

**If B is skipped,** the GPU stages on the psd config give 1.55× (broad phase), 2.71–2.77× (+ assembly), 3.64–3.84× (+ trial updates), 5.33–5.78× (+ CCD), and 7.11× at zero cost.

**Effort and risk:**

**A. The setting (small).**
- *Work:* the setting is parked on `cut-6c` (`pipeline.gd`) and needs the verify pass.
- *Quality, measured here* (`runs/guest_psd_gap.txt`, with cut-6c's `eval_gap.py`):
  - energy 0.0112336 against 0.0112073 (+0.23%);
  - gap mean 1.834 against 1.840 voxels (−0.005);
  - p95 5.357 against 5.215 voxels (+0.14);
  - guard **PASS**; CHECK: no intersections.
- *Risk:* the guest's phase 0 under psd ran 50 Newton, to its cap, where native converged in 28 (`guest-psd-prof.txt`; `solo-psd`).

**B. CPU fixes (medium).** The changes go in the org forks of ipc-toolkit, polysolve and cloth-fit. Each is measurable natively in about a minute and in the guest in about 10 minutes.
- *Risk:* the removal shares are guesses anchored on section 1, not measurements.
- *Risk:* the guest trajectory moves, so `trace_diff.py` and the guard must be rerun.

**C. GPU broad phase and assembly (large).**
- *Kernels, as Lean expressions (rule 2):*
  - `similarity_hessian` and ipc-toolkit's autogen distance Hessians;
  - a 12×12 Jacobi eigensolver;
  - a fixed-slot CSR gather, extending `lean/Cloth/Avbd/AdjacencyKwise` and `VbdGather*` from diagonal blocks to block pairs;
  - the broad-phase count, scan and write.
- *Plumbing:* fit.elf becomes a guest fiber that yields once per Newton. Today `fit_step` is one blocking vmcall per phase on a GDScript worker thread. The yields are pumped on that thread through its own local RenderingDevice (see the gates below).
- *Risks:* the GPU's fused multiply-adds move the trajectory; the fp32 box superset adds narrow-CCD work on the CPU.

**D. Trial updates and SDF fill on the GPU (medium).**
- *Blocked on* fp64 on RenderingDevice, which is unverified on the 4090, the 3090, dzn and RunPod.
- *Risk:* a brute-force winding number differs from `igl::fast_winding_number` near the sign boundary; Gate 6a's sign check covers that.

**E. Narrow CCD on the GPU (largest).**
- *Work:* an interval branch-and-bound in fp64.
- *When:* only if the loop's phase 0 stays CCD- and broad-phase-heavy after A and B.
- *Risk:* the CPU result depends on candidate order through the shrinking `tmax`; the GPU version needs a fixed `tmax` per pass.

**Rule 5:** each stage keeps its cpp twin, the same Lean source built with `slangc -target cpp`, and uses a measured garment-size threshold for choosing CPU or GPU.

### 6. Can more code go to ggml-rd? No.

| part | ggml-rd | Lean on `rd_compute` | CPU |
|---|---|---|---|
| broad phase | no: variable-length pair output; no scan/compaction op | **yes** | fallback (cpp twin) |
| Hessian assembly | no: per-element gathers, 12×12 PSD, CSR scatter, no F64 | **yes** | contact Hessian (fp64, 0.6%) |
| trial updates, SDF resample | no: fp64, per-candidate branching | **yes**, after the fp64 gate | fallback |
| SDF brick fill | no: winding number and point-triangle distance are not ggml ops | **yes** (fp32) | or a second Sandbox (section 7) |
| narrow CCD | no | later (fp64) | **today** |
| sparse LDLT | no: ggml has no factorisation; a GPU sparse direct solve at n = 2797 is latency-bound | no | **yes** |
| line-search and Newton control | no | no | **yes** |

**Why none of it is a ggml op.**
- The fit's GPU-shaped work is irregular gather/scatter, enumeration and branch-and-bound; none of it is `mul_mat`, `soft_max` or `rope`.
- The only dense pieces are 9×9 and 12×12 blocks. ggml-rd's host cost of 4.98–5.43 µs per graph node (G3.cost) would exceed their work.
- Wrapping Lean kernels as ggml custom ops would add a graph and gallocr for nothing.

**What to take from ggml-rd (cut-3):**
- the fiber pump protocol (WAIT_GPU / COOP / UPLOAD / READ, `guest/pump`);
- `rdc::Device::buffer_get_into` (≤ 16 MiB staging);
- caching of permanent-RID pipelines and uniform sets;
- barrier elision.

Its pump runs from the main thread's `_process`. The fit needs a worker-thread variant, because its CPU segments are about 1 s of guest time each. Sharing the plumbing means merging `cut-3` first.

### 7. Wins outside the GPU

Amdahl leaves 13–15% serial, so the Newton count and the per-Newton serial work set the floor. All native times here are from cut-6c's budget runs on the loop skirt (`runs/budget_table.txt`). The quality guard is the Gate 6 budget guard against the loop's guest fit: energy within 10%, gap mean within ±0.25 voxel, p95 within ±0.5 voxel.

**Measured:**

| change | fit time | Newton (AL / reduced) | quality | verdict |
|---|---|---|---|---|
| **psd** (guest) | 1393 → **613 s, 2.27×** | 144/109 → 50/100 | guard PASS (section 5, A) | **take it (stage A)** |
| psd (native) | 80.8 → 41.3 s, 1.96× | 100/159 → 28/103 | PASS against the guest reference | — |
| psd + `grad_norm` 0.03 (native only) | 33.3 s, **2.43×** | 28/71 | energy +1.6%; gap 1.857/5.600; PASS against both references | next to try in the guest |
| coarser mesh, 585 vertices (`e040`) | 40.4 s, 2.0× | — | gap 2.20/6.86 voxels | **FAIL** |
| coarser mesh, 417 vertices (`e050`) | 15.8 s, 5.1× | — | gap 2.46/7.85 voxels | **FAIL** |
| `grad_norm` 0.03 without psd | 52.4 s | — | energy +66% | FAIL |
| AL cap 25 | 63.7 s, 1.27× | — | p95 6.25 voxels | FAIL |
| AL cap 30 in the guest | — | 5 minimizes | the reduced solve throws | FAIL (Gate 8) |
| reduced cap 100 | 58.6 s, 1.38× | — | energy +14%, ‖∇f‖ 0.62 | FAIL |
| line-search limiter 1 | 79.9 s, 1.01× | — | — | no gain |
| hash-grid broad phase | 72.3 s, 1.12× | — | — | with psd, no gain (43.6 against 41.7 cpu s) |
| iterative solvers, Jacobi preconditioner (foxgirl p0, measure.md) | — | — | energy 68–348× the direct solve's; every unregularised solve hit the 1000-iteration cap and was rejected | **negative**: GPU PCG over the existing `lean/Cloth` CG kernels is not a drop-in |

**Why psd works.** The Newton count is the lever: fit time = Newton count × time per Newton.

- In the default config, phase 0 is bound by its cap, not by convergence. Both native AL minimizes stop at 50 iterations, and the guest needs 3 minimizes and 144 Newton.
- With psd, phase 0 converges natively in 28 Newton.

**Not measured:**

- **Garment-only broad-phase queries,** with the avatar's BVH built once per phase (the avatar is static in phase 1 and affine in α in phase 0).
  - Why: 85% of the query boxes are avatar primitives (map).
  - What it removes: the 6.4% rejection filter outright, plus most of the 22.1% box-search self time.
- **Fixed-pattern, in-place assembly.** It drops triplets, `set_from_triplets`, the sparse sums and PᵀHP.
  - In foxgirl's profile these are 5.0 of 7.2 s of assembly (measure.md). In loop p0 the sums alone are 6.1% of the phase.
  - It is also the prerequisite for stage C.
- **Reusing `analyze_pattern`** when the sparsity pattern is unchanged: at most 2.3% of the phase.
- **Reusing allocations** across Newton iterations. 12.7% of loop p0 samples are under an allocator frame, and guest malloc/free are host syscalls (AGENTS).
- **Canonical candidate order.** Not a speedup, but it tests Gate 6's tie hypothesis.
- **SDF brick fill in a second Sandbox** on another host thread during phase 0, or cached per avatar across fits.
  - Why: the fill depends only on the avatar.
  - Size: 20.9e9 instructions, about 19 s (map); 1.5% of the loop fit, 3% under psd.
  - Constraint: guest threads are serialised (Gate 0C), so a second Sandbox is the only CPU parallelism available.
- **The linear solve itself:** 9.8–11.5% of the time, so even a free solver gives at most 1.12×. A supernodal factorisation (CHOLMOD) is not in the org (rule 1: ask before forking).
- **Warm start of a re-fit.** Phase 0 grows the avatar from its shrunken copy (`fit_driver.cpp` `run_phase`: `next_avatar_v = (nc − skinny)·α + skinny`). Skipping it requires the previous fit to be intersection-free on the full body, plus a vertex map across curvenet remeshes.
- **The interpreter.** Every part runs 9–21× slower in the guest (section 2), so a faster execution mode would multiply everything, the serial part included.
  - libriscv has a binary-translation mode. Not checked: whether the org's godot-sandbox build enables it without a host DLL (which AGENTS bans).

## Negative results

- The **iterative linear solvers** fail on phase 0 (measure.md).
- A **coarser garment** fails the gap guard at 585 and 417 vertices.
- **Caps, the step limiter and the hash-grid broad phase** gain at most 1.38× and fail or do nothing.
- **Guest threads** give no parallelism (Gate 0C).
- **ggml-rd** has no op that fits (section 6).
- **CCD is not a small, stay-on-the-CPU part everywhere.**
  - It is 7% in loop p1, which is how the map found it.
  - It is 20.9% in loop p0 and 35.6% in foxgirl p0. Because phase 0 is 73% of the loop, leaving CCD on the CPU caps the GPU plan at about 3× on today's config.

## Answer

1. **Ceiling:**
   - 6.6–7.7× if all parallel work is free;
   - 5.1–6.0× modelled with rule-4 round trips (0.6–16.7 ms each), dispatch cost (about 8 µs each), readbacks and CPU leftovers;
   - 2.8–3.0× if narrow CCD stays on the CPU.

   The serial remainder (LDLT + control, 13–15%) and the fp64 parts (trial updates, CCD) bound it, not frame latency.
2. **Plan:**
   - A, psd: measured at 2.27× in the guest.
   - B, CPU fixes: estimated 3.6× cumulative.
   - C, GPU broad phase + assembly, fp32-capable Lean kernels: about 6.4×.
   - D, after the fp64 gate, trial updates + SDF fill: about 9×.
   - E, fp64 CCD, only if phase 0 still needs it: about 13–14×.
   - Floor: about 16×, about 86 s.

   Only A is measured. B–E are this model's estimates.
3. **ggml-rd:** nothing moves into it. The GPU parts are Lean→Slang kernels on `rd_compute`; ggml-rd lends its pump and buffer plumbing, which needs `cut-3` merged.
4. **Outside the GPU:** the best measured lever is fewer Newton iterations. psd gives 2.27× in the guest, and psd + `grad_norm` 0.03 gives 2.43× natively. After that come the CPU fixes to the broad phase and assembly, and a faster interpreter, which would speed up every part, the serial remainder included.

## Gates before building (rule 7)

1. **Stage A in the loop.** Ship psd through Gate 8, flat and VR, with the guard. Run psd + `grad_norm` 0.03 in the guest.
2. **Loop phase 0 per component in the guest.** Run `fit_prof.elf` on the default and psd configs. polysolve prints no timing when a minimize hits its cap, which both loop p0 minimizes do, so this needs a timing print on the cap exit in the fork, or `rdinstret` brackets.
3. **fp64 on RenderingDevice.** One double dispatch with a float control, on the 4090, the 3090, dzn (the desk loop worker) and RunPod.
   - Today `kernels/fit/gen.sh` validates SPIR-V Float64, but nothing dispatches it.
   - Only stages D and E depend on this gate.
4. **Worker-thread pump.** A local RenderingDevice and the fiber pump for fit.elf on its worker thread; measure frames per round trip in flat and XR.
5. **CPU fixes, one at a time.** Measure each natively, then in the guest, with `trace_diff.py` and the guard.
6. **Per-kernel rd-vs-cpp thresholds** by garment size (rule 5).

## Files

**Added by this README** (`gates/6-fit/amdahl/`):

- `run_loop_p0.sh`: runs the loop skirt's phase 0 natively; `samply` or `counters` mode; `LOOP_ROOT` is a cut-6c checkout.
- `runs/loop-p0-direct.samply.json.gz` and `.samply.json.syms.json`, `runs/loop-p0-direct-samply.log`, `runs/loop-p0-direct.buckets.md`: the loop p0 profile and its buckets (`reduce_samply.py --newton 100`).
- `runs/loop-p0-direct-counters.log` and `runs/loop-p0-direct-counters.summary.txt` (`counters_summary.py`): n, nnz, candidates, collisions.
- `amdahl_model.py` and `runs/amdahl_model.txt`: every S in sections 4 and 5.
- `runs/budget_table.txt`: cut-6c `table.py` over `C:/b/budget/*`.
- `runs/guest_psd_gap.txt`: cut-6c `eval_gap.py` on the guest psd fit.

**Already here:** `measure.md` and its runs (foxgirl p0, the iterative-solver trial).

**Cited, read only:**
- `gates/8-loop/README.md` (fit budget table, frames);
- `gates/6-fit/README.md` (6.P, ldlt8k);
- `gates/0c-threads`, `gates/0f-runtime`, `gates/1-rd-compute`, `gates/2-avbd`;
- `cut-3:gates/3-ggml-rd/README.md`;
- `cut-6c:gates/6-fit/budget/guest-psd-prof.{log,txt}`;
- `C:/b/budget/{solo-base,solo-psd}/run.log`;
- the map report: the Amdahl map returned by the workflow's map agent (not saved to the repo), whose evidence is `guest-psd-prof.log`, `C:/b/budget/psd/run.log` and `gates/6-fit/results.txt`.

## Decision since this was written (user, 2026-09-23)

Stages D and E are modelled here in fp64. The decision is float-float instead: the forms, trial updates and PCG in df32 Lean kernels, and narrow-phase CCD as **additive CCD in df32** with its conservative margins recomputed for df32 error, escalating to qf32 (4 x f32) only where df32 margins are too loose. Gate G6g.0 (gates/6g-polyfem-gpu/g0-df32) tests that before any kernel is written.


## Verifier notes

I did not append the Verifier notes: `gates/6-fit/amdahl/README.md` does not exist on disk (the README says it was returned as text and not saved). I left it absent rather than save someone else's draft. The finished `## Verifier notes` section is at `C:\Users\ernest.lee\AppData\Local\Temp\claude\C--interactor-dress-on\5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5\scratchpad\amdahl-verifier-notes.md`, ready to append. `map.md` does not exist either. The map-only inputs cannot be checked: 1.99 trials per Newton, d² 1e-7..4e-6, 85% avatar boxes, and the fp64 eigensolve estimate. The P1 timers were checked directly against `guest-psd-prof.log`. Nothing tracked was modified.

**These match their sources:**
- 1393/1013/380 s, 144/109 Newton, 72.7%.
- Every S and fit time in sections 4 and 5. I reran `amdahl_model.py` and its output is byte-identical to `runs/amdahl_model.txt`.
- The section 1 columns, which each sum to 100 ± 0.1.
- The effective-s column and the per-component ratios (per Newton, at 0.893 ns per instruction).
- Every budget-table row, the guest psd gap and energy, and the counters.
- 1,674 frames/s.
- cut-3 at f1a4babbb, 32 commits ahead of gate-0h, in neither main nor gate-0h, and the worktree path.
- G3.cost 4.98–5.43 µs per node and 7.97–8.41 µs per dispatch; Gate 1's 6–12 µs; 0F's 1.9–3.4 GB/s; Gate 0C.
- `fit_driver.cpp:346-351` and the `run_phase` formula.

**Findings, most severe first:**

1. **The guest phase-0 composition is not measured, and the model does not reproduce it.**
   - Native loop p0 × the phase-1 ratios predicts 4.28 guest s per Newton. Gate 8 measured 7.03 s (1013/144), so the model explains 61%.
   - Guest time per instruction differs by phase: 1.237 ns in p0 against 0.934 ns in p1 (Gate 8), and 1.116 against 0.893 ns (psd run). P1's instruction shares are therefore not time shares.
   - Gate 6's wall-timed ldlt8k is 25.5×, against the 10.1× linear ratio derived here from instructions.
   - So "everything is in guest seconds" is false for 73% of the fit. The 6.6–7.7× and 5.1–6.0× ranges rest on a native proxy, with gate 2 as their precondition.

2. **The "serial remainder" is a design choice, not an Amdahl bound.**
   - "A GPU sparse direct solve at n = 2797 is latency-bound" has no numbers behind it.
   - The guest LDLT costs 0.33 s per Newton in p1 psd and about 0.81 s in p0.
   - A dense fp64 factorisation at n = 2797 (62 MB, ~7.3 GFLOP) plus one round trip is never evaluated. It would need the fp64 gate and a Lean kernel.
   - "rest" (f, ∇f, line-search energies) is element-parallel work.
   - **Correction:** "at most 6.6–7.7× with LDLT and energy evaluations kept in the guest".

3. **`merge-cut-3` is stale.** It is at b37e8ee41 (08:42:43), on top of c92ca9a86 "Merge cut-3…" (08:41:24), not at e36a152f7. It is local only (worktree `scratchpad/ido-merge3`), not on origin, and not in main or gate-0h.

4. **The SDF brick fill.**
   - **The fp32 claim has the wrong basis.** SdfGrid computes the distance in double; only the winding sign is float (`SdfGrid.hpp:10`). Gate 6 shows that SDF value differences of ≤0.058 voxel move the trajectory, and that f32 inputs move the energy −10.4%. Section 3 says fp32, while stage D puts the fill behind the fp64 gate.
   - **It is not "once per phase".** The native solo-psd fill counter reads 1.38 s (671 bricks), but the phase-1 outer `constraint_set_update` is 0.803 s. So at least 42% of the fill happens lazily, later in the phase.
   - **The "sdf" bucket is not only the fill.** It is the minimize's first `solution_changed` (`Solver.cpp:273`), which also builds the collision set.
   - **The model charges it nothing.** A fill at `fit_begin` would have to cover bricks nobody knows in advance, and the model gives it 0 s and 0 round trips.
   - **Its guest/native ratio is 23×, not the 10 in RATIO.** This does not change any output.

5. **The fp32 Hessian in stage C ("no fp64 needed") is assumed, and no gate covers it** (rule 7). A native fp32-rounding gate with `trace_diff.py` and the guard is needed.

6. **The pump does not address rule 4.**
   - The two round-trip cases (0.6 ms, 16.7 ms) are main-thread frame periods. The plan runs fit.elf's pump on its worker thread with a local RenderingDevice, which has no frames. That round trip is unmeasured, and nothing shows how it avoids syncing in the tick that submits.
   - "Yields once per Newton" contradicts the model's 3.03–3.99 round trips per Newton.
   - The rd_compute name pool is already full on cut-3 (G3: 32 of 32 host cache slots), which the plumbing list ignores.

7. **"The map's 3.82e9 double-counts the line-search sub-timers" is wrong.**
   - 3.82e9 is the disjoint top-level sum: 2.64 + 4.48 + 156 + 198 + 20.9 = 382.0e9 over 100 Newton.
   - The README's 3.748e9 drops about 7.0e9 of untimed line-search work: LS begin's energy and gradient, the post-step energy, and LS end (`LineSearch.cpp`). That work is serial.
   - With it restored: ceiling 6.45× (7.48× weighted); model 5.01–5.28× (5.55–5.87× weighted).
   - P0psd double-counts the line-search constraint-set update inside `classical_line_search` (`LineSearch.cpp:256`); the effect is negligible (floor 7.14 → 7.19×).

8. **"Fast and 60 Hz differ by at most 6%" is false for section 5.** Stage E differs by 9.1% (6.28× against 5.75×; 98 s against 106 s), and the no-B +CCD case by 8.3%. "4–5 round trips per Newton, 75–95 ms" counts CCD as a round trip that the model does not charge. The model's figures give 3.03–3.99 round trips and 59–77 ms at 60 Hz.

9. **The weighted serial remainder is 12.9%, not 12.7%** (12.94%, consistent with 1/7.73).

10. **The psd P1 profile stands in for the default config's phase 1** without being disclosed. Native timers differ between the two: CCD is 10.1% against 4.7%, and assembly 31% against 40%.

11. **Phase-1 trials per Newton are 2.46 from the guest log** (native 2.32), not the map's 1.99. The same estimator gives 1.02 on loop p0, against 1.03 from the counters. The effect is at most 1%.

12. **Stage A's 2.27× compares one run with one run.** It sets one fit_prof.elf run against one Gate 8 fit.elf run. The numerics are the same, but guest time varies about ±10% between runs (Gate 6: 596–725 s), and the load differed (2340 against 1674 frames/s). The plain fit.elf psd run never finished. **Correction:** "about 2.1–2.5×, one run each".

13. **The modelled headline hides an assumption.** "Modelled 5.1–6.0×" rests on s_ccd = 10, assumed and not measured. At 60 Hz, s_ccd = 3–30 gives 4.25–5.43×.

14. **"Every part runs 9–21× slower" mixes two measures.** The per-component ratios are 9.0–15.6×. The 21× is a whole-phase ratio over a different trajectory.

15. **Minor.**
    - ggml-rd also accepts I16.
    - `gates/3-ggml-rd/census` exists on gate-0h.
    - box_search self time is 22.5% of the phase window (22.1% is of the thread).
    - A free solver gives 1.13×, not 1.12×.
    - The SDF fill is 1.34% of the fit, not 1.5%.
    - The largest broad-phase readback (4.2 MB) exceeds g's 1 MB allowance.
    - The quoted upload rate is host GDScript `buffer_update`, not data sent from the guest.
    - The guard is cut-6c's parked, ungated one.
    - The per-node-cost argument against ggml ignores that `mul_mat` batches. The conclusion still holds, because ggml-rd has no F64, no scatter and no eigensolver.

**Rules:**
- **Rule 2:** no violation. Every GPU part is proposed as Lean→Slang with a cpp twin.
- **Rule 4:** not addressed for the worker-thread pump (finding 6).
- **Rule 5:** no violation, because thresholds are deferred to gate 6. But every staged speedup assumes the GPU wins at 932 vertices, which that gate has yet to show.
