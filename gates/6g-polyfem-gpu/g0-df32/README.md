# Gate 6g.0 — is float-float (df32) enough for PolyFEM's forms and CCD? [G6g0-26]

**Question.** The fit's GPU port (cut 6g-C) must not use fp64 (consumer GPUs
run it at 1/64 rate). Do PolyFEM's forms (elastic/similarity, IPC barrier,
SDF), the line search's energies and the CCD keep the solve converging in
df32 (two float32s, ~48-bit significand, float32 exponent range), with the
CCD as additive CCD (ACCD) whose conservative margins are recomputed for
df32's error?

**How.** A scratch build of fit_native (`C:/b/g6g0`, sources in
`scratchpad/g6g0/g6g`, patch not vendored) evaluates the hot form functions
through a scalar type that rounds every operation to a chosen significand
width with float32 range checks (`df32.hpp`; 48 bits = df32, 44 = a
pessimistic model), and runs ACCD in that type next to tight-inclusion and
double ACCD with an audit of every query. `run_arm.sh` runs one capped arm;
`analyze.py` compares trajectories. Arms marked `-psd` have polysolve's
`force_psd_projection` on (stage A, shipped in `6d6964f`).

## Results (foxgirl, native, 2026-09-23)

**Without psd** (`a*`, `b*` logs, earlier scratch run): forms in df32 match
double to 1e-11…1e-15 relative per evaluation, no overflow; ACCD in df32
(plain and recomputed margins) vs tight-inclusion and double ACCD over
4.53 M point-triangle + 11.66 M edge-edge queries: **0 missed, 0 extra**,
time of impact within 1.14e-12 (p50 4.6e-15), identical step decisions in
all 41 line searches. But the full phase-0 solve with forms in df32 at 48
bits hit the 50-iteration cap (energy 30× the control's) while the 44-bit
model converged in 24 iterations (double: 41).

**With psd, phase 0** (`c*` logs): every arm converges by gradient norm
(status 4, none at the cap), no intersections — c0 double 10 Newton,
energy 4.86e-4, 11.4 s; c1 df48 10 Newton, 6.11e-4 (+26%), 24.7 s; c2 df44
12 Newton, 1.70e-3, 32.3 s; c3 df48 + ACCD(df32) 8 Newton, 4.02e-3, 22.8 s.
The trajectories part from the first Newton step (analyze: first record
> 1e-6 at iteration 1). Phase-0 energy is not a usable quality measure on
this chaotic solve: double itself moves 44% between psd off and on.

**With psd, all phases** (`f*` logs): c0 double finishes every phase in
64 s (10/17/7/… Newton). **The df32 arms crash (exit 139) in phase 1**
after converging in phase 0 — a fault in the scratch wrappers for the
reduced solve's forms, not a numerical result. The fit-gap check
(`fit_gap.py`) was not applied: fit_native writes the garment in the solve
frame, and the tool wants the avatar frame.

## Verdict

**Inconclusive past phase 0; parked** (user, 2026-09-23, weekly quota). What
stands: df32 evaluations are accurate, df32 ACCD is conservative on 16.2 M
queries, and with psd the phase-0 Newton converges in df32. What is missing
for a PASS: the wrapper crash in phase 1 fixed, all four phases run for c1
and c3, and the final garments compared in the avatar frame under the Gate 6
guard (gap mean within 0.25 voxel, p95 within 0.5, no intersections). Until
then, cut 6g-C's kernels are not started.
