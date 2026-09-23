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

**With psd, all phases, second run** (`g*` logs, `run_psd_full2.sh`, 2026-09-23):
the phase-1 crash was a scratch-wrapper fault, as suspected, and not a
numerical one. `ALSolver::solve_reduced` asks the freshly enabled `FitForm`
for its value (the "Failed to apply boundary conditions" check) before the
solver's first `solution_changed`; the double path reads `totalP`, sized and
zero-filled at construction, while the df32 wrapper indexed an empty
per-form sample vector (`fit_df.cpp`, `g_samples[this]`) and segfaulted.
The fix (`src/fit_df.cpp`): absent samples read as SdfHess's zero default,
and a form erases any samples a form at a reused address left (`reset`
from the constructor). All three arms then finish every phase, none at the
cap, no intersections (`fit_check_intersections` on the final garment):

| arm | Newton per phase | total | final energy (17 digits) | fit gap mean / p95 (voxels) | wall |
|---|---|---|---|---|---|
| g0 double, psd (control) | 10 / 17 / 7 / 47 | 81 | 0.0015371630972068565 | 1.9042 / 3.9322 | 48.5 s |
| g1 df48, psd | 10 / 19 / 10 / 35 | 74 | 0.0015841669043766085 (+3.1 %) | 1.9172 / 3.9019 | 162.7 s |
| g3 df48 + ACCD(df32), psd, audited | 8 / 35 / 11 / 50 | 104 | 0.0014004200927798165 (-8.9 %) | 1.7964 / 3.9848 | 224.1 s |

The fit gap is measured in the solve frame against the oracle's final
avatar (`C:/b/cf-up-out1/step_avatar_252.obj`, the avatar `gate_fit.gd`
uses), `runs/g-gap.txt`. Against the double control the df48 arm moves the
mean by +0.013 voxel and the p95 by -0.030; the ACCD arm by -0.108 and
+0.053. Both are inside the Gate 6 guard (mean within 0.25, p95 within 0.5),
and all three sit inside the guard around upstream's 1.8467 / 3.9207 as
well. The wall times are the emulation's (every df32 operation is a rounded
double operation plus bookkeeping), not a forecast.

ACCD in df32 with recomputed margins drove all 104 line searches of g3:
6,016,252 point-triangle and edge-edge queries, 627 hits, and the audit
(Tight Inclusion on [0, step], minimum separation 0, tolerance 1e-6)
flagged **0** of them. The df32 event counters saw no overflow, NaN or
infinity in any form; only underflows into the low word (the second float
of a pair falling below the float range, a loss of the last bits of a term
near 1e-38 relative), 1.2e7 in 2.4e10 similarity operations.

Newton counts and energies differ between the arms from the first step
(`runs/g-analyze.txt`: the trajectories part at record 1 and the
collision-free steps agree in 19 of 74 line searches); the solve is chaotic
(Gate 6), so per-iteration agreement was never the criterion. What the
gate asks is whether df32 keeps the solve converging to a garment the guard
accepts, and it does, in every phase, with the CCD in df32 too.

## Verdict

**PASS: float-float carries the whole fit.** The forms, the SDF sampler,
the line-search energies and the CCD all run in df32 (48-bit significand,
float32 range) through all four foxgirl phases with psd on, converge by
gradient norm in every phase, and land inside the Gate 6 guard, with the
df32 ACCD conservative on 6.0 M audited queries. No stage needs qf32.
Cut 6g-C starts its kernels in df32 (`lean/Fit/`), with the CPU double
path as the flat control.

What the earlier runs established stands: forms in df32 match double to
1e-11 to 1e-15 per evaluation; ACCD in df32 against Tight Inclusion and
double ACCD over 16.2 M queries: 0 missed, 0 extra, time of impact within
1.14e-12.

Reproduce: the scratch build (`C:/b/g6g0`, sources under `scratchpad/g6g0`;
`src/` here carries the two files the fix touched), then
`bash run_psd_full2.sh`. Negative arms kept: `f1`, `f3` (the crash, exit
139), `b-df48` (no psd: 50-iteration cap).
