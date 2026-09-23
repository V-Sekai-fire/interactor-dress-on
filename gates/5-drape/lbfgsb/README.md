# Gate 5 G1 and G2: the in-guest L-BFGS-B driver against LBFGSpp

**Result: G1 PASS on both backends. G2 FAIL, 19 of 20 traces on both backends.**
The failing trace is `rosen_n2_m10_dc`, and the flat control below shows
that LBFGSpp itself misses the same trace once its inputs are rounded to
float32. `RESULT: FAIL` in [`results.txt`](results.txt) comes from that one
trace.

The driver is `guest/drape/lbfgsb.{h,cpp}`. It holds LBFGSpp 0.3.0's
`LBFGSBSolver::minimize` and `LineSearchMoreThuente` as scalar control flow
and uses reverse communication. Every vector and matrix operation is one of
the Lean-emitted kernels in `kernels/drape`, dispatched through
`guest/drape/vec_cpu.cpp` (the slangc cpp emits) or `guest/drape/vec_rd.cpp`
(the SPIR-V over `rdc::Device`). A phase is one list of ops. On the GPU that
list becomes one compute list and one submit, and the scalars are read back
on the next tick (rule 4).

## Runs

The main run is:

```
godot --path project --script gate_lbfgsb.gd --rendering-driver vulkan --xr-mode off   # results.txt, run.log
```

Godot 4.7.2 ran it on an RTX 4090. The host hands the 45 oracle files
(`gates/5-drape/oracle`) to the guest through `drape_job_data`. The jobs
`lbfgsb_components` (G1) and `lbfgsb_problems` (G2) then run on `cpu` and
`rd`.

Two host-native checks sit beside it:

- `tests/lbfgsb_driver/build.sh` builds the same `lbfgsb.cpp`,
  `lbfgsb_gate.cpp` and `vec_cpu.cpp` natively and writes
  [`host_native.log`](host_native.log). Its G1 and G2 numbers equal the
  guest's cpu numbers digit for digit.
- `tests/lbfgsb_oracle/sensitivity.sh` is G2's flat control. It runs LBFGSpp
  (double, unmodified) from float32-rounded x0, lb and ub and writes
  [`sensitivity.log`](sensitivity.log).

## G1: component fixtures (rel <= 1e-4; drt, dg, step_max <= 5e-4; sets identical)

Each fixture is replayed in the order the driver dispatches:

1. reset;
2. for each pair: five `lb_multi_dot`s, then `lb_compact` and `lb_ring_store`;
3. M by columns, and M·v;
4. pg_inf, `lb_cauchy`, and drt0 with its step_max;
5. `lb_subspace`, g·drt and step_max.

The rd path issues 395 submits over the 40 runs.

| backend | fixtures | control (θ ×1.25) | worst xcp | worst drt0 | worst vecc | worst M / Mv |
|---|---|---|---|---|---|---|
| cpu | 20/20 | fails 20/20 | 6.6e-6 | 6.4e-6 | 2.7e-6 | 5.3e-7 / 4.9e-7 |
| rd | 20/20 | fails 20/20 | 9.0e-5 | 8.6e-5 | 3.7e-5 | 3.8e-7 / 4.8e-7 |

The cpu numbers equal Task K's host run (`gates/5-drape/kernels`).

**Found on the way: FMA contraction on the GPU.** On the first rd run,
fixture comp_03 failed with xcp at 1.4e-4 and drt0 at 1.4e-4. comp_03 is a
tight box whose Cauchy sweep crosses four breakpoints before its final step.
The sweep amplifies rounding order about a thousandfold. Two checks show
this:

- Compiling the same cpp emits on the host with `-mfma -ffp-contract=fast`
  moves comp_03's xcp from 6.6e-6 to 6.0e-5.
- slangc's SPIR-V left every float op contractible, so the driver was free to
  fuse.

`kernels/drape/gen.sh` now compiles SPIR-V with `-fp-mode precise`. That
decorates each float op `NoContraction` (123 in `lb_cauchy`, 244 in
`lb_subspace`; see `../kernels/l1_spirv_val.log`, all 11 still valid and
using only the Shader capability). With it, comp_03 comes in at 9.0e-5. The
remaining gap to the CPU's 6.6e-6 is Vulkan's division and sqrt precision,
which may be off by up to 2.5 ulp. **The margin is thin: 0.90 of the limit on
one fixture.**

## G2: whole problems (f within 1e-6 abs+rel, x within 1e-3, final L/U identical, iterations within max(2, 25%))

On both backends the iteration counts equal LBFGSpp's on all 20 traces (0
of the allowed margin). The evaluation counts are equal too, except for one
extra line-search evaluation on rd in `rosen_n100_m5_tight` (51 against 50).
The final active sets match on all 20. The worst x error is 0.21 of its
limit.

| trace | cpu f err | rd f err | LBFGSpp from float inputs | limit |
|---|---|---|---|---|
| rosen_n2_m10_dc | 2.1e-5 | 2.1e-5 | 2.5e-6 (outside) | 1.05e-6 |
| every other trace (worst) | 2.1e-6 (boxqp, limit 1.9e-3) | 1.8e-6 (rosen_n10, limit 2.3e-5) | within | |

Why `rosen_n2_m10_dc` fails:

- The oracle's problems use double inputs (x0 = −1.2, ub = 0.6), and float32
  cannot hold those values exactly.
- The dc parameter set stops on the delta test partway down the valley
  (pg 0.09), so the stopping point depends on the path.
- Our f stays within 1e-6 of LBFGSpp's through iteration 4 and parts from it after
  that (first divergence > 1e-6 at iteration 5). At the stop it is 2.1e-5
  away.

LBFGSpp in double, started from the same float32-rounded inputs, also lands
outside the band on this trace (2.5e-6) and on no other. So the problem is
what amplifies the input rounding here, not the port. The same problem with
LBFGSpp's default delta (`rosen_n2_m10_tight`) matches to 6.9e-9.

## Cost per iteration (host wall time of a one-trace job ÷ its iterations)

| n | cpu ms/iter | rd ms/iter | rd/cpu |
|---|---|---|---|
| 2 | 0.05–0.06 | 2.7–2.8 | 41–55 |
| 10 | 0.04–0.09 | 2.4–2.8 | 28–60 |
| 25 | 0.09–0.13 | 2.6–3.2 | 20–36 |
| 100 | 0.11–0.15 | 3.1–4.1 | 23–29 |
| 1000 | 0.72–1.03 | 7.3–9.9 | 8–10 |

An rd iteration costs three frames, one per phase: direction, trial point,
and post-evaluation. Its cost is dominated by frames and submits, while the
cpu cost grows with n. The ratio falls as n grows, but no crossover appears
up to n = 1000. For that reason, `drape_queue_optimize`'s `vec=auto` and the
`lbfgsb_*` jobs' `auto` pick cpu. A threshold beyond 1000 would need a
measurement, not an extrapolation.

## The rest of the run

- **`drape_queue_optimize`** was run end to end on the sphere demo, as an rd
  session over 60 steps. The target was mu 0.3; the start was mu 0.539770,
  in native mode, with DiffCloth's m 10, delta 1e-3 and max_linesearch 20.
  It ends DONE after 1 iteration and 3 evaluations at mu 0.236, with loss
  4.5e-7, stopped by the delta test. At these loss magnitudes the
  `max(|f|, 1)` in that test makes delta absolute, which is also why
  DiffCloth's own run stops after one iteration (`native/backwardLog.txt`).
  This is information only; G7 is the gate for the mu sequence.
- **`main.gd` wrappers.** `lbfgsb_load_oracle` followed by
  `drape_job("lbfgsb_problems", "cpu", "only=rosen_n2_m5_tight")` gives PASS.
- **Rule 4.** `rd_rule4` reports `same_frame_syncs=0` over 3432 submits.
  `rd_close` frees the device with 0 permanent slots left.
- **Rules 2 and 3.** `llvm-nm -C drape.elf` shows 0 Eigen and 0 LBFGSpp
  symbols.

## Reproduce

```
./build.sh                                   # runs kernels/drape/gen.sh --no-emit (SPIR-V with -fp-mode precise)
tests/lbfgsb_driver/build.sh                 # host_native.log
tests/lbfgsb_oracle/sensitivity.sh           # sensitivity.log (ARIA=<clone> reuses a checkout)
cd project && godot --path . --script gate_lbfgsb.gd --rendering-driver vulkan --xr-mode off > ../gates/5-drape/lbfgsb/run.log 2>&1
```
