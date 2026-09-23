# Gate 5, kernel level: the Lean-emitted L-BFGS-B against LBFGSpp

**Result: PASS.** All 20 LBFGSpp component fixtures pass at L2, and the
corrupted-θ control fails all 20.

This is G1 ("components vs LBFGSpp within 1e-4; active sets identical;
control: a corrupted θ fails") checked on the host, on the same cpp emits
the guest's CPU path compiles. It does not run the kernels in drape.elf or
on the GPU; the SPIR-V is only validated here (L1).

## What runs

- **L0.** `lake build` in `lean/` checks the `native_decide` pins of the 12
  `Drape.SlangCodegen.Lb*` modules, along with every Cloth pin. The build
  passes (91 jobs). A tampered pin fails the build: changing `il + dtm` to
  `il - dtm` in LbCauchy's `expected` makes `native_decide` fail.
- **L1.** `kernels/drape/gen.sh` compiles the kernels.
  - `slangc -target spirv`: 11 kernels. [`l1_spirv_val.log`](l1_spirv_val.log)
    shows all 11 valid under `spirv-val --target-env vulkan1.2`, using only
    the `Shader` capability (no Float64).
    Since Task B the SPIR-V is compiled with `-fp-mode precise`, which marks
    every float op `NoContraction` (counts in the log). Without it the GPU
    could fuse ops into FMAs, which moved comp_03's Cauchy point to 1.4e-4
    in the guest (gates/5-drape/lbfgsb).
  - `slangc -target cpp`: 11 kernels.
  - `tests/drape_kernels/build.sh` also compiles the host test's
    translation unit for riscv64, object only.
- **L2.** `tests/drape_kernels/build.sh` builds `test.cpp` against the cpp
  emits. It replays each fixture through the kernels in the order the
  drape driver dispatches them:
  1. `lb_compact` (reset).
  2. For each history pair: 5× `lb_multi_dot_serial`, then `lb_compact`,
     then `lb_ring_store`.
  3. `lb_apply_m`, on unit vectors (M) and on the three `mv_in` vectors.
  4. `lb_pg_inf_serial`.
  5. `lb_cauchy`.
  6. `saxpby` + `dot_reduce_serial` + `lb_step_max_serial`, giving drt0 and
     step_max0.
  7. `lb_subspace` (max_submin 10), then `dot_reduce_serial` for dg and
     `lb_step_max_serial` for step_max.
  8. `lb_box_project`, which must leave the in-box x bit-identical.

  Everything runs in float32, and the fixtures' inputs are float32-exact.
  The pass criteria:
  - rel ≤ 1e-4 for θ, M, Mv, pg_inf, xcp, vecc, drt0 and step_max0;
  - rel ≤ 5e-4 for drt, dg and step_max;
  - ncorr and ptr equal;
  - every pair passes the curvature filter;
  - every Cholesky pivot is positive;
  - newact, the free set, brk0 and the subspace L/U/P sets identical as sets.

  Here rel is ‖got − ref‖∞ / ‖ref‖∞, and values ≥ 1e30 must match as ±inf.

## Numbers ([`l2_fixtures.log`](l2_fixtures.log))

20/20 fixtures pass. The worst relative error over all 20:

| quantity | worst | quantity | worst |
|---|---|---|---|
| θ | 9.2e-8 | drt0 | 6.4e-6 |
| M | 5.3e-7 | step_max0 | 1.1e-6 |
| Mv | 4.9e-7 | drt | 3.0e-7 |
| pg_inf | 1.7e-7 | dg | 8.0e-8 |
| xcp | 6.6e-6 | step_max | 1.0e-7 |
| vecc | 2.7e-6 | | |

Every set matches. That includes the tight-box fixtures 09 and 15, where
the Cauchy search crosses every breakpoint, and the empty-history
fixture 19.

Subspace exit paths across the 20 fixtures:

| exit path | fixtures |
|---|---|
| unconstrained y feasible | 9 |
| primal-dual loop converged | 9 |
| free set empty | 2 |

No fixture reaches LBFGSpp's fallbacks.

## Control ([`l2_control.log`](l2_control.log))

The control scales θ by 1.25 after the history is loaded. `lb_compact` then
refactors T with the wrong θ; zeroed products make its curvature test
reject, so it only rebuilds L, T and the factor. θ's own check is turned
off, so the control is judged only on what θ feeds.

All 20 fixtures fail:

- M, Mv and vecc fail on every fixture with history (0.07 to 0.7);
- drt fails on 19;
- xcp fails on 18;
- the Cauchy sets differ on 9.

## Found on the way

In fixture 00, vecc first came out at 9.8e-5, just inside the tolerance.
After the sweep there, every breakpoint has been crossed and the free
coordinate has d = 0. In exact arithmetic p = Wᵀd is 0 at that point, so
LBFGSpp's last step does nothing. In float, p, fp and fpp are rounding
residue, and `−fp/fpp · p` moved c by 1e-4. `lb_cauchy` now skips the last
step when no coordinate is left moving, and fixture 00's vecc error drops to
8.9e-8. Keeping fp and fpp in double did not help: the error is in p, not
in fp.

Later, in the integrated gate (G7, `../README.md`): two absolute guards
that LBFGSpp writes with `numeric_limits<double>::epsilon()` had been
emitted with FLT_EPSILON. They are the Cauchy step's "fpp is numerically
zero" guard (`lb_cauchy`) and the subspace fallbacks' descent test
g.d <= -eps (`lb_subspace`). On the sphere demo the first gradient is about
1.2e-5, so fpp = g.g is 1.4e-10. That is below FLT_EPSILON (1.2e-7), so the
guard fired, divided by 1.2e-7 instead of by fpp, and shrank the Cauchy step
8000 times. The step, 1.5e-8, is under half an ulp of mu = 0.54, so xcp = x,
d = 0, and the driver stopped with "the moving direction does not decrease
the objective". Both guards now use `dblEps` (2^-52, exact in float32), which
is LBFGSpp's own threshold. The 20 fixtures are unchanged (identical
`l2_fixtures.log` and `l2_control.log`); only `lb_cauchy` and `lb_subspace`
re-emitted (2 lines each), and `l1_spirv_val.log` was regenerated.
`lb_compact`'s curvature test s.y > eps y.y stays relative, with FLT_EPSILON.

## Reproduce

```
kernels/drape/gen.sh              # lake exe emit_drape -> slang/ -> cpp/, spv, table, embed
python kernels/drape/pin.py --check
(cd lean && lake build)
tests/drape_kernels/build.sh      # rewrites l2_fixtures.log and l2_control.log
```
