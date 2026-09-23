# L-BFGS-B oracle (host-native LBFGSpp)

Reference data for Gate 5 G1 (components) and G2 (whole problems). Made by
`tests/lbfgsb_oracle/gen.cpp` through `tests/lbfgsb_oracle/build.sh`, which
writes this directory and `gen.log`. The output is deterministic: two runs
give byte-identical files, and so do the two Eigen copies below.

## Source

- LBFGSpp 0.3.0 comes from `V-Sekai-fire/interactor-aria-lbfgspp` @ `10086b6b2022802d7942ba16842c43063b3cff91`, under `thirdparty/LBFGSpp/include`.
  - Its `include/` tree is byte-identical to `C:/cloth-dynamics-standalone/external/LBFGSpp/include` (`diff -r` is empty). The whole `thirdparty/LBFGSpp` directory is identical too.
  - The headers are used unmodified. `#define private public` around `#include <LBFGSB.h>` exposes `LBFGSBSolver::max_step_size`, `proj_grad_norm` and `BFGSMat::m_ptr`.
- Eigen 3.4.90 comes from the same clone, `thirdparty/eigen`. The files here were first made with `cloth-dynamics-standalone/external/eigen` (@ e361584, also 3.4.90, 58 files differ); a fresh clone and build with the org copy reproduces every file, `gen.log` included, byte for byte (`diff -r` empty).
- LBFGSpp and Eigen are used only by this host-native generator. Neither reaches any guest ELF; the in-guest L-BFGS-B is Lean-emitted (AGENTS.md rules 2 and 3).
- Compiler: llvm-mingw 20260826 `x86_64-w64-mingw32-clang++ -O2 -std=c++17 -static`. All arithmetic is in `double`.

## Conventions (all files)

- The text is line-oriented. Each line starts with a key.
- A vector is written as `key <len> v0 v1 …`.
- An index set is written as `key <count> i0 i1 …`, with 0-based indices.
- Numbers use `%.9g`. Infinite bounds are written `inf` / `-inf`.
- Every **input** (bounds, x, g, s, y, the `mv_in` vectors, problem data) is rounded to float32 before use. The `%.9g` text therefore holds each input exactly, as a float32 and so as a double. Outputs are double results printed at 9 significant digits.
- `#` lines are comments.

## `components/comp_NN.txt` (20 fixtures, NN = 00..19)

A single state (x, g, bounds, BFGS history) is pushed through `BFGSMat<double,true>`, then `Cauchy::get_cauchy_point`, then `SubspaceMin::subspace_minimize` (max_submin = 10). This is exactly what one `minimize` iteration does after `add_correction`.

Coverage:

| field | values |
|---|---|
| n | cycles 5, 37, 300 |
| m | 10 (NN%4 < 2), else 5 |
| npairs | {3, m, 1, m+3, 2m+1, 7, 2} by NN%7; NN=19 has 0 (empty W, theta = 1) |

- About 15% of coordinates have lb==ub, 5% are (-inf, inf), 5% are one-sided, and the rest are finite boxes.
- About 20% of x start at lb and 20% at ub.
- About 3% of coordinates have g = 0, which gives brk = inf.
- NN = 3, 9 and 15 are "tight boxes": finite bounds, nonzero g, |g| up to 50. In 9 and 15 the Cauchy search crosses every breakpoint (`fv` is empty: the `crossed_all` branch).
- The history comes from a convex quadratic H = diag(d) + u1u1' + u2u2' with y = H s, so s'y > 0. Every pair passes the solver's `s'y > eps*y'y` filter.

| key | meaning |
|---|---|
| `seed` | splitmix64 seed (`0x5EED0000 + 7919*NN`) |
| `n`, `m` | dimension, history size |
| `npairs` | number of `add_correction` calls, in order |
| `ncorr` | `min(npairs, m)`, the retained pairs |
| `ptr` | `BFGSMat::m_ptr` after the last add. The most recent pair sits in slot `(ptr-1) % m` |
| `theta` | `y'y / s'y` of the last pair (1 if no pairs) |
| `lb`, `ub`, `x`, `g` | inputs, length n. x lies in [lb, ub] |
| `s P slot J <n> …` / `y P slot J <n> …` | pair P in insertion order. It is stored in ring slot J = P % m, and a later pair overwrites the slot |
| `M K K …` | the 2ncorr×2ncorr matrix applied by `apply_Mv`, row-major (column j = `apply_Mv(e_j)`) |
| `mv_in t <2ncorr> …`, `mv_out t <2ncorr> …` | three random v and `apply_Mv(v)` |
| `pg_inf` | `‖P(x−g) − x‖∞` (`proj_grad_norm`) |
| `xcp` | generalized Cauchy point |
| `vecc` | c = Wᵀ(xcp − x), length 2ncorr, W layout below |
| `brk0` | coordinates with breakpoint 0 (lb==ub, or x on a bound with −g pointing out). These are in neither set below |
| `newact` / `newact_sorted` | coordinates that become active at their breakpoint during the GCP search (solver order / ascending) |
| `fv` / `fv_sorted` | the free set after the GCP: brk = inf first, then the uncrossed breakpoints in sort order (solver order / ascending) |
| `drt0`, `step_max0` | `normalize(xcp − x)` (the first-iteration direction) and `max_step_size(x, drt0)` |
| `drt` | direction from `subspace_minimize` (x_sm − x) |
| `sub_L`, `sub_U`, `sub_P` | the free coordinates the subspace step leaves at lb (drt = lb−x exactly), at ub, or strictly inside |
| `dg` | gᵀ drt |
| `step_max` | `max_step_size(x, drt, lb, ub)`. It is `inf` if drt is 0 |

**Layout of W, M, vecc and mv.** A 2ncorr vector is `[Yᵀv ; θ·Sᵀv]`.
- Entry j < ncorr belongs to **ring slot j** (not to insertion order), and entry ncorr+j is slot j's S part.
- While npairs ≤ m, slot order equals insertion order.

**Comparing sets.** `newact` and the tail of `fv` come from `std::sort` on breakpoints. `std::sort` is not stable, so equal breakpoints may come out in a different order. Compare the sets as sets: use the `_sorted` lines.

## `problems/<name>.txt`

These files contain `n`, `lb`, `ub` and `x0`.

- **`rosen_n{2,10,100}`**: the classic chained Rosenbrock, f = Σ_{i<n−1} 100(x_{i+1} − x_i²)² + (1 − x_i)².
  - Bounds: lb = −1.5, ub = 1.5. If i%3 == 1, then ub = 0.6.
  - For n ≥ 10: if i%5 == 4, the coordinate is free (−inf, inf), and x[n/2] is fixed at lb = ub = 0.5.
  - x0: −1.2 at even i, 1.0 at odd i. It is projected into the box by `minimize`.
- **`rosenbox_upstream_n25`**: LBFGSpp's own `examples/example-rosenbrock-box.cpp`, verbatim.
  - f = (x0 − 1)² + Σ 4(x_i − x_{i−1}²)².
  - Bounds [2, 4], with x2 free. x0 = 3, except x0 = x1 = 2 and x5 = x7 = 4.
- **`boxqp_n1000`**: f = ½xᵀAx − bᵀx, with A tridiagonal (`A_diag`, off-diagonal `A_offdiag` = −1) and bounds [−1, 1]. x0 = 0.
  - It is built from a known KKT point `xstar` in which half the coordinates are active:
    - i%4 == 0 sits at lb, with g* ≥ 0.25;
    - i%4 == 1 sits at ub, with g* ≤ −0.25;
    - the rest are free in [−0.5, 0.5], with g* = 0.
  - All data are dyadic, so b = A·xstar − g* is exact. The file also holds `b`, `fstar` and `gstar`.

## `traces/<problem>_m<M>_<tag>.txt` (20 files)

Problems × m ∈ {10, 5} × tag:

- **`dc`** uses the diffcloth parameters (BackwardTaskSolver.cpp:49-52): delta = 1e-3, max_linesearch = 20.
- **`tight`** is identical except delta = 1e-10, which is LBFGSpp's default.

All other parameters are LBFGSpp defaults. The `param` line spells them out. The solver is `LBFGSBSolver<double, LineSearchMoreThuente>`.

Header lines:

| key | meaning |
|---|---|
| `niter` | the value `minimize` returns (max_iterations = 0) |
| `nfev` | total f/g evaluations |
| `status` | `converged-grad`: ‖pg‖∞ ≤ ε or ≤ ε_rel‖x‖ |
| | `converged-delta`: \|f_{k−1} − f_k\| ≤ δ·max(\|f\|, \|f_{k−1}\|, 1) |
| | `exception: …`: the line search threw (none occur) |
| `f_final`, `pg_final` | at the last iterate |
| `xstar_err_inf` | ‖x_final − xstar‖∞ (QP only) |

Then one block per iterate k = 0..niter:

```
iter k nfev N f <f_k> pginf <‖P(x_k−g_k)−x_k‖∞>
x <n> …
L <c> i…     # active at lb: x_i ≤ lb_i + 1e-9·max(1,|lb_i|)  (lb==ub coordinates land here)
U <c> i…     # active at ub: x_i ≥ ub_i − 1e-9·max(1,|ub_i|)
```

- Iterate 0 is x0 projected into the box. Its `nfev` is 1.
- Iterate k is the x that the **unmodified** solver returns with `max_iterations = k`. The algorithm is deterministic, so this is exactly the k-th iterate of the full run. The generator checks that the full run's `niter` and final x equal the last prefix run bit for bit (no mismatches).
- The iterate is taken before the solver's next `force_bounds`. A step that ends at `step_max` can therefore sit up to an ulp off the bound, which is why the active-set test uses 1e-9.
- `nfev` is cumulative up to the end of iteration k.

## Results (m = 10 / m = 5)

| problem | dc niter | tight niter | tight f_final | final \|L\|/\|U\| |
|---|---|---|---|---|
| rosen_n2 | 19 / 19 | 21 / 20 | 0.0505960984 | 0/1 |
| rosen_n10 | 11 / 10 | 40 / 45 | 21.9430592 | 1/1 |
| rosen_n100 | 12 / 10 | 43 / 43 | 118.811118 | 1/1 |
| rosenbox_upstream_n25 | 10 / 10 | 13 / 13 | 360.283586 | 22/1 |
| boxqp_n1000 | 5 / 5 | 11 / 11 | −1932.98535 | 250/250 (xstar err 8.3e-5 / 7.8e-5) |

## `f32io_control.txt` (G2's flat control, Cut 5c)

This file is written by `tests/lbfgsb_oracle/sensitivity.sh`, not by
`gen.cpp`. Each line is `<trace> <err>`, where err is |f_final − f_trace|
for unmodified LBFGSpp (double) on that trace under the float32 interface
the guest has:

- float32-rounded x0, lb and ub;
- f and g evaluated at float32 x;
- g handed back as float32.

For each trace, G2's f band is max(1e-6 (1 + |f|), 2 × err). Only
`rosen_n2_m10_dc` has 2 × err above the 1e-6 band: its err is 4.83e-6, so
its band is 9.66e-6. Every other trace keeps the 1e-6 band. The gate hands
this file to the guest as `f32io_control`, and G2 refuses to run without it.
The evidence for this band (the storage arm and the stochastic-rounding
ensemble) is in `../lbfgsb/sensitivity.log` and in `../README.md` under G2.

## `inverse_min/case_<name>.txt` (G3, 2 files)

Made by `tests/inverse_min_oracle/build.sh` (`oracle.cpp`), not by `gen.cpp`:
LBFGSpp 0.3.0 (the same clone) on cloth-dynamics' `test_avbd_inverse_min`
objective, compiled for the host from the guest's own sources
(`guest/drape/inverse_min.h` over `guest/avbd/avbd_cpu*.cpp`, llvm-mingw
clang++ -O2 -ffp-contract=off). One file per case: `k_tri` (truth 2, start
0.5, lb 0.05) and `k_bend_density` (truth (1.5, 1.25), start (0.4, 2.5), lb
0.02 each). Keys: `param` (LBFGSpp's defaults with m 10, max_linesearch 20,
max_iterations 100), `truth`, `start`, `lb`, `target` (the host AvbdCpu's
rollout at the truth), one `eval` line per objective evaluation (x, f, g),
then `niter`, `nfev`, `status`, `f_final`, `x`, `err` (|x - truth|_inf), the
same with a `zero_` prefix for the zero-gradient arm, and `gd_x` / `gd_err`
(upstream's own backtracking gradient descent on this objective, for
information). `oracle.log` is the run's summary: LBFGSpp recovers both
cases (6 iterations, err 1.1e-6; 13 iterations, err 4.2e-6), its
zero-gradient arm stays put (err 1.5 and 1.25), and upstream's descent
reaches err 0 and 6e-7.
