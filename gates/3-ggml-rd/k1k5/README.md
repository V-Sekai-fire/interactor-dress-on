# Gate 3, families K1 and K5: element-wise ops and ROPE on ggml-rd

**Result: PASS.** All 68 f32 cases of test-backend-ops for SILU, GELU,
GELU_ERF, SIGMOID, NEG, SCALE, DIAG_MASK_INF and ROPE (NEOX) pass in the
guest (`ggml_test.elf`, `-b RD0`) against the in-guest ggml-cpu, with
0 FAIL. The results are the same with a barrier after every dispatch. All
15 census rows, at the census shapes, are within test-backend-ops' NMSE of
ggml-cpu. Both fault controls fail every case they must fail. L0, L1 and
L2 pass, and the swapped-stride control catches every case where the swap
moves an address.

Setup: Godot 4.7.2, RTX 4090 (device #0). While these runs went on, other
Gate 3 family agents used the same GPU at 75% utilisation, so the timings
below include that contention. The branch is `cut-3-k1k5`, on `cut-3`
(40079d0).

## Kernels (`lean/Ggml/SlangCodegen/`)

| kernel | Lean | ggml | what |
|---|---|---|---|
| `silu_f32` | `Unary.lean` | UNARY SILU | `x / (1 + exp(-x))` |
| `gelu_f32` | `Unary.lean` | UNARY GELU | `0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2)))`, tanh argument clamped to [-10, 10] |
| `gelu_erf_f32` | `Unary.lean` | UNARY GELU_ERF | `0.5 x (1 + erf(x / sqrt 2))`, erf from A&S 7.1.26 (a Lean helper, `erf_as`) |
| `sigmoid_f32` | `Unary.lean` | UNARY SIGMOID | `1 / (1 + exp(-x))` |
| `neg_f32` | `Unary.lean` | UNARY NEG | `-x` |
| `scale_f32` | `Unary.lean` | SCALE | `x * s` when `b == 0` (so -0 stays -0, as in `ggml_vec_scale_f32`), else `x * s + b` |
| `diag_mask_inf_f32` | `Unary.lean` | DIAG_MASK_INF | `-inf` where `i0 > n_past + i1`, else `x` |
| `rope_neox_f32` | `Rope.lean` | ROPE, mode NEOX | ggml-cpu's `rope_yarn` line for line, with its running-product angle |

- **K1 kernels.** Each kernel uses one thread per destination element. Both
  tensors are 4-D strided, using Binary's `unravel4` and `off4`. There are
  no barriers and no groupshared memory, so there is no Serial sibling: the
  cpp and spirv targets are the same module.
- **K5 kernel.** It uses one thread per pair of elements in a row. Thread
  `j < n_dims/2` rotates the pair `(j, j + n_dims/2)`. Any other thread
  copies the pair `(2j, 2j+1)`, which is ggml-cpu's pass-through. Each
  thread reads only the two elements it writes, so the kernel is safe in
  place (test-backend-ops has an in-place case).
- **The rope angle.** It is ggml-cpu's own `theta = p; theta *= theta_scale`
  loop, run `j` times, so `theta_extrap` is bit-identical to ggml-cpu's.
- **Rope constants from the packer.** `theta_scale`, `corr_dims` and
  `mscale` come from the packer. It computes them with ggml-cpu's C
  expressions (`powf`, `ggml_rope_yarn_corr_dims`, `logf`), so the GPU's
  `pow` and `log` never set them.
- **What `supports_op` accepts.** It accepts f32 only. f16 is not
  supported, and every census row is f32. For ROPE it accepts mode NEOX
  only, with no src2 (frequency factors), i32 positions (at least `ne2`),
  even `n_dims <= ne0` and even `ne0`. ROPE_BACK has no packer.

### erf, measured against the thresholds

A&S 7.1.26 has an absolute error of at most 1.5e-7 in exact arithmetic.
Evaluated in f32, the maximum absolute error is 5.2e-7, measured over
2,000,001 points in [-8, 8]. The worst point is x = -0.035, where
`1 - q e^{-x^2}` cancels. In GELU_ERF that error is multiplied by `0.5 x`,
and x is small there. The measured NMSE is:

- 2.1e-15 against ggml-cpu (`erff`) on the census rows over [-6, 6];
- 2.4e-15 against a double-precision `erf`, where ggml-cpu's own is 1.0e-15;
- 1.4e-19 on test-backend-ops' [-150, 150] cases in L2.

That is eight orders of magnitude inside the 1e-7 threshold, so no better
erf was needed.

### GELU: ggml-cpu is the less accurate side

ggml-cpu rounds GELU through an f16 table for |x| < 10 (`GGML_GELU_FP16` in
`vec.h`). This kernel computes it in f32. On the census row [8192,4096]
over [-6, 6], the NMSE against a double-precision GELU is:

| backend | NMSE vs double |
|---|---|
| RD | 8.5e-16 |
| ggml-cpu | 5.5e-8 |

The 5.5e-8 between the two backends is therefore ggml-cpu's rounding. It is
still inside 1e-7. test-backend-ops draws from [-150, 150], where most
inputs are past the table, and gets 1.2e-11.

## Verdicts

| level | what | verdict | numbers |
|---|---|---|---|
| L0 | `lake build Ggml` pins, gen.sh check | **PASS** | `kernels/lean-build.log`. `silu_f32` and `rope_neox_f32` are pinned in full. Every other K1 kernel is pinned as silu's text with its helper and its one result line changed. The erf, gelu and scale helper texts are pinned, as are the rope word numbers (55-60, 43, 44). Controls: `kernels/lean-negative-control.log`. |
| L1 | spirv-val, fixed layout, riscv64 cpp | **PASS** | `kernels/l1.log`: 10 kernels on the layout. Object sizes are 3544-4552 bytes. |
| L2 | cpp emits + guest packers vs native ggml-cpu | **PASS** | `kernels/l2.log`: 89/89 cases (48 of them K1/K5: 28 and 20). Every case is below 1e-7. SIGMOID, NEG, SCALE and DIAG_MASK_INF are bit-exact. The worst ROPE NMSE is 2.2e-11 and the worst GELU is 5.5e-8 (above). |
| L2 control | src0's nb1 and nb2 swapped after packing | **PASS** | `kernels/l2-control.log`: 83 cases detected, 0 missed, 6 no-ops. Of the K1/K5 cases, 46 are detected and 2 are no-ops (ne1 = 1). |
| L3 G3.ops | test-backend-ops in the guest | **PASS** | `../ops/k1k5/results.txt`: OK=68, FAIL=0. The OK cases are 4 each for SILU, GELU, GELU_ERF, SIGMOID, NEG and SCALE, 3 for DIAG_MASK_INF and 41 for ROPE. The 427 "not supported" cases are f16 and the other ROPE modes and frequency factors; every ROPE f32 NEOX case with ff=0 is OK. `ops_main` took 99 s and `ops_barrier_all` gave the same 68. |
| L3 control | `GGML_RD_FAULT=1` | **PASS** | FAIL=65, OK=0 on the seven ops test-backend-ops can judge. DIAG_MASK_INF is left out on purpose (see "test-backend-ops cannot fail DIAG_MASK_INF" below); the census control covers it. |
| census | the census shapes vs ggml-cpu (`census_ggml_rd.gd`) | **PASS** | `../ops/k1k5/census.txt`: 15/15 rows OK. NEG, SCALE and DIAG_MASK_INF are bit-exact. SILU, GELU_ERF and SIGMOID are at most 2.1e-15, GELU is 5.5e-8 (above) and ROPE [128,8,514] is 2.9e-11. The big SCALE row is checked at ne1 = 64 (27.6M elements) and timed at full size. |
| census control | the same under `GGML_RD_FAULT=1` | **PASS** | 15/15 rows FAIL, DIAG_MASK_INF included (NMSE 2.0, inf-aware). |
| rule 4 | syncs in the submit frame | **PASS** | `rule4_same_frame_syncs=0` in every run. `permanent_slots=0` at close. |
| probes | chain, independent, files, alias | **PASS** | These are unchanged from Cut 3 and re-run in `../ops/k1k5/results.txt`. |

### test-backend-ops cannot fail DIAG_MASK_INF on values

test-backend-ops' `nmse()` sums `(a - b)^2` over every element. It skips
nothing, so a masked element contributes `-inf - -inf = NaN`, and
`NaN > max_err` is false. Under `GGML_RD_FAULT=1` all three DIAG_MASK_INF
cases still read OK.

The inf checks before the sum do catch a wrong mask position or sign; they
do not catch wrong values. That is why:

- L2's comparison skips infinities that match in sign, as test-backend-ops'
  inf check does, and fails on any other infinity;
- the census probe compares the same way, and its fault control fails
  DIAG_MASK_INF (NMSE 2.0).

The same blind spot applies to any op whose output holds matching
infinities. The fact is in AGENTS.md.

## Performance (host-timed, `../ops/k1k5/census.txt`)

The guest clock is not a clock, so the host times each graph.
`InferHost.wait_us` runs from the end of the vmcall that submitted the
graph to the end of the vmcall that synced it; the job yields COOP right
after the sync.

Each perf row is a graph of R in-place applications, which has R - 1
barriers. The run does one warm-up and five timed graphs, and reports the
median. Row 0 is one 256-element NEG, the floor: frame gap, submit and
sync, 0.34-0.66 ms. The per-dispatch time is (median - floor) / R.

| census row | count per run | R | graph median | per dispatch | effective |
|---|---|---|---|---|---|
| SCALE [54000,512,8] | skin-tokens x822 | 8 | 22.3 ms | 2.70 ms | 656 GB/s (1.77 GB per dispatch) |
| GELU [8192,4096] | Pixal3D x60 | 16 | 7.9 ms | 452 us | 594 GB/s |
| ROPE [128,8,514] | skin-tokens x21560 | 512 | 3.1 ms | 4.7 us | including a barrier |
| DIAG_MASK_INF [514,514,16] | skin-tokens x56 | 64 | 2.3 ms | 25 us | L2-resident |
| GELU_ERF [4096,1029] | Pixal3D x24 | 32 | 1.1 ms | 14.5 us | |
| NEG [1,64,12,4096] | Pixal3D x120 | 64 | 1.3 ms | 10.4 us | |
| SILU [3072,514] | skin-tokens x56 | 64 | 0.96 ms | 4.7 us | |
| SILU [3072,1,2], SCALE [515,1,16,2], SIGMOID [1,16384] | skin-tokens x10724, x10724, x212 | 256 | 0.55-0.74 ms | below the floor | 256 dependent dispatches fit inside one frame gap, so each is under about 2.5 us |

The two DRAM-bound rows (SCALE and GELU) move 594-656 GB/s. That is about
60-65% of the 4090's 1008 GB/s, measured while the GPU was 75% busy with
other agents' runs.

A contiguous fast path was tried: skip `unravel4` when both tensors are
contiguous, flagged by a derived word. It measured 666 vs 656 GB/s on
SCALE and 610 vs 594 GB/s on GELU, which is within the contention noise.
It also made the stride control blind on contiguous cases (the kernel no
longer reads the strides), so it was removed. The kernels are memory-bound,
not index-bound.

## Files

- `kernels/`: L0, L1, L2 and L2-control logs for this family's tree. The
  shared `gates/3-ggml-rd/kernels/*.log` are left as the lead committed
  them; the lead re-runs them after merging.
- `../ops/k1k5/`:
  - `results.txt`: G3.ops. `run-ops_*.log` and `run-probe_*.log` are each
    job's output, and `run.log` is Godot's stdout.
  - `census.txt`: census, its control, and perf. `run-census-all.log`,
    `run-census_fault-all.log` and `run-perf-<row>.log` are each job's
    output, and `run-census.log` is Godot's stdout.

To reproduce:

```
kernels/ggml/gen.sh --update && gates/3-ggml-rd/kernels/l0.sh && gates/3-ggml-rd/kernels/l1.sh
tests/ggml_rd_kernels/build.sh
./build.sh && godot --path project --headless --import
godot --path project --script gate_ggml_rd.gd --rendering-driver vulkan --xr-mode off -- \
  --ops=SILU,GELU,GELU_ERF,SIGMOID,NEG,SCALE,DIAG_MASK_INF,ROPE \
  --fault-ops=SILU,GELU,GELU_ERF,SIGMOID,NEG,SCALE,ROPE --out=ops/k1k5
godot --path project --script census_ggml_rd.gd --rendering-driver vulkan --xr-mode off -- --out=k1k5
```

## Changes outside the family's own files

These are small, and the lead merges them.

- `tests/ggml_rd_kernels`:
  - `L2Case::init` fills a leaf itself, for ROPE's i32 positions.
  - The comparison follows test-backend-ops' inf rule: matching infinities
    are skipped in the NMSE and any other infinity fails. Before this,
    `-inf - -inf` made every DIAG_MASK_INF case NaN.
  - The swapped-stride control no longer runs a dispatch whose swapped
    strides reach past the memory block, and counts it as detected. The
    cpp emits do not check bounds, and SIGMOID [1,16384] crashed the
    control arm.
- `project/gate_ggml_rd.gd` takes `--ops=`, `--fault-ops=` and `--out=`
  user arguments, so a family runs its own ops without overwriting the
  lead's `ops/` evidence. The defaults are unchanged.
- `project/infer_host.gd` has `wait_us`, the host-timed GPU waits.
- `project/census_ggml_rd.gd`, `guest/ggml_test/census.{h,cpp}`: the census
  and perf probes, registered in `probes.cpp`, with `census.cpp` added to
  `ggml_test`'s sources.
- `project/main.gd` has the preset wrappers `ggml_ops_k1k5`,
  `ggml_probe_census`, `ggml_probe_census_fault` and `ggml_probe_perf`.
  They add no new guest entry point.
