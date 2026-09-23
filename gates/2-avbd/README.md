# Stage 2: AVBD in the guest, forward and backward, one source, two targets

**Result: PASS** (`results.txt`, `run.log`, 2026-09-23, RTX 4090, Godot 4.7.2).

All 25 AVBD kernels are emitted from cloth-dynamics lean (vendored as `lean/` in the next commit) by `kernels/avbd/gen.sh`.
That is the 13 forward and dual kernels, the self-collision scan, the 10 backward
kernels and `saxpby`. slangc compiles each kernel twice:
- `-target cpp` for `AvbdCpu`, the CPU path inside the guest;
- `-target spirv` for `AvbdRd`, the GPU path over `rd_compute`.

Both backends run the same kernel for every step of the backward, including its
prologue (deltaX = positions − positionsPre) and epilogue (the inertial and direct
paths), which are `saxpby` on both.

The solver lives in `drape.elf`, which has its own Sandbox (rule 6).
`dress_on.elf` keeps the Stage 1 probes.

The gate is frame-driven:
- Every check is a **job**: a stage queue in the guest (`guest/jobs.h`, `guest/drape/avbd_jobs.cpp`).
- The host ticks the job once per `_process` (`project/gate_avbd.gd`: vsync off, `max_fps` 0, a 600 s wall clock).
- A GPU submit ends the tick. Every readback therefore lands on a later frame, so rule 4 holds by construction, and the counter below proves it.
- At the end the gate calls `rd_close`, which drops the job, frees the device and checks that no permanent RID slot is left (`permanent_slots=0`).

```
godot --path project --script gate_avbd.gd --rendering-driver vulkan --xr-mode off > ../gates/2-avbd/run.log 2>&1
```

## What PASS means (plan Cut A)

| criterion | cpu | rd | evidence |
|---|---|---|---|
| Forward still exact (4-vertex oracle, 1e-5) | max_abs_diff 0 | 2.38e-7 | `job fixture` |
| Backward smoke finite and equal to native | 12/12 arrays finite, 18/18 values within 5e-3 | same | `job backward_smoke`, `--- checks` |
| gradcheck 5/5 against FD, and within 1e-4 of native | 5/5; worst 1.04e-6 vs native | 5/5; worst 9.6e-7 | `job gradcheck`, `--- checks` |
| stategrad 12/12 + 12/12 | yes | yes | `job stategrad` |
| two_vertex: cpu == rd | 0 of 29 floats differ in both arms, max 2.98e-8 | | `job two_vertex_bwd` |
| Self-collision pairs exact, cpu == rd | 4/4 cases | 4/4, same hashes | `job self_collision` |
| Rule 4: `rd_rule4 == 0`, and the probe raises it | | same_frame_syncs=0 of 339 syncs; probe 0 → 1 | `--- checks` |
| Shutdown releases everything | | `CLOSED device=freed permanent_slots=0` | `rd_close` line |

The negative controls:
- **Falsifiability.** A doubled `predictedGrad` is rejected on 9 of 12 components, on both backends; the native run also rejects 9 of 12.
- **Radius 0.07 gives 0 pairs.**
- **No padding reproduces the race** (details below).

`gradcheck_duals` is **not** a negative control; it is a regression tripwire (below).

## The gradients against the native Vulkan tests (the flat control)

`run_native.sh` runs cloth-dynamics' own `test_avbd_{solver,gradcheck,stategrad}.exe` on **our** SPIR-V (`build/spv`). The logs are `native_*.log`. If the guest disagreed with these, the kernels would be ruled out and the guest driver blamed. It does not disagree.

gradcheck (L = 0.5·|x_out|², one step, central differences with h = 1e-3). The `rel` column is the native test's own measure, |a−fd| / max(1, |a|, |fd|):

| parameter | native analytic | cpu analytic | rd analytic | rd FD | rel (rd) |
|---|---|---|---|---|---|
| restLen | 0.370625 | 0.370625019 | 0.370624989 | 0.370400398 | 2.3e-4 |
| k_spring | −0.178328 | −0.178328186 | −0.178328171 | −0.17877008 | 4.4e-4 |
| k_attach | 3.4668 | 3.46679688 | 3.46679688 | 3.46690416 | 3.1e-5 |
| k_tri | −6.08297 | −6.08297062 | −6.08297062 | −6.08214668 | 1.4e-4 |
| k_bend | −5.62509 | −5.62508869 | −5.62508821 | −5.62573128 | 1.1e-4 |

The rd finite differences reproduce the native ones digit for digit, because both run on the same GPU and the same SPIR-V. The cpu finite differences differ in the 4th digit (for example k_spring −0.178484), because the interpreter's float rounding differs. Both are within 5e-2 of the analytic value.

Backward smoke (loss = x₀.z). Native prints these values at `%.3g`:

| | native | cpu | rd |
|---|---|---|---|
| dL/dx v0.z, v1.z | 0.149, 0.0298 | 0.148809582, 0.029761903 | 0.148809582, 0.0297618955 |
| dL/dk_tri, dλ0_tri.z | −0.49, 0.143 | −0.489795893, 0.142857134 | same |
| dL/dk_bend | −0.245 | −0.244897947 | same |
| spring and attachment | 0 | 0 | 0 |

stategrad (positions and predicted are finite-differenced separately; rel is taken against max(1, |fd|)):
- positions 0/12 and predicted 0/12 disagree on both backends;
- the bisect by constraint family (inertia, attachment, spring, triangle, bending) gives 0/12 for each family;
- the rd analytic columns are within 3.9e-6 of native, and the rd FD columns within 2.7e-6;
- this is 174 evaluations, one per tick on rd (181 ticks, 0.21 s).

## Padding: the race and its negative control

`AvbdRd` pads every vertex and constraint buffer to `roundUp(n+1, 64)`, the same capacity upstream's `AvbdSolverVk` uses. Index padding points at the dummy vertex nV, and the pad rows of hScratch hold the identity. Without this padding:
- the 63 ragged-tail lanes of `attachment_force_al_backward`, whose writes are 1:1 and unguarded, write `v_positions[vertIdx[c]]`;
- when the padding points at a real vertex, that becomes vertex 0.

`two_vertex_bwd_nopad` does exactly this with `setPadFillForTest(0)`. **The race reproduces on this GPU:**
- In both arms, exactly the 3 floats of vertex 0's positions gradient disagree. In this run: cpu (0.3333, 0.1283, 0.1283) against rd (1.2029, 1.0374, 1.0374).
- The other 26 floats are equal.
- With the default padding, `two_vertex_bwd` gives 0 of 29 floats different in both arms.

The rd values are a race, so they are not a fixed result: they vary from run to run. (The two gate runs recorded here, 01:45 and 02:54, happened to give the same values.) The job only asks whether any float disagrees, and it reports "race did NOT reproduce" as a negative result instead of failing.

## The dual snapshot cannot be tested (the job is a regression tripwire)

Both backends copy the triangle and bending duals before each step, and the backward binds the copies. This is an adaptation: upstream snapshots only the positions (`AvbdSolverVk.cpp` 1241–1247) and binds the live duals.

**The snapshot cannot be tested today**, because the two AL backward kernels never read the duals:
- `triangle_membrane_force_al_backward` and `triangle_bending_force_al_backward` declare `lambda0`/`lambda1`/`lambda` but never read them.
- The AL force is affine in the dual, with a coefficient that does not depend on the positions: dF/dx is constant for the membrane, and the stencil weights are constant for the bending.
- So no adjoint output depends on which duals are bound, and binding the live duals instead (`setLambdaSnapshotForTest(false)`) gives a bit-identical result (max |Δ| = 0).

What `gradcheck_duals` does check:
1. A warm-up `run(1, true)` makes the duals non-zero; they are live in the forward (L = 47.52 with warm duals, against 32.97 for the cold fixture).
2. The state is reset, and the measured `run(1, true)` is checked against FD over the start positions: 12/12 on both backends.
3. The live-dual arm must be bit-identical to the snapshot arm.

Step 3 is a tripwire, not a control: it cannot tell a right binding from a wrong one today. It fails the day a backward kernel starts reading the duals, which is when the snapshot starts to matter and needs a real test.

## Self-collision

| case | K | pairs | cpu = rd hash |
|---|---|---|---|
| 4 verts, r = 0.05, spacing 0.05 and 0.08 | 4 | {(0,1),(2,3)} exactly | 4fd51986a7e59503 |
| 6-vertex cluster, all in contact | 3 | {(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)}: each vertex keeps its first 3 j in ascending order | 8966e364a513bfa3 |
| 8×8 panel (spacing 1/7), r = 0.08 | 8 | 112 = the grid edges exactly, no diagonals (0.202 > 0.16) | 6a85798fccec1743 |
| 8×8 panel, r = 0.07 (the negative control) | 8 | 0 | 14650fb0739d0383 |

On rd the scan is `submitSelfCollisionScan` in one tick and `collectSelfCollisions` in the next. `AvbdRd` has no one-call `detectSelfCollisions` (it would sync in its submit's frame); `AvbdCpu` keeps one.

## Rule 4, measured

`rdc::Device` records the process frame at each `submit()` and counts every `sync()` that runs in the same frame.

After every job on both backends (339 submits and 339 syncs on rd), `same_frame_syncs` is 0. `rd_rule4_probe`, which submits and syncs in one call as the positive control, raises it from 0 to 1.

The earlier `_init`-driven gate (`run-split.log`) counted 223 of 223, which shows the counter has teeth.

The one-shot MCP entry points `avbd_fixture` and `avbd_bench` are cpu only now. On rd the same work is the jobs `fixture`, `bench_fwd` and `bench_bwd`, through `avbd_job_start` and `avbd_job_tick` (wrappers in `project/main.gd`).

## A finding that voided an assumption: RIDs are per-vmcall in godot-sandbox

The first frame-driven run threw on every rd job. The error was `Parameter "shader" is null`, and on the next call `uniform_set_create`: "Cannot convert argument 2 from PackedByteArray to RID".

`rd_compute.h` assumed that an RID is a plain integer that survives across vmcalls. It is not:
- godot-sandbox lists RID among the scoped Variant types (`GuestVariant::is_scoped_variant`).
- The guest therefore holds an index into the current call's Variant table, not the RID's id.
- In a later call, that index names whatever that call created at the same index.

The single-vmcall gate never showed this, because every RID it used was born and died inside one call.

The fix in `rd_compute`:
- Every RID the device returns is moved to permanent storage with `Variant::make_permanent`, which gives it a negative index that stays valid across calls.
- `free_rid()` releases the slot through `ECALL_VSTORE_GLOBAL`. The guest API has no wrapper for it, so `rd_compute.cpp` declares one.
- `forget()` releases the slot of an RID that Godot has already freed.
- `AvbdRd::invalidate_sets()` frees the uniform sets that are still valid and forgets the ones that are not, so slots do not leak. There are only 65534 permanent slots.
- `Device::permanent_slots()` counts the slots held; `rd_close` reports it, and the gate requires 0.

The bisect below shows this costs nothing measurable.

## The rd regression: a host name-cache collision (bisect)

The first frame-driven run had rd at about 0.9 s per substep, flat from 64 to 4096 vertices. The explanation then recorded here, that godot-sandbox's host-call path is bimodal between processes, was **refuted**: in one Godot process, HEAD's `dress_on.elf` (`fdf18e1`) ran `avbd_bench rd 32x32` at 13–18 ms per substep, while the new `drape.elf` took 600–1500 ms.

**The cause.** godot-sandbox resolves an Object call's method name through a 32-entry direct-mapped cache keyed by the guest **address** of the name string: slot = ((address · 2654435761) >> 8) & 31.
- A miss rebuilds the entry and drops the MethodBind cached beside it, so the call re-resolves through ClassDB, at 2–5 ms.
- Two hot method names whose literals the linker placed in the same slot evict each other on every call.
- The link layout decides it, not the code. HEAD had no hot pair in one slot. In the pre-fix builds, `compute_list_bind_compute_pipeline` shared a slot with `compute_list_dispatch` (dress_on) or with `compute_list_add_barrier` (drape). So every dispatch of the recording loop paid a re-resolve.

**The fix** (`guest/rd_compute.cpp`): every RenderingDevice method name that `rdc::Device` calls is copied once into a static pool, at an offset whose address has a slot of its own (27 names, 32 slots). The host still reads and compares the text, so the calls are unchanged.

The bisect: every ELF in its own Sandbox, in **one** Godot process, host-timed (`perf-bisect.log`: variant definitions, slot maps, all raw lines). Run A has all variants, 2 rounds; run B has fixed and head only, 4 rounds, fixed loaded first. The probe column leaves out each Sandbox's first call, which is warm-up (21,000–29,000 µs in every ELF).

| ELF | what differs | probe, 64 dispatches (µs) | submit loop (µs/submit) | bench rd 32×32 (ms/substep) | bench rd-batched 32×32 |
|---|---|---|---|---|---|
| head (`fdf18e1`) | | 7,300–10,900 | 59–98 | 8.2–9.9 | 7.7–8.6 |
| now (pre-fix) | | 151,000–180,000 | 92–142 | 491–702 | 499–509 |
| prefix (now, rebuilt) | | 154,000–167,000 | 2,392–2,617 | 490–492 | 483–509 |
| noperm | (a) no `make_permanent`/`VSTORE_GLOBAL` | 156,000–201,000 | 2,400–2,819 | 512–526 | 508–510 |
| noframe | (b) no `get_process_frames` at submit/sync | 158,000–190,000 | 2,459–2,747 | 496–518 | 500–512 |
| forced | the fix, with `dispatch` forced into `bind_compute_pipeline`'s slot | 148,000–167,000 | 62–185 | 758–817 | 769–849 |
| **fixed** | the fix | **630–2,070** | 62–175 | **7.1–8.5** (run A: 8.0, 17.1) | **6.3–6.9** (run A: 6.9, 11.6) |

The rows read as follows:
- Candidates (a) and (b) change nothing: noperm and noframe are as slow as prefix.
- Candidates (c) padding, (d) `buffer_clear`/`saxpby`, (e) the CMake link (`add_stage_elf`) and (f) `references_max` are identical in prefix and fixed, so they cannot account for a 60–100× gap.
- Putting one hot pair back into one slot (forced) brings the whole regression back. That is the causal test.
- The submit-loop column is the old "bimodal submit+sync" (~70 µs or ~2.4 ms, AGENTS.md). prefix's `dress_on.elf` has `compute_list_end` and `sync` in one slot and pays 2.4–2.8 ms per submit. now's has not and pays 92–142 µs. It is per build, not per process.
- The fixed ELF is **at or below HEAD** in the same process. In run B, round by round, rd 32×32 is 0.81–0.88× HEAD, rd-batched 32×32 0.80–0.84×, and 64×64 0.91–1.0×. The one slow fixed row (17.1 ms, 2.1× HEAD) is run A's second round, where fixed ran last of 13 Sandboxes; it did not recur in run B.
- A single RenderingDevice call now costs 0.8–2.4 µs (`rd_calls`, fixed) against 6.7–9.8 µs on HEAD.

## The bench: frame-driven, host-timed, and the cpu/rd threshold (rule 5)

The setup is the pinned panel under gravity: 10 substeps of 10 iterations at each size, one substep per tick.
- On rd, the readback of substep k and the submit of k+1 share a tick.
- On cpu, each substep yields.
- The timing uses the host clock passed in with each tick. It runs from the tick that submits substep 0 to the tick after the last readback: 11 frames for 10 substeps.
- The idle frame period is 0.41 ms.

| panel | verts | cpu fwd (ms/substep) | rd fwd | cpu bwd (runWithBackward) | rd bwd |
|---|---|---|---|---|---|
| 8×8 | 64 | 6.52 | 3.15 | 4.71 | 3.75 |
| 16×16 | 256 | 30.4 | 3.38 | 18.8 | 3.90 |
| 32×32 | 1024 | 76.8 | 3.75 | 80.6 | 4.27 |
| 64×64 | 4096 | — | 4.83 | — | 6.39 |

Every size is finite, and ymin matches free fall. Before the fix, rd was 814–970 ms per substep in this table.

**The threshold.** On the frame-driven path on the RTX 4090, rd is faster at every size measured, from 64 vertices up: forward 3.15 against 6.52 ms, backward 3.75 against 4.71 ms at 8×8. The verifier's rerun on the RTX 3090 (Godot's device #0 in that session) had cpu winning 3 of 4 comparisons at 8×8, which is also the bench's first size and absorbs warm-up; from 256 vertices rd won every run on both GPUs. So the drape uses rd from 256 vertices and cpu below; below 64 nothing is measured. rd is nearly flat (3.2 → 4.8 ms from 64 to 4096 vertices), so the frame-driven floor is set by the ~1,200 RenderingDevice calls per substep (the ab5 trace in `perf-bisect.log`: 1,600 pipeline binds, 1,600 set binds, 1,600 dispatches and 1,000 barriers per 5 substeps), not by the GPU.

The one-shot synchronous path crosses later: between 64 vertices (cpu 3.7–4.8, rd-batched 4.2–4.5 ms) and 256 (cpu 16.6–17.5, rd-batched 4.5–5.1 ms), from `perf-bisect.log` run B.

## Limitations (recorded, not gated)

- **The backward differentiates only the last iteration** of `run(iters, duals)` (and of `runWithBackward`). The earlier iterations and the dual updates are treated as constants. This is upstream's truncation.
- **The colored Gauss–Seidel backward is exact only for a single colour.**
  - `vbd_solve_apply_backward` and the scatter run over the whole mesh at once.
  - Within one iteration, a later colour's solve reads positions that an earlier colour just moved. The adjoint does not model that dependence.
  - Every FD check here uses the 4-vertex fixture, which is one colour (no `buildColoring`). The benches, which use 4 colours, check finiteness only.
- `run.log` ends with one leaked ObjectDB instance, down from 4. It is `JSONRPC` from the MCP addon: a no-op script in the same project leaks the same one (`--verbose`).

## Files

- `results.txt`, `run.log`: this gate.
- `perf-bisect.log`: the same-process A/B bisect of the rd regression.
- `run-split.log`: the forward gate after the ELF split, `_init`-driven.
- `native_*.log`, `run_native.sh`: the flat control.
- `import.log`: the ELF import.
- `project/gate_avbd.gd`: the frame-driven runner. `project/main.gd`: the no-argument wrappers.
- `guest/jobs.{h,cpp}`: the stage queue. `guest/drape/avbd_jobs.cpp`: the jobs. `guest/drape/main.cpp`: the job API and `rd_close`.
- `guest/rd_compute.{h,cpp}`: the GPU layer, with the method-name pool and the permanent-slot count.
- `guest/avbd/`: `avbd_cpu`, `avbd_cpu_backward`, `avbd_rd`, `avbd_rd_backward`, `avbd_topology`, `avbd_sim.h`, `cloth_grid`, `slang-rt`, `CITATION.cff`.
- `kernels/avbd/`: `gen.sh`, `kernels.txt` (25 kernels), the committed `slang/` and `cpp/`, and `AvbdKernelTable.inc`.
