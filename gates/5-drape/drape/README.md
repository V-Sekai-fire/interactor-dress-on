# Gate 5, Task D: the drape simulation and its backward, in drape.elf

**Result: FAIL, on one criterion.** G4 passes. G6 passes. G5 passes at μ 0.01,
and misses its 5e-2 tolerance at μ 0.539770 (7.9%) and μ 0.375146 (5.1%). The
flat control shows why. The same port on the other backend moves dL/dμ by
18% and 21% at those two μ. Nothing in the backward differs between the two
backends: the spread comes from the forward's own sensitivity. The native
number sits inside that spread (details below).

The run: `results.txt` and `run.log`, 2026-09-23, Godot 4.7.2, RTX 4090 (the
GPU the native reference ran on).
`results_rtx3090.txt` is an earlier run of the same gate on the machine's
other GPU (without the cpu G5 control), and its numbers are bit-identical.

```
godot --path project --script gate_drape.gd --rendering-driver vulkan --xr-mode off --gpu-index 1 > ../gates/5-drape/drape/run.log 2>&1
```

## What was built

The drape runs in `guest/drape/`, with no Eigen, over `AvbdCpu` and `AvbdRd`.
- `drape_config.h` mirrors cloth-dynamics' AvbdConfig and is printed into
  every log. Its defaults are iters 16, colours on, AL off, contact and the
  friction predictor on, self-collision with 2 passes and K = 16 neighbours,
  bwd truncation K = 20, and clipping 16·nV.
- `drape_scene.*` builds the sphere demo exactly as DiffCloth does. It also
  takes a mesh from the host (positions, triangles, pins and a material).
- `primitives.*` holds Sphere, Plane and Capsule, each with isInContact and
  the projection Jacobian.
- `drape_sim.h` is the staged step. Each step keeps a record of:
  - the predictor s, which is what upstream's records and OBJ frames hold;
  - s_blend, x_avbd, x and v;
  - the friction sensitivities;
  - the contacts and pushes, in the order they were applied.
- `drape_backward.h` has two losses, MATCH_TRAJECTORY and TARGET_POINTS, and
  three modes:
  - **native** is the upstream port. It works on the last solver state and the
    last iteration only, with K = 20, clipping, dL/dv = 0, and dL/dμ taken from
    the previous record. Its μ carry is added twice per step, as upstream
    does.
  - **step** recomputes each step and differentiates the last iteration.
  - **unrolled** recomputes each step and unrolls all 16 iterations from
    per-iteration snapshots.

  Both recompute modes chain through the self-collision pushes, the
  projection, the velocity response, v = (x_avbd - x)/h and the friction
  blend.
- `drape_session.h` and `drape_jobs.*` hold the stage-queue session behind the
  `drape_*` API, and four jobs: `sphere_forward`, `sphere_backward`,
  `sim_gradcheck` and `bench_drape`.
- `project/main.gd` has a no-argument wrapper for each entry point. The gate
  drives them in its "main.gd wrappers" check.

## G4: the forward against the native sphere demo

Faces: our 1152 triangles equal those of `native/iter0/0.obj`, in order.
Frame 0 is equal to 4e-22.

Max |dx| against native `iter0`, at μ 0.539770:

| frame | 1 | 10 | 50 | 100 | 350 |
|---|---|---|---|---|---|
| rd, 350 steps | 1.2e-10 | 4.9e-6 | 6.4e-6 | 4.0e-3 | 0.12 |
| cpu, 100 steps | 1.2e-10 | 4.9e-6 | 6.2e-6 | 0.082 | — |
| limit | 1e-5 | 1e-3 | | | |

Up to frame 50 the differences are the OBJ files' 6-digit print resolution.
The per-step |Δx|_max matches the native `[avbd-step]` lines to every printed
digit for steps 1–5, 10, 20 and 50 (rd, in `results.txt`). The host harness
(cpu) also matches steps 60 and 70. Step 80 first differs, in the sixth
digit:

| step | 1 | 2 | 3 | 20 | 50 | 80 (host cpu) |
|---|---|---|---|---|---|---|
| ours | 0.000302433 | 0.000604831 | 0.000907192 | 0.00604185 | 0.0150594 | 0.0283020 |
| native | 0.000302433 | 0.000604831 | 0.000907192 | 0.00604185 | 0.0150594 | 0.0283022 |

The friction-event counts per step match too, starting with one event on the
second step.

The two runs part ways at the self-collision onset. Upstream resolves its
first pairs on the 69th step, and so do we (host cpu). We record 5 pushes there,
upstream 6. From then on, a pair that lands at the threshold goes one way or
the other, and the difference grows. cpu against rd, which is the same code
in a different float order, grows the same way, to 0.082 at frame 100.

**Control:** μ 0.3 sits further from the native frames than ours at μ 0.54
does:

| frame | μ 0.3 vs native | ours vs native |
|---|---|---|
| 50 | 6.8e-3 | 6.4e-6 |
| 100 | 0.144 | 4.0e-3 |
| 350 | 0.184 | 0.122 |

All 350 steps are finite on rd, and all 100 on cpu.

## G5: native-mode dL/dμ against `native/backwardLog.txt`

The target is our own μ 0.3 run of 350 steps, because the native run never
wrote its target to disk. The loss is MATCH_TRAJECTORY, with K = 20.

| μ | backwardLog | ours (rd) | rel | cpu (control) | rd vs cpu | loss native / ours |
|---|---|---|---|---|---|---|
| 0.010000 | −50.45588 | −50.674986 | **0.43%** | −51.088782 | 0.8% | 1.65198 / 1.64973 |
| 0.539770 | 0.01153 | 0.012444 | 7.9% | 0.010230 | 17.8% | 0.00132918 / 0.00141238 |
| 0.375146 | 0.00781 | 0.0074085 | 5.1% | 0.0058319 | 21.3% | 0.00047714 / 0.00060653 |

Why the tolerance holds at μ 0.01 and not at the other two:
- dL/dμ comes from the friction events of the last 20 steps (frames 331–350).
- Upstream adds the carry twice per step, so the last step weighs 2^19 times
  the 20th-last. Summed once instead, the sums are 2.8e-7, −1.3e-3 and 1.7e-7.
- At μ 0.54 and 0.375 the gradient is a small difference between two
  trajectories that are both past the chaotic self-collision onset. At
  μ 0.01 it is large and robust.

The losses carry the same spread (6% and 27% against native), and so does
our own cpu-against-rd control (18% and 21%). The 5e-2 tolerance is tighter
than the forward's reproducibility at those two μ.

Evidence that the port is right, not merely close:
- The μ 0.01 agreement is 0.4%.
- The double-carry reproduction is right. Without it, the answer would be
  about 10^4 times smaller.

## G6: unrolled mode against central finite differences (single colour)

The scenes are an 8x8 panel of 1 m and 20 steps at h = 1/180. The loss is
MATCH_TRAJECTORY against the same scene run at (μ 0.25, kTri 110,
density 0.36). The base point is (0.4, 150, 0.3).
- panel: pinned at two corners, no primitive.
- plane: unpinned, moving at (0.5, −0.4, 0.2) m/s, over a plane tilted 5.7°
  that cuts through one side of it. Every step has contact, projection and
  friction.

The FD step is relative, 1e-3. rel = |a − fd| / max(|a|, |fd|).

| scene | param | FD | unrolled | rel | step (rel) | native (rel) |
|---|---|---|---|---|---|---|
| panel | kTri | 1.7123e-7 | 1.7122e-7 | 9.5e-5 | 0.98 | 0.99 |
| panel | density | −8.5636e-5 | −8.5634e-5 | 1.8e-5 | 0.98 | 1.0 |
| panel | μ | 0 | 0 | 0 | 0 | 0 |
| plane | μ | 5.1155e-5 | 5.1210e-5 | 1.1e-3 | 0.81 | 0.99 |
| plane | kTri | 6.8150e-9 | 6.5186e-9 | 0.044 | 0.94 | 1.0 |
| plane | density | −3.3318e-6 | −3.2594e-6 | 0.022 | 0.94 | 1.0 |

The plane's kTri and density FD values move by 4% and 1.5% between eps and
10·eps (6.815e-9 against 6.536e-9), because contact makes the loss only
piecewise smooth. Unrolled sits between the two FD values.

**Multi-colour (information: this picks the default mode).**
- On the panel, unrolled is still within 1.8e-4. That is Gauss-Seidel with
  weak coupling.
- On the plane it drifts to rel 0.58 on kTri and 0.38 on density, but only
  0.015 on μ. The backward treats each iteration as Jacobi.
- step and native are off by 80–100% everywhere. They differentiate one
  iteration, and native does so on one state for every step.

**Default: unrolled.** Use single colour whenever stiffness gradients have to
be exact.

## Rule 4, rule 5 and shutdown

- `rd_rule4` shows `same_frame_syncs=0` of 5986 syncs after every job. The
  probe raises it, 0 → 1.
- `rd_close` gives `CLOSED device=freed permanent_slots=0`.
- The rd steps are frame-driven: about 2.8 ticks per step (solve, contact and
  scan, then pass 2). That is 967 ticks for 350 steps.

`bench_drape` measures ms per step on the host clock, frame-driven, with the
sphere-demo scene at n x n:

| n x n (nV) | 8x8 (64) | 16x16 (256) | 24x24 (576) | 32x32 (1024) | 48x48 | 64x64 |
|---|---|---|---|---|---|---|
| cpu | 8.0 | 41.1 | 100.4 | 178.1 | — | — |
| rd | 16.1 | 17.7 | 15.8 | 19.2 | 28.6 | 40.6 |

- rd has a floor of about 16 ms per step, which is three frames of submit and
  readback.
- cpu crosses it between 64 and 256 vertices.
- `auto` picks rd from 256 vertices, which is rule 5's threshold from
  gates/2-avbd, confirmed here.

## Found on the way (it cost the most time)

A `std::vector<std::vector<T>>` stored in a record, and kept alive in a
`std::vector` of records across stages, corrupted guest libc state. The
symptoms were an "Illegal opcode" at an `ecall` in `memmove`, `memcpy`,
`puts` or `fflush`, with garbage syscall numbers. This happened with one
empty inner vector.

What does not trigger it:
- the same nested vector as a local;
- a flat vector of the same element type kept in the record.

The host build of the same code under ASan and UBSan is clean
(`tests/drape_host`). So the push log, the MATCH_TRAJECTORY target and the
unrolled snapshots are all flat arrays now. The root cause, in the sandbox's
heap or its memmove/memcpy host calls, is not isolated.

## Files

- `results.txt` and `run.log`: this gate on the RTX 4090.
- `results_rtx3090.txt`: the same on the RTX 3090, bit-identical.
- `../native/`: the reference (see its README).
- `tests/drape_host/`: a host-native harness for the same sources, for
  ASan/UBSan and fast iteration. It gives no verdict.
