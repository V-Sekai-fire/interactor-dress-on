# Gate 5 native reference: the DiffCloth sphere demo on the AVBD build

**Result: recorded, and reproducible bit for bit.** This is the reference
drape.elf is checked against in Gate 5 (G4 frames, G5 dL/dμ, G7 μ sequence).
It is the upstream DiffCloth "rotating sphere" system-identification demo,
running on the AVBD solver of cloth-dynamics-standalone. The run was repeated
and matches the previous run byte for byte.

## Provenance

| | |
|---|---|
| source | `C:/cloth-dynamics-standalone`, commit `e361584c6e52a1b6f2635b93a8dff80eafc00052` ("Remove sparse Eigen from the simulation", 2026-09-22 13:29 -0700), clean tree, branch `feature/vulkan-avbd-windows-backend` |
| binary | `build-win/tool_cloth_dynamics.exe`, sha256 `b4f70230430d76226f4872593d103720e99583937891bba1f30a2a8c9a3b0bb3` |
| build | Release, Ninja, clang 23.1.1 (scoop `mingw-mstorsjo-llvm-ucrt`). Rebuilt at `e361584` for this run with the tree's own `ninja -C build-win` (19 steps, 31 s); the `build-win` binary already there was from 09:06, before `8a4965f`, and was stale |
| command | `cd C:/cloth-dynamics-standalone && ./build-win/tool_cloth_dynamics.exe -demo sphere -seed 1` (it must run from the repo root, which is where it looks for `src/assets/` and writes `output/`); no `AVBD_*` or `OMP_NUM_THREADS` variables set |
| date | 2026-09-23T00:24:34Z (2026-09-22 17:24 -0700); exit 0, 127 s wall |
| device | NVIDIA GeForce RTX 4090, Vulkan 1.4 |
| output dir | `output/rotating_sphere-randseed-1-20260922_172514-forwardThresh--9.0-LBFGS` (`source_dir.txt`) |

`e361584` exists only in the local checkout: no remote branch contains it. The
checkout's only remote is `v-sekai/TOOL_cloth_dynamics`, which is under
V-Sekai, not V-Sekai-fire.

The solver config line from `stdout.log`:
```
[avbd-config] solver=AVBD iters=16 damp=1 relax=1 colors=1 drive=1 | membrane=1 bending=1 rawStiffness=0 | al=0 gamma=default | contact=1 frictionPred=1 selfColl=1 passes=2 gpuSelf=1 | bwd=AVBD truncateK=20 ift=0
```

## What the run does

The first thing the tool builds is a hat-scene system (579 vertices), which the
demo does not use. It then builds the sphere scene: 625 vertices, 1152
triangles, 1680 bendings, 1875 DoF, h = 1/180 s, 350 steps. `stdout.log`
contains four forward passes of 350 steps:

| pass | μ | stdout lines | loss | dL/dμ (`backwardLog.txt`) | exported as |
|---|---|---|---|---|---|
| target | 0.300000 (ground truth) | 39–2279 | — | — | not exported |
| LBFGS iter 0 | 0.539770 (seed-1 initial guess) | 2305–4560 | 0.00132918 | 0.01153 | `iter0/` |
| LBFGS iter 1 | 0.010000 | 4568–6850 | 1.65198194 | -50.45588 | (`iter1/param.txt`) |
| LBFGS iter 2 | 0.375146 | 6858–9116 | 0.00047714 | 0.00781 | (`iter2/param.txt`) |

LBFGS stops after 1 iteration with x = 0.375146 and f = 0.000477142.
`iter3/` is the same μ exported again. No step reports `nan`.

**Watch out: `iter0/` is μ = 0.539770, not the ground truth.** The μ = 0.3
target trajectory is generated inside the process and never written to disk.
To compare frames with this directory, drape.elf must run at μ = 0.539770. To
compute the G5 loss, it has to generate its own μ = 0.3 target.

Per-step `[avbd-step]` lines (`|Δx|_max` for each step) come from the target
pass:

| step | 1 | 10 | 100 | 350 |
|---|---|---|---|---|
| `|Δx|_max` | 0.000302433 | 0.00302272 | 0.0413634 | 0.0277782 |

Steps 1 and 10 print the same `|Δx|_max` in all four passes.

## Reproduction

A previous run with the same command existed:
`output/rotating_sphere-randseed-1-20260922_132726-forwardThresh--9.0-LBFGS`,
the latest of 28. It was made at 13:27 with the `build-fix` binary, which
`ninja -n` reports as up to date at `e361584`. The new run reproduces it
exactly (`compare_132726.log`):

- Frames 0, 1, 10, 100 and 350: max |dx| = 0, and the faces are identical.
- `iter0` to `iter3` (354 files each) and `last_frame_meshes` (5 files) are
  byte-identical.
- `backwardLog.txt`, `task_info.txt`, `scene-config.txt` and `iters.txt` are
  identical.
- `forwardLog.txt` differs only in its runtime lines. This run's forward pass
  took about 25 s per iteration against 8.4 s before, because other builds
  were using the CPU at the time.

Earlier runs: `iter0` frames 1, 10, 100 and 350 of every run since 09:16 are
identical to this one. The `backwardLog.txt` of every run since 12:14 is
identical.

**Control: how much μ moves the frames.** This uses the same run's `iter0`
against its other iterations:

| frame | 1 | 10 | 20 | 50 | 100 | 350 |
|---|---|---|---|---|---|---|
| μ 0.54 vs 0.01 | 0 | 1.79e-4 | — | — | 0.212 | 3.64 |
| μ 0.54 vs 0.375 | 0 | 4.1e-5 | 1.87e-4 | 7.76e-3 | 0.135 | 0.180 |

**Frames 1 and 10 barely depend on μ,** because friction has not acted yet.
Frame 1 is identical for every μ. At frame 10 even μ 0.01 is only 1.8e-4 away,
which is inside G4's 1e-3 tolerance. So G4's control, "μ 0.3 diverges from the
native frames", cannot be judged at frame 1 or 10. It has to be judged at frame
50 or later (7.8e-3 at frame 50 between μ 0.54 and 0.375, and 0.13 at frame
100).

## Files

- `run_native.sh` runs the demo and collects the output. `FROM=<output dir>`
  collects an existing run without re-running it.
- `stdout.log` is the full stdout and stderr, including ANSI colour codes.
- `iter0/` holds frames 0, 1, 10, 50, 100 and 350 of the cloth (625 vertices,
  1152 faces) plus `1-SPHERE.obj` and `param.txt`, about 0.36 MB. The run wrote
  all 351 frames (13 MB); only the frames G4 is judged at are kept, and
  `run_native.sh` collects only those. Frame 50 is where the μ control first
  separates (see above).
- `iter{1,2,3}/param.txt` hold the μ iterate sequence (G7).
- `backwardLog.txt`, `forwardLog.txt`, `task_info.txt`, `scene-config.txt`,
  `iters.txt` and `perf.txt` are copied unchanged from the output dir.
- `compare_frames.py` computes per-frame max |dx| and face equality for two
  `iter0`-style directories. `compare_132726.log` holds its results.
