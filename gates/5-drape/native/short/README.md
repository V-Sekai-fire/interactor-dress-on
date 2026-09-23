# Gate 5 native reference at 60 steps (G5's pre-chaos horizon)

**Result: recorded, and reproducible bit for bit.** This is the reference
G5 is gated against since Cut 5c. It is the sphere demo of
[`../README.md`](../README.md) (same checkout, same binary objects, seed 1,
no `AVBD_*` variables) run for 60 steps instead of 350.

## Why 60 steps

drape.elf reproduces native digit for digit through step 70, where the
self-collision onset sends one pair across the threshold (see
[`../../README.md`](../../README.md), "Where the sphere demo leaves native").
After that the trajectory is chaotic: moving μ by 4.4e-9 moves the 350-step
dL/dμ by 7.8%, and the cpu and rd backends differ by 7.8% at μ₀. A 5% gate on
the 350-step gradient therefore measures the chaos, not the port. The 60-step
horizon ends before the onset. It is also the horizon over which G7's
recompute modes recover the ground truth μ = 0.3.

## How it was made

`run_native_short.sh`. The native tool has no step-count option, because
`rotatingSphereScene.stepNum` is a constant. Its `backwardLog.txt` prints
dL/dμ with 5 decimals, which would read 0.0000x at this horizon, where the
loss is about 2e-6. The script leaves the standalone checkout and its
`build-win/` untouched. It copies two translation units into
`build/native_short/` and patches the copies:

- `OptimizationTaskConfigurations.cpp`: `rotatingSphereScene.stepNum = 60`.
- `OptimizeHelper.cpp`: after "Loss is:", one line
  `[g5-native] mu=%.17g loss=%.17g dL/dmu=%.17g`.

It compiles them with `build-win`'s own flags (`ninja -t commands`), links
them with `build-win`'s other objects, and runs the result from the
standalone root. The script checks first that the checkout is at `e361584`
and that `build-win` is up to date (`ninja -n`: no work to do). That is the
state in which `build-win/tool_cloth_dynamics.exe` hashes to the recorded
sha256 `b4f70230…`.

## The run (`s60/`)

| | |
|---|---|
| date | 2026-09-23, RTX 4090, 13 s and 19 s of wall time (two runs) |
| output dir | `s60/source_dir.txt` |
| config line | the same `[avbd-config]` as the 350-step run (iters 16, contact, friction prediction, self-collision passes 2, bwd AVBD truncate K 20) |

`s60/evaluations.txt` holds every evaluation made by the native L-BFGS:

| evaluation | μ | loss | dL/dμ |
|---|---|---|---|
| iter 0 | 0.5397701956236457 (μ₀) | 1.9988183e-06 | 0.026005736 |
| iter 1 | 0.010000000000000009 | 1.9070384e-05 | −0.23403413 |
| iter 2 | 0.23597573936785088 | 4.5123996e-07 | −0.016010246 |

As at 350 steps, the L-BFGS stops after one iteration, on the delta test.

**Reproduction.** The run was made twice. Both runs print identical
`[g5-native]` lines and identical per-step `[avbd-step]` statistics; only the
wall-time fields differ.

`s60/` also keeps `stdout.log`, `backwardLog.txt`, `iters.txt`,
`task_info.txt` and `scene-config.txt`.
