# Gate 6G.1: rd_compute from the fit worker thread

**Result: PASS, with two negatives that change how the fit must hold its GPU.**
`results.txt`, `results-run2.txt` (RTX 3090, idle) and `results-4090.txt`
(RTX 4090, shared with another job at 30–45% load) each end
`SUMMARY: PASS=67 FAIL=0 INFO=15`.

**The question.** PolyFEM stays the fit (decided 2026-09-23) and moves its
arithmetic to the GPU through `rd_compute`. fit.elf runs each phase as one
vmcall on a GDScript worker Thread. Can such a vmcall do these things?

- create and hold an `rdc::Device`;
- dispatch Lean kernels, sync and read back;
- do both while the main thread renders and drape.elf uses RD on the main
  thread.

And what does one round trip cost from there?

**The answers.**

- **Yes.** A worker-thread vmcall opened a device and ran 5,445 round trips of
  Lean kernels, per run. Every one of the 3,630 checked readbacks was exact.
  Meanwhile the main thread rendered 400 shadowed spheres at 60 Hz. With
  drape.elf's rd job also ticking on the main thread, every drape job passed
  and its `same_frame_syncs` stayed 0.
- **Cost.** One round trip (submit + sync + a 16-byte readback) takes
  **259–283 µs p50** on the worker and 268–333 µs as a mean inside one vmcall.
  Ending the round trip in `buffer_get_data` alone takes **137–162 µs p50**.
  Each dispatch recorded into the list adds 6–9 µs.
- **Budget.** 10% of a Newton iteration's native time allows about **80–150
  round trips of up to 10 dispatches** for the 932-vertex skirt, each ending in
  `buffer_get_data`; with submit + sync + read it is 53–83. For foxgirl it
  allows about 300–550.
- **Negative 1: the device is bound to its thread.** Godot ties a local
  RenderingDevice to the OS thread that created it. A device opened on the
  main thread returns nothing to a worker. A device opened by one worker
  Thread returns nothing to the next Thread, and `stage_base.start()` makes a
  new Thread per call.
- **Negative 2: opening a device holds up the main thread.** Opening one on a
  worker stalls the main thread's frame for about **200–290 ms**, and freeing
  it for 30–60 ms. GDScript without a sandbox does the same.

**Recommendation.**

- PolyFEM's synchronous Newton loop should **sync on the worker thread,
  inside the phase vmcall**. The fiber/WAIT_GPU pattern would pay a frame,
  16.4 ms, per round trip.
- The fit stage needs **one persistent worker Thread that owns the device for
  the whole session**. Feed it through a queue, and open and free the device
  on it once.
- Rule 4 needs a scope: the main thread. The proposed wording is below.

Setup: Godot 4.7.2-stable, the vendored `godot_sandbox` addon, Ryzen 7 3800X,
Vulkan, vsync on a 60 Hz display, `--xr-mode off`. `--gpu-index 1` gave the
3090 and `--gpu-index 0` the 4090 on this boot; the log's adapter line and the
first results line name the device. Other agents shared the machine.

```
BUILD_DIR=<dir> BUILD_FIT=0 BUILD_TARGETS=rd_worker ./build.sh          # project/rd_worker.elf
godot --path project --headless --import                                   # first run after adding the ELF
godot --path project --script gate_rd_worker.gd --rendering-driver vulkan --xr-mode off --gpu-index 1 \
      -- --reps=300 --out=results.txt > gates/6g-polyfem-gpu/g1-rd-worker/run.log 2>&1
```

`project/rd_worker.elf` sha256 `50c7dea5723fc496d626063335e0fc18779c9c95397ead6c05e0af38c0bf444e`.

## What a round trip is

`rd_worker.elf` (`guest/rd_worker/main.cpp`) records k dispatches into one
compute list:

- k − 1 `saxpby` dispatches, a ping-pong chain with a barrier after each;
- then one `dot_reduce`, the df32 dot, into a 16-byte result buffer.

The chain is exact by construction. After D saxpbys every element is D, so
the dot must read back n·D (n = 2,796 = 3 × 932, the loop skirt's vector).
Every readback is compared with that value. A round trip ends one of three
ways:

| mode | ending | fences |
|---|---|---|
| `sync_get` | `submit` + `sync` + `buffer_get_data` of the 16-byte result | 2 (`buffer_get_data` flushes and stalls on its own) |
| `get` | `buffer_get_data` alone | 1 |
| `sync` | `submit` + `sync`, nothing read | 1 |

Each "single" round trip is one vmcall, timed on the host (300 per cell, after
5 warm-ups). Each "batch" is 300 round trips in one vmcall, the shape a
Newton loop inside a phase vmcall has. Nothing is timed in the guest.

## Verdicts

| arm | what | verdict |
|---|---|---|
| A | guest, main thread (control) | **PASS**: 3,630 exact readbacks + 1,815 unread round trips. The ladder held the main thread 2.1–2.8 s, and 0 frames passed |
| FA | flat control: the same kernels and calls from GDScript, main thread, no sandbox | **PASS** |
| N1 | device opened on the main thread, used from a worker vmcall | **PASS (negative reproduced)**: codes `[-2, -2, -2]` (nothing came back). The log has `This function (compute_list_begin) can only be called from the render thread.` and the same for `bind_*`, `dispatch`, `compute_list_end`, `submit`, `sync` and `buffer_get_data`. Back on the main thread the device is intact: the next round trip is exact and the close leaves 0 permanent slots |
| FN1 | N1 in GDScript, no sandbox | **PASS (negative)**: codes `[-2, -2]`, the same errors. The refusal is Godot's, not the sandbox's |
| R0 | the scene alone, 6 s | INFO: frames p50 16.66–16.68 ms, p99 17.1–20.5 ms, 0 over 25 ms |
| **B** | guest, worker Thread: open, ladder, close on one Thread | **PASS**: 3,630 exact + 1,815 unread. The main thread ran at 56–57 fps. The only long frames were at the open (205–282 ms) and the close (40–59 ms) |
| FB | flat control on a worker Thread | **PASS** |
| O | 3 × (`rw_open` + `rw_close`) on a worker, then 3 × bare GDScript `create_local_rendering_device` + `free` | **PASS**, and **negative 2**: every create lines up with a 199–269 ms main-thread frame and every free with a 33–60 ms one, in the guest and GDScript alike. On the worker, an open takes 213–327 ms and a close 57–96 ms |
| N2 | device opened by worker Thread 1, used by Thread 2 | **PASS (negative reproduced)**: Thread 1 is exact, Thread 2 gets `[-2, -2]`. The device is orphaned: no thread can free it (`free_rid` and teardown are guarded too). It is the likely source of the logs' "1 resources still in use at exit": that line appeared when N2 ran alone, and in neither of two runs without N2 (A alone; B, C0, C and E). The leaked ObjectDB count at exit is 10 for the full run, 7 for N2 alone and 2 for A alone |
| Q | one persistent worker Thread fed by a Mutex + Semaphore queue: `rw_open`, three `rw_rounds`, `rw_stats` and `rw_close` posted 10 frames apart | **PASS**: 1 thread id for all six vmcalls; one device held from open to close; 230/230 exact |
| C0 | scene + drape.elf `bench_fwd` on rd, one tick per frame, 10 s | **PASS**: 11 jobs PASS, `same_frame_syncs=0`. Frames p50 16.6 ms, p99 77–84 ms. All 13–14 long frames overlap a drape tick over 20 ms (job restarts) |
| **C** | C0 + B's worker ladder at the same time | **PASS**: the worker got 3,630 exact readbacks and the drape 3 jobs PASS with `same_frame_syncs=0`. Frames p50 16.6 ms. Of 175–181 frames, 6–7 were long: the open, plus frames that overlap a drape tick over 20 ms. One 33 ms frame in run 2 overlaps neither |
| E | frame-paced on the main thread: `rw_submit` in frame f, `rw_collect` in f+1 (rule 4, and what a fiber yielding WAIT_GPU per submit costs) | **PASS**: 120/120 exact, `same_frame_syncs=0`. Submit to result **16.43–16.45 ms p50** |

## What a round trip costs (µs)

Worker thread, guest (arm B). p50 / p90 / p99 of single round trips, and the
batch mean; ranges over the three runs:

| k | mode | p50 | p90 | p99 | batch mean |
|---|---|---|---|---|---|
| 1 | sync_get | 259–283 | 392–512 | 1,203–1,333 | 268–333 |
| 1 | **get** | **137–162** | 165–306 | 468–761 | **147–197** |
| 1 | sync | 115–145 | 151–268 | 485–592 | 123–161 |
| 10 | sync_get | 297–375 | 396–648 | 1,255–1,318 | 342–417 |
| 10 | **get** | **200–242** | 270–426 | 775–841 | **218–269** |
| 10 | sync | 178–214 | 237–351 | 865–1,070 | 188–241 |
| 100 | sync_get | 830–1,050 | 1,091–1,503 | 1,670–2,071 | 844–1,119 |
| 100 | get | 692–961 | 874–1,183 | 1,204–1,534 | 744–1,075 |
| 100 | sync | 695–1,110 | 909–1,427 | 1,269–1,923 | 778–1,067 |

**Against the controls** (the full tables are in each results file):

- **Worker vs main thread.** The worker's p50 is the main thread's (A):
  139–160 µs at k=1 get. Its tail is longer (p99 468–761 against 256–593 µs),
  in GDScript as well (FB 450–962): that is the OS scheduling a second busy
  thread, not the sandbox.
- **Guest vs GDScript (the sandbox's share).** At k=1 the guest is within
  ±20 µs of GDScript (A 139–160 against FA 142–156 p50, get). Per dispatch
  recorded, the guest costs 5.5–8.9 µs against GDScript's 4.0–6.2 µs:
  (k=100 − k=1) / 99 at the batch mean. Each dispatch is four host calls
  (bind pipeline, bind set, dispatch, barrier).
- **Under load (C).** With the main thread rendering and draping, the worker's
  k=1 get is 155–166 µs p50 and 180–184 µs mean. The p99 rises to 870–895 µs.
  k=100 rises to ~1 ms.
- **Frame-paced (E).** 16.4 ms per round trip at 60 Hz, **60–120× the
  worker's**.

**A model that fits every arm:** a round trip ≈ 115–160 µs (one fence and a
16-byte readback) + 6–9 µs per dispatch recorded. A second fence
(`sync_get` against `get`) adds ~110–160 µs.

## How many round trips a Newton iteration can afford

The iteration times come from committed gates:

- **foxgirl** (2,682 vertices), phase 0 native: 33.6 s / 41 Newton =
  **820 ms** (Gate 6.P).
- **the loop's skirt** (932 vertices), in the guest: 1,393 s / 253 Newton =
  5.5 s (Gate 8). Divided by Gate 6.P's per-Newton guest/native ratio of 24.6,
  that is **≈ 224 ms native**. The skirt was never run natively (Case C of the
  uncommitted Amdahl notes), so this figure is derived, not measured.

At 10% of that time, using the worker's batch means (B):

| round trip | skirt, 22.4 ms | foxgirl, 82 ms |
|---|---|---|
| k=1, get | 113–152 | 415–556 |
| k=1, sync_get | 67–83 | 245–306 |
| k=10, get | 83–102 | 304–376 |
| k=10, sync_get | 53–65 | 196–239 |
| k=100, get | 20–30 | 76–110 |
| frame-paced (E) | **1** (1.4 frames is the whole budget) | 5 |

The same budget in one formula, for the skirt: R round trips and D dispatches
per iteration fit when R × 0.15–0.2 ms + D × 0.006–0.009 ms ≤ 22 ms. For
example, R = 20 with D ≈ 2,000, or R = 100 with D ≈ 250.

- **A Newton iteration needs far fewer.** It needs one readback per
  line-search trial energy, one for the CCD step bound, one for the
  convergence norm, and one every m CG iterations if the CG scalars stay on
  the GPU (lean/Cloth's `CGBeta`). That is on the order of 5–20 per iteration.
- **The budget shrinks as the GPU takes work.** It is 10% of today's CPU
  iteration. If the GPU makes the iteration 10× faster, the same round trips
  cost 10× more of it. The structural rule stays: keep scalars on the GPU and
  read back only what the CPU branches on.
- **Against the guest, the cost is negligible.** The fit's CPU side is
  interpreted: a skirt Newton iteration is 5.5 s in the guest. Against that,
  100 round trips of 0.3 ms are 0.5%.

## Findings that change code elsewhere

1. **A local RenderingDevice is bound to the OS thread that created it.**
   - **The mechanism:** Godot's `RenderingDevice` stores `render_thread_id`
     at construction, and `ERR_RENDER_THREAD_GUARD` refuses other threads.
     The guarded calls include:
     - every `compute_list_*` call;
     - `buffer_get_data`, `buffer_update`, `buffer_copy` and `buffer_clear`;
     - `submit` and `sync`;
     - `free_rid`, and `finalize` (the device's own teardown).
   - `make_current()` would move a device to another thread, but it is not
     bound to scripts.
   - **What a refused call does:** it logs `can only be called from the
     render thread` and returns null or empty. The guest sees an empty
     readback.
   - **Consequence for `stage_base.start()`:** it makes a new Thread per
     call. A device opened in one fit phase's Thread is unusable, and
     unfreeable, in the next.
   - **Consequence for the fit stage:** either it keeps one persistent worker
     (arm Q) or it opens and frees a device inside every phase vmcall.
2. **Opening a device holds up the main thread** for 200–290 ms, and freeing
   one for 30–60 ms, from a worker as well.
   - The GDScript control shows the same, with no shader or buffer: it is
     device creation itself.
   - Where the main thread waits was not instrumented.
   - In XR a stall of that size is a visible freeze. Open the fit's device
     where a hitch is acceptable (at stage open, behind a loading state), and
     keep it for the session, not per phase.
3. **The rule-4 counter is not a rule-4 signal on a worker.**
   - `rdc::Device::sync()` counts a sync as same-frame when
     `Engine.get_process_frames()` has not moved since its submit.
   - On the worker, every sync counts: 3,630 of 3,630 in B and C. Yet no
     frame waited (B's main thread ran 56–57 fps with 0 long frames during
     the ladder).
   - **Proposed, not done here** (`rd_compute` is shared by every ELF):
     - `rdc::Device` notes the thread that opened it (the host's
       `OS.get_thread_caller_id()`, through its name table);
     - it counts `same_frame_syncs` only when that thread is the main thread;
     - it reports a worker's syncs as `worker_syncs`.
   - Until then, read a worker's `same_frame_syncs` as "syncs made on a
     worker".
4. **End a round trip in one `buffer_get_data`, not submit + sync + read.**
   - `buffer_get_data` does its own flush and stall (`_flush_and_stall_for_all_frames`),
     so a submit + sync before it is a second fence: +110–160 µs.
   - It stages the whole source buffer (Gate 0F finding 3), so read a small
     result buffer.

## Recommendation: how PolyFEM's Newton loop should talk to the GPU

**Sync on the worker thread, inside the phase vmcall.** Do not use the fiber
/ WAIT_GPU pattern of ggml-rd.

- **The fiber protects a thread the fit does not block.** The fiber exists so
  that a guest running on the *main* thread never blocks a frame. fit.elf
  already runs on a worker. B and C show that a worker's sync leaves the main
  thread alone: frame p50 16.6 ms, and no long frame outside the device open
  and close, and the drape's own restarts.
- **The fiber is 60–120× dearer per round trip.** A frame-paced round trip is
  16.4 ms (E), against 0.15–0.4 ms on the worker. A 20-round-trip iteration
  would wait 330 ms for frames, against 3–8 ms.
- **PolyFEM is synchronous C++.** polysolve's line search and ipc-toolkit's
  CCD call and wait. Yielding at every GPU call would thread a fiber through
  both for no gain.

**The shape:**

1. **One fit worker Thread per session, fed by a queue** (arm Q: Mutex +
   Semaphore). It replaces `stage_base.start()`'s Thread per call for the fit
   stage. The host posts each phase as a job and polls for the result. It
   never joins a live Thread.
2. **Open the device once, on that Thread,** at stage open. It costs
   ~0.21–0.35 s there and a 200–290 ms main-thread frame, so do it where that
   is acceptable. Keep the device across phases, and free it on the same
   Thread at session end: the free is guarded too, and costs ~0.06–0.1 s and
   a 30–60 ms frame.
   - **Fallback with today's `stage_base`:** open and free a device inside
     every phase vmcall. It works, but costs that hitch per phase: fine on the
     desk, not in a headset.
3. **Round trips:** batch the dispatches of each GPU step into one compute
   list and end in one `buffer_get_data` of a small result buffer. Keep
   reductions and CG scalars on the GPU (df32, Lean kernels), and read back
   only what the Newton loop branches on. Plan for about 5–20 round trips per
   iteration; the skirt's budget is ~80–150.
4. **A failed round trip must say so.** A guarded refusal returns an empty
   readback, not an exception. Check the size of every readback (as
   `read_check` does) and fail the phase loudly.

## Proposed rule-4 wording (AGENTS.md is not edited here)

Now:

> 4. **State machines and queues, not waits.** Never `sync()` in the frame that
> `submit()`s; the host advances each guest's state machine from `_process`
> and reads back once the fence is known-done. Batch iterations into one
> compute list (`AvbdRd::run`).

Proposed:

> 4. **State machines and queues, not waits, on the main thread.** Nothing on
> the main thread `sync()`s in the frame that `submit()`s: the host advances
> each main-thread guest's state machine from `_process` and reads back once
> the fence is known-done. A guest that runs on its stage's own worker Thread
> (fit.elf) may submit, sync and read back inside its vmcall, because that
> wait blocks only the worker (Gate 6G.1). Its RenderingDevice is created,
> used and freed on that one OS thread (Godot refuses a local device on any
> other), so such a stage keeps one persistent worker fed by a queue. The host
> polls that worker and never joins a live one. Batch dispatches into one
> compute list per round trip (`AvbdRd::run`), and end a round trip in one
> `buffer_get_data` of a small result buffer.

And two lines for "Facts that cost time":

> - A local RenderingDevice works only on the OS thread that created it:
>   compute lists, `buffer_get_data`/`update`/`copy`/`clear`, `submit`, `sync`
>   and `free_rid` log "can only be called from the render thread" and return
>   nothing on any other (`make_current` is not bound). A new GDScript Thread
>   is a new OS thread.
> - Creating a local RenderingDevice stalls the main thread's frame
>   ~200–290 ms (freeing it ~30–60 ms), even from a worker thread
>   (gates/6g-polyfem-gpu/g1-rd-worker).

## Kernels (AGENTS.md rule 2)

Both kernels are Lean's, taken from the drape set that `kernels/drape/gen.sh`
embeds into `drape_kernels.inc`:

- `saxpby` is `Cloth.SlangCodegen.Saxpby`;
- `dot_reduce` is `Cloth.SlangCodegen.DotReduce`, a df32 (two-sum /
  two-prod) dot with a 256-thread tree.

Nothing was hand-written or copied in. The flat control runs the same SPIR-V,
which the guest hands over through `rw_spirv`.

## Files

- **Evidence:**
  - `results.txt`, `results-run2.txt` and `results-4090.txt`: the three runs,
    streamed.
  - `run.log`, `run2.log` and `run-4090.log`: their stdout and stderr, with
    the adapter line, the render-thread errors of N1, FN1 and N2, and the
    exit-time leak of N2's orphaned device.
  - `wrappers.txt`: `tests/probe_main_wrappers.gd` with rd_worker.elf added to
    the audit: 122 of 122 entry points wrapped. It is written to
    `gates/8-loop/` and copied here; Gate 8's own file is left as it was.
- **`guest/rd_worker/main.cpp`:** the probe ELF, 8 entry points. `rw_open`,
  `rw_round`, `rw_rounds`, `rw_submit`, `rw_collect`, `rw_stats` and
  `rw_close` go through `rdc::Device` only. `rw_spirv` is for the flat
  control.
- **`CMakeLists.txt`:** the `rd_worker` target. `build.sh` lists
  `rd_worker.elf`.
- **`project/gate_rd_worker.gd`:** the gate. It is frame-driven, with a wall
  clock in every branch, and polls Threads without ever joining a live one.
- **Rule 8:**
  - `project/main.gd`: the `rw_*` delegates, every argument defaulted.
  - `project/stages/dress_on_stage.gd`: `rw()`, the Sandbox made on first use.
  - `project/tests/wrapper_audit.gd`: maps `guest/rd_worker/main.cpp` to
    `dress_on`.
