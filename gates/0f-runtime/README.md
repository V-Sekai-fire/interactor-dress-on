# Gate 0F — what the godot-sandbox runtime actually does

**Result: PASS, with nine findings that change later cuts.**
`results.txt`: `SUMMARY: PASS=31 FAIL=5 INFO=27 DEFERRED=0` in 263 s. The five
FAIL lines are all probe 3: the guest cannot open a file by any path. That is
a real negative result, not a harness fault. Every probe has a control.
Cut 4 added probe 17 and rebuilt every ELF against the fixed aligned
allocator. A full re-run gave `SUMMARY: PASS=33 FAIL=5 INFO=27 DEFERRED=0`
in 164 s (`results-memalign.txt`): the same five probe-3 FAILs, and probe
17's two PASS lines. The table's other numbers are from `results.txt`.

This is the corrected run. A verifier found three claims of the first run
wrong or untested (the memory ceiling, the `references_max` rows, the
in-place loss) and one plan probe missing (set-0 sharing). Probes 6 and 13
now use a fresh Sandbox per arm, probe 16 is new, and the in-place loss has
a root cause.

Setup: Godot 4.7.2-stable, the vendored `godot_sandbox` addon, Ryzen 7
3800X, Vulkan. This machine has an RTX 4090 and an RTX 3090, and Godot's
device #0 changes between sessions. The first run got the 4090; this run got
the **3090**. The machine is shared with other agents, so wall times are
noisy (probe 8's speed-up was 1.68× in the first run and 1.21× here).
`probes.elf` is one guest ELF in its own Sandbox. It shares only
`rd_compute` and runs only Lean-emitted kernels (below).

```
godot --path project --script gate_runtime.gd --rendering-driver vulkan --xr-mode off [-- --only=N,M]
```

## Verdicts

| # | probe | verdict | numbers |
|---|---|---|---|
| 1 | exceptions through `std::function` | **PASS** | Thrown through two `std::function`s plus a typed rethrow, and caught. Control (no throw) returns 11. |
| 2 | `fesetround` | **INFO: not honoured** | `fegetround` reads back UP=3 and DOWN=2, yet 1/3 comes out `3fd5555555555555` under UPWARD, DOWNWARD and NEAREST alike (float too). |
| 3 | file I/O from the guest | **FAIL (negative)** | `ifstream` and `fopen` both fail on an absolute path (`/` and `\`), `res://`, a relative path and `/etc/hostname`: `errno=9 EBADF`. libriscv refuses `openat` unless `permit_filesystem` is set, and godot-sandbox never sets it. Control: GDScript `FileAccess` reads the same file (403 B). |
| 4 | threads | **PASS** | `hardware_concurrency=2`; spawn and join work (value 42). Gate 0C already showed they run serialized. |
| 5 | `execution_timeout` | **PASS** | It is `instructions_max` (reading one reads the other); default 8000, in units of 2^20 instructions. A 1e12-step loop is killed at every limit: lim 1 → 8 ms, 10 → 29 ms, 100 → 208 ms, 1000 → 1.81 s. The sandbox stays usable after a kill. Control: 1e6 steps match the GDScript LCG exactly, with 0 timeouts. The first call runs at 110 M steps/s and later calls at 831 M steps/s. At the warm rate, 2 s of work (1.66e9 steps) finishes at the default limit (2.18 s). The first run's "~2 s" row used the first-call rate and so ran only 342 ms. |
| 6 | `memory_max` ladder | **PASS** | Default 512 MiB. It survives `program=` (set 2048 first and it reads 2048 after). Each arm is a fresh Sandbox with the limit set before `program=`. X MiB under a limit of 2X: 64 to **2048 MiB ok**. **Control:** at a limit of X/2 the allocation is refused for every X from 64 to 2048. **Ceiling:** the heap is 0.8 × `memory_max` and must end below 4 GiB. 4096 → 3275 MiB ok, 3277 null; 5000 → 3999 ok; 5100 → 4079 ok; **5112 → 4088 MiB ok**, 4090 null. At 5114 and above the program does not load (`Native heap exceeds 32-bit address range`). **After `program=`:** raising the limit applies (2048 → 1024 MiB ok). Lowering it is **ignored** (32 → 256 MiB still ok); 32 set before `program=` refuses 256. |
| 7 | vmcall on a GDScript `Thread` | **PASS** | A 227 ms guest loop on a Thread. The main thread advanced 6 frames meanwhile and the result matched. Control: the same call on the main thread advances 0 frames. |
| 8 | two Sandboxes on two Threads at once | **PASS** | Both results are exact against serial references. Serial 195 ms vs concurrent 161 ms, a **1.21×** speed-up with no crash (1.68× in the first run, on a less loaded machine). |
| 9 | typed argument echo | **PASS** (unboxed) | With `unboxed_arguments=true` (the default) every value round-trips bit-exactly: 9 floats including −0, subnormal, max, NaN and ±inf; 7 ints including 2^53+1 and INT64_MIN; bools; a UTF-8 string with a non-BMP code point; PackedFloat32Array; PackedByteArray. With `false`, typed signatures read garbage (every float → 0.0) and only `Variant` signatures work (25/25). |
| 10 | heap readings | **PASS** | `get_heap_usage` 74 736 → 67 183 600 after `p_hold(64)` (+64.00 MiB exactly) → 74 736 after release. `monitor_heap_*` agree. |
| 11 | **fiber across vmcalls** | **PASS** | Hand-written riscv64 switch (`guest/fiber/`). 100 rounds of {+1 dispatch, submit, yield WAIT_GPU}; the next vmcall, one frame later, resumes and syncs (rule 4). It read back 100 over 101 vmcalls, with throw/catch inside the fiber at round 50 and nothing escaping. Control: the same job as a state machine also gives 100. |
| 11b | a RID kept across vmcalls | **PASS** | A 16-byte buffer's RID kept in a guest static. Made permanent (`rdc::Device` does this for every RID it returns), it reads back exactly in the next vmcall. Left as the host scoped it (`set_permanent_rids(false)`, a gate hook), it works in its own call; in the next call that call throws `GuestVariant::toVariantPtr(): 23 (RID) idx=1 is not known/scoped` (`run.log`). This reproduces the first run's fiber failure. |
| 12 | RD buffers up to 4 GiB−256 | **PASS** | Each buffer is created empty (`storage_buffer_uninit`), cleared with `buffer_clear`, then run through a whole-buffer Lean `saxpby` (up to 4 194 304 groups). The last element reads back as 4.5 and the rest as 0 at 256 MiB, 1 GiB, 2 GiB and **4 GiB−256**, all exact (14–184 ms). The readback goes through `buffer_copy` into a 16-byte buffer; see finding 3. Host `buffer_update` into the guest-held device: 1 GiB in 564 ms including submit+sync, **1.9 GB/s** on this run's 3090 (3.4 GB/s on the 4090 in the first run). |
| 13 | `references_max` under 10k dispatches | **PASS** | 10 000 dispatches × 3 binds + 2 499 barriers in one vmcall, four ping-pong counters, one fresh Sandbox per arm with setup split over four vmcalls. All counters read exactly 2500 at `references_max` **100** (73 ms), **4096** (97 ms) and **65536** (54 ms). So recording (binds, dispatches, barriers on permanent RIDs) uses no scoped references. It does need `execution_timeout` raised: at the default 8000, n=4000 (~17k host calls) passes and n=10000 (~42.5k) is killed. **Control:** creating 12 uniform sets and 28 buffers in one vmcall trips at 100, whether 100 is the default or set after `program=`, and passes at 4096 set before or after. **The limit only grows:** set 65536 then 100 and the Sandbox reads 100, yet the same setup passes. |
| 14 | f16 storage read on the GPU | **PASS** | Lean-emitted `half_load` (`lean/Probes/HalfLoad.lean`, pinned by `native_decide` to its own text and to lean-slang `emit-fp`'s `halfLoadShader`). 64 halves cover ±0, normals, max, min normal, subnormals, ±inf and NaN, and match a CPU decode bit for bit (0 mismatches, NaN→NaN). |
| 15 | ggml-cpu in the guest | **PASS** | `V-Sekai-fire/ggml` @04b55bba, static, rv64gc, 1 thread, no OpenMP/llamafile/RVV/Zfh/Zvfh/Zicbop/Zihintpause. 256³ f16×f32 `mul_mat` + `soft_max`, two calls with identical results: **190 and 215 ms** in the full run, but **637 and 675 ms** alone (`--only=15`, `results-only15.txt`). The first run saw the same split (218 vs 685 ms). The solo process is ~3× slower on both calls; probe 5's first-call vs warm rates show the same kind of gap, which fits libriscv still compiling in the background (this was not isolated). The worst relative difference of four checksums against the llvm-mingw native build (`ggml_host.txt`, AVX2/FMA/F16C) is **1.7e-8** (limit 1e-6). Control: a Zfh TU (`fmadd.h`) **traps** in the full run. Alone it returns **1.5 with no trap, where 4.875 is right**. Either way Zfh is not executed correctly. |
| 16 | set-0 uniform sets shared across pipelines | **PASS** | Lean `probe_add` and `probe_scale` (`lean/Probes/Set0.lean`) share Cut 3's layout: b0 params, b1–b3 sources, b4 destination. Built with **`slangc -O0 -preserve-params`**, a set made for `probe_add` binds under `probe_scale` and gives 256/256 exact. **Control:** `-O0` without the flag drops the unused sources, and Godot refuses the dispatch (`Uniforms supplied for set (0) ... not the same format`); D is unchanged. **`-preserve-params` at the default `-O1` is also refused:** the optimiser strips the sources again (`gen.log`: 3 bindings, not 5). One dispatch with the same RID read-only at b1 and read-write at b4 is exact (256/256). |
| 16 | in-place ops across barriers | **PASS** (hazard reproduced, with controls) | 1000 rounds of +1 over 4096 elements, one compute list, a barrier after every round, twice per shape. **Aliased** (X at b1 read-only and b4 read-write): 0/4096 exact, values 12–477 where 1000 is right. **Read-only first:** X read by one dispatch, then written by the next, in the same span between barriers: 0/4096, values 449–622. **Controls, all 4096/4096:** the same +1 with X bound once read-write (`probe_acc`); ping-pong between two buffers; the write dispatch first, then the read. See finding 4. |
| 17 | aligned allocation (added with Cut 4) | **PASS** (bug reproduced by the control) | `results-memalign.txt`. 1000 blocks at each of 64, 128 and 4096 alignment, rotating through `posix_memalign`, `aligned_alloc`, `memalign` and aligned `operator new`, each filled with its own tag byte. A third of the steps free a random live block (997) and every seventh reallocs one (206). Of 3000 blocks, 0 are misaligned, 0 overlap and 0 are corrupted, and 0 calls return null. The heap reads 74736 bytes before and after (after a warm-up call). The whole call takes 49 ms. **Control:** the same sequence through a copy of upstream's fallback. 2953 of 3000 calls exhaust the 16 tries and return a block they have already freed. Among the 2963 blocks still held, 2953 are misaligned, 2945 overlap and 2938 are corrupted. The fix is in `vendor/sandbox-api` (see its `CITATION.cff` and AGENTS.md). |

## What these verdicts set

- **`memory_max`:** set it before `program=` on a fresh Sandbox (a lower
  value later is ignored). At most ~5112 MiB, which gives a ~4088 MiB heap
  (0.8 × the limit, ending below 4 GiB). Weights stay out of the guest heap,
  as planned; that is now a hard limit as well as a design choice.
- **infer.elf can use the fiber.** Straight-line job code with WAIT_GPU
  yields works across vmcalls, and exceptions work inside it. It relies on
  permanent RIDs (probe 11b), which `rd_compute` now provides.
- **File feed:** the guest has no filesystem at all. Every byte comes through
  the host: `FileAccess` → `rd.buffer_update` for weights (1.9–3.4 GB/s), or
  a PackedByteArray argument for small inputs.
- **fit.elf worker model:** there is no real parallelism inside one guest
  (Gate 0C: threads are serialized). There is parallelism across guests:
  separate Sandboxes on GDScript Threads run concurrently (1.2–1.7× for two)
  and leave the main thread free. Run PolyFEM single-threaded
  (`POLYFEM_THREADING=NONE`). Split work across Sandboxes if it pays.
- **`IPC_TOOLKIT_WITH_FILIB=OFF`.** filib's interval bounds rely on directed
  rounding, and the guest silently rounds to nearest whatever `fesetround`
  reports.
- **ggml in the guest:** rv64gc only, never `GGML_RV_ZFH`. Add
  `-U__riscv_v_intrinsic` to ggml-cpu (finding 5).
- **Cut 3's set-0 layout works, with two conditions.** Compile at
  `-O0 -preserve-params` (or make every kernel touch every binding). In-place
  ops bind the buffer once, read-write (finding 4).
- **`references_max`:** set 4096 or more before the first object-creating
  vmcall. Long recordings do not need more; raise `execution_timeout`
  instead.

## Findings that change code elsewhere

1. **A RID held across vmcalls must be made permanent.** A RID the host
   returns reaches the guest as a *scoped* Variant index, valid for one
   vmcall. Probe 11b reproduces the failure (`idx=1 is not known/scoped`).
   Main's `rd_compute` now makes every returned RID permanent and releases
   the slot in `free_rid`/`forget`. The probes' own `keep()` is removed, and
   every RenderingDevice call in `probes.elf` goes through `rdc::Device`
   (whose method-name pool avoids the host's 32-slot cache collisions).
   `rd_compute` gains `storage_buffer_uninit` (probe 12's empty 4 GiB
   buffer), `object()` (the device handed to the host) and
   `set_permanent_rids` (the probe 11b hook; default on).
2. **A killed vmcall leaves its compute list open.** Reproduced in probe 13:
   after a `refs_run` killed by `execution_timeout`, the next `buffer_update`
   is refused (`Updating buffers is forbidden during creation of a compute
   list.`). `p_list_end` recovers, and the setup then passes. `rd_compute`
   needs a recovery path on this pattern.
   **Done in Cut 3:** `rdc::Device` keeps a list-open flag in guest memory,
   which survives the killed call, and ends an orphaned list before its
   next `buffer_update`/`buffer_copy`/`buffer_clear`, `list_begin` or
   `submit`. Probe 13 now kills the call twice: with the recovery off
   (`p_recovery(false)`, a gate hook) the next `buffer_update` is refused as
   above; with it on, it goes through with no `p_list_end` and one recovery
   counted. That run (`gates/3-ggml-rd/regression/0f-runtime-results.txt`)
   reads `SUMMARY: PASS=34 FAIL=5`, the five being probe 3's; `results.txt`
   and `results-memalign.txt` here are the runs before the change.
3. **`buffer_get_data` stages the whole source buffer.** One 4-byte direct
   read adds ~60 ms at 256 MiB, ~200 ms at 1 GiB and ~410 ms at 2 GiB over the
   copy path. From the host, a 4-byte read out of a 1 GiB buffer takes
   199–245 ms. At 4 GiB−256 it fails (`Can't create buffer of size:
   4294967040 (VkResult error -2)`). Read through `buffer_copy` into a small
   staging buffer instead: exact at every size.
4. **A buffer's first binding in a compute list sets its usage for the whole
   list.** This is a real Godot hazard, not a driver race and not a probe
   bug.
   - **The mechanism:** `compute_list_add_barrier` is `compute_list_end()` +
     `compute_list_begin()`, so every span between barriers is its own
     render-graph command. In that command,
     `RenderingDeviceGraph::add_compute_list_usage` keeps one usage per
     buffer: the first one recorded. Development builds reject a second,
     different usage (`Tracker can't have more than one type of usage in the
     same compute list`); release builds, like 4.7.2-stable, drop it
     silently.
   - **What goes wrong:** if a buffer is first bound read-only, the span is
     recorded as only reading it. The graph then does not order later spans
     after its writes, so they overlap on the GPU and increments are lost.
   - **The evidence:** probe 16's aliased and read-only-first shapes lose,
     while the read-write-only, ping-pong and read-write-first shapes are
     exact. The same arithmetic on the same buffer bound once is exact, and
     each thread touches only its own element, so no dispatch races with
     itself. The first run's saxpby shape (y at b2 read-only, dst at b3
     read-write, one buffer) still loses in probe 13: 710/674/195/605 of
     2500.
   - **The rules:**
     - Within one span between barriers, a buffer that is written anywhere
       must be bound read-write the first time it appears.
     - ggml-rd in-place ops (Cut 3) must not bind one buffer as a b1–b3
       source and as the b4 destination. They take an in-place kernel
       variant with the one read-write binding (`probe_acc`), or ping-pong.
     - AVBD's forward recording (`AvbdRd::record_iteration`,
       `record_duals`) was checked against this rule and passes: positions,
       lambdas and the scratch rows are never read-only-first within a span
       that writes them. The backward recording was not checked against the
       rule; it does match its oracle on rd.
5. **clang predefines `__riscv_v_intrinsic` for every riscv64 target,** with
   or without V. ggml-cpu keys its RVV paths on that macro alone, so at rv64gc
   it fails on `vfloat32m8_t`. `CMakeLists.txt` undefines it for ggml-cpu
   only. Only ggml's libraries are rv64gc (their CMake sets it). The probe
   TUs build at sandbox-api's `-march=rv64gcv_zba_zbb_zbs_zbc`, and
   `zfh_probe.cpp` at `rv64gc_zfh`.
6. **Host calls are charged against `execution_timeout`.** Roughly 0.2–0.5M
   instructions of budget go per host call: ~42k host calls exhaust the
   default 8000 in ~25–50 ms of wall time. A long recording in one vmcall
   needs the budget raised; 1e6 was used here.
7. **`references_max` is consumed by object and Variant creation** (buffers
   with data, `RDUniform`s, returned RIDs) within one vmcall, not by
   recording calls on permanent RIDs. The capacity only grows: a value lower
   than one already set reads back but is not applied. The first run's row
   "exact at 4096" had been run at 65536 capacity. It is now tested in a
   fresh Sandbox and passes.
8. **`memory_max`: 0.8 × the limit, ending below 4 GiB, and it only grows
   after `program=`.** The first run's "any limit ≥ 4096 fails to load" was
   wrong: 4096 loads, and so does 5112. Its X/2 control had set a lower
   limit after `program=`, which is ignored, so the rows at 512 and above
   were untested, and its "floor" (256 MiB under a 32 MiB limit) was the
   same artefact.
9. **`slangc -preserve-params` does nothing at the default optimisation
   level** (slangc 2026.13.1). The SPIR-V optimiser removes the unused
   bindings again. It takes `-O0 -preserve-params`
   (`kernels/probes/gen.sh`), or kernels that touch every binding.

## Differences from the plan text

- The plan's ladder is "set after `program=`". Setting it after
  `program=` applies only when the limit grows, so the ladder sets it before
  `program=` on a fresh Sandbox per arm. Loading does not reset a value set
  earlier.
- The plan's ladder ran to 16 GB. The ceiling is ~5112 MiB of `memory_max`
  (~4088 MiB of heap), so larger limits are recorded as load failures.
- Probe 14 was allowed to be DEFERRED. It ran, because `emit-fp` was pushed
  with the HalfLoad fixture before this gate.
- Cut 3's "`-preserve-params`" needs `-O0` with it (finding 9).

## Kernels (AGENTS.md rule 2)

Every kernel `probes.elf` runs comes from `lean/` through
`lake exe emit_probes` (`lean/EmitProbes.lean`):

- `saxpby`: `Cloth.SlangCodegen.Saxpby`, the AVBD kernel;
- `half_load`: `lean/Probes/HalfLoad.lean`;
- `probe_add`, `probe_scale`, `probe_acc`: `lean/Probes/Set0.lean`.

Each is pinned by `native_decide`. By default `kernels/probes/gen.sh`
re-emits them and **fails if any differs** from the committed
`kernels/probes/slang/`; `--update` writes them and `--no-emit` skips Lean.
`gen.log` has the check, a negative control (a tampered `half_load.slang`
fails with the diff) and the binding counts per slangc flag set.
`build.sh` runs `gen.sh --no-emit` unless `PROBES_EMIT=1`.

## Files

- `results.txt` is the full run, streamed; `run.log` is its stdout and
  stderr. `run.log` holds the scoped-RID exception, the refused
  `buffer_update`, the refused set format, the Zfh trap and the
  heap-ceiling exceptions. `results-only15.txt` and `run-only15.log` are
  probe 15 alone, where Zfh silently returns 1.5. `import.log` is the ELF
  import, and `gen.log` is the kernel check. Absolute paths in the logs are
  replaced by `<project>`, `<repo>` and `<godot-sandbox>`.
- `ggml_host/` is the host-native twin: `build.sh` → `ggml_host.txt`, with
  `GGML_SRC` defaulting to `vendor/ggml` (vendored by Cut 3; probes.elf
  builds against it unconditionally since).
- `project/gate_runtime.gd` is the frame-driven runner, with a wall-clock
  quit. `project/main.gd` has the no-argument wrappers for all 37 entry
  points (rule 8; `p_recovery` came with Cut 3).
- `guest/probes/`:
  - `main.cpp`;
  - `ggml_probe.cpp`, shared with the native twin;
  - `zfh_probe.cpp`, the only Zfh TU.
- `guest/fiber/`: `fiber.h`, `fiber.cpp` and `fiber_riscv64.S`.
- `kernels/probes/`: `gen.sh`, `kernels.txt` and `slang/*.slang`.
- `lean/Probes/` and `lean/EmitProbes.lean` are the kernels' source.
