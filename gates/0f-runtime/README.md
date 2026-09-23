# Gate 0F — what the godot-sandbox runtime actually does

**Result: PASS, with seven findings that change later cuts.**
`results.txt`: `SUMMARY: PASS=25 FAIL=5 INFO=23 DEFERRED=0` in 152 s. The five
FAIL lines are all probe 3: the guest cannot open a file by any path. That is
a real negative result, not a harness fault. Every probe has a control.

Setup: Godot 4.7.2-stable, the vendored `godot_sandbox` addon, RTX 4090
(Vulkan), Ryzen 7 3800X. `probes.elf` is one guest ELF in its own Sandbox.
It shares only `rd_compute` and uses two Lean-emitted kernels (below).

```
godot --path project --script gate_runtime.gd --rendering-driver vulkan --xr-mode off [-- --only=N,M]
```

## Verdicts

| # | probe | verdict | numbers |
|---|---|---|---|
| 1 | exceptions through `std::function` | **PASS** | Thrown through two `std::function`s plus a typed rethrow, and caught. Control (no throw) returns 11. |
| 2 | `fesetround` | **INFO: not honoured** | `fegetround` reads back UP=3 and DOWN=2, yet 1/3 comes out `3fd5555555555555` under UPWARD, DOWNWARD and NEAREST alike (float too). |
| 3 | file I/O from the guest | **FAIL (negative)** | `ifstream` and `fopen` both fail on a Windows absolute path (`/` and `\`), `res://`, a relative path and `/etc/hostname`: `errno=9 EBADF`. Control: GDScript `FileAccess` reads the same file (403 B). |
| 4 | threads | **PASS** | `hardware_concurrency=2`; spawn and join work (value 42). Gate 0C already showed they run serialized. |
| 5 | `execution_timeout` | **PASS** | It is `instructions_max` (reading one reads the other); default 8000. A 1e12-step loop is killed at every limit: lim 1 → 9 ms, 10 → 29 ms, 100 → 193 ms, 1000 → 1.76 s (about 0.24e9 four-instruction steps, so **1 unit ≈ 1M instructions**). The sandbox stays usable after a kill. Control: 1e6 steps match the GDScript LCG exactly, with 0 timeouts. |
| 6 | `memory_max` ladder | **PASS** | Default 512 MiB. **Not reset by `program=`** (set to 2048 first and it reads 2048 after). Allocate-and-touch X MiB under a limit of 2X: 64 MiB to **2048 MiB ok**. Any limit ≥ 4096 fails to load: `Native heap exceeds 32-bit address range`. At 4095: 3072 MiB ok, 3584 MiB null. Control: with the limit at X/2, the allocation is refused for every X ≥ 512. There is a floor: at a 32 MiB limit, 256 MiB still succeeds and 512 MiB does not. |
| 7 | vmcall on a GDScript `Thread` | **PASS** | A 256 ms guest loop on a Thread; the main thread advanced 19 frames meanwhile and the result matched. Control: the same call on the main thread advances 0 frames. |
| 8 | two Sandboxes on two Threads at once | **PASS** | Both results are exact against serial references. Serial 296 ms vs concurrent 177 ms: **1.68× speed-up**, no crash. |
| 9 | typed argument echo | **PASS** (unboxed) | With `unboxed_arguments=true` (the default) every value round-trips bit-exactly: 9 floats including −0, subnormal, max, NaN and ±inf; 7 ints including 2^53+1 and INT64_MIN; bools; a UTF-8 string with a non-BMP code point; PackedFloat32Array; PackedByteArray. With `false`, typed signatures read garbage (every float → 0.0) and only `Variant` signatures work (25/25). |
| 10 | heap readings | **PASS** | `get_heap_usage` 74 736 → 67 183 600 after `p_hold(64)` (+64.00 MiB exactly) → 74 736 after release. `monitor_heap_*` agree. |
| 11 | **fiber across vmcalls** | **PASS** | Hand-written riscv64 switch (`guest/fiber/`). 100 rounds of {+1 dispatch, submit, yield WAIT_GPU}; the next vmcall, one frame later, resumes and syncs (rule 4). Read back 100 over 101 vmcalls, with throw/catch inside the fiber at round 50 and nothing escaping. Control: the same job as a state machine also gives 100. |
| 12 | RD buffers up to 4 GiB−256 | **PASS** | Created empty, `buffer_clear`, a whole-buffer Lean `saxpby` (up to 4 194 304 groups), last element read back as 4.5 and the rest as 0: 256 MiB, 1 GiB, 2 GiB and **4 GiB−256** all exact (76–209 ms). The readback goes through `buffer_copy` into a 16-byte buffer; see finding 3. Host `buffer_update` into the guest-held device: 1 GiB in 317 ms, **≈3.4 GB/s** including submit+sync (0.7 GB/s cold at 64 MiB). |
| 13 | `references_max` under 10k dispatches | **PASS** | 10 000 dispatches × 3 binds + 2 499 barriers in one vmcall, four ping-pong counters: all exactly 2500, at `references_max` 4096 (123 ms) and 65536 (83 ms). They need `execution_timeout` raised: at the default 8000, n=4000 (~17k host calls) passes and n=10000 (~42.5k) is killed. The recording itself holds no references, so even 100 passes. Control: setup, which creates 12 uniform sets and 28 buffers, trips at 100 (default or set after `program=`) and passes at 4096. |
| 14 | f16 storage read on the GPU | **PASS** | Lean-emitted `half_load` (lean-slang `emit-fp` @e0e96da, `TestFp.halfLoadShader`, pinned by `native_decide`): 64 halves covering ±0, normals, max, min normal, subnormals, ±inf and NaN, bit-exact against a CPU decode (0 mismatches, NaN→NaN). |
| 15 | ggml-cpu in the guest | **PASS** | `V-Sekai-fire/ggml` @04b55bba, static, rv64gc, 1 thread, no OpenMP/llamafile/RVV/Zfh/Zvfh/Zicbop/Zihintpause. 256³ f16×f32 `mul_mat` + `soft_max` in **218 ms**. The worst relative difference of four checksums against the llvm-mingw native build (`ggml_host.txt`, AVX2/FMA/F16C) is **1.7e-8** (limit 1e-6). Control: a Zfh TU (`fmadd.h`) **traps** in the full run. Alone (`--only=15`, `results-only15.txt`) it returns **1.5 with no trap, where 4.875 is right**. Either way Zfh is not executed correctly. |

## What these verdicts set

- **`memory_max`:** at most 4095 (MiB). About 3 GiB is usable in one guest,
  and the setting survives `program=`. Weights stay out of the guest heap, as
  planned: that is now a hard limit as well as a design choice.
- **infer.elf can use the fiber.** Straight-line job code with
  WAIT_GPU yields works across vmcalls, and exceptions work inside it. This
  holds only with the RID rule below.
- **File feed:** the guest has no filesystem at all. Every byte comes through
  the host: `FileAccess` → `rd.buffer_update` for weights (≈3.4 GB/s), or a
  PackedByteArray argument for small inputs.
- **fit.elf worker model:** there is no real parallelism inside one guest
  (Gate 0C: threads are serialized). There is parallelism across guests:
  separate Sandboxes on GDScript Threads run concurrently (1.68× for two) and
  leave the main thread free. Run PolyFEM single-threaded
  (`POLYFEM_THREADING=NONE`). Split work across Sandboxes if it pays.
- **`IPC_TOOLKIT_WITH_FILIB=OFF`.** filib's interval bounds rely on directed
  rounding, and the guest silently rounds to nearest whatever `fesetround`
  reports.
- **ggml in the guest:** rv64gc only, never `GGML_RV_ZFH`. Add
  `-U__riscv_v_intrinsic` to ggml-cpu (finding 5).

## Findings that change code elsewhere

1. **A RID held across vmcalls must be made permanent.** A RID the host
   returns reaches the guest as a *scoped* Variant index (`RID idx=3 is not
   known/scoped`). A guest static keeping it names whatever the next call
   scopes at that index; the first fiber run got a PackedByteArray where a
   shader RID was expected. `Variant(rid).make_permanent()` fixes it
   (`keep()` in `guest/probes/main.cpp`). **`rd_compute.h`'s comment "what
   persists is RIDs, which are integers" is wrong.** Stage 2 never noticed
   because AvbdRd creates and uses everything inside one vmcall. Cut A's jobs
   and Cut 3's cached uniform sets need this.
2. **A killed vmcall leaves its compute list open.** Every later
   `list_begin`/`buffer_update`/`buffer_clear` on that device is then refused
   (`Only one compute list can be active`), which cascaded through four probes
   on the first run. `rd_compute` needs a recovery path (`p_list_end` here).
3. **`buffer_get_data` stages the whole source buffer.** One 4-byte read adds
   ~35 ms at 256 MiB, ~185 ms at 1 GiB and ~300 ms at 2 GiB over the copy
   path; from the host, a 4-byte read out of a 1 GiB buffer takes 175–340 ms. At 4 GiB−256 it
   fails (`Can't create buffer of size 4294967040`, VK -2). Read through
   `buffer_copy` into a small staging buffer instead: exact at every size.
4. **In-place read-modify-write across barriers loses writes.** The same
   10 000-dispatch recording, with `y` and `dst` bound to one buffer
   (bindings 2 and 3 of one set), read 497/499/502/490 instead of 2500 each (other runs: 245–786).
   Ping-ponging between two buffers is exact. Stage 1's `accumulate` (one RW
   binding) was exact with barriers, so the loss is tied to one buffer bound
   twice in one set. ggml-rd in-place ops (Cut 3) must not alias src and dst.
   AVBD kernels should be audited for the same pattern.
5. **clang predefines `__riscv_v_intrinsic` for every riscv64 target,** with
   or without V. ggml-cpu keys its RVV paths on that macro alone, so at rv64gc
   it fails on `vfloat32m8_t`. `CMakeLists.txt` undefines it for ggml-cpu
   only.
6. **Host calls are charged against `execution_timeout`.** Roughly 0.2–0.5M
   instructions of budget go per host call: ~42k host calls exhaust the
   default 8000 in ~50 ms of wall time. A long recording in one vmcall needs
   the budget raised; 1e6 was used here.
7. **`references_max` is consumed by object and Variant creation** (buffers
   with data, `RDUniform`s, returned RIDs), not by recording calls. Setting it
   after `program=` takes effect. Lowering it live after the first call did
   not trip 256 uniform-set creations. Whether that is because creations are
   cheap or because a live change is not applied was not separated.

## Differences from the plan text

- The plan said memory_max is "set after `program=` (loading resets it)".
  Here it was **kept** across `program=`, and both orders gave the same
  ceiling.
- The plan's ladder ran to 16 GB. The ceiling is **< 4 GiB** (32-bit native
  heap), so 4–16 GiB are recorded as load failures.
- Probe 14 was allowed to be DEFERRED. It ran, because `emit-fp` was pushed
  with the HalfLoad fixture before this gate.

## Files

- `results.txt` (the full run, streamed) and `run.log` (stdout and stderr,
  with the Zfh trap and the heap-ceiling exceptions). `results-only15.txt`
  and `run-only15.log` are probe 15 alone, where Zfh silently misreturns 1.5.
  `import.log` is the ELF import.
- `ggml_host/`: the host-native twin (`build.sh` → `ggml_host.txt`, built in
  `C:/b/ggml-host-0f`).
- `project/gate_runtime.gd`: the frame-driven runner, with a wall-clock quit.
  `project/main.gd`: the no-argument wrappers for all 29 entry points
  (rule 8).
- `guest/probes/`: `main.cpp`, `ggml_probe.cpp` (shared with the native
  twin) and `zfh_probe.cpp` (the only Zfh TU). `guest/fiber/`: `fiber.h`,
  `fiber.cpp`, `fiber_riscv64.S`.
- `kernels/probes/`: `gen.sh`, `kernels.txt` (with provenance), and
  `slang/{saxpby,half_load}.slang`, both emitted by Lean.
