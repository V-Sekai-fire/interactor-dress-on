# Gate 3 — ggml-rd: ggml over RenderingDevice, kernels from Lean

**Result: G3.ops PASS for all 22 census ops** (and DUP and MEAN, which
share their kernels), after the six op families merged into cut-3.
`ops/results.txt` (2026-09-23, RTX 4090, Godot 4.7.2, `--xr-mode off`):
`test-backend-ops -o ADD,MUL,CPY,DUP,CONT,GET_ROWS,CONCAT,REPEAT,MUL_MAT,
FLASH_ATTN_EXT,IM2COL,CONV_3D,NORM,RMS_NORM,MEAN,SOFT_MAX,SILU,GELU,GELU_ERF,
SIGMOID,NEG,SCALE,DIAG_MASK_INF,ROPE -b RD0` in the guest reports **1700/1700
OK, 0 FAIL, `Backend RD0: OK`** (7230 not supported: quantized types,
masks, sinks, frequency factors, f16 ADD/MUL; none a census row), in 1590 s.
The same 1700 pass with a barrier after every dispatch (1612 s). Every
census-required type row (the gate's `REQUIRED`: MUL_MAT f32/f16/bf16 x f32,
f16 x f16 and permuted src1; CONT f16; GET_ROWS f32/f16/bf16; ROPE NEOX
without frequency factors; FLASH_ATTN_EXT D=128 H=12 f32 K/V, no mask,
prec f32; CONV_3D f16 kernel; IM2COL f16 out; SOFT_MAX without mask; the
f32 unary, norm and move ops) matches at least one OK case and no "not
supported" one. The census probe (K1/K5's 15 rows vs the in-guest ggml-cpu)
passes. Controls: `GGML_RD_FAULT=1` fails 54/54 ADD cases and 129/129
data-movement cases; headless, the same ELF prints `no RD device` and runs no
RD0 case (`ops/results-headless.txt`). Rule 4: 0 same-frame syncs, 0
permanent slots at close. L0 (all pins, three negative controls), L1 (67
kernels spirv-val and layout, 48 cpp emits for riscv64, 19 group-shared
kernels skipped for their siblings) and L2 (356/356 cases; swap-nb control
detects 336, misses 0, 20 no-ops) pass on the merged tree. Per-family
detail is in the sections below; their own evidence folders (`ops/k1k5`,
`ops-k3k4`, `ops-k7`, `ops/fa-serial`) keep their runs.

**G3.graph PASS, G3.cost measured** (`graph/results.txt`, section
[G3.graph and G3.cost](#g3graph-and-g3cost-the-apps-own-graphs)): the
apps' own graph builders (skin-tokens-ggml, pixal3d-ggml, copied into
`guest/ggml_test/app_graphs/`) on random weights, **run in the guest on
ggml-rd only**, their outputs dumped to the host and compared there with a
host-native oracle built from the same builders and seeds
(`tests/ggml_graph_oracle`): a Qwen3 decoder layer (f16) at rel-L2 2.10e-4
from host ggml-cpu, a sparse-conv level (f16) at 2.74e-4, and the full
4096 x 1536 DiT block (bf16) at 8.76e-4 from ggml-vulkan on the RTX 4090
(limit 1e-3; 8.77e-4 at 8^3 tokens), every gap the reference's own
rounding of activations to the weight type: with the weights widened to
f32 the three read 9.2e-6, 4.8e-7 and 3.1e-7 (limit 1e-4). Barrier elision
is bit-identical to a barrier after every dispatch on all of them; dropping
one barrier elision placed changed the output in 20 of 39, 16 of 48 and 5 of
48 runs (the rest are races that did not happen, see finding 3 there). **No
graph reference runs on the in-guest ggml-cpu any more** (AGENTS.md rule 10:
guest inference runs on ggml-rd; oracles on the host): the parked run's
in-guest reference took 334 s for the sparse level and stalled for hours on
the DiT block (`graph/parked-in-guest-ref/`); the whole G3.graph run now
takes 381 s, 302 of them host ggml-cpu on the sparse level. A skin-tokens decode step (28 layers) costs 10.6 ms of
ggml-rd host time (4.98 us per node, 8.4 per dispatch), 9.2 ms of GPU and
one frame; a Pixal3D flow forward (30 blocks) 16.9 ms of host time
(5.4 us per node) and 1.32 s of GPU, one frame.

**ADD and MUL** (the first reference run, before the families):
ggml's own `test-backend-ops -o ADD,MUL -b RD0`, run inside the guest
(`ggml_test.elf`) against the in-guest ggml-cpu, reports **100/100 tests
passed, 0 FAIL, `Backend RD0: OK`**; the 90 f16 cases report "not supported"
(f16 ADD/MUL are not in the census, and the reference kernel is f32). The
same 100 pass with a barrier after every dispatch. The control
`GGML_RD_FAULT=1` (every dispatch reads a source one element off) fails
**54/54** ADD cases. Headless, with no RenderingDevice, the same binary
prints `no RD device` and runs no RD0 case (`ops/results-headless.txt`):
the flat control that separates "the GPU path is not there" from "the
kernels are wrong". L0, L1 and L2 pass, and on the rebuilt ELFs Stage 1,
Stage 2, Gate 0F, Gate 4 and the lean gate still pass (`regression/`).

Setup: Godot 4.7.2-stable, the vendored `godot_sandbox` addon, Ryzen 7
3800X; Godot's device #0 was the RTX 4090 (the machine also has an RTX
3090, and #0 changes between sessions). ggml is `V-Sekai-fire/ggml`
@04b55bba, a squashed subtree at `vendor/ggml`. The branch sits on main
37d9c3a (after Cut 4); every number below is from runs on that tree.

```
./build.sh                                   # ELFs; kernels/ggml/gen.sh --no-emit runs first
godot --path project --headless --import     # once, after adding an ELF
godot --path project --script gate_ggml_rd.gd --rendering-driver vulkan --xr-mode off   # ops/results.txt
godot --headless --xr-mode off --path project --script gate_ggml_rd.gd                   # ops/results-headless.txt
gates/3-ggml-rd/kernels/l0.sh                # L0: kernels/lean-build.log, kernels/lean-negative-control.log
gates/3-ggml-rd/kernels/l1.sh                # L1: kernels/l1.log
tests/ggml_rd_kernels/build.sh               # L2, host-native: kernels/l2.log, kernels/l2-control.log
kernels/ggml/gen.sh --update                 # re-emit the kernels from lean/ after a Lean change
```

## Verdicts

| level | what | verdict | numbers |
|---|---|---|---|
| L0 | `native_decide` pins (`lean/Ggml`) and the committed emission | **PASS** | `lake build Ggml`: `add_f32`'s text pinned, MUL = ADD with one operator, the shared helpers, the generated `ggml_rd_params.h`; `kernels/ggml/gen.sh` (check mode) finds the committed `slang/` and header identical to Lean's emission (`kernels/lean-build.log`). Controls (`kernels/lean-negative-control.log`): a pin with one character changed is rejected by `native_decide`, and a copy of `kernels/ggml` whose `add_f32.slang` says `255u` for `256u` is refused with the diff. |
| L1 | spirv-val, fixed layout, riscv64 cpp | **PASS** | Both kernels pass `spirv-val` and the SPIR-V layout check (`kernels/l1.log`); the cpp emits compile for riscv64 (3784-byte objects). Controls: `add_f32` built at slangc's default -O1 is refused (s2 dropped from the SPIR-V) and `kernels/avbd`'s saxpby is refused (another layout); the read-only-source control kernel is refused by `gen.sh`. |
| L2 | cpp emits + the guest's packers vs native ggml-cpu | **PASS** | ADD/MUL: 41/41 cases, every one bit-exact (NMSE 0): broadcast in each dimension, permuted src1, overlapping views, odd sizes, a strided in-place destination and a two-node chain. With K2's 116 the log is 157/157, all bit-exact (`kernels/l2.log`). Control: src0's nb1/nb2 swapped after packing is caught in all 152 cases where the swap moves an address; the other 5 have nb1 = nb2 or both extents 1 (`kernels/l2-control.log`). |
| L3 | G3.ops, test-backend-ops in the guest | **PASS** | ADD,MUL: 100 OK (ADD 54, MUL 46), 0 FAIL, 90 not supported (f16), in 101 frames (one WAIT_GPU per submit), 149 s. Barrier-all: the same 100. Fault control: 54/54 ADD FAIL. The same verdicts in five runs this session, four of them before the rebase onto Cut 4. With K2 (the committed run): 425 OK, 0 FAIL, 698 not supported, 345 s; barrier-all the same 425; K2's fault control 129/129 FAIL. |
| L3 | probes (`guest/ggml_test/probes.cpp`) | **PASS** | chain: 256 in-place ADDs on one tensor, one graph, x = 256 exactly (256/256), 255 barriers, elided or all. independent: 64 ADD/MULs into separate outputs, 64000/64000 exact, **0 barriers** with elision, 63 with barrier-all. files: READ of a 16 KiB host file, then `ggml_backend_rd_tensor_upload` of the same file into a tensor at byte 512 of its RD buffer (UPLOAD), 4096/4096 exact, and x + x on the GPU 4096/4096 exact (including -0, 1e-30 and FLT_MAX + FLT_MAX = inf). alias: see finding 1. |
| rule 4 | no sync in its submit's frame | **PASS** | `rule4_same_frame_syncs=0` over every run; `close: permanent_slots=0`. |
| rule 8 | `main.gd`'s wrappers, as MCP calls them | **PASS** | `ops/wrappers.txt` (`project/probe_ggml_wrappers.gd`): `ggml_attach`, the chain/files/alias presets and a short fault run through `ggml_ops_start`, each pumped by `main.gd`'s own `_process`; a second `ggml_pump` in the start frame does not pump again; `ggml_rd_close` ends at `permanent_slots=0`, rule-4 counter 0 over 23 submits. |
| control | headless, no RenderingDevice | **PASS** | `no RD device`, `Testing 1 devices`, no RD0 case run, 0.5 s. Without `--xr-mode off` the headless process hangs after OpenXR fails to start and never runs the script: twice, killed at 900 s and at 300 s (`ops/run-headless-xr-default-hung.log` is the second). AGENTS.md's `--xr-mode off` holds for headless runs too. |
| regression | Stage 1, Stage 2, Gate 0F, Gate 4 and lean on the rebuilt ELFs | **PASS** | `regression/`: see below. |
| G3.graph | the apps' graphs, guest ggml-rd vs a HOST oracle | **PASS** | Qwen3 layer 2.10e-4 (f16) / 9.2e-6 (f32 arm) vs host ggml-cpu; sparse-conv level 2.74e-4 / 4.8e-7 vs host ggml-cpu; DiT block 4096 x 1536 8.76e-4 (bf16) / 3.1e-7 vs ggml-vulkan (8^3 tokens: 8.77e-4 / 3.5e-7); the dumped inputs bit-identical to the oracle's; elision bit-identical to barrier-all and to itself on all four; the dropped-barrier control detected in 20/39, 16/48, 8/48, 5/48 runs. Oracle checks: ggml-vulkan vs host ggml-cpu 5.57e-4 / 4.5e-5 on the DiT block (both sizes), 2.09e-4 / 1.4e-5 (Qwen), 2.74e-4 / 4.8e-7 (sparse). Control: ggml-vulkan in its default mode misses the f32 arm (1.10e-3 > 1e-4). |
| G3.cost | host us per node, dispatches, frames, GPU per graph | **measured** | decode step: 2135 nodes, 1265 dispatches, 1123 barriers, 1 frame, 10.6 ms host (4.98 us/node), 9.2 ms GPU; flow forward: 3107 nodes, 2115 dispatches, 1903 barriers, 1 frame, 16.9 ms host (5.43 us/node), 1.32 s GPU. |
| rule 8 | `main.gd`'s G3 presets | **PASS** | `graph/wrappers.txt` (`project/probe_ggml_wrappers_graph.gd`): `ggml_graph_qwen`, `ggml_cost_decode`, `ggml_cost_dit`, `ggml_graph_sconv`, `ggml_graph_dit` (8^3 tokens) each RESULT: PASS through `main.gd`'s own pump, and each graph's outputs leave through `ggml_graph_dump` (no arguments; the new guest entry points `ggml_dump_list`/`ggml_dump_chunk` behind it); rule 4 at 0, `permanent_slots=0`, 22.8 s (1273 s with the in-guest reference). |
| rule 10 | every in-guest ggml-cpu vmcall capped at ~5 min; a timeout is a FAIL | **PASS** (subset) | `merge-main/results.txt` (2026-09-23, main + cut-3 merged, RTX 4090): each pump vmcall of a job that runs ggml-cpu (test-backend-ops, the census probe) is capped at 214,577 units (300 s x 0.75e9 instructions/s / 2^20; `InferHost.GGML_CPU_TIMEOUT_UNITS`), the ggml-rd-only jobs keep 1,000,000. Under it: probe_census PASS (25.7 s, 46 pumps), `-o SILU,NEG,SCALE,RMS_NORM,SOFT_MAX,ROPE` 95/95 OK, 0 FAIL (84.4 s, 96 pumps), ADD's fault control 54/54 FAIL, all 11 probes PASS, rule-4 counter 0. Control `ops_cpu_cap_control` (always last): the cap lowered to 8 units kills the first pump after 11.2 ms and the run ends `ggml_pump killed by execution_timeout (8 units, the ggml-cpu cap of AGENTS.md rule 10)`. The subset leaves out ops_barrier_all and ops_fault_move, whose verdicts then FAIL by design, so the file reads RESULT: FAIL. Not yet rerun under the cap: the full 1700-case list (its pumps averaged 0.93 s, 1590 s over 1708). Not yet enforced on the host: G3.graph's sparse-level oracle ran 302.0 s of host ggml-cpu in one process (four arms, 72 + 138 + 47 + 41 s), over rule 10's `timeout 300`; `gate_ggml_graph.gd` still allows 1200 s. |

The in-guest reference dominates G3.ops' time: ops_main is 125-164 s of
vmcalls over 101 frames (five runs on a shared machine), most of it
ggml-cpu at rv64gc on the 16.7M-element cases. That is where it stops:
G3.ops' single-op cases are the only in-guest ggml-cpu reference; every
graph is compared on the host (G3.graph below).

## What this cut found (and fixed)

1. **Read-write sources avoid Gate 0F's lost in-place writes; the alias
   probe is the A/B.** Gate 0F finding 4 names the mechanism:
   `RenderingDeviceGraph::add_compute_list_usage` keeps a buffer's FIRST
   usage in a compute list (only a DEV build reports a second, different
   one), and a uniform set lists its trackers in binding order. In ggml one
   buffer holds a node's sources and its destination nearly always, so
   ggml-rd declares b1-b3 `RWStructuredBuffer` (kernels only read them):
   every list's usage of a data buffer is then a write, and consecutive
   lists are ordered. The alias probe records x += 1 in place 1000 times,
   x bound at b1 and b4 of one set, a barrier between rounds: with
   `add_f32` (read-write sources) 4096/4096 end at exactly 1000; with the
   control `ctl_add_f32_rosrc` (the same kernel, read-only sources) 0/4096
   do (mean 35-99 of 1000 over five runs). test-backend-ops' in-place and
   fused chains (nf up to 16) pass. This settles 0F's open question for
   ggml-rd: a buffer bound at two bindings of one dispatch is coherent when
   both are read-write, and no kernel reads an element another thread of
   the same dispatch writes.
2. **slangc keeps an unused binding only at -O0.** At the default -O1 with
   `-preserve-params` the unused s2 disappears from the SPIR-V while the
   reflection JSON still lists it (Gate 0F finding 9); set-0 sets would then
   not bind across kernels. gen.sh compiles with `-O0 -preserve-params` and
   the table generator parses the SPIR-V itself (and the JSON), refusing
   any kernel that differs from the fixed layout.
3. **The host views at most 16 MiB of guest memory per syscall.** A
   PackedByteArray made from guest memory or fetched into it faults at
   17 MiB (16 MiB works), and so did test-backend-ops' 64 MiB `memcpy`
   (`__wrap_memcpy` is a syscall). `rdc::Device` splits `buffer_update`,
   `storage_buffer` and staged reads at 16 MiB, `pump::read` asks at most
   16 MiB per READ, and `vendor/sandbox-api` splits memcpy, memmove, memset
   and memcmp.
4. **The sandbox API's aligned allocation returned freed pointers.**
   Upstream `memalign` tried 16 plain mallocs and, when none was aligned,
   returned its last candidate after freeing it: ggml's 64-byte
   `posix_memalign` context buffers failed as double frees on the second
   job. Cut 4 met the same bug in Geogram and fixed it in
   `vendor/sandbox-api` (an over-allocating memalign with a side table
   that `free`/`realloc` consult; Gate 0F probe 17). This cut first
   carried an equivalent fix of its own; rebased onto main after Cut 4, it
   runs on Cut 4's, and adds only the 16 MiB mem* split (finding 3).
5. **An Object argument must be declared `Object`.** With unboxed arguments
   the host passes the handle in a register; a `Variant` parameter read it
   as a pointer and the device arrived as Nil.
6. **A local RenderingDevice drops what is recorded between submit and
   sync** (`sync()` runs `_begin_frame`, which clears the graph). Every
   ggml-rd entry point that touches the device first waits for a pending
   graph (WAIT_GPU, then sync on the next frame), and
   `ggml_backend_rd_tensor_upload` waits before it yields UPLOAD, so the
   host only uploads while the device is idle.
7. **The reference CPU backend has to run one thread.** test-backend-ops
   creates it with `GGML_DEFAULT_N_THREADS` (4) and guest threads are
   serialized (Gate 0C), so ggml's spinning barriers would never release.
   `ggml_test.elf` links with `--wrap=ggml_backend_init_by_type` and sets 1.
8. **`buffer_get_data` stages the whole buffer** (0F finding 3): reads go
   through `Device::buffer_get_into`, a `buffer_copy` into a staging buffer
   of at most 16 MiB and a `buffer_get_data` of that buffer only.
9. **A pump yield outside the job would jump into a dead stack.** A host
   call that closes the backend after the job returned (a pending graph,
   then `ensure_idle`) reached `wait_gpu` with no fiber running. The pump
   now knows when the job is running; outside it, wait_gpu and coop return
   at once (the caller syncs in that call), READ returns nothing and UPLOAD
   is refused.
10. **The committed ELFs named the build machine and the commit.** ggml's
    asserts expand `__FILE__`, so probes.elf and ggml_test.elf carried the
    builder's home directory and worktree (12 and 15 strings), and ggml
    stamps `ggml_commit()` with `git rev-parse HEAD` of the enclosing repo
    (this one), so they changed with every commit. The guest build passes
    `-ffile-prefix-map=<checkout>/=` and stamps ggml with its pin
    (`04b55bba`): the ELFs name sources as `vendor/ggml/src/...`, carry no
    machine path, and two builds in different directories are byte
    identical.

## Family K2: data movement (CPY, DUP, CONT, GET_ROWS, CONCAT, REPEAT)

**Result: PASS.** `ops/results.txt` (2026-09-23, RTX 4090): test-backend-ops
`-o ADD,MUL,CPY,DUP,CONT,GET_ROWS,CONCAT,REPEAT -b RD0` in the guest,
**425/425 OK, 0 FAIL** (CPY 161, CONCAT 80, CONT 36, GET_ROWS 20, REPEAT 18,
DUP 10, plus ADD 54 and MUL 46), `Backend RD0: OK`; the same 425 with a
barrier after every dispatch. Every census row of these ops is among the OK
cases (all f32, plus CONT f16 -> f16 of a row-strided view); the 698 "not
supported" are the quantized types (CPY, GET_ROWS, CONCAT), i8 and i64
CONCAT, and CPY f32 <-> i32, none of them in the census. The fault control
`GGML_RD_FAULT=1` on DUP, CONT, GET_ROWS, CONCAT and REPEAT fails **129/129**
(i32 cases excluded: test_get_rows fills an i32 source like its row indices,
mostly zeros, so a shifted index picks another zero row; with them, 2 of 4
i32 GET_ROWS cases passed the fault, 2026-09-23 first run).

| level | verdict | numbers |
|---|---|---|
| L0 | **PASS** | `lake build Ggml`: every helper, entry and `val` text pinned, and each of the 15 kernels pinned as the concatenation of pinned pieces (`kernels/lean-build.log`); gen.sh's check finds the committed `slang/` equal to Lean's emission. |
| L1 | **PASS** | 17 kernels pass spirv-val and the fixed-layout check; the 15 new cpp emits compile for riscv64 (3632-5640-byte objects) (`kernels/l1.log`). |
| L2 | **PASS** | 157/157 cases (116 new), **every one bit-exact** (NMSE 0 where test-backend-ops asks 0, and also the conversions): all 9 type pairs among f32/f16/bf16, permuted and strided sources and destinations (16-bit ones with odd strides: the word-ownership path), reshaping copies, f16 subnormals and +-65000, DUP/CONT nodes, GET_ROWS of f32/f16/bf16/i32 with views and batches, CONCAT on every dim with non-contiguous operands, REPEAT on every dim, and the census's hot shapes (KV-cache concat [128,8,514]+[128,8,1], CONT f16 [1024,1,1024], REPEAT [128,8,1,514] -> x2) (`kernels/l2.log`). Control swap-nb: 152 cases DETECTED, 0 missed, 5 no-ops (`kernels/l2-control.log`). |
| L3 | **PASS** | G3.ops above; barrier-all identical; fault control 129/129 FAIL. |
| perf | numbers | `probe_perf` (below), with and without a barrier per dispatch. |

### How the kernels work

- **One `val(i)` per kernel, two entries** (`lean/Ggml/SlangCodegen/Move.lean`).
  Every storage binding is `uint`, so a same-type move is a bit copy and
  the conversions are integer code with ggml-cpu's rounding: f16 -> f32 exact
  (subnormals normalised by a shift loop), f32 -> f16 round-to-nearest-even
  with ggml's NaN (`0x7E00 | sign`), bf16 = `h << 16`, f32 -> bf16 ggml's
  `(u + 0x7fff + ((u >> 16) & 1)) >> 16`. L2 is bit-exact for all of them.
- **16-bit destinations without atomics.** The cpp target has no
  InterlockedAnd/Or, and an f16 shares its word. The thread of the element at
  a word's even address owns the word: it stores its half and, when the next
  address is in the destination, that element's half too (one store), else
  it keeps the other half (read-modify-write); an odd-address thread returns
  when the previous address is in the destination, else owns its word the
  same way. "Next/previous address in dst" is a carry over dst's dimensions
  sorted by stride, which the packer arranges (`guest/ggml-rd/ops/move.h`):
  iteration order = dst dims by stride, size-1 last; word 55 = the merge
  chain; a 16-bit dst is supported when that order is nested
  (`nb[k+1] >= ne[k] nb[k]`): every permutation, transpose and strided
  sub-block of a contiguous tensor is; an interleaved view such as
  ne [3,2], nb [2,3] is not, and is "not supported" there. The sort also makes every op's writes coalesced whatever dst's
  permutation.
- **CPY/DUP/CONT** (8 kernels: `cpy_b32`, `cpy_b16` and the six conversions)
  find a destination element's source by its ggml linear index (words 56-59:
  dst's linear weights in iteration order), unravelled over src0, so shapes
  may differ (reshaping copies) and both sides may have any strides.
- **CONCAT** (`concat_b32`, `concat_b16`) needs no dimension word: an index
  is past src0 in at most one dimension, so `i < src0.ne` on all four picks
  src0, else src1 at `i - src0.ne` on the dimension that is past. The packer
  permutes dst, src0 and src1 alike.
- **REPEAT** (`repeat_b32`, `repeat_b16`): `src0[i mod src0.ne]`.
- **GET_ROWS** (`get_rows_b32`, `_f16`, `_bf16` -> f32): one thread per dst
  element in dst's order, the row read from src1 (i32), every operand strided.
- **A 16-bit RMW writes 2 bytes outside dst's range at its ends**, so
  `rd_graph.cpp`'s hazard ranges are now whole 4-byte words.

### GPU time per op (`probe_perf`, RTX 4090)

A graph of K copies of the op (each into its own output, K up to 256 and
about 512 MiB of outputs) against a graph of one, both timed on the GPU by
timestamps around the compute list (`ggml_backend_rd_set_timestamps`, new in
`rd_compute`: `capture_timestamp`, read after the sync); per op =
(T_K - T_1) / (K - 1). GB/s counts bytes read + written by one op; the
sources are the same buffer every time, so their reads hit the 72 MB L2 and
the rates above ~1 TB/s are cache-assisted. Numbers from the committed run
(`ops/results.txt`); an earlier run the same day on the shared machine
measured the large cases 1.6-2.1x slower (the sub-microsecond ones alike).

| shape (census source) | per op, independent | per op, barrier after each |
|---|---|---|
| CONCAT f32 [128,8,514]+[128,8,1] dim 2 (skin-tokens KV cache, 213752/run) | 3.87 us (1089 GB/s) | 4.85 us |
| CONCAT f32 [128,8,515,1]x2 dim 3 (skin-tokens, 192304/run) | 6.43 us (1313 GB/s) | 7.25 us |
| CPY f32 [128,8] (skin-tokens KV write, 213752/run) | 0.14 us | 2.69 us |
| REPEAT f32 [128,8,1,514] -> [128,8,2,514] (skin-tokens, 43008/run) | 7.37 us (857 GB/s) | 8.47 us |
| CONT f32 permute(0,2,1,3) [128,8,2,514] (skin-tokens C:R, 21560/run) | 7.49 us (1124 GB/s) | 8.68 us |
| CONT f32 transpose [128,515,16,2] (skin-tokens C:N, 10724/run) | 28.7 us (587 GB/s) | 29.0 us |
| GET_ROWS f32 [896,33036] x10 (skin-tokens embedding) | 0.18 us | 1.66 us |
| GET_ROWS f32 [512,246] x246 (pixal3d) | 1.20 us (839 GB/s) | 3.43 us |
| CONT f16 permute [1024,1024] -> [1024,1,1024] (pixal3d, 1080/run) | 9.69 us (433 GB/s) | 11.2 us |
| CONCAT f32 [1,64,12,4096]x2 dim 0 (pixal3d rotate-half) | 37.3 us (1349 GB/s) | 37.6 us |
| CPY bf16 -> f32 [1536,4096] (a bf16 weight widened) | 41.1 us (919 GB/s) | 41.3 us |
| CPY f32 -> f16 [1536,4096] | 84.0 us (449 GB/s) | 64.7 us |

A skin-tokens run's K2 GPU time is then about 213752 x 4.85 + 192304 x 7.25 +
213752 x 2.69 + 43008 x 8.47 + 21560 x 8.68 + 10724 x 29 us = 3.9 s, 63%
of it the KV cache's two concats (which copy the whole cache every
token: the model's O(n^2), not the kernel's). What is left on the table: the
transpose reads uncoalesced (587 GB/s; a groupshared tile would about double
it, 0.3 s/run), and 16-bit destinations run twice the `val` of an f32 one
(the f16 CONT at 433 GB/s).

### Files (K2)

```
lean/Ggml/SlangCodegen/Move.lean      the shared pieces: integer conversions, ld16, entry32/entry16, pins
lean/Ggml/SlangCodegen/{Cpy,GetRows,Concat,Repeat}.lean   15 kernels, each pinned
guest/ggml-rd/ops/move.h              iteration order, nesting, merge chain, block permutation
guest/ggml-rd/ops/{cpy,get_rows,concat,repeat}.cpp        packers (CPY, DUP, CONT share cpy.cpp)
tests/ggml_rd_kernels/cases/move.cpp  116 L2 cases
guest/ggml_test/probes.cpp            probe perf (ggml_probe_start("perf", "move", ""))
```

Shared-layer changes: `rd_compute` gains `capture_timestamp`,
`timestamps_count` and `timestamp_gpu_ns` (three names in the name pool,
which is now **full: 32 of 32 host cache slots**; the next name needs one
retired or a cold path outside the pool); ggml-rd gains
`ggml_backend_rd_set_timestamps` / `ggml_backend_rd_last_gpu_ns` and
`GGML_RD_TIMESTAMPS=1`; hazard ranges are word-aligned; the L2 harness
compiles the emits with `SLANG_ENABLE_BOUND_ZERO_INDEX` (the swap-nb control
moved the large cases' addresses past the 256 MiB block and faulted);
`main.gd` gains `ggml_ops_move`, `ggml_ops_move_fault`, `ggml_probe_perf`.
Every ELF links `rd_compute`, so all five were rebuilt (a second build in a
fresh directory is byte-identical) and the earlier gates re-run on them
(`regression/`, overwritten with these runs): Stage 1 PASS (held device,
barriered counts exact, the no-barrier arm still loses counts); Stage 2
PASS (gradcheck 5/5 cpu and rd, worst rel 1.04e-6 and 9.59e-7,
backward_smoke 18/18 both, `same_frame_syncs=0` over 339 submits); Gate 0F
`PASS=34 FAIL=5` (probe 3's five, as committed; 135 s); Gate 4 guest ==
native 9/9, rule 8 23/23; lean gate clean, 0 DIFF, control 1 DIFF. Rule 8
for ggml (`ops/wrappers.txt`): PASS.
## K6: MUL_MAT

**Result: G3.ops PASS for MUL_MAT.** `test-backend-ops -o ADD,MUL,MUL_MAT
-b RD0` in the guest: **728/728 passed, 0 FAIL, `Backend RD0: OK`**, of
which MUL_MAT **628 OK** (f32 x f32 209, f16 x f32 208, bf16 x f32 149,
f16 x f16 62), in 981 s; the same 728 with a barrier after every dispatch
(953 s). No case of a census type row reports "not supported" (the gate's
`REQUIRED` check: 0). The 917 MUL_MAT cases not supported are the
quantized types (855) and f32 x f16 (62), which neither the census needs
nor ggml-cpu computes (it refuses f32 x f16 and bf16 x f16, so no reference
could check them).

Kernels (`lean/Ggml/SlangCodegen/MulMat*.lean`, 16, one module each):

| kernel | threads | work | used when |
|---|---|---|---|
| `mul_mat_tiled_<a>_<b>` | 16 x 16 | a 64 x 64 dst tile, BK = 16, `As/Bs[16][65]` f32 group memory, 4 x 4 register block, converted on load, every edge bounds-checked, batch = `gid.z` | ne11 > 4 |
| `mul_mat_vec_<a>_<b>` | 32 x 8 | 8 rows x 32 lanes, up to 4 columns, lane partials over k = l, l+32, ..., a group-memory tree 16/8/4/2/1 | ne11 <= 4 (decode, beams) |
| `mul_mat_serial_{tiled,vec}_<a>_<b>` | same | the same sums in the same order, no group memory | the cpp target (L2) |

for (a, b) in f32 x f32, f16 x f32, bf16 x f32, f16 x f16. f16 and bf16
stay in `uint` words and widen exactly on load (bf16 = `h << 16`); every sum
is f32 (GGML_PREC_F32 holds). Every operand is read through its strides, so
permuted src1, non-contiguous src0 rows (RC, RR) and broadcast
(r2 = ne12/ne02, r3 = ne13/ne03, words 55/56) need no copy. The packer is
`guest/ggml-rd/ops/mul_mat.cpp`. slangc's cpp target rejects the group
barrier (E36107), so `kernels/ggml/cpp_siblings.txt` names each GPU
kernel's serial sibling: gen.sh emits cpp for the siblings only, the table
check requires each pair's thread groups to match, and the L2 harness runs
the sibling wherever the packer chose the kernel.

| L0 | **PASS** | pins: the tiled and vec f16 x f32 texts whole, the other pairs as those texts with two declarations and two load helpers replaced, the siblings as the kernels' preamble/stores around their own loops, the load helpers of each type, the sibling pairs; control (1b): the tiled pin with `As[16][64]` is rejected. |
| L1 | **PASS** | 18 kernels spirv-val and layout OK; the 8 siblings' cpp compile for riscv64 (6.5-11.5 KB objects); controls: slangc -target cpp refuses `mul_mat_tiled_f16_f32` (E36107 at GroupMemoryBarrierWithGroupSync), and a sibling pair with different thread groups is refused by the table check. |
| L2 | **PASS** | 64/64 cases (23 MUL_MAT: every pair, vec at n = 1..4 and tiled from n = 5, broadcast, the three permutations, k views, o = 3, one past the tile on every side, m = 1, skin-tokens' decode and RC attention, an RR case, a scaled Pixal3D block) within NMSE 5e-4 of ggml-cpu (worst 2.3e-6, bf16: ggml-cpu rounds src1 to bf16, the kernels keep it f32). Control swap-nb: detected in 58, missed 0 (6 no-ops: nb1 = nb2). |
| L3 | **PASS** | as above; fault control (ADD) 54/54 FAIL; rule 4: 0 same-frame syncs; headless control PASS. |
| perf | **PASS** | `probe_mm_perf`, below; every shape within nmse 1e-8 of a double sum over the same inputs (worst 5.2e-13). |

GPU time per MUL_MAT (RTX 4090, `probe mm_perf`): each shape runs a short
and a long graph of r1 < r2 independent MUL_MATs over the same operands,
timed on the host clock from graph_compute to its synchronize (a frame
later), best of 3; per-op = (long - short) / (r2 - r1). Independent
dispatches may overlap on the GPU, so this is throughput, and the smaller
shapes move between runs (two runs of the full gate this session):

| shape (census calls) | a x b | M x N x K, batch | kernel | per op | GFLOP/s |
|---|---|---|---|---|---|
| benchmark | f16 x f32 | 4096 x 1024 x 1536 | tiled | 1281-1402 us | 9193-10061 |
| benchmark | bf16 x f32 | 4096 x 1024 x 1536 | tiled | 1231-1301 us | 9906-10470 |
| benchmark | f32 x f32 | 4096 x 1024 x 1536 | tiled | 1337-1374 us | 9375-9637 |
| skin-tokens decode (32172) | f16 x f32 | 2048 x 1 x 896, 2 (bcast) | vec | 6.0-8.6 us | 857-1230 |
| skin-tokens decode (21448) | f16 x f32 | 1024 x 1 x 896, 2 (bcast) | vec | 7.4-8.3 us | 442-496 |
| skin-tokens decode (21448) | f16 x f32 | 896 x 1 x 2048, 2 (bcast) | vec | 4.0-10.1 us | 726-1828 |
| skin-tokens attention RC (10724) | f32 x f32 | 515 x 1 x 128, 16 x 2, permuted KV | vec | 14.0-23.1 us | 183-301 |
| skin-tokens mat (3183) | f16 x f32 | 1024 x 54000 x 512 | tiled | 5463-5810 us | 9746-10365 |
| Pixal3D bf16 (120) | bf16 x f32 | 4608 x 4096 x 1536 | tiled | 5014-5043 us | 11498-11564 |
| Pixal3D f16 (663) | f16 x f32 | 1024 x 1029 x 1024 | tiled | 101-236 us | 9134-21419 |
| Pixal3D f32 attention (24) | f32 x f32 | 1029 x 1029 x 64, 16 | tiled | 318-329 us | 6591-6819 |

About 10 TFLOP/s on the large tiles, 12% of the 4090's f32 peak: the
simple 4 x 4 block is the first version, not a tuned one (a wider block
and vectorised f16 loads are the obvious next steps, gated by this probe).
## Family K7: IM2COL and CONV_3D

**Result: G3.ops PASS for IM2COL and CONV_3D** (`ops-k7/results.txt`:
RESULT: PASS, RTX 4090). `test-backend-ops -o IM2COL,CONV_3D -b RD0` in the
guest: **350/350 OK** (IM2COL 92, CONV_3D 258), 0 FAIL, **0 not supported**,
so every census row (IM2COL f32 image -> f16 columns with an f16 kernel;
CONV_3D f16 kernel x f32 input) runs on the GPU. The same 350 pass with a
barrier after every dispatch. The control `GGML_RD_FAULT=1` (src1, the
image or input, read one element off) fails **350/350**. The earlier probes
and rule 4 (`rule4_same_frame_syncs=0`, `permanent_slots=0`) pass in the
same run.

Kernels (`lean/Ggml/SlangCodegen/Conv.lean`), one thread per output, no
barriers, no groupshared (so no Serial sibling):

| kernel | op | what |
| `im2col_f32` | IM2COL -> f32 | one thread per column element, 2-D and 1-D, stride/padding/dilation from op_params, the image through its strides |
| `im2col_f16` | IM2COL -> f16 | one thread per 32-bit word of a contiguous dst (two halves; a half outside dst keeps the old bits), f32 -> f16 rounded to nearest even in integer arithmetic |
| `conv3d_f32` | CONV_3D, f32 kernel | the IC x KD x KH x KW window summed in f32 in ggml-cpu's order; params read once, offsets advanced per loop level |
| `conv3d_f16` | CONV_3D, f16 kernel | the same, the kernel read as f16 halves of `uint` words and the input rounded to f16 first, as ggml-cpu's im2col-into-f16 does |

| L0 | **PASS** | the four kernels' whole texts pinned by `native_decide`, `im2col_at` shared verbatim by both IM2COL kernels, and `f32_to_f16`'s integer steps on 12 boundary values (65504, the 65520 tie to inf, 2^-14, 2^-24, the 2^-25 tie to 0, ties to even, NaN -> 0x7E00); `gen.sh` check finds the committed emission identical (`kernels/lean-build.log`). |
| L1 | **PASS** | spirv-val and the fixed layout for all six kernels; the cpp emits compile for riscv64 (`kernels/l1.log`). |
| L2 | **PASS** | 44 K7 cases (20 IM2COL, 24 CONV_3D) of 85, all within threshold: every IM2COL case bit-exact, f16 columns included (the stride/padding/dilation sweep, 1-D, Whisper's [3000,128] conv, odd-length f16 output, a permuted image, the census' 512x512x3 DINO patch embedding); CONV_3D NMSE <= 2.7e-13 against 5e-4 (the sweep, dilation, the asymmetric 5x1x3 kernel, a 1x1x1 kernel, a permuted input, both census shapes). Control: swapped strides detected in 40 of the 44, the 4 no-ops being CONV_3D kernels whose swapped strides are equal (KH = 1) or never used (1x1x1) (`kernels/l2-control.log`). |
| L3 | **PASS** | 350/350 OK, barrier-all 350/350, fault 350/350 FAIL, 129 s of ops_main (the in-guest ggml-cpu reference dominates). Rule 8: `main.gd`'s K7 presets (`ggml_ops_conv`, `ggml_ops_conv_fault`, `ggml_probe_conv_perf`) through `project/probe_ggml_wrappers_k7.gd`: conv_perf PASS, 350/350, rule 4 at 0 (`ops-k7/wrappers.txt`). ADD,MUL on this ELF: unchanged, 100 OK, 90 not supported, fault 54/54. |
| perf | **PASS** | `conv_perf` probe (`guest/ggml_test/probe_conv.cpp`) on the census' hottest shapes, see below; 4096 sampled outputs of each checked (f16 columns bit for bit, CONV_3D NMSE <= 1.9e-13). |

GPU time per op on the RTX 4090, from the gate run (vsync off; host
`Time.get_ticks_usec()` around 10 graph computes of 1 and of 17 independent
copies, after at least 300 ms of untimed ones; per_op = the slope):

| census shape | kernel | outputs | MACs | per op | one-op graph, frame included |
| IM2COL 512x512x3, 16x16 stride 16 -> [768,32,32] f16 (Pixal3D DINO) | im2col_f16 | 786,432 | - | 0.006 ms | 0.50 ms |
| CONV_3D 16^3 x 8 -> 512, 3^3, f16 kernel (Pixal3D ss_dec, 19 per run) | conv3d_f16 | 2,097,152 | 453 M | 0.81 ms | 1.23 ms |
| CONV_3D 64^3 x 32 -> 1, 3^3, f16 kernel (Pixal3D ss_dec) | conv3d_f16 | 262,144 | 226 M | 0.55 ms | 0.91 ms |

The warm-up matters: the same probe without it (5 timed computes after 1)
read 1.48 and 0.71 ms, and 2.14 and 1.11 ms before the params words were
hoisted into locals and the offsets advanced per loop level. Those runs
paid the GPU's clock ramp, so only their order is comparable. Under
`main.gd` (`ops-k7/wrappers.txt`, vsync on) every graph takes one 16.7 ms
frame whatever it holds, so the slope there is noise: the gate is the
measurement. What the inner loop still pays is the exact f16 rounding of
the input (ggml-cpu's semantics), once per multiply-add in a
one-thread-per-output kernel.

What K7 found:

1. **slangc's C++ `f32tof16` rounds ties away from zero** (it adds the
   first dropped bit and ignores the rest), and SPIR-V leaves a half
   conversion's rounding to the driver. ggml rounds to nearest even. The
   kernels round in integer arithmetic (`f32_to_f16`), so f16 columns are
   bit-identical to ggml-cpu on both targets (L2 and the probe).
2. **A half-precision store would need 16-bit storage and `Float16`**
   (LeanSlang: slangc declares both for a `half` buffer). `im2col_f16`
   writes `uint` words instead, one thread per word; the packer launches
   one thread per word from dst's first half to its last, so an odd offset
   or length is handled and nothing outside dst changes.
3. **IM2COL reads src0 for its shape only**, so the L2 swap-nb control on
   src0 changes no address; an `L2Case` now names the source the control
   swaps (`control_src`, 1 for IM2COL). And a swapped stride can point far
   outside the harness's memory block: the harness builds the emits with
   `SLANG_ENABLE_BOUND_ZERO_INDEX` (an out-of-range index reads element 0,
   as a robust Vulkan device reads zeros), so the control fails a case
   instead of the process.

Run it: `godot --path project --script gate_ggml_rd.gd --rendering-driver
vulkan --xr-mode off ++ --ops=IM2COL,CONV_3D --fault-ops=IM2COL,CONV_3D
--out=ops-k7 --probe=conv_perf:all` (the gate's user arguments are new:
`--ops`, `--fault-ops`, `--out`, `--probe`, so an op family gates its own
ops into its own folder; with none, the gate runs as before, with `OPS`
now `ADD,MUL,IM2COL,CONV_3D`).

## G3.graph and G3.cost: the apps' own graphs

**Result: G3.graph PASS on all three graphs (the DiT block at 4096 and at
512 tokens), G3.cost measured.** `graph/results.txt` (2026-09-23, RTX 4090,
`gates/3-ggml-rd/graph/run.sh` -> `project/gate_ggml_graph.gd`): the graphs
are built by the apps' own builder functions, copied into
`guest/ggml_test/app_graphs/` (skin-tokens-ggml `src/qwen.cpp` @097a0cc,
pixal3d-ggml `trellis2.cpp` @1c22f5e; `CITATION.cff` names each function and
the adaptations), on random weights. **The guest runs them on ggml-rd only**
(`ggml_test.elf`); the reference is computed on the host by
`tests/ggml_graph_oracle`, which compiles the same builders and the same
net code (`guest/ggml_test/graph_nets.cpp`, shared by both) against a
host-native build of the vendored ggml (V-Sekai-fire/ggml @04b55bba) with
ggml-cpu and ggml-vulkan:

| graph (builder) | shape, weights | nodes / dispatches | barriers: elided / all | host oracle (check) | rel-L2, native weights (limit 1e-3) | rel-L2, f32 arm (limit 1e-4) | elision vs barrier-all, repeat | dropped-barrier control | guest run / oracle |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3 decoder layer (`decode_layer`) | skin-tokens: hidden 896, 16/8 heads, 128, MLP 2048; 2 beams, KV cache past 514; f16 | 76 / 45 | 39 / 44 | ggml-cpu, 16 threads (ggml-vulkan) | **2.10e-4** (layer out 6.7e-5, new K/V cache rows 2.0-2.1e-4) | **9.2e-6** | bit-identical, bit-identical | 20 of 39 detected | 2.0 s / 9.8 s |
| sparse-conv level (`shape_dec_run`, level 1) | C 256 (prev 512, next 128), 2 ConvNeXt blocks + child head + up part A, L = 632 shell voxels (6200 of 17064 neighbour slots real); f16 | 789 / 571 | 359 / 570 | ggml-cpu (ggml-vulkan) | **2.74e-4** (subdiv, h1; x 1.6e-4) | **4.8e-7** | bit-identical, bit-identical | 16 of 48 detected | 2.9 s / 302 s |
| DiT block, 8^3 tokens | 512 tokens, otherwise as below | 103 / 70 | 62 / 69 | ggml-vulkan, precise (ggml-cpu) | **8.77e-4** | **3.5e-7** | bit-identical, bit-identical | 8 of 48 detected | 4.4 s / 10.8 s |
| DiT block (`trellis2_ss_flow_forward`'s block + Pixal3D's `proj_linear`) | 4096 tokens x 1536, 12 heads, MLP 8192, cross-attention over 5 global tokens, proj [1024, 4096]; bf16 | 103 / 70 | 62 / 69 | ggml-vulkan, precise (ggml-cpu) | **8.76e-4** (residual branch 1.63e-3) | **3.1e-7** | bit-identical, bit-identical | 5 of 48 detected | 14.6 s / 24.8 s |

Every arm is built by the builder from scratch on its backend (ggml's
gallocr allocates it, as the apps do), and every leaf is filled from a seed of
its name, so both sides get the same bytes: weights uniform within
PyTorch `nn.Linear`'s 1/sqrt(fan_in), norm gains in [0.8, 1.2], inputs in
[-1, 1], the RoPE tables, neighbour indices and masks computed as the apps
compute them. The oracle checks that: the block input the guest dumped is
bit-identical to the one it built (all four graphs). rel-L2 is
||rd - ref|| / ||ref|| over each output; the table gives the worst output.
Per graph:

- **the guest** (`probe_graph.cpp`): the native arm (the weights in the
  model's type: f16; bf16 for the DiT, the type of the chibifire Pixal3D
  GGUFs) and the f32 arm (the same weight values, rounded through f16/bf16,
  stored as f32) on ggml-rd; both arms' outputs stay in the guest heap and
  leave through `ggml_dump_list`/`ggml_dump_chunk` (8 MiB chunks, finding 3),
  which `project/graph_dump.gd` writes under `build/graph-dumps/<run>/`.
  The guest runs no reference: its ggml-cpu (rv64gc, one thread, about
  0.1 GFLOP/s) took 334 s for the sparse level's two reference arms and
  never finished the 4096-token DiT block's (about 440 GFLOP each, hours),
  which is where the previous run was parked (`graph/parked-in-guest-ref/`).
- **the oracle** (`tests/ggml_graph_oracle`, a child process of the gate,
  polled each frame): rebuilds both arms on the reference backend (host
  ggml-cpu for the Qwen layer and the sparse level, ggml-vulkan on the RTX
  4090 for the DiT block), and on the other one as a check, then compares:
  rd native vs ref native (limit 1e-3), rd f32 vs ref f32 (limit 1e-4), and
  the info rows (`graph/oracle-graph_*.txt`). **ggml-vulkan is an oracle
  only**: it compiles its own GLSL with glslc, never ships and never runs in
  the guest, so AGENTS.md rule 2 (shipped kernels come from Lean) does not
  apply to it. It runs "precise" (`GGML_VK_DISABLE_F16`, `_COOPMAT`,
  `_COOPMAT2`, `_BFLOAT16`, `_INTEGER_DOT_PRODUCT`): f32 accumulation from
  f32 shared memory.
- **elision**: on the native RD net, a run with elision, one with
  `GGML_RD_BARRIER_ALL=1` and a second elision run are compared bit for bit,
  every output. Before every RD run the graph's compute buffers (gallocr's)
  are cleared to 0, so a dispatch that races its producer reads zeros, not
  the previous run's identical result.
- **control** (`GGML_RD_DROP_BARRIER=<k>`, new in `rd_graph.cpp`): the k-th
  barrier elision places is recorded as placed (the hazard segments reset)
  but left out of the compute list. One run per barrier (at most 48, spread
  over the graph), each compared bit for bit with the barrier-all run; a
  run that differs has detected its dropped barrier.

What G3.graph found:

1. **The native-weight gap is the reference's rounding, not ggml-rd's.**
   ggml-cpu converts a matmul's activations (src1) to the weight's
   `vec_dot_type` (f16, bf16) before the dot product, and ggml-vulkan does
   the same for bf16 weights (src1 is converted to bf16 for its bf16 x bf16
   matmul); ggml-rd widens the weights on load and keeps the activations
   f32 (K6). The arms separate the two: ggml-rd's native-weight outputs are
   bit-identical to its own f32 arm (0 in every graph), sit as close to the
   reference's f32 arm (Qwen 9.2e-6, sparse 4.8e-7, DiT 3.1e-7) as that arm
   does, and the reference's native arm is as far from its own f32 arm
   (2.09e-4, 2.74e-4, 8.76e-4) as ggml-rd is from it. For bf16 that rounding
   (8 significant bits) alone is 8.76e-4 of the 4096-token block's output
   and 1.63e-3 of its residual branch (output minus input): the 1e-3 limit
   is a limit on the reference here, and it holds with 12% to spare, at
   every size (8.77e-4 at 512 tokens; 9.31e-4 against ggml-cpu's rounding at
   216 and 512 tokens).
2. **The oracle is checked against the other host backend, and only the
   precise ggml-vulkan is exact enough for the f32 arm.** Precise
   ggml-vulkan and ggml-rd agree to 3.1e-7 (4096 tokens) and 3.5e-7 (512) on
   the DiT block's f32 arm; host ggml-cpu is 4.5e-5 from both GPUs there
   (its GELU reads an f16 table, `GGML_GELU_FP16`, the likely cause; not
   isolated), and ggml-vulkan vs ggml-cpu on the native arm is 5.57e-4 at
   both sizes (two different bf16 roundings of the activations). On the
   Qwen layer ggml-vulkan vs ggml-cpu reads 2.09e-4 / 1.4e-5, on the sparse
   level 2.74e-4 / 4.8e-7. The control: ggml-vulkan in its default mode
   (f16 accumulation for f16 matmuls, coopmat2's f16 conversion of f32
   operands on the RTX 4090) is 1.10e-3 from ggml-rd's f32 arm on the 8^3
   block and FAILs the 1e-4 limit (`graph/oracle-control-vk-default-dit8.txt`).
3. **Barrier elision is exact on the apps' graphs, and saves little.** Every
   output of every graph is bit-identical with and without elision, and two
   elision runs are bit-identical (the kernels are deterministic: no atomics,
   fixed reduction orders). The apps' graphs are chains: elision drops 5 of
   44 barriers in the Qwen layer, 211 of 570 in the sparse level (the 27
   independent get_rows/mask/mul_mat gathers of a conv), 7 of 69 in the DiT
   block.
4. **A dropped barrier is a race; the comparison catches it when it
   happens, and it does not always happen.** 20 of 39 (Qwen), 16 of 48
   (sparse), 8 of 48 and 5 of 48 (DiT, 512 and 4096 tokens) single dropped
   barriers changed the output. The ones that did not are of three kinds,
   all visible in `run-graph_*.log`: the guarded write stores what the
   buffer already held (the KV-cache CPY writes the same row into a
   persistent leaf every run, so the CONCATs that read it see the right
   bytes either way: all 6 CONCAT drops in the Qwen layer); a
   write-after-read hazard (gallocr reuses a buffer; the later writer only
   corrupts a reader it overtakes); and a short producer that finishes
   before its consumer is scheduled (in the sparse level only the barriers
   before the accumulating ADDs show, never the ones before the mask MUL or
   the MUL_MAT). The same drop is detected in one run and not in another
   (Qwen: 17, 19, 20 and 21 of 39 over four runs; DiT: 9, 8 and 5 of 48,
   the 4096-token block the fewest). So a missing barrier cannot be
   tested for by running; it is prevented by construction (elision places
   one on every byte-range RAW/WAR/WAW overlap, and barrier-all is the
   reference it must match). The detected ones changed the output by rel-L2
   8.2e-3 or more in the Qwen and sparse runs, but a DiT dev run had one at
   6.8e-4, under the 1e-3 tolerance: bit-identity, not a tolerance, is the
   check that catches them.

**G3.cost** (`graph/parked-in-guest-ref/run-cost_*.log`, RD only: the
cost probes never ran a reference, so the parked run's numbers stand; the
rule-8 rerun on this build reads the same, 10.5 ms and 16.1 ms of host
time, `graph/wrappers.txt`): the graph as
the app builds it every call (a context, the graph, a gallocr, the inputs,
`ggml_backend_graph_compute`, the output read), 5 (decode) or 3 (DiT) timed
calls after a cold one, then one call profiled per dispatch. Host
microseconds are the host's clock (`rdc::host_usec`, `Time.get_ticks_usec`
through a name-cache slot no recording name uses: 0.12 us per reading), GPU
time is RenderingDevice timestamps around the compute list, frames are the
pump's WAIT_GPU + COOP yields from building the graph to reading its output.
Medians:

| | skin-tokens decode step | Pixal3D flow forward |
|---|---|---|
| graph | `qwen_graph_evaluator::decode`: embedding rows, 28 decode layers, norm, logits [33036 x 2]; 2 beams, past 514; f16 | `trellis2_ss_flow_forward` with Pixal3D's proj: stem, t-embedder, 30 blocks, head; 4096 tokens; bf16 blocks |
| nodes / dispatches / barriers per graph | 2135 / 1265 / 1123 | 3107 / 2115 / 1903 |
| frames per graph | **1** (cold: 4, the pipelines and uniform sets COOP) | **1** (cold: 4) |
| ggml-rd host time, graph_compute entry to submit | **10.6 ms** (cold 29.0): pack 3.7, prepare 0.3, params upload 0.06, record 3.6, submit 3.0 | **16.9 ms** (cold 37.4): pack 5.2, prepare 0.4, upload 0.1, record 5.9, submit 5.1 |
| per node / per dispatch | **4.98 us / 8.41 us** | **5.43 us / 7.97 us** |
| the app's graph build + gallocr (guest) | 7.4 ms + 5.7 ms | 10.5 ms + 9.7 ms |
| GPU time | **9.2 ms** (steady; 97-100 ms on the first steps while the GPU clocks up, 15.7 ms in a dev run on the shared GPU) | **1.32 s** (1.84 s in a dev run) |
| whole call (vsync off) | 33.7 ms | 1.36 s |

Per dispatch (the profiled call, `kernel=` rows in the logs), packing is
1.9-3.9 us (FLASH_ATTN_EXT 5.9) and recording (pipeline and set binds, the
barrier, the dispatch: host calls into RenderingDevice) 2.7-3.3 us, the same
for every kernel; Godot's `submit` adds 2.3-2.4 us per dispatch (its render
graph). What that means for the apps:

- **The decode step is host-bound and latency-bound, not GPU-bound.** Of
  33.7 ms, 24 ms is guest CPU work (the app's graph build and allocation,
  13 ms; ggml-rd's pack/record/submit, 10.6 ms) and 9.2 ms is the GPU, whose
  1265 dispatches are 1123 barrier-separated steps of about 8 us each (the
  vec matmuls are 4-10 us, K6). A skin-tokens run's 383 decode steps are
  then about 13 s. The obvious next steps are the app's: build the decode
  graph once and reuse it (llama.cpp's graph reuse), which removes 13 ms per
  step; ggml-rd's recording could then be cached as well (the params table
  changes only in the positions).
- **The flow forward is GPU-bound**: 1.32 s of GPU against 37 ms of host
  work. It is 13.1 TFLOP (10.05 in the token-wise matmuls, 335 GFLOP a
  block; 3.09 in the 4096 x 4096 self-attention, 103 GFLOP a block), so
  about 10 TFLOP/s overall, the rate K6's tiled bf16 matmul reaches on its
  own; the per-kernel split is not measured here (the timestamps bracket
  the whole list). pixal3d-ggml's sparse-structure sampler (12 Euler steps
  by default, CFG inside part of the interval) is 12-24 forwards: 16-32 s of
  GPU at this rate. K6's and K8's next steps (wider register blocks, f16
  loads, float4 groupshared rows) are the levers.
- **One frame per graph** (rule 4): the submit's frame ends at the WAIT_GPU
  that ggml's `ggml_backend_graph_compute` (compute, then synchronize)
  yields, and the next frame's sync blocks for the rest of the GPU time
  (`wait_us`: 9.4 ms and 1.32 s).

Run it:
```
tests/ggml_graph_oracle/build.sh             # the host oracle, once (C:/b/...; ggml-vulkan.cpp alone is ~20 min of clang -O3)
gates/3-ggml-rd/graph/run.sh                 # builds the oracle if needed, runs the gate: graph/results.txt
gates/3-ggml-rd/graph/run.sh runs=graph_qwen,graph_sconv,graph_dit8,graph_dit   # this run
godot --path project --script probe_ggml_wrappers_graph.gd --rendering-driver vulkan --xr-mode off   # graph/wrappers.txt (rule 8)
<oracle>.exe --graph=dit:8 --dump=build/graph-dumps/graph_dit8 --ref=vulkan --vk=default --check=cpu   # the default-mode control
```
User arguments after `++` (run.sh passes its own through): `runs=...`
(only those; verdicts for those only), `--dit=dit:8` (graph_dit at 8^3
tokens), `--out=<folder>`, `--dump=<dir>`, `--wall=<s>` (default 3600),
`--oracle=<exe>` (run.sh sets it; without it every graph run FAILs). The
oracle gets 1200 s per graph and is killed past it. The gate's Sandbox keeps
`memory_max` 3600 MB (the DiT block's RD outputs, 25 MB per output at 4096
tokens, several runs of them for the bit comparisons) and
`execution_timeout` 4000000.

## The CPU fallback: ggml-rd with no RenderingDevice (Linux, no GPU)

**Result so far (2026-09-24, a Linux container with no GPU, the Linux addon
built from the org's godot-sandbox fork): the fallback runs the same graphs
the GPU runs, and the host oracle accepts them.** With no RenderingDevice,
RD0 still exists: every dispatch goes through the kernel's
`slangc -target cpp` emit on guest memory (`guest/ggml-rd/rd_cpu.cpp`), the
same packers, the same 64 params words, the runner the host L2 harness
generates (`tests/ggml_rd_kernels/gen_host_kernels.py`, one memory block per
storage binding). Rule 2's second target, not ggml-cpu: rule 10 is untouched.
`GGML_RD_CPU_FALLBACK=0` on the first job turns it off (the no-device
control, `gate_ggml_rd.gd -- fallback=off`).

| run | on the fallback | on the RTX 4090 |
|---|---|---|
| host L2 (`tests/ggml_rd_kernels`, Linux clang 18) | 403/403, control 380 detected, 23 no-op | 356/356 (before the five MotionBricks kernels) |
| `-o ADD -p ne=[1,1,1,1]` | OK=2 FAIL=0, Backend RD0: OK, 4.6 s | |
| `-o ADD`, all 54 cases | 30 OK, then the `[1,1,65536,1] x 256` case hit rule 10's per-vmcall cap (614 s) | 1590 s for all 1700 cases |
| G3.graph Qwen3 decoder layer, f16, vs host ggml-cpu | rel-L2 7.508e-4 (f16 weights), 1.665e-7 (f32 arm); 45 dispatches in 6.9 s per run | 2.10e-4, 9.2e-6 |

What the profile says (`GGML_RD_PROFILE=1` now prints the fallback's graph,
kernel and buffer-op times; the gate takes `--env=GGML_RD_PROFILE=1`): on a
2M-element ADD the kernel takes 0.95 s, the buffer ops 20 ms, and the case
127 s, so the time is test-backend-ops' own work on the guest CPU (the cost
the GPU runs pay too: their pumps averaged 0.93 s). The 16.7M-element ADD
cases do not fit rule 10's cap on the interpreter at all: the killed vmcall
had done its allocation and one 64 MiB upload in under 0.1 s and spent the
rest before the first readback, in the harness's graph copy. Native
translation of the ELF is the fix, not a wider cap.

**Native translation, measured (2026-09-24, the same container):**
`ggml_test.elf`'s translation (61 MB of C from `project/tools/bintr_emit.gd`,
6.4 min with clang 18 -O2, 20 MB object) against the interpreter, on the same
ELF (hash 63ab8527) and the same case:

| run | interpreter | translated | speed-up |
|---|---|---|---|
| `ADD [1,1,1920,1] x [32,32,1,1]`, whole case | 126.4 s | 18.1 s | **7.0x** |
| the same case, the `add_f32` kernel alone | 0.916 s | 0.123 s | **7.4x** |
| `ADD [1,1,65536,1] x 256`, time to rule 10's cap | 605.6 s | 85.0 s | 7.1x, still killed |

The capped case is still killed because rule 10's cap counts guest
instructions (214,577 units of 2^20), and a translation runs the same
instructions faster, not fewer: the ~5 minutes the cap stands for is ~40 s of
translated work. A per-vmcall cap for translated ELFs needs its own number
(a measured gate, rule 5), not the interpreter's. The Qwen layer did not
speed up (7.2 s a run, 6.9 s before) because it ran untranslated: its gate
makes the Sandbox with `memory_max` 3600, the translation's defines include
the arena size, so its hash differs from the 2048 MiB one baked here. A
translation is per (ELF, memory_max): bake one per Sandbox configuration.

Two things the CPU path judges differently: the dropped-barrier control is
not applicable (no barrier is placed, `drop_control n/a` in the SUMMARY),
and the rd-vs-rd f16/f32 arm comparison is exactly 0 (the same emit reads
the same values).

Two Linux findings on the way: the Linux addon's Sandbox defaults
(`allocations_max` 4000, `memory_max` 32 MiB) and Ubuntu clang 18's rv64gcv
code (`vsetivli` + `vl1r.v` struct copies trap as `Illegal opcode` on the
Linux addon; guest ELFs are built with `SANDBOX_RISCV_EXT_V=OFF` now).

## How ggml-rd works

- **One params table, no push constants.** Every dispatch of a graph owns a
  256-byte slot (64 words, `ggml_rd_params.h`, generated from
  `lean/Ggml/SlangCodegen/Common.lean`): kernel id; dst, s0, s1, s2 blocks
  (ne, nb, offset, in elements); op_params raw; derived words 53-63 (53 =
  threads, 54 = groups along x). One `buffer_update` of the whole table
  precedes `compute_list_begin`; the table starts at 8192 slots and doubles
  when a graph needs more. A pooled 16-byte uniform buffer per slot, bound
  as set 1, tells the kernel its base.
- **Fixed descriptor layout.** Set 0: b0 params (read-only), b1-b3 sources,
  b4 destination (all read-write); set 1: b0 slot. Set-0 uniform sets are
  cached by their five buffer RIDs and bind under every pipeline; an unused
  source binding names the destination's buffer (no new dependency).
- **One ggml buffer = one RD storage buffer**, created empty and cleared on
  the GPU; ggml sees a fake base `0x1000 + (index << 40)`; alignment 256,
  16 bytes of slack per tensor, 2 GiB maximum by default
  (`GGML_RD_MAX_BUFFER_MB` raises it; 0F showed 4 GiB - 256 working).
  set_tensor is `buffer_update`, get_tensor `buffer_get_into`, memset and
  clear `buffer_clear` (or bytes when non-zero or unaligned), cpy_tensor
  `buffer_copy`.
- **graph_compute** packs every node (layout-only and empty nodes skipped),
  makes the missing pipelines and sets (COOP every 256), uploads the
  table, records one compute list with a barrier only on a byte-range
  RAW/WAR/WAW overlap since the last barrier (`GGML_RD_BARRIER_ALL=1`: after
  every dispatch), submits and returns. `synchronize` and every read run
  the wait hook first. `GGML_RD_FAULT=<n>` moves a source of every n-th
  dispatch one element (the control).
- **The pump.** A job (test-backend-ops, a probe, later a model) runs on a
  guest fiber (`guest/fiber`, from Gate 0F). It yields WAIT_GPU after a
  submit, COOP during long setup, READ/UPLOAD for host files (the guest
  cannot open files, 0F probe 3), DONE or ERROR (a ggml abort becomes ERROR
  through `ggml_set_abort_callback`). `project/infer_host.gd` serves it once
  per frame, reads and uploads in the same frame up to 512 MB.
  Weights load with `ggml_backend_rd_tensor_upload(tensor, off, path,
  file_off, n)`: the host reads the file and `buffer_update`s the RD buffer,
  so they never enter the guest heap.

## File layout

```
lean/Ggml.lean                          Ggml.kernels / Ggml.controls: one import + one ++ line per family
lean/Ggml/SlangCodegen/Common.lean      the fixed layout, the params words, helpers, entry1D, paramsHeader
lean/Ggml/SlangCodegen/Binary.lean      ADD/MUL f32 with broadcast: the reference kernel (+ its control)
lean/Ggml/SlangCodegen/Move.lean        K2's shared pieces; Cpy, GetRows, Concat, Repeat: its 15 kernels
lean/Ggml/SlangCodegen/MulMat*.lean     MUL_MAT (K6): MulMat (shared), MulMatTiled, MulMatVec, MulMatSerial
lean/Ggml/SlangCodegen/Conv.lean        IM2COL (f32, f16 columns) and CONV_3D (f32, f16 kernel), family K7
lean/EmitGgml.lean                      lake exe emit_ggml <outDir> [<params header>]
kernels/ggml/kernels.txt                kernel names; a kernel's id is its line index
kernels/ggml/controls.txt               control kernels, gates only
kernels/ggml/cpp_siblings.txt           <kernel> <serial sibling>: group-memory kernels' cpp stand-ins
kernels/ggml/gen.sh                     check (default) | --update | --no-emit (build.sh)
kernels/ggml/gen_ggml_kernel_table.py   the fixed-layout check -> GgmlKernelTable.inc
kernels/ggml/{slang,cpp}/, GgmlKernelTable.inc   generated, committed
guest/ggml-rd/ggml-rd.h                 the public API: registry, attach, hooks, upload, stats
guest/ggml-rd/ggml-rd.cpp               registry, device, buffer type, buffer, backend (API v2)
guest/ggml-rd/rd_graph.cpp              graph_compute
guest/ggml-rd/rd_kernels.cpp            pipelines, the params table, slot sets, the set-0 cache
guest/ggml-rd/rd_pack.{h,cpp}           the packer core (no RenderingDevice; the L2 harness links it)
guest/ggml-rd/ggml_rd_params.h          generated from Common.lean, committed
guest/ggml-rd/ops/<op>.cpp              one packer file per op family; ops/binary.cpp is the template
guest/ggml-rd/ops/move.h                K2's iteration order (16-bit destinations), shared by its packers
guest/ggml-rd/ops/mul_mat.cpp           MUL_MAT: vec (ne11 <= 4) or tiled, r2/r3, the grid
                                        (ops/im2col.cpp, ops/conv3d.cpp: K7)
guest/pump/, guest/fiber/               the pump protocol on Gate 0F's fiber
guest/ggml_test/                        ggml_test.elf: test-backend-ops and the probes on the pump
                                        (probe_conv.cpp: K7's timed census shapes;
                                        probe_graph.cpp: G3.graph and G3.cost)
guest/ggml_test/app_graphs/             the apps' graph builders, copied (CITATION.cff)
project/infer_host.gd                   the host side of the pump
project/gate_ggml_rd.gd                 this gate (OPS lists the ops under test)
project/gate_ggml_graph.gd              G3.graph and G3.cost (graph/)
tests/ggml_rd_kernels/                  L2: main.cpp, l2.h, cases/<family>.cpp, build.sh
```

## Adding a kernel (the checklist for the op families)

1. **Lean.** `lean/Ggml/SlangCodegen/<Family>.lean`: build each kernel with
   `Common.kernelModule t0 t1 t2 td helpers (entry1D TG body)` (or your own
   entry over the same globals); element types are per binding (f16/bf16
   inputs can stay `uint` words read with `fnLdF16 "s0"` / `fnLdBf16 "s0"`,
   bf16 as uint16 << 16); read words with `pwN k` / `pfN k` (op_params from
   37; derived 53-63, of which 53 and 54 belong to `entry1D`'s grid and
   55-63 are the op's own); pin every kernel's text with `native_decide`; export
   `kernels : List (String × SlangShaderModule)`. Add one import and one
   `++ <Family>.kernels` line to `lean/Ggml.lean`. A kernel that shares
   group memory (`GroupMemoryBarrierWithGroupSync`) has no cpp emit
   (slangc E36107): give it a serial sibling with the same words,
   thread-group size and grid (`MulMatSerial.lean` is the pattern) and a
   `<kernel> <sibling>` line in `kernels/ggml/cpp_siblings.txt`; gen.sh then
   emits cpp for the sibling only, the table check requires equal
   thread groups, and L2 runs the sibling wherever the packer picked the
   kernel.
2. **List.** One line per kernel in `kernels/ggml/kernels.txt`.
3. **Emit.** `kernels/ggml/gen.sh --update` writes `kernels/ggml/slang/`,
   `cpp/` and `GgmlKernelTable.inc` and fails on any spirv-val error or
   layout difference; `ggml_rd_params.h` changes only if Common does. Its
   default mode (and `GGML_EMIT=1 ./build.sh`) then checks that nothing
   committed differs from Lean.
4. **Packer.** `guest/ggml-rd/ops/<op>.cpp`, including `../rd_pack.h` only:
   `supports(op)` accepts exactly what the kernel computes (types, shapes,
   `tensor_fits`, at most 3 sources); `pack(p)` sets
   `p.kernel = kernel_index("<name>")`, the derived words, and the grid
   (`grid_1d(p, threads)`, or `p.groups` with each dimension <= 65535);
   `GGML_RD_OP(ident, GGML_OP_X, sub or -1, supports, pack)`. The device's
   `supports_op` is the registered packers' `supports`, so there is no
   other list to edit, and no CMake edit (globbed). `ops/binary.cpp` is the
   template.
5. **L2.** `tests/ggml_rd_kernels/cases/<family>.cpp` with `L2_CASES`
   (~20 shapes: permuted, broadcast, odd sizes; test-backend-ops thresholds;
   `cases/binary.cpp` is the template; set `control_src = 1` when the op
   reads src0 for its shape only), then
   `tests/ggml_rd_kernels/build.sh`: `l2.log` and `l2-control.log` PASS
   (each checkout builds in its own `C:/b/ggml-rd-l2-<hash>`).
6. **L0, L1.** `gates/3-ggml-rd/kernels/l0.sh` and `l1.sh` PASS.
7. **L3.** `./build.sh`, add the op to `OPS` in `project/gate_ggml_rd.gd`
   (the lead merges that line), run the gate: 0 FAIL, and the per-op line
   shows the op's cases OK. `++ --ops=<yours> --fault-ops=<yours>
   --out=ops-<family>` gates only your ops, into your own folder.

A kernel with groupshared memory and workgroup barriers has no cpp emit
(slangc refuses the barrier on the cpp target, E36107). Give it a
`<k>_serial` sibling in the same Lean module (one thread per output row, no
groupshared, the same arithmetic), list both in kernels.txt, and have the
packer pick the sibling under `GGML_RD_SERIAL=1`: gen.sh then skips the
tiled kernel's cpp and requires the sibling, the L2 harness (which sets the
switch) runs the sibling, and the same switch is the GPU A/B
(`ops/flash_attn_ext.cpp` and `project/gate_ggml_rd_serial.gd` are the
example). Census shapes that test-backend-ops' own list misses go in
`guest/ggml_test/census_cases.inc`.

Rules a kernel must keep:
- A source and the destination are usually the same RD buffer, bound twice.
  In place (dst is src0) a thread must read what it needs before it writes;
  no thread may read an element another thread of the same dispatch writes.
- No push constants, no extra bindings; set 0 and set 1 exactly as Common.
- f16 output: declare `dst` as `half`, or write two halves per word from one
  thread; the cpp target has no InterlockedAnd/Or.
- Group counts <= 65535 per dimension; `grid_1d` splits 1-D launches.
- Group-shared memory and barriers: slangc's cpp target rejects them
  (E36107), so a kernel `<k>` that uses them lists a sibling `<k>_serial`
  in kernels.txt, with the same params and grid, one thread per group, and
  the same arithmetic in the same order. `gen.sh` then emits no cpp for
  `<k>`, and the host tests run `<k>_serial` in its place. `Rows.lean`
  (families K3/K4) is the template; `ops-k3k4/README.md` has its gate.
- A family runs the gate on its own ops, into its own folder:
  `gate_ggml_rd.gd -- --ops=<OPS> --fault=<ops> --out=gates/3-ggml-rd/ops-<family> [--probe=<name>[:arg]]`.

## K8: FLASH_ATTN_EXT (no mask)

**Result: PASS.** ggml's test-backend-ops in the guest, `-o FLASH_ATTN_EXT
-b RD0`: **150/150 OK, 0 FAIL**, including the 6 census cases (D = 128,
H = 12, f32 Q/K/V, no mask, prec F32: Lk = 5, 1029, 4096, 912; Lq up to
912). The same 150 pass with `GGML_RD_SERIAL=1` (the serial siblings on the
GPU, `ops/fa-serial/`). The not-supported cases are masks, sinks, head sizes
other than 64 and 128, and K/V types other than f32/f16 (bf16, quantized, or
K and V of different types), none of which the census uses.

| check | verdict | numbers |
|---|---|---|
| L0 | **PASS** | `lean/Ggml/SlangCodegen/FlashAttn.lean`: 8 kernels, the f32 tiled D = 64 and D = 128 and the f32 serial D = 64 pinned in full, the other five pinned as text relations of those (f16 = the f32 text with `uint` K/V, `ld_f16_*` reads and Q rounded to f16; serial D = 128 = D = 64 with wider loops), groupshared 43,264 bytes at D = 128 (<= 48 KiB). |
| L1 | **PASS** | all 8 pass spirv-val and the fixed-layout check; the 4 serial emits compile for riscv64; the 4 tiled kernels have no cpp (slangc refuses workgroup barriers on the cpp target, E36107), which gen.sh and l1.sh now require and report. |
| L2 | **PASS** | 21 FA cases vs native ggml-cpu (`cases/flash_attn_ext.cpp`: D 64/128, f32/f16 K/V, GQA in dims 2 and 3, permuted Q/K/V, K/V as views of a longer cache, softcap, prec DEFAULT, Lk = 2, 5, 31, 32, 33, 113 ... 1029, N = 1, 3, 15, 16, 17, 75, FA then ADD), NMSE <= 5e-4 (f32 K/V: <= 4.3e-13; f16: <= 2.1e-5, ggml-cpu accumulates f16 V in f16). Control: Q's nb1/nb2 swapped is caught in 19/19 FA cases where it moves an address (2 have N = 1). |
| L3 | **PASS** | `ops/results.txt`: ADD,MUL,FLASH_ATTN_EXT 250 OK (FLASH_ATTN_EXT 150), 0 FAIL, the same 250 with a barrier after every dispatch, the fault control 54/54 ADD FAIL, rule 4 0, 1654 s; headless control PASS. ops_main took 883 s, 150 more frames than ADD,MUL alone: the in-guest reference is the cost. |
| A/B | **PASS** | `ops/fa-serial/results.txt` (`project/gate_ggml_rd_serial.gd`): the same run with `GGML_RD_SERIAL=1`, the 4 serial pipelines on the GPU: 150 OK, 0 FAIL, 762 s, rule 4 0. |

**The kernel.** One work group = 16 queries of one head, 128 threads as 16
rows × 8 lanes; K/V tiles of 32 keys in groupshared; each lane scores 4 keys
of its row, then every lane of the row runs the online softmax (running max
and sum) over the 32 scores and accumulates its own D/8 output columns in
registers (unrolled in Lean). Scale and softcap come from op_params (the
scale divided by the softcap on the host, as ggml-cpu does); grouped-query
attention is the ratio words 55-58. With f16 K, Q is rounded to f16 first,
as ggml-cpu's `vec_dot_type` does. The `_serial` sibling does the same
arithmetic per query row in one thread, without groupshared; it is what the
host L2 harness runs (`GGML_RD_SERIAL=1`, set by the harness), and on the
GPU it is the A/B.

**Timing** (`perf/fa.txt`, `project/perf_ggml_fa.gd`, RTX 4090 shared with
other agents' runs): a one-node graph computed repeatedly, one submit and one
WAIT_GPU per frame; the median interval between WAIT_GPUs is an upper bound
on the node's GPU time (the Lq = 16, Lk = 5 floor is 0.72 ms).

| shape (D = 128, H = 12, f32, no mask) | census calls | tiled, median ms | TFLOP/s | serial, median ms |
|---|---|---|---|---|
| Lq = Lk = 4096 (the hottest) | 60 | 29.3 (min 28.1) | 3.5 | 75.0 |
| Lq = 4096, Lk = 1029 | - | 12.5 | 2.1 | - |
| Lq = 912, Lk = 1029 | 30 | 2.34 | 2.5 | 8.79 |
| Lq = Lk = 912 | 30 | 2.29 | 2.2 | - |
| Lq = 4096, Lk = 5 (proj cross-attention) | per block | 0.84 (floor 0.72) | - | - |

The tiled kernel runs at 3-4 TFLOP/s f32 at the hottest shape, about 4-5 %
of the 4090's f32 peak: it is latency-bound (43 KiB of groupshared leaves 2
groups of 4 warps per SM, 3 barriers per 32 keys), not bound by shared-memory
bandwidth (about 6 ms of LDS traffic at 4096 × 4096). The next steps, if
G3.cost puts FA on the critical path: float4 groupshared rows (4x fewer LDS
instructions), Q held in registers with a subgroup reduction of the
partial dot products (drops the Q tile, raising occupancy), and 64-key tiles.

**Shared changes K8 made** (for the lead's merge):
- `kernels/ggml/gen.sh`: a kernel whose Slang has
  `GroupMemoryBarrierWithGroupSync` gets no cpp emit, and kernels.txt must
  list its `<k>_serial` sibling; `gates/3-ggml-rd/kernels/l1.sh` skips its
  riscv64 compile and checks the sibling instead.
- `tests/ggml_rd_kernels`: `gen_host_kernels.py` writes runners only for
  kernels with an emit; `main.cpp` sets `GGML_RD_SERIAL=1`.
- `vendor/ggml/tests/test-backend-ops.cpp`: a third guarded edit,
  `GGML_GUEST_EXTRA_EVAL_CASES` includes `guest/ggml_test/census_cases.inc`
  at the end of the eval list (upstream tests f32 K/V only at head sizes 64
  and 72); CITATION.cff says so. Other families can add their census shapes
  there.
- `guest/ggml_test`: the `fa_perf` probe; `GGML_RD_SERIAL` is one of the
  switches a job's env resets.
- `project/gate_ggml_rd.gd`: `OPS` gains FLASH_ATTN_EXT, and `WALL_S` goes
  from 1800 to 3600 s (FLASH_ATTN_EXT's in-guest reference alone is about
  720 s per run, and the gate runs it twice).

## Regression: the shared layers

`guest/rd_compute` gained `storage_buffer_empty`, `buffer_get_into`,
`memory_usage`, `device_name`, compute-list recovery and 16 MiB splits,
`vendor/sandbox-api` the 16 MiB mem* split, and the guest build relative
source paths and ggml's own commit stamp; every ELF links or takes all of
them. Every earlier gate was re-run on the rebuilt ELFs (`regression/`,
each gate's own result files copied, the committed ones left as they
were):

| gate | runner | verdict | numbers |
|---|---|---|---|
| Stage 1 | `gate_rd_compute.gd` | **PASS** | `regression/1-rd-compute.log`: the held-device probe passes and every barriered count is exact; the no-barrier arm still loses counts (75 of 256), as Stage 1 recorded. |
| Stage 2 | `gate_avbd.gd` | **PASS** | `regression/2-avbd-results.txt`: gradcheck 5/5 on cpu and rd (worst rel 1.04e-6 and 9.59e-7), backward_smoke 18/18 on both, self-collision pair sets identical (4 cases), `same_frame_syncs=0` over 339 submits. |
| Gate 0F | `gate_runtime.gd` | **PASS** | `regression/0f-runtime-results.txt`: `SUMMARY: PASS=34 FAIL=5 INFO=27` in 185 s; the five FAILs are probe 3's (no guest filesystem), as committed; Cut 4's probe 17 (memalign) passes with its control. Probe 13's killed-call row now has two arms: recovery off reproduces the refused `buffer_update`, recovery on lets it through with one recovery counted. Before the arms were split, a re-run read one FAIL more: that row expected the refusal `rd_compute` now prevents. |
| Gate 4 | `gate_curvenet.gd`, `probe_curvenet_wrappers.gd` | **PASS** | `regression/4-curvenet-*`: guest vs native identical in 9/9 checks (verdicts, integers, float signatures), a repeat byte-identical, 0 `Eigen::` in 13,547 symbols, rule 8 23/23 wrappers; curvenet.elf is rebuilt with this cut's CMake. |
| lean | `gates/lean/verify.sh` | **PASS** | `regression/lean-*`: the tree builds clean, the 24 AVBD kernels re-emit with 0 DIFF, and the numthreads control gives exactly 1 DIFF (vbd_init). |

G3.graph and G3.cost changed three shared pieces, and every ELF was rebuilt:
`rd_compute` gained `rdc::host_usec()` (Time.get_ticks_usec, its method name
placed in the host name-cache slot of `create_local_rendering_device`, which
no recording calls; the 32 RenderingDevice names keep their 32 slots, as
every run's stats line says), `rd_graph.cpp` gained `GGML_RD_DROP_BARRIER`
and `GGML_RD_PROFILE` (neither set by default; at profile level 0 no clock
is read), and the pump counts its yields. Re-run on the rebuilt ELFs
(`graph/regression/`):

| gate | runner | verdict | numbers |
|---|---|---|---|
| G3.ops subset | `gate_ggml_rd.gd ++ --ops=ADD,MUL,CPY,DUP,CONT,GET_ROWS,CONCAT,REPEAT` | **PASS** | `graph/regression/ops/results.txt`: every probe (chain, independent, files, alias rw and its ro control, perf, mm_perf, census) PASS; 425/425 OK, the same 425 barrier-all, fault controls 54/54 ADD and 129/129 data-movement FAIL; rule 4 at 0. |
| Stage 1 | `gate_rd_compute.gd` | **PASS** | `graph/regression/1-rd-compute.log`: the held-device probe passes, every barriered count exact, the no-barrier arm still wrong in 4 of 7. |
| Gate 0F | `gate_runtime.gd` | **PASS** | `graph/regression/0f-runtime-results.txt`: `SUMMARY: PASS=34 FAIL=5 INFO=27` in 134 s, the five FAILs probe 3's, as committed. |
| rule 8 | `probe_ggml_wrappers_graph.gd` | **PASS** | `graph/wrappers.txt`. |

Stage 2, Gate 4 and the lean gate were not re-run for this change: they
link `rd_compute` but call none of what changed.

## Differences from the plan text

- The plan's set 0 has read-only sources (b1-b3); they are read-write
  (finding 1).
- The plan's "-preserve-params" needs `-O0` with it (finding 2, Gate 0F
  finding 9).
- Weights are uploaded through `ggml_backend_rd_tensor_upload` and an
  upload hook, which the pump serves as UPLOAD; the plan named the request
  only.
- G3.graph's DiT block is Pixal3D's (TRELLIS.2's block with ProjectAttention's
  `proj_linear` added, cross-attention over 5 global tokens), built by
  pixal3d-ggml's TRELLIS.2 builder plus that one line; the sparse-conv level
  is one whole shape-decoder level (four convs: the child head's conv2, two
  ConvNeXt blocks' and the up-block's conv1; 108 gathers), not a single
  conv.
  "A dropped barrier is detected" became a sweep over every barrier (up to
  48), because a single drop is a race that may not happen (finding 3 of
  G3.graph).

## Files

- `ops/results.txt` (the GPU run, streamed; last line RESULT), `ops/run.log`
  (Godot's stdout and stderr), `ops/run-<name>.log` (each job's captured
  output), `ops/results-headless.txt` and `ops/run-headless.log` (the
  control), `ops/run-headless-xr-default-hung.log` (the same without
  `--xr-mode off`, which hung).
- `kernels/l0.sh` → `lean-build.log`, `lean-negative-control.log`;
  `kernels/l1.sh` → `l1.log`; `tests/ggml_rd_kernels/build.sh` →
  `kernels/l2.log`, `kernels/l2-control.log`.
- `ops/wrappers.txt` (`project/probe_ggml_wrappers.gd`): rule 8.
- `regression/`: the earlier gates' results on this cut's ELFs.
- `graph/results.txt` (G3.graph and G3.cost, streamed; last line RESULT),
  `graph/run.log`, `graph/run-<name>.log` (each probe's output: every
  comparison, every dropped barrier, the per-kernel cost tables),
  `graph/wrappers.txt` (rule 8), `graph/regression/` (this change's
  regression: the rule-8 run's log, a G3.ops subset with every probe and
  control, Stage 1, Gate 0F).
