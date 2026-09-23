# Gate 3 — ggml-rd: ggml over RenderingDevice, kernels from Lean

**Result: G3.ops PASS for ADD and MUL.** `ops/results.txt`: RESULT: PASS.
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
| L2 | cpp emits + the guest's packers vs native ggml-cpu | **PASS** | 41/41 cases, every one bit-exact (NMSE 0): broadcast in each dimension, permuted src1, overlapping views, odd sizes, a strided in-place destination and a two-node chain (`kernels/l2.log`). Control: src0's nb1/nb2 swapped after packing is caught in all 37 cases where the swap moves an address; the other 4 have nb1 = nb2 (`kernels/l2-control.log`). |
| L3 | G3.ops, test-backend-ops in the guest | **PASS** | ADD,MUL: 100 OK (ADD 54, MUL 46), 0 FAIL, 90 not supported (f16), in 101 frames (one WAIT_GPU per submit), 149 s. Barrier-all: the same 100. Fault control: 54/54 ADD FAIL. The same verdicts in five runs this session, four of them before the rebase onto Cut 4. |
| L3 | probes (`guest/ggml_test/probes.cpp`) | **PASS** | chain: 256 in-place ADDs on one tensor, one graph, x = 256 exactly (256/256), 255 barriers, elided or all. independent: 64 ADD/MULs into separate outputs, 64000/64000 exact, **0 barriers** with elision, 63 with barrier-all. files: READ of a 16 KiB host file, then `ggml_backend_rd_tensor_upload` of the same file into a tensor at byte 512 of its RD buffer (UPLOAD), 4096/4096 exact, and x + x on the GPU 4096/4096 exact (including -0, 1e-30 and FLT_MAX + FLT_MAX = inf). alias: see finding 1. |
| rule 4 | no sync in its submit's frame | **PASS** | `rule4_same_frame_syncs=0` over every run; `close: permanent_slots=0`. |
| rule 8 | `main.gd`'s wrappers, as MCP calls them | **PASS** | `ops/wrappers.txt` (`project/probe_ggml_wrappers.gd`): `ggml_attach`, the chain/files/alias presets and a short fault run through `ggml_ops_start`, each pumped by `main.gd`'s own `_process`; a second `ggml_pump` in the start frame does not pump again; `ggml_rd_close` ends at `permanent_slots=0`, rule-4 counter 0 over 23 submits. |
| control | headless, no RenderingDevice | **PASS** | `no RD device`, `Testing 1 devices`, no RD0 case run, 0.5 s. Without `--xr-mode off` the headless process hangs after OpenXR fails to start and never runs the script: twice, killed at 900 s and at 300 s (`ops/run-headless-xr-default-hung.log` is the second). AGENTS.md's `--xr-mode off` holds for headless runs too. |
| regression | Stage 1, Stage 2, Gate 0F, Gate 4 and lean on the rebuilt ELFs | **PASS** | `regression/`: see below. |

Not in this cut: G3.graph (a Qwen3 layer, a DiT block, elision vs
barrier-all bit-identity on real graphs) and G3.cost. The in-guest
reference dominates the time: ops_main is 125-164 s of vmcalls over 101
frames (five runs on a shared machine), most of it ggml-cpu at rv64gc on
the 16.7M-element cases.

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
lean/EmitGgml.lean                      lake exe emit_ggml <outDir> [<params header>]
kernels/ggml/kernels.txt                kernel names; a kernel's id is its line index
kernels/ggml/controls.txt               control kernels, gates only
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
guest/pump/, guest/fiber/               the pump protocol on Gate 0F's fiber
guest/ggml_test/                        ggml_test.elf: test-backend-ops and the probes on the pump
project/infer_host.gd                   the host side of the pump
project/gate_ggml_rd.gd                 this gate (OPS lists the ops under test)
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
   `++ <Family>.kernels` line to `lean/Ggml.lean`.
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
   `cases/binary.cpp` is the template), then
   `tests/ggml_rd_kernels/build.sh`: `l2.log` and `l2-control.log` PASS
   (each checkout builds in its own `C:/b/ggml-rd-l2-<hash>`).
6. **L0, L1.** `gates/3-ggml-rd/kernels/l0.sh` and `l1.sh` PASS.
7. **L3.** `./build.sh`, add the op to `OPS` in `project/gate_ggml_rd.gd`
   (the lead merges that line), run the gate: 0 FAIL, and the per-op line
   shows the op's cases OK.

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

## Differences from the plan text

- The plan's set 0 has read-only sources (b1-b3); they are read-write
  (finding 1).
- The plan's "-preserve-params" needs `-O0` with it (finding 2, Gate 0F
  finding 9).
- Weights are uploaded through `ggml_backend_rd_tensor_upload` and an
  upload hook, which the pump serves as UPLOAD; the plan named the request
  only.

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
