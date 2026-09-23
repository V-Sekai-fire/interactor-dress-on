# AGENTS.md — how to work in interactor-dress-on

Standing rules for any agent (or person) touching this repo. Amend this file
when a rule is added or changed; the plan and READMEs cite it, they do not
duplicate it.

## What this is

A garment authoring loop for Godot: image → curvenet → dress-on → drape, as
**godot-sandbox RISC-V guest ELFs** that require only the `godot_sandbox`
addon and GPU rendering. No engine fork, no engine module, no GDExtension of
our own, no host DLL. The GPU is reachable only through Godot's
`RenderingDevice`, via `guest/rd_compute` (Stage 1, measured).

**One scoped exception (user, 2026-09-23): Stage 7 calls out.** Image →
mesh is not in the sandbox: the loop calls the org's Pixal3D model service
over HTTP (`V-Sekai-fire/interactor-pixal3d-image-to-textured-mesh`;
`POST /predict`, `/extract`) from GDScript. It answers with **OpenUSD**
(a `.usdz`: USDC layer, UsdPreviewSurface material, textures), never GLB,
and the loop reads it in the guest (the org's `flow-*` USD importer as a
sandbox ELF; Gate 0G). The service runs natively from a **pixi**
environment (win-64 on the desk, not Docker; a Linux image only for RunPod);
no card is pinned, each launch takes the next GPU in round robin
(`tools/services/svc_common.py`). Multi-view Pixal3D (the `_mv`
checkpoints) is the next stretch goal; VoxHammer comes after it, as an edit
agent over the loop's meshes scored with MaskScore (user, 2026-09-23). No GDExtension, no host DLL: curvenet, fit and drape stay guest
ELFs, the rig (4b) stays on ggml-rd. Without a reachable service the INFER
state fails loudly, never a silent fixture.

## Rules

1. **Stay inside github.com/V-Sekai-fire.** Every dependency comes from the
   org's repo or fork (`ggml`, `cloth-fit`, `pixal3d-ggml`, `entities-godot`,
   `godot-sandbox`, `transport-xr-grid`, `transport-godot-mcp`,
   `interactor-mujoco-sandbox-demo`, `skin-tokens-ggml`; for `lean/`:
   `contract-lean-slang`, `plausible`, `plausible-witness-dag`). If only a `V-Sekai/`
   or upstream copy exists, ask before forking it in. Push only to org remotes.
2. **One source, two targets: Lean → Slang → `cpp` | `spirv`.** Kernels are
   generated from `lean/` (a squashed git subtree of cloth-dynamics' `lean/`,
   cited in `lean/CITATION.cff`) by `lake exe emit_shaders`
   (`kernels/avbd/gen.sh`; `CLOTH_LEAN` overrides the path); `slangc -target cpp` is the in-guest CPU path
   (`AvbdCpu`), `slangc -target spirv` is the GPU path (`AvbdRd` over
   `rd_compute`). Never copy prebuilt kernels in; never hand-write a kernel
   that Lean could emit. Optimizers too: L-BFGS-B is written in Lean→Slang, not
   pulled in as a library.
3. **No Eigen in the drape ELF.** The drape is Lean-generated kernels plus an
   Eigen-free driver (`guest/avbd/avbd_sim.h` lineage). Eigen survives only
   inside `fit.elf` (PolyFEM), a separate ELF.
4. **State machines and queues, not waits.** Never `sync()` in the frame that
   `submit()`s; the host advances each guest's state machine from `_process`
   and reads back once the fence is known-done. Batch iterations into one
   compute list (`AvbdRd::run`).
5. **Batch / CPU / GPU chosen per problem.** Small meshes run on `AvbdCpu`,
   large ones on `AvbdRd`; thresholds come from a measured gate, never a
   guess.
6. **Composition over one monolith.** One ELF per stage in its own Sandbox
   node (`curvenet`, `infer`, `fit`, `drape`), composed by `project/main.gd`;
   meshes cross through the host as packed arrays; `rd_compute` is a static
   lib they share.
7. **Gates before building on assumptions.** Anything that could void the plan
   gets a runnable gate under `gates/` with its logs kept as evidence, a
   README that states the result (including negative ones), and a flat
   control that separates "the sandbox/VR blocked it" from "nothing was there".
8. **Every guest entry point gets a no-argument wrapper in `project/main.gd`**
   so MCP `call_method` needs no argument marshalling (Gate 0E). main.gd is a
   thin root: the wrapper lives in the stage file and main.gd keeps a
   same-named delegate with the same defaults; `tests/probe_main_wrappers.gd`
   FAILs on any `ADD_API_FUNCTION` without one.
9. **Do not touch the user's machine config.** OpenXR runtime is selected per
   process with `XR_RUNTIME_JSON` (OXRSys, Windows port, at
   `tools/oxrsys/build/windows/runtime/oxrsys-runtime.json`; its Qt simulator
   shows the stream); the system default and SteamVR's settings are not ours to
   flip.
10. **Guest inference runs on ggml-rd.** ggml-cpu only where a measured
    profile shows it much faster, per graph and recorded (like rule 5).
    Every ggml-cpu run has a hard ~5-minute timeout, and a timeout is a
    FAIL: on the host `timeout 300`; in the guest the Sandbox's
    `execution_timeout`, set for every vmcall of a job that runs ggml-cpu
    (`infer_host.gd`, `GGML_CPU_TIMEOUT_UNITS`). Chaining resumable
    sub-5-minute jobs is allowed but frowned upon: a last resort, documented
    in the gate's README.

## Facts that cost time (do not relearn)

- Run Godot with `--rendering-driver vulkan`; `--headless` hands back a null
  RenderingDevice. `--xr-mode off` on non-VR runs, headless ones included:
  without it a headless run hung after OpenXR failed to start, twice
  (Gate 3).
- Godot buffers stdout when redirected: **poll the port**, or write results to
  a file from GDScript; never wait on the log. `--quit-after` never fires under
  `--xr-mode on`; quit on a wall clock, in every branch.
- RenderingDevice enums are pinned by hand in `guest/rd_enums.h`
  (`SHADER_STAGE_COMPUTE=4`, `UNIFORM_TYPE_UNIFORM_BUFFER=7`,
  `UNIFORM_TYPE_STORAGE_BUFFER=8`). `RID::RID(const Variant&)` links only
  while unused; use `v.operator ::RID()`.
- The render graph does **not** order same-buffer dispatches inside one
  compute list; `compute_list_add_barrier` is mandatory between dependent
  dispatches (it is `end()+begin()`+rebind, there is no cheaper form).
- Each span between barriers is one graph command, and it keeps a buffer's
  **first** usage only (release builds drop a second one silently). A buffer
  first bound read-only and then written in the same span is recorded as
  read-only, and later spans race its writes (Gate 0F finding 4: 0 of 4096
  exact). Bind a written buffer read-write first; never alias a read-only
  source binding with the read-write destination for an in-place op. A
  storage binding that can share a buffer with a written one is declared
  read-write even when the kernel only reads it: ggml-rd's sources are, and
  one buffer at b1 and b4 over 1000 dependent spans is exact where the
  read-only control loses (`gates/3-ggml-rd/`).
- `buffer_update` is refused inside a compute list; `RDUniform` binds whole
  buffers; every buffer is created with contents (zeros if none), except
  `rdc::Device::storage_buffer_uninit` for sizes a zero array cannot reach
  (4 GiB−256), which must be `buffer_clear`ed (`storage_buffer_empty` does
  both). `buffer_get_data` stages the whole buffer; read big buffers through
  `Device::buffer_get_into` (a `buffer_copy` into a small staging buffer).
- A guest static can hold the RenderingDevice across vmcalls (handle = engine
  instance id in unrestricted mode); RefCounted helpers are per-call only.
- An RID (any handle a host call returns) is a per-vmcall scoped Variant: the
  guest holds an index into that call's Variant table, which names something
  else in the next call. Anything kept across vmcalls must be made permanent
  (`Variant::make_permanent`, as `rdc::Device` does); permanent slots are
  min(references_max + guest globals, 65534) and must be freed (`ECALL_VSTORE_GLOBAL`, `Device::forget`).
- The guest clock is not a clock (it jumps between time bases). Time on the
  host, around the vmcall.
- `Sandbox.references_max` defaults to 100; ~30 uniform sets in one call trip
  it. Host scripts set 4096 (or more). It only grows: a lower value set later
  reads back but is not applied. Recording on permanent RIDs uses none.
- `Sandbox.memory_max` (MiB, default 512): the heap is 0.8 x it and must end
  below 4 GiB, so ~5112 is the most that loads (~4088 MiB of heap). Set it
  before `program=`; after `program=` only a larger value is applied.
- The guest has no filesystem (`openat` → EBADF); every byte comes through
  the host. `fesetround` is accepted and ignored.
- `slangc -preserve-params` keeps unused bindings only at `-O0`; at the
  default `-O1` the optimiser strips them again (Gate 0F finding 9) while
  the reflection JSON still lists them. Shared-layout kernels compile at
  `-O0 -preserve-params`, and the layout check reads the SPIR-V
  (`kernels/ggml/gen_ggml_kernel_table.py`).
- `slangc -target cpp` rejects `GroupMemoryBarrierWithGroupSync` (E36107):
  a kernel that shares group memory has no cpp emit. It gets a serial
  sibling (same words, thread-group size and grid, same sums in the same
  order, no group memory; e.g. `lean/Ggml/SlangCodegen/MulMatSerial.lean`)
  named in `kernels/ggml/cpp_siblings.txt`, which the L2 harness runs in
  its place. A sibling with another thread group is named `<k>_serial`
  instead: run in place when it is one thread per group over the same
  grid (NORM, SOFT_MAX), else chosen by its packer under
  `GGML_RD_SERIAL=1` (FLASH_ATTN_EXT; `tests/ggml_rd_kernels/gen_host_kernels.py`).
- godot-sandbox caches an Object call's method name in a 32-slot direct-mapped
  cache keyed by the guest ADDRESS of the name string; two hot names in one
  slot evict each other and each call re-resolves (~2-5 ms). It is decided
  by the link layout, so it moves between builds, not processes: the old
  "bimodal submit+sync" (~70 µs or ~2.4 ms) reproduces as 62-174 µs vs
  2.4-2.8 ms per submit with `compute_list_end` and `sync` in one slot, and
  it was the 50-86x rd regression of Cut A (`gates/2-avbd/perf-bisect.log`).
  Call RenderingDevice only through `rdc::Device`, whose method names sit at
  addresses with a slot each. That pool is **full** (32 names, 32 slots,
  since Gate 3's timestamps): a new name needs an old one retired first.
- GPU time: `rdc::Device::capture_timestamp` before `list_begin` and after
  `list_end`; on a local device both are readable right after that submit's
  `sync()` (`get_captured_timestamp_gpu_time`, nanoseconds). ggml-rd wraps
  it as `ggml_backend_rd_set_timestamps` / `GGML_RD_TIMESTAMPS=1`.
- Cross-compile: `build.sh` (riscv64 clang from scoop, lld, the org's
  `riscv64-sysroot` via `RISCV64_SYSROOT`). First Godot run after adding an
  ELF: `godot --path project --headless --import`.
- `lean/` is a subtree. To take upstream changes, split in a **scratch clone**
  of cloth-dynamics (`git subtree split --prefix=lean -b lean-split`, ~2 min
  for 425 commits; the clone keeps the split refs out of the real checkout),
  then `git subtree pull --prefix=lean <clone> lean-split --squash` here. Keep
  the V-Sekai-fire URLs in `lean/lakefile.lean` and `lake-manifest.json` when
  resolving. `lake build` from an empty `lean/.lake` takes 50-134 s (three
  packages fetched and built; `gates/lean/`); `lean/.lake/` is ignored.
  After `gates/lean/verify.sh`, `git status` must show only its logs.
- LeanSlang is `V-Sekai-fire/contract-lean-slang` (the renamed `lean-slang`;
  the old URL redirects, but pin the canonical one) at branch `emit-fp`,
  **pinned by SHA** (60532ae), not `main`. `emit-fp` is v0.0.6 plus `half`,
  `double`, `litHalf`, `litInt`, `cast` and `litFloatExact`/`litDoubleExact`,
  additive, so the AVBD emission is byte-identical. `litFloat` prints six
  decimals (1e-12 emits as 0.000000): any literal that is not a multiple of
  1e-6 must use `litFloatExact` (binary32) or `litDoubleExact`. `main` adds a libslang FFI `extern_lib` as a default target
  (vendored SDK headers, Linux link flags) that breaks `lake exe` on Windows.
  Changing the URL: delete `lean/.lake/packages/LeanSlang` first, then
  `lake update LeanSlang` (only that package; the other revs must not move).
- The guest heap also caps **live allocations**: `Sandbox.allocations_max`
  defaults to 10000 ("Too many arena chunks"). fit.elf holds ~79k after
  `fit_begin`; `stages/fit_stage.gd` sets 4,000,000 before `program=`.
- Unqualified `abs(double)` binds to C's `int abs` under the guest's
  libstdc++ (clang `-Wabsolute-value`) but to the double overload under
  llvm-mingw's libc++, so native and guest silently differ. Treat that
  warning as an error in anything shared between them.
- libstdc++ (guest) and libc++ (native) leave **tied sort keys** in different
  orders (`std::sort`, `nth_element`, `partial_sort`), and a sum taken in that
  order differs by one ULP. fit.elf and fit_native part after 7 Newton
  iterations while inputs, LDLT and the libm the solver calls are bitwise
  equal (`gates/6-fit`, 6.0); SimpleBVH's Morton sort, which has such ties,
  is the *likely* cause (hypothesis: no call site instrumented yet). Test an
  index tie-break before relying on it. Confirmed at one call site: Geogram's
  Hilbert sort (`nth_element` on one coordinate) inserted a skirt panel's
  tied boundary points in another order, DMWT tiled the panel differently,
  and an index tie-break in `Hilbert_vcmp` made curvenet.elf bit-identical to
  its native control again (`gates/4-curvenet/README.md`, skirt (c)).
- **Guest out-of-memory is not `std::bad_alloc`.** Below the heap floor a
  failed allocation is a `Protection fault` at the malloc ecall (the vmcall
  aborts) or a segfault that kills Godot (exit 139), and the heap's meminfo
  cannot see in-phase peaks. Size `memory_max` with a ladder of fresh
  Sandboxes, then add 1.25x headroom. For fit.elf on foxgirl the floor is
  352 MiB, run at 440.
- `execution_timeout` (2^20-instruction units) must cover a whole vmcall. One
  fit phase is up to 5.4e5 units, 67x the 8000 default. Read a call's
  instructions from the guest with `rdinstret`; libriscv counts from the
  vmcall's start.
- Starting several Godot processes in the same second segfaulted one once.
  Stagger launches by a few seconds.
- Godot imports every `.obj` under `res://` as a mesh and fails on line-only
  ones (skeletons). Data OBJs live under a `.gdignore`d directory
  (`project/fixtures/`) and are read as text (`util/obj_io.gd`).
- Make a stage's Sandbox with `stages/sandbox_util.gd` (memory_max,
  references_max, execution_timeout before `program=`; a missing ELF or
  entry point is a reason, not an error). In XR the root viewport reads back
  black: screenshot a SubViewport on the same World3D.
- The GPU and the guest CPU round the same Slang differently (the driver
  forms FMAs slangc's cpp build does not), so a sum at float noise can be
  exactly 0 on rd and not on cpu. Guard every division by a computed norm in
  the Lean kernel (Gate 5 G10: a bending hinge went NaN on rd only).
- A local RenderingDevice drops whatever is recorded between `submit()` and
  `sync()` (sync's `_begin_frame` clears the graph): sync before any
  buffer_update/copy/clear or compute list, and let the host upload only
  while the guest is idle.
- A vmcall killed while recording (execution_timeout, references_max, a
  trap) leaves its compute list open, and Godot then refuses every
  buffer_update and list_begin on that device. `rdc::Device` ends the
  orphaned list before its next such call (`recoveries()` counts them;
  Gate 0F probe 13 has the hazard and the fix as two arms).
- The host views at most **16 MiB** of guest memory per syscall: a
  PackedByteArray made from guest memory or fetched into it, and a
  memcpy/memset/memmove/memcmp, fault above that ("Protection fault").
  `rdc::Device` splits buffer_update and staged reads, the pump splits READ,
  and `vendor/sandbox-api` splits the mem* wrappers.
- **Why rule 10:** the guest CPU (rv64gc, one thread) runs ggml-cpu at
  ~0.1 GFLOP/s: a 4096-token DiT block's two reference arms would be hours
  (the parked Gate 3 run stalled there). ggml-rd has no CPU fallback (an op
  it does not support is refused, never sent to ggml-cpu), and no graph has
  a recorded profile that moves it to ggml-cpu; the in-guest ggml-cpu is
  only G3.ops' single-op reference (test-backend-ops, the census probe),
  each vmcall capped at 214,577 units (300 s at ~0.75 G instructions/s).
  Oracles for anything bigger
  run on the HOST: the guest dumps its outputs (`main.gd`'s
  `ggml_graph_dump`, `project/graph_dump.gd`) and `tests/ggml_graph_oracle`
  rebuilds the same net from the same seeds (`guest/ggml_test/graph_nets.cpp`,
  compiled for both) on host ggml-vulkan (the GPU, large graphs) or host
  ggml-cpu (small ones). ggml-vulkan is an oracle only (its glslc shaders
  never ship; rule 2 is about shipped kernels) and must run with f16,
  coopmat, coopmat2, bf16 and integer dot off (`--vk=precise`): by default
  its f32 arm is 1.1e-3 off (f16 accumulation, coopmat2's f16 conversion).
  Host ggml-cpu is not an exact f32 reference either: 4.5e-5 from both GPUs
  on the DiT block's f32 arm, where ggml-rd and precise ggml-vulkan agree to
  3.5e-7 (its GELU reads an f16 table, `GGML_GELU_FP16` in ggml-cpu's
  `vec.h`, the likely cause; not isolated). `ggml-vulkan.cpp` takes ~20 min to compile with
  llvm-mingw clang -O3: build the oracle once, outside the checkout (C:/b).
- With unboxed arguments (the default) declare an Object parameter as
  `Object`, never `Variant`: the host passes a bare handle, and a Variant
  parameter reads it as a pointer (it arrives as Nil).
- Guest threads are serialized (Gate 0C), so ggml's spin barriers never
  release: every in-guest ggml-cpu backend runs one thread (ggml_test.elf
  wraps `ggml_backend_init_by_type` to set it; the default is 4).
- Bash heredocs with apostrophes and long scripts fail in this harness; write
  scripts with the Write tool and run them.
- godot-sandbox's guest heap has no aligned entry point (malloc/calloc/
  realloc/free are syscalls into a host heap that keeps its bookkeeping
  outside guest memory and hands out 16-byte alignment). Upstream's
  `memalign` fallback (behind `posix_memalign`, `aligned_alloc`, aligned
  `new`) returned an already-freed block whenever 16 malloc tries missed a
  > 16-byte alignment: nearly always at 4096, sometimes at 64 (Gate 4's
  "Possible double-free" in Geogram). Fixed in `vendor/sandbox-api`'s
  `native.cpp`: over-allocate, return the aligned address, and map it back
  to the host block in a side table that the wrapped `free`/`realloc`
  consult (a header below the block cannot work: the host only frees the
  pointer it returned). Gate 0F probe 17 checks it (ggml's 64-byte
  buffers need it too, Gate 3); re-vendoring sandbox-api must keep that
  patch and Gate 3's 16 MiB mem* split, or the bugs return silently.
- A native flat control built with llvm-mingw links libc++; the guest links
  libstdc++. `std::shuffle` and `std::uniform_*_distribution` differ
  between them from the same seed: use the engine's raw output (Gate 4).
- In a guest, a `std::vector<std::vector<T>>` kept alive in a long-lived
  record (a `std::vector` of records, across stages) corrupted libc state:
  "Illegal opcode" at an `ecall` in memmove/memcpy/puts/fflush, garbage
  syscall numbers. As a local it is fine, and ASan on the host is clean. Keep
  long-lived guest data flat (gates/5-drape/drape/README.md). Godot's
  `--gpu-index` order is not stable (index 1 was the RTX 4090 on one boot,
  index 0 on the next): check the adapter line in the log. The native
  references ran on the 4090; the drape is bit-identical on the 3090.
- DiffCloth's sphere demo is chaotic after its self-collision onset (step
  ~70): a 9.7e-9 change of mu (mu0 vs float32(mu0)) moves the 350-step
  dL/dmu by 4.4%, and a 4.4e-9 change moves it by 7.8% and the loss by 6.5%.
  Compare with the native run at its exact seed-1 mu, 0.5397701956236457
  (`kNativeSphereMu0`), never the printed 0.539770; at the exact value every
  printed per-step statistic matches native to step 70 (gates/5-drape/trace).
- An L-BFGS-B guard that LBFGSpp writes with `numeric_limits<double>::epsilon()`
  as an absolute threshold keeps 2^-52 in the float32 kernels (`dblEps`), not
  FLT_EPSILON: the sphere demo's gradients are ~1e-5, so fpp = g.g ~ 1e-10,
  and FLT_EPSILON there shrank the first Cauchy step to nothing.
- test-backend-ops cannot fail an op on values where its output holds
  infinities: `nmse()` sums `-inf - -inf = NaN`, and `NaN > max_err` is
  false (DIAG_MASK_INF passes with every source read one element off).
  Its inf check still catches a wrong position or sign. Judge such ops
  with an inf-aware NMSE, as L2 and the census probe do
  (gates/3-ggml-rd/k1k5).

## Conventions

- Commit messages: plain prose, five whys, the numbers. **No Claude
  annotations** — no `Co-Authored-By`/`Claude-Session` trailers, no
  assistant attributions in files.
- Plan of record: `~/.claude/plans/declarative-soaring-cake.md` (the user's
  plan file); status table at its top. Amend it when a decision changes.
- Vendored code carries a `CITATION.cff` in the org's form (title, abstract,
  authors, repository-code, commit, license) naming the source and the local
  adaptations; never a PROVENANCE.txt.
