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
   so MCP `call_method` needs no argument marshalling (Gate 0E).
9. **Do not touch the user's machine config.** OpenXR runtime is selected per
   process with `XR_RUNTIME_JSON` (OXRSys, Windows port, at
   `tools/oxrsys/build/windows/runtime/oxrsys-runtime.json`; its Qt simulator
   shows the stream); the system default and SteamVR's settings are not ours to
   flip.

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
  its place.
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
