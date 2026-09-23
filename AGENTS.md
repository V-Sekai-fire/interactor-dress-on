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
   process with `XR_RUNTIME_JSON` (OpenXR-Simulator at
   `tools/openxr-simulator/openxr_simulator.abs.json`); the system default and
   SteamVR's settings are not ours to flip.

## Facts that cost time (do not relearn)

- Run Godot with `--rendering-driver vulkan`; `--headless` hands back a null
  RenderingDevice. `--xr-mode off` on non-VR runs.
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
- `buffer_update` is refused inside a compute list; `RDUniform` binds whole
  buffers; every buffer is created with contents (zeros if none).
- A guest static can hold the RenderingDevice across vmcalls (handle = engine
  instance id in unrestricted mode); RefCounted helpers are per-call only.
- The guest clock is not a clock (it jumps between time bases). Time on the
  host, around the vmcall.
- `Sandbox.references_max` defaults to 100; ~30 uniform sets in one call trip
  it. Host scripts set 4096 (or more).
- `submit+sync` from the guest is bimodal per process (~70 µs or ~2.4 ms,
  empty list, unexplained). Rule 4 makes it a latency, not a stall.
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
  **pinned by SHA** (e0e96da), not `main`. `emit-fp` is v0.0.6 plus `half`,
  `double`, `litHalf`, `litInt` and `cast`, additive, so the AVBD emission is
  byte-identical. `main` adds a libslang FFI `extern_lib` as a default target
  (vendored SDK headers, Linux link flags) that breaks `lake exe` on Windows.
  Changing the URL: delete `lean/.lake/packages/LeanSlang` first, then
  `lake update LeanSlang` (only that package; the other revs must not move).
- Bash heredocs with apostrophes and long scripts fail in this harness; write
  scripts with the Write tool and run them.

## Conventions

- Commit messages: plain prose, five whys, the numbers. **No Claude
  annotations** — no `Co-Authored-By`/`Claude-Session` trailers, no
  assistant attributions in files.
- Plan of record: `~/.claude/plans/declarative-soaring-cake.md` (the user's
  plan file); status table at its top. Amend it when a decision changes.
- Vendored code carries a `CITATION.cff` in the org's form (title, abstract,
  authors, repository-code, commit, license) naming the source and the local
  adaptations; never a PROVENANCE.txt.
