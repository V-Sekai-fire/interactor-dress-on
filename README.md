# interactor-dress-on

A garment authoring loop for Godot: image → curvenet → dress-on → drape.

**The binding constraint is independence.** Every stage the loop runs
in-process is a **godot-sandbox RISC-V guest ELF** that needs only the
`godot_sandbox` addon and GPU rendering: no engine fork, no engine module, no
GDExtension of our own, no host DLL. The GPU is reachable only through Godot's
`RenderingDevice` (`guest/rd_compute`), so the cloth solver's kernels and
ggml's compute backend (ggml-rd) share one GPU layer, and every kernel is
generated from Lean (`lean/` → Slang → `cpp` | `spirv`).

**One scoped exception: Stage 7 calls out.** Image → mesh (Pixal3D) runs as
the org's HTTP model service, launched from a **pixi** environment, and hands
its result back as **OpenUSD** (a `.usdz` with real geometry, material and
textures). The loop reads USD in the guest: Gate 0G shows OpenUSD runs as
a sandbox ELF, so the org's `flow-*` importer can become one. See [AGENTS.md](AGENTS.md).

## Where it stands

**The loop runs end to end** (Gate 8: `PASS FIXTURE:infer,rig`, flat, in VR
and over MCP, and again from a fresh clone of `main`): a scripted pen authors
a skirt on the body, `curvenet.elf` builds a welded 932-vertex tube,
`fit.elf` dresses it onto the body intersection-free, and `drape.elf` drapes
it on the GPU against the body mesh. The body and the rig are still the
FoxGirl fixtures; Stage 7 (image → mesh) and Stage 4b (the rig) replace them.

## Gates

Things that could void the plan run first, and stay in the tree as runnable
evidence (logs, a README that states the result, a flat control).

| gate | question | result |
|---|---|---|
| [0A](gates/0a-renderingdevice/) | Can a sandbox guest drive RenderingDevice compute? | **PASS** — stock Godot 4.7.2, RTX 4090 |
| [0B](gates/0b-crosscompile/) | Does the heavy C++ cross-compile for riscv64? | **PASS** — ggml, PMP, Geogram; cloth-fit (OpenVDB later replaced) |
| [0C](gates/0c-threads/) | Do guest threads run? | **PASS-SEQUENTIAL** — they complete, never overlap |
| [0D](gates/0d-openxr/) | Does stock Godot's OpenXR reach a runtime with an HMD? | **PASS** — per-process `XR_RUNTIME_JSON`; the loop's VR runs use OXRSys |
| [0E](gates/0e-mcp/) | Drive the guest over transport-godot-mcp | **PASS** — `call_method(/root/Main, …)` |
| [0F](gates/0f-runtime/) | What does the sandbox runtime allow? | **PASS 33 / FAIL 5 (expected)** — no guest filesystem (host uploads instead); heap = 0.8 × `memory_max` < 4 GiB; worker-thread vmcalls; a fiber across vmcalls; 4 GiB RD buffers; shared descriptor sets need `slangc -O0 -preserve-params`; a buffer bound read-only and read-write in one span loses writes |
| [lean](gates/lean/) | Is `lean/` (the vendored emitter tree) the one kernel source? | **PASS** — AVBD emission byte-identical; LeanSlang = `V-Sekai-fire/contract-lean-slang` emit-fp (half, double, exact literals) |
| [0G](gates/0g-openusd/) | Can OpenUSD (the org's `flow-*` USD importer) run as a sandbox ELF? | **PASS** — static OpenUSD 26.05 + oneTBB at rv64gc (a 4-file riscv64/plugin patch), a 31.8 MB probe ELF; plugins registered from memory; USDA and USDC load from bytes, 9/9 equal to host usd-core, corrupt input fails cleanly; hand the guest USDC (6.7 ms vs 836 ms) |

## Stages

| stage | what | result |
|---|---|---|
| [1](gates/1-rd-compute/) | `rd_compute`, the one GPU layer | **PASS** — device held across vmcalls; barriers mandatory; ~1–2 µs per RenderingDevice call once method names sit in their own slots of godot-sandbox's 32-slot name cache (a collision cost 2–5 ms per call: the old "bimodal submit+sync") |
| [2](gates/2-avbd/) | AVBD in the guest: `AvbdCpu` (Lean → cpp) and `AvbdRd` (Lean → SPIR-V) | **PASS** — forward, duals, backward (gradcheck 5/5, stategrad 12/12 + 12/12, within 1.04e-6 of native) and self-collision exact on both |
| [3](gates/3-ggml-rd/) | ggml-rd: ggml over RenderingDevice, every kernel Lean → Slang (`guest/ggml-rd`, gated in `ggml_test.elf`) | **PASS** — G3.ops: test-backend-ops 1700/1700 on the 22 census ops vs the in-guest ggml-cpu (the only in-guest reference: single ops), fault and headless controls fail as they must; G3.graph: the apps' own builders run in the guest on ggml-rd only, checked on the host (`tests/ggml_graph_oracle`): Qwen3 decode layer f16 and sparse-conv level f16 vs host ggml-cpu at rel-L2 2.1e-4 / 2.7e-4, the full 4096 × 1536 DiT block bf16 vs ggml-vulkan on the RTX 4090 at 8.8e-4 (the references' own activation rounding; f32 arms ≤ 9.2e-6, the DiT 3.1e-7), barrier elision bit-identical to barrier-all; G3.cost: ~5 µs host per node, one frame per graph, decode step 10.6 ms host + 9.2 ms GPU, flow forward 17 ms host + 1.32 s GPU |
| [4](gates/4-curvenet/) | `curvenet.elf`: Cassie (pen → curvenet → mesh, mesh → curvenet) on a namespaced godot-lite shim, Geogram Delaunay, PMP without Eigen, Lean curve kernels | **PASS** — 10 checks, guest == native bit for bit, including `skirt_tube` (rings drawn as boundary strokes are openings, the panels are the patches) |
| 4b | The rig: skin-tokens on ggml-rd | not started (next, after Cut 3) |
| [5](gates/5-drape/) | `drape.elf`: DiffCloth's sphere demo forward and backward, L-BFGS-B as Lean kernels, Eigen-free, cpu and rd, triangle-mesh body collider | **7 of 10** — G1, G3, G4, G6, G7, G8, G10 pass (LBFGSpp components 20/20; inverse_min exact; the demo matches native to print resolution through frame 50; unrolled backward vs FD ≤ 0.043; native's μ sequence reproduced; the rd NaN at scale 10 fixed). Open: G2 (float32 driver on Rosenbrock n=2), G5 (a 350-step gradient where 4.4e-9 in μ moves it 7.8%), G9 (the cpu/rd threshold drifts with machine load) |
| [6](gates/6-fit/) | `fit.elf`: cloth-fit's garment retarget (PolyFEM, ipc-toolkit) in the guest, OpenVDB replaced by a lazy brick-grid SDF with a Lean tricubic sampler | **PASS** (five-part criterion) — foxgirl in 42 min, intersection-free, 0 file opens, fit gap equal to the oracle's; guest ≠ native bitwise after Newton 7 (open, likely sort-tie order) |
| 7 | Image → mesh: Pixal3D (with MoGe-3 fov, NAF, BiRefNet_HR-matting) as a pixi-launched HTTP service returning OpenUSD (`.usdz`) | **in progress** (branch `services-win`: both services on native Windows under pixi) |
| [8](gates/8-loop/) | The loop: body → rig → pen → curvenet → mesh → fit → check → drape, `project/stages/` composed by `pipeline.gd` | **PASS FIXTURE:infer,rig** — flat, VR (OXRSys) and MCP; the fit takes 10.5 min for a 932-vertex skirt with `force_psd_projection` (23 min without; `gates/8-loop/flat-psd.txt`, `gates/6-fit/amdahl/`); drape 100 GPU steps at 21.5 ms; drop-seam and push-vertex controls fail as they should; 114 guest entry points, each with a no-argument `/root/Main` wrapper |

**Stretch goals:** multi-view Pixal3D (its `_mv` checkpoints); then VoxHammer as
an edit agent scored with MaskScore; MotionBricks
(`V-Sekai-fire/motion-bricks-ggml`) on ggml-rd in the sandbox, so the
garment drapes on a moving body.

## Rules that shape the design

The standing rules are in [AGENTS.md](AGENTS.md). The ones that shape
everything after Stage 1: **state machines and queues, not waits** (no
`sync()` in the frame that `submit()`s; the host advances each ELF from
`_process`); **batch / CPU / GPU chosen per problem** from a measured gate;
**ggml-rd by default, ggml-cpu only where a profile shows it much faster**,
and every ggml-cpu run under a hard 5-minute timeout; **composition over one
monolith** — one ELF per stage in its own Sandbox node, meshes crossing
through the host as packed arrays.

## Build and run

```sh
bash build.sh                          # every ELF into project/ (BUILD_FIT=0 skips fit.elf)
godot --path project --headless --import
godot --path project --rendering-driver vulkan --xr-mode off --script gate_avbd.gd      # a gate
godot --path project --script gate_loop.gd --rendering-driver vulkan --xr-mode off -- \
    --gate=loop --out=../gates/8-loop/results.txt --wallclock=3600 --allow-fixture=infer,rig
```

Weights live in the gitignored `models/`, pinned by `tools/models/manifest.tsv`.
`--headless` gives a null RenderingDevice; poll result files rather than
stdout (Godot buffers it). VR runs set `XR_RUNTIME_JSON` per process only.

## Layout

```
guest/     rd_compute (the GPU layer), jobs (frame-driven job queue), pump/ (fiber jobs),
           drape/ curvenet/ fit/ probes/ ggml_test/ — one ELF per stage; ggml-rd/ (ggml's
           backend over rd_compute); godot_lite/ (Cassie's shim)
lean/      the Lean emitter tree (subtree of cloth-dynamics lean/) + Cassie/ Drape/ Fit/ Ggml/ Probes/
kernels/   Lean-emitted Slang and its cpp: avbd/ cassie/ drape/ fit/ ggml/ probes/
vendor/    sandbox-api, xr-grid, cassie + geogram/pmp/mwt subsets, godot-core-subset,
           cloth-fit and ggml (subtrees); on branches: trellis2, skin-tokens
project/   the stock-Godot project: main.gd (thin root) + stages/*.gd, fixtures/,
           gate_*.gd / probe_*.gd, xr_main.tscn
gates/     every gate and stage record with its logs
tests/     host-native controls (curvenet, fit, godot_lite, lbfgsb oracle, drape kernels,
           ggml-rd kernels, the G3.graph oracle)
tools/     forks, native oracle recipes, model manifest, OXRSys (gitignored)
build.sh   riscv64 cross-build of every ELF into project/
```
