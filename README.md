# interactor-dress-on

A self-contained godot-sandbox guest for garment authoring: image → curvenet →
dress-on → drape.

**The binding constraint is independence.** This is a RISC-V ELF that requires
only the `godot_sandbox` addon and GPU rendering. No engine fork, no engine
modules, no GDExtension of our own, no host DLL. Everything else — Cassie's
curvenet and surfacing, cloth-fit's retargeting, DiffCloth's AVBD solver, ggml
and TRELLIS.2 — is vendored into the ELF.

That constraint has one consequence that shapes the whole design: the GPU is
reachable only through Godot's `RenderingDevice`. So ggml gets a RenderingDevice
compute backend, and AVBD's SPIR-V kernels use the same layer. One GPU
abstraction serves both the inference and the cloth solve.

## Gates

Things that could void the plan run first, and stay in the tree as runnable
evidence.

| gate | question | result |
|---|---|---|
| [0A](gates/0a-renderingdevice/) | Can a sandbox guest drive RenderingDevice compute? | **PASS** — stock Godot 4.7.2, RTX 4090 |
| [0B](gates/0b-crosscompile/) | Does the heavy C++ cross-compile for riscv64? | **PASS** — ggml, PMP, Geogram clean; cloth-fit 179/184 TUs, single blocker (OpenVDB → libigl); Godot Delaunay2D → Geogram's |
| [0C](gates/0c-threads/) | Do guest threads run? | **PASS-SEQUENTIAL** — they complete, never overlap |
| [0D](gates/0d-openxr/) | Does stock Godot's OpenXR reach a runtime with an HMD? | **PASS** — OpenXR-Simulator 1.5.0 via per-process `XR_RUNTIME_JSON`; SteamVR/VDXR present no HMD without a headset |
| [0E](gates/0e-mcp/) | Drive the RD probe over transport-godot-mcp | **PASS** — `tools/call` → `call_method(/root/Main, rd_probe)` returns the 0A result |

## Stages

| stage | what | result |
|---|---|---|
| [1](gates/1-rd-compute/) | `rd_compute`, the one GPU layer (`guest/rd_compute.{h,cpp}`, a static lib) | **PASS** — device held across vmcalls; barriers mandatory (the graph does not order same-buffer dispatches); 6–13 µs per in-list call; the old "bimodal submit+sync" was godot-sandbox's 32-slot method-name cache colliding by string address (fixed in `rd_compute`, Cut A) |
| [2](gates/2-avbd/) | AVBD in the guest: Lean → Slang → `cpp` (`AvbdCpu`) and `spirv` (`AvbdRd`), one driver | **PASS** — forward, duals, backward (gradcheck 5/5, stategrad 12/12 + 12/12, within 1.04e-6 of native) and self-collision exact on both; rd from 256 vertices (frame-driven, 3.2–4.8 ms/substep to 4096 vertices) |
| [5](gates/5-drape/) | `drape.elf`: DiffCloth's Simulation (sphere demo) forward and backward (native, step, unrolled) and L-BFGS-B, every vector op a Lean-emitted kernel, Eigen-free, cpu and rd | **FAIL, 2 of 9**. G1, G3, G4, G6, G7, G8 and G9 pass: the LBFGSpp components (20/20); inverse_min (the target to 1.2e-6 and 4.1e-6, in LBFGSpp's iteration counts); every printed per-step statistic matching native to step 70 at native's exact μ; unrolled against FD within 0.043; native's μ sequence 0.539770 → 0.010000 → 0.375146 from backwardLog's values; and `auto` = rd from 160 vertices (the 90 fps crossover). G2 is 19/20 (one delta-stopped trace, 2.1e-5 vs 1e-6; LBFGSpp itself moves 4.8e-6 under float32 I/O). G5 has dL/dμ 5.02% off at μ₀ (limit 5%) and the loss at 0.01 printing 1.650 vs 1.652, where moving μ by 2e-7 moves dL/dμ 7.5%: the demo is chaotic after its self-collision onset |
| [6](gates/6-fit/) | `fit.elf`: cloth-fit's garment retarget (PolyFEM, ipc-toolkit, Lean SDF sampler) in the guest, one phase per vmcall on a worker Thread | **PASS** on the five-part criterion — foxgirl in 42 min (219 Newton, ~20–25× native), no intersections, 0 file opens, fit gap 1.77/3.87 voxels, Hausdorff 7.42 to the oracle; `fit_weight=0` and the 5 cm push caught. Guest ≠ native bitwise after Newton 7 (not libm; likely libstdc++/libc++ sort-tie order, a hypothesis). memory_max floor 352 MiB (run at 440), phase ≤ 5.4e5 × 2^20 instructions; Stage 8: ≤ 1k-vertex garment, ~60 Newton |
| [8](gates/8-loop/) | The loop: body → rig → pen → curvenet → mesh → fit → check → drape, `project/stages/` composed by `pipeline.gd` | **PASS FIXTURE:infer,rig** — flat, VR (OXRSys, per-process `XR_RUNTIME_JSON`) and MCP. The scripted pen's rings are boundary strokes, so curvenet gives 2 cycles, 2 openings and the 2 panels, one welded tube of 932 vertices (2 loops); fit.elf fits it in 1393 s (2 phases, 253 Newton; the plan's "30-iteration caps" made it slower and then threw), `OK none` (control INTERSECTS); drape.elf drapes 100 finite rd steps at scale 10 against the body mesh collider, 21.5 ms/step. VR equals flat (integers equal, fitted vertices Δ = 0). Controls: drop-seam → FAILED(MESH), 0 cycles; push-vertex → INTERSECTS. 114 guest entry points, each with a `/root/Main` wrapper. Infer and rig are still the FoxGirl fixtures |

The standing rules are in [AGENTS.md](AGENTS.md).

Three rules shape everything after Stage 1: **state machines and queues, not
waits** (no `sync()` in the frame that `submit()`s; the host advances each
ELF's state machine from `_process`); **batch / CPU / GPU chosen per problem**
(the in-guest `AvbdCpu` is a first-class backend for small meshes); and
**composition over one monolith** — one ELF per stage in its own Sandbox node
(`curvenet`, `infer`, `fit`, `drape`), meshes crossing through the host as
packed arrays.

## Host-side addons (stock Godot, GDScript only)

- **VR pen** — `V-Sekai-fire/transport-xr-grid`'s `procedural_3d_grid`: XR controller
  strokes → `pen_begin / pen_point / pen_end` on the guest.
- **MCP** — `V-Sekai-fire/transport-godot-mcp`: editor bridge on 8788, in-game
  bridge on 8789, so an agent can play the project and `call_method` into the
  sandbox.

Neither touches the ELF. VR adds an OpenXR runtime to the host requirements.

## Layout

```
guest/     rd_compute (static lib), rd_enums.h, main.cpp (dress_on.elf, Stage 1)
           later: curvenet/ infer/ fit/ drape/ — one ELF per stage
vendor/    sandbox-api (guest API + cmake helper), xr-grid; later ggml, trellis2,
           skin-tokens, cassie subset, cloth-dynamics, cloth-fit
kernels/   .slang sources and their .spv (probe, accumulate; AVBD's to come)
project/   the stock-Godot project: main.gd composes the Sandbox nodes,
           gate_*.gd / probe_*.gd / control_*.gd are the runnable evidence
gates/     0A–0E and the stage records, with their logs
tools/     OXRSys runtime + Qt simulator, Windows port (gitignored; tools/oxrsys/scripts/windows_build.ps1)
build.sh   riscv64 cross-build of every ELF into project/
```
