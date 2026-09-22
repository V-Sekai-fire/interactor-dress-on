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
| [1](gates/1-rd-compute/) | `rd_compute`, the one GPU layer (`guest/rd_compute.{h,cpp}`, a static lib) | **PASS** — device held across vmcalls; barriers mandatory (the graph does not order same-buffer dispatches); 6–13 µs per in-list call; `submit+sync` bimodal per process (open) |

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
tools/     OpenXR-Simulator (gitignored; see gates/0d-openxr/)
build.sh   riscv64 cross-build of every ELF into project/
```
