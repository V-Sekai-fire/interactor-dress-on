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
| [0D](gates/0d-openxr/) | Does stock Godot's OpenXR reach SteamVR? | plumbing works; no HMD present on either runtime |
| 0E | Drive the RD probe over transport-godot-mcp | next |

## Host-side addons (stock Godot, GDScript only)

- **VR pen** — `V-Sekai/transport-xr-grid`'s `procedural_3d_grid`: XR controller
  strokes → `pen_begin / pen_point / pen_end` on the guest.
- **MCP** — `V-Sekai-fire/transport-godot-mcp`: editor bridge on 8788, in-game
  bridge on 8789, so an agent can play the project and `call_method` into the
  sandbox.

Neither touches the ELF. VR adds an OpenXR runtime to the host requirements.

## Layout

```
guest/     the ELF: ADD_API_FUNCTION surface, rd_compute, pipeline
vendor/    ggml + trellis2 + cassie subset + cloth-dynamics + cloth-fit
kernels/   AVBD .spv and ggml vulkan-shaders, embedded
project/   minimal stock-Godot project
gates/     the gates, kept as runnable evidence (0A–0D in)
tests/     host-side native tests that need no sandbox
```
