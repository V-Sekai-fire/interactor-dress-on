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

Two things could have voided the plan. They run first.

| gate | question | result |
|---|---|---|
| [0A](gates/0a-renderingdevice/) | Can a sandbox guest drive RenderingDevice compute? | **PASS** — stock Godot 4.7.2, RTX 4090 |
| 0B | Does the heavy C++ cross-compile for riscv64, and fit in the guest heap? | in progress |

## Layout

```
guest/     the ELF: ADD_API_FUNCTION surface, rd_compute, pipeline
vendor/    ggml + trellis2 + cassie subset + cloth-dynamics + cloth-fit
kernels/   AVBD .spv and ggml vulkan-shaders, embedded
project/   minimal stock-Godot project
gates/     the two gates, kept as runnable evidence
tests/     host-side native tests that need no sandbox
```
