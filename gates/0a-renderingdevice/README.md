# Gate 0A — can a godot-sandbox guest drive RenderingDevice compute?

**Result: PASS.**

Everything in this repo rests on this. The guest must be independent of engine
modules, so the GPU is reachable only through Godot's `RenderingDevice`, called
out over the sandbox boundary. ggml's inference backend and AVBD's 24 SPIR-V
kernels both sit on that one path. If it did not work, no amount of porting
would help.

## What was proved

A statically linked riscv64 ELF, loaded by the prebuilt `godot_sandbox` addon
into **stock Godot 4.7.2** — not the engine fork, no modules, no GDExtension of
our own — reached `RenderingServer`, created a local `RenderingDevice`, compiled
a SPIR-V compute shader, allocated a storage buffer, dispatched, and read back
the value the kernel wrote.

```
Vulkan 1.4.351 - Forward+ - Using Device #0: NVIDIA - NVIDIA GeForce RTX 4090
spirv bytes: 552
PASS: GPU compute reached from the guest
```

The constant is `0x00C0FFEE`, chosen so the readback distinguishes "the kernel
ran" from "the readback succeeded against a zeroed buffer". Zero or a small
integer would not.

## Run it

```sh
# build the guest ELF (needs a riscv64-capable clang on PATH)
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=<mujoco-sandbox-demo>/third_party/riscv64-sysroot/toolchain.cmake
cmake --build build
cp build/rdprobe project/rdprobe.elf

# first run only: let Godot register the GDExtension
godot --headless --path project --import

# the gate itself -- NOT headless, see below
godot --path project --script gate.gd --rendering-driver vulkan
```

## Two things that cost time, recorded so they do not again

**`--headless` has no RenderingDevice.** The first run of this gate reported
`create_local_rendering_device: not an Object` and looked like a sandbox
restriction. It was not. Headless Godot uses a dummy renderer and returns null
to *anyone* who asks. `control.gd` is the control that settles it: run under
`--headless` it prints `<Object#null>`, and under `--rendering-driver vulkan` it
prints a live `RenderingDevice`. Keep that control. A gate that cannot tell "the
sandbox blocked it" from "there was no GPU to begin with" is not a gate.

**`UNIFORM_TYPE_STORAGE_BUFFER` is 8, not 6.** Six is `IMAGE_BUFFER`. The
validator says so precisely — *"Expected 'StorageBuffer', supplied:
'ImageBuffer'"* — which is worth remembering when wiring the rest of the
RenderingDevice enums by hand from the guest.

Also settled: `RID::RID(const Variant &)` is declared in the sandbox API headers
but never defined in `libsandbox_api`, so it links only if unused. Convert with
`v.operator ::RID()` instead.
