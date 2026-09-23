# Gate 0H — the loop on a RunPod GPU worker (stock Godot, Linux, no display)

**Question.** Serverless for the whole flow (user, 2026-09-23) needs the loop's
Godot half — the guest ELFs driving RenderingDevice compute — on a RunPod
Linux GPU worker. `--headless` hands back a null RenderingDevice (AGENTS.md), and
a worker has no display. Does stock Godot 4.7.2 + the godot_sandbox addon get a
Vulkan RenderingDevice on the worker's NVIDIA GPU, and does the loop run there?

**How.** `run_pod.py` starts a plain `ubuntu:22.04` pod (community cloud, one
GPU from a cheap list) with `NVIDIA_DRIVER_CAPABILITIES=all`, copies this
repo's `project/` (plus the gitignored `fit.elf`), `gates/2-avbd` and
`vendor/xr-grid` at HEAD, runs `remote.sh` over ssh, brings the logs back into
`runN/` and
terminates the pod in every branch. `remote.sh` installs Xvfb, the Vulkan
loader and Mesa, downloads the official Godot 4.7.2 Linux build, imports, and:

| variant | what | expect |
|---|---|---|
| V0 | `vulkaninfo`, no display, NVIDIA ICD | the NVIDIA device is there |
| V1 | `godot --headless` Gate 1 (control) | FAIL: no RenderingDevice |
| V2 | Xvfb + NVIDIA ICD, Gate 1 (`gate_rd_compute.gd`) | PASS on the NVIDIA adapter |
| V3 | Xvfb + lavapipe, Gate 1 (control) | tells "GPU path blocked" from "RD never works here" |
| V4 | Xvfb + NVIDIA, Gate 8 (`gate_loop.gd`, FIXTURE:infer,rig) | the loop, end to end |

## Run 1 (2026-09-23 14:30 UTC) — RD PASS, loop FAIL

Pod: NVIDIA RTX 4000 Ada Generation (20 GB, driver 565.57.01), 48 × Xeon
E5-2650 v4 @ 2.20 GHz, community cloud, $0.20/h; 182 s from create to
terminate, about $0.01. Logs: `run1/`.

| variant | result |
|---|---|
| V0 | PASS: `deviceName = NVIDIA RTX 4000 Ada Generation` with no display |
| V1 | FAIL as it should: `FAIL at create_local_rendering_device: not an Object` |
| V2 | **PASS**: Gate 1 rc 0, `Using Device #0: NVIDIA - NVIDIA RTX 4000 Ada Generation` |
| V3 | PASS on `llvmpipe (LLVM 15.0.7, 256 bits)`: RenderingDevice works on CPU Vulkan too |
| V4 | **FAIL** in 8 s: `FAILED MESH: mesh_build: <null>` — curvenet.elf faulted with `Too many arena chunks (data: fa0)` in `pen_end`/`curvenet_build` |

**Answer to the question: yes.** Stock Godot gets a Vulkan RenderingDevice on
the worker's NVIDIA GPU through an Xvfb display; the guest's rd_compute gate
passes on it. The container toolkit granted the NVIDIA Vulkan driver with
`NVIDIA_DRIVER_CAPABILITIES=all`; `/etc/vulkan/icd.d` was a read-only mount, so
a missing ICD file is written under `/tmp` and named with `VK_ICD_FILENAMES`.

**Why V4 failed (two findings, both fixed for run 2):**

1. The addon's default `allocations_max` is 4000 live guest heap chunks
   (`MAX_HEAP_ALLOCS`, `fa0`). The Linux build ran out inside curvenet on the
   loop's skirt; the Windows build of the same ELF passes Gate 4 and Gate 8 at
   that default. `stages/curvenet_stage.gd` now sets 1,000,000, as
   `fit_stage.gd` sets its own (Gate 4 re-run on the desk: PASS, results
   unchanged).
2. On Linux and macOS the addon looks up a faulting line with `addr2line`
   inside the `ghcr.io/libriscv/cpp_compiler` Docker container on every guest
   fault, ignoring `sandbox/toolchain/docker_enabled`: three failed
   `docker pull`/`docker run` cycles for one fault. Patched in the org fork
   (V-Sekai-fire/godot-sandbox PR #4: the lookup honours the setting), and
   `project.godot` sets `toolchain/docker_enabled=false`.

## Desk runs on the desk's GPUs (2026-09-23) — Linux image, Vulkan through dzn

The user asked that nothing goes to RunPod before it passes locally, and not on
the CPU. Docker Desktop's Linux containers reach the RTX cards through WSL2's
D3D12; Mesa's dzn is Vulkan on top of it (`tools/runpod/loop/wsl-test`: Arch +
`vulkan-dzn` around the loop image's own `/opt/godot` and `/app`). vulkaninfo:
`Microsoft Direct3D12 (NVIDIA GeForce RTX 3090)` and `(RTX 4090)`, discrete,
Vulkan 1.2.354. NVIDIA's own Linux Vulkan driver is not reachable there (the
driver store's `nv-vk64.json` is the Windows ICD); Gate 0H V2 covered it.

| run | result |
|---|---|
| full job through the handler, 4090, fit.elf | AUTHOR 2 cycles / 2 openings; MESH 932 v; **FIT 329 Newton, 1601 s**; **CHECK `OK none`** (push-vertex control INTERSECTS); **DRAPE FAILED** at its first call: `Too many arena chunks (fa0)` in drape.elf — the same Linux-only 4000-chunk default as curvenet in run 1 |
| fix | `stages/sandbox_util.gd` gives every stage `allocations_max` 1,000,000 (fit keeps 4,000,000); curvenet's own override removed |
| MESH + DRAPE, fit as fixture | **DRAPE 100 steps, finite**, 119 ms/step through dzn (40.6 ms on the same card natively on Windows), 0 arena faults |
