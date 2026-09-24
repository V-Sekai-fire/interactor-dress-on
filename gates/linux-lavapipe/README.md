# Linux, no GPU: the gates on Mesa lavapipe

**Question.** Does the loop run on a Linux machine with no GPU? The Linux
godot-sandbox addon (a45da9f) drives the guest ELFs, and Mesa's lavapipe
(llvmpipe, a software Vulkan driver) stands in for the RenderingDevice.

**Result: yes.** Gate 8 ends `RESULT: PASS FIXTURE:infer,rig` in 177.6 s,
against 19.8 s on the RTX 4090 (`gates/8-loop/flat-avbd.*`). The fitted skirt
is within 8.1e-7 of the 4090's on all 932 vertices (mean 1.1e-7).

Two portability bugs in the gate scripts had to be fixed first. Both are in the
commits on this branch.

- **The live-allocation limit.** The Linux addon allows 4000 live guest heap
  chunks by default. The gates that make their own Sandbox died with "Too many
  arena chunks". They now set `allocations_max` 1,000,000, as `stages/sandbox_util.gd` does.
- **The heap default.** The Linux addon defaults `memory_max` to 32 MiB, where
  the Windows addon uses 512. The gates that took the default now pin 512.

## Setup

Godot 4.7.2-stable Linux x86_64 and `mesa-vulkan-drivers` 25.2.8 (llvmpipe,
LLVM 20.1.2, 256 bits), on an Intel Xeon at 2.80 GHz. There is no GPU and no
`/dev/dri`. `fit.elf` is the v0.1.0 release asset, whose sha256 matches
`tools/runpod/loop/Dockerfile`.

```sh
apt-get install mesa-vulkan-drivers
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
godot --path project --headless --import
xvfb-run -a godot --path project --rendering-driver vulkan --xr-mode off --script <gate>.gd
```

`--headless` still gives a null RenderingDevice, so the rd gates run under Xvfb.
Gate 4 needs no device and runs headless.

## Results

| gate | Linux result | against the recorded Windows / RTX 4090 run |
|---|---|---|
| 1 `gate_rd_compute` | **PASS** | same. Its no-barrier control is weak here: llvmpipe runs the dispatches in order, so an unordered pair cannot show |
| 2 `gate_avbd` | **PASS** (13 s) | same |
| 4 `gate_curvenet` | FAIL on 2 float signatures | 10/10 checks PASS, verdicts and integer outputs equal to native in 10/10. `crossing_split` and `skirt_tube` differ in float bits from the Windows native log. See below |
| 5 `gate_lbfgsb` | FAIL (G2 19/20) | the same verdicts as the 4090 line for line. G2 is open there too (README of gate 5) |
| 5 `gate_drape only=G4,G8 quick` | FAIL (G4 control) | G8 PASS. G4 rd and cpu equal native to frame 10 (4.888e-6, as on the 4090), but rd frame 50 is 9.4e-3 from native, where the 4090 gave 5.0e-6. See below |
| 6G.1 `gate_rd_worker` | 66 PASS, 1 FAIL | the FAIL is C0: its 10 s window ran 20 frames on llvmpipe (587 on the 4090), too few for a drape job to finish. Arm C runs the same job and passes |
| 8 `gate_loop` | **PASS FIXTURE:infer,rig** | fit residual 0.00213 (same); drape ymin 5.723832 (5.723830); friction events 53338 (53339); 9x slower |
| 0F `gate_runtime` | killed | P06's memory ladder asks for 4 GiB heaps; the container's cgroup killed Godot at 3.4 GB resident. P15 ran before the heap pin and failed from the 32 MiB default |
| `tests/probe_main_wrappers` | **PASS** | same |

## What differs, and why

- **curvenet's two float signatures.** curvenet.elf reaches the host's
  C library for `sin`, `cos`, `acos` and `atan2`. It calls them through the
  sandbox's math ecalls, syscalls 533 and 534 (`ECALL_MATH_OP32` and
  `ECALL_MATH_OP64`); `llvm-objdump` finds 16 call sites. glibc and the
  Windows UCRT round some of those in the last bit, so the guest's floats
  depend on the host. The result is stable across Linux processes. "Guest ==
  native bit for bit" holds per host, not across hosts.
- **The drape's frame 50 on rd.** Both backends match native at frames 1 and 10, and
  friction events start between frames 10 and 50. llvmpipe and NVIDIA fuse
  multiply-adds differently (AGENTS.md: the GPU and the guest CPU round the
  same Slang differently). A friction branch that flips on one of them would
  grow into this gap. That is a hypothesis: no step between 10 and 50 has
  been compared yet.
