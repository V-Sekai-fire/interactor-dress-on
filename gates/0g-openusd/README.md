# Gate 0G: can OpenUSD read a stage inside a godot-sandbox guest?

The question is whether the core of the org's flow-* USD importer
(V-Sekai-fire/idtx-flow, a GDExtension over OpenUSD 26.05) could run as a
riscv64 guest ELF.

**Result: PASS.** OpenUSD 26.05 and oneTBB 2021.12.0 cross-compile static for
riscv64 once a 28-line patch is applied. The probe ELF links. The plugin
registry and schema registry come up from memory with no file opened. USDA
and USDC stages open from host bytes. Every guest line matches the host's
OpenUSD byte for byte, and corrupt input produces a clean error instead of a
crash.

| sub-question | verdict | evidence |
|---|---|---|
| compiles? | **yes, after a 3-file arch port**. Pristine v26.05 fails 538/554 TUs on `arch/defines.h:54 #error "Unsupported architecture.  x86_64 or ARM64 required."` (and `arch/math.h:89`, `arch/stackTrace.cpp` `#error Unknown architecture`) | `build.log` run A / run B |
| links? | yes: `usd_probe.elf` 31,808,800 B (text 17.1 MB, data 0.37 MB, bss 69.6 MB) | `build.log` |
| plugin registry without files? | yes, through a hook patch in `plug/info.cpp` (below) | `results.txt`: `usd_init ok plugins=8 resolvers=2 resources=11 errors=0` |
| USDA from bytes? | yes, both through `SdfLayer::CreateAnonymous(".usda")->ImportFromString` (mode 0) and through the in-memory resolver (mode 1) | `results.txt` |
| USDC from bytes? | yes, through an in-memory `ArResolver` (`ArInMemoryAsset`), with no patch needed | `results.txt` |
| guest == host? | yes: 9 of 9 good loads give the same counts and FNV-1a checksum as the host, and 4 of 4 corrupt loads give the same `ERR: fmt=` | `results.txt` vs `native-control.log` |

## Source (rule 1 exception, recorded)

No org fork of OpenUSD or oneTBB exists. For this gate both are used
**read-only, upstream, outside the repo, and not vendored**:

- OpenUSD `v26.05`, commit `2095fafafd033fa23386d7ec6d58c7cc33974518`. This is
  the tag idtx-flow's `SConstruct`/`scons/openusd.py` clones
  (`openusd_version = "26.05"`). No `openusd-26.05-src` was already downloaded
  under idtx-flow's `thirdparty/`, which does not exist in the local checkout,
  so it was cloned to `C:/b/usd2605`. No `interactor-flow` checkout exists
  locally.
- oneTBB `v2021.12.0`, the version OpenUSD's `build_usd.py --onetbb` pins
  (zip sha256 `fe6ca052…931db`), unpacked to `C:/b/oneTBB-2021.12.0`.

The local adaptation is `openusd-26.05-rv64.patch` (4 files, +28/−3), which
`patch_usd.py` applies. If this goes past a gate, it should become an org fork
of OpenUSD, and the user has to decide that.

## What was needed

1. **arch port (riscv64 is not a supported CPU).** `defines.h` gets
   `ARCH_CPU_RISCV` and puts it under `ARCH_BITS_64`. `math.h` adds it to the
   `sincos` branch. `stackTrace.cpp` gets the riscv64 Linux raw `execve`
   (`a7=221`, `ecall`) for the non-locking fork used in crash reports.
   `ARCH_SPIN_PAUSE` falls back to empty, and timing already uses
   `std::chrono`. After this there were **0 failures in 554 OpenUSD TUs and
   33 oneTBB TUs**, with only upstream-style warnings. oneTBB needed no change.
2. **Configure** (`build_usd_rv64.sh`): the repo's toolchain file at
   `-march=rv64gc -U__riscv_v_intrinsic`, `BUILD_SHARED_LIBS=OFF`, no Python,
   imaging, usdImaging, tools, exec or validation. Targets are `usdSkel`,
   `usdShade`, `usdGeom` and `kind`, which pull in arch, tf, gf, js, trace,
   work, plug, vt, ts, ar, sdf, pcp, usd and sdr. Watch for this: a
   command-line `CMAKE_CXX_FLAGS` replaces the toolchain's
   `CMAKE_CXX_FLAGS_INIT`, and the libstdc++ `-isystem` paths go with it.
   Every C++ TU then fails on `<cstddef>`, so the script repeats them.
3. **plugInfo.json from memory.** `Plug_InMemoryPlugInfoHook` is a function
   pointer in `plug/info.cpp`, and `_ReadPlugInfoObject` asks it before
   `std::ifstream`. The guest serves plugInfo.json at the install layout's
   paths (`/usd/<lib>/resources/plugInfo.json`) and calls
   `PlugRegistry::RegisterPlugins` with them. `gen_resources.py` embeds the
   build tree's configured plugInfo.json files plus each schema library's
   `generatedSchema.usda`: 11 resources, 157,359 B. Static libraries have an
   empty `LibraryPath`, so `PlugPlugin::Load` does not dlopen. The pxr
   libraries link `--whole-archive`, as OpenUSD's own static link does.
4. **Assets from memory.** `UsdProbe_MemResolver` is an `ArResolver`, declared
   through a probe plugInfo, set with `ArSetPreferredResolver`, and returning
   `ArInMemoryAsset`s. It serves `generatedSchema.usda` to the schema registry
   and the input to `SdfLayer::FindOrOpen("/mem/inputN.usd{a,c}")`. The crate
   reader needed nothing else.
5. **Threads.** `WorkSetConcurrencyLimit(1)` is set before anything runs, and
   TBB spawned no workers. The loads are single-threaded, since guest threads
   serialize anyway (Gate 0C).
6. **Probe-side details.** The pxr-facing TU is `-std=gnu++17`, because
   sandbox-api's `gnu++23` rejects OpenUSD's `unique_ptr<UsdPrimDefinition>`
   to an incomplete type. The guest warns twice at startup, harmlessly:
   `ArchWarn: Unable to read /proc/self/exe` and
   `ARCH_CACHE_LINE_SIZE != Arch_ObtainCacheLineSize()`, both because there is
   no filesystem (`run.log`).

## Numbers (`results.txt`, host-timed around the vmcall, `memory_max` 1024 MiB)

| step | time | guest heap after |
|---|---|---|
| ELF load (`program=`) | 743 ms | 0.75 MB |
| `usd_init` (plugins, resolver) | 16 ms | 1.05 MB |
| first load, skel_quad.usda (includes schema-registry init) | 46 ms | 1.62 MB |
| blendshape_test.usda (7 KB, 11 prims) | 20 ms / 12.5 ms (mode 0 / 1) | 1.65 MB |
| blendshape_test.usdc (4.8 KB) | 7.7 ms | 1.69 MB |
| morph_stress_test.usda (766 KB, 1528 pts, 7236 fvi) | 836 ms / 491 ms | 2.47 MB |
| morph_stress_test.usdc (282 KB) | 6.7 ms, repeat 7.3 ms | 3.06 MB |
| corrupt / truncated / garbage | 1.3–56 ms, a clean `ERR:` | 3.42 MB |

Guest heap stays under 3.5 MB across all 15 loads. USDC is about 100× faster
than USDA for the same stage, so a converter should ask the host for USDC.

## Flat control and its limits

`host_control.py` runs the host's OpenUSD, **usd-core 26.08 through Python**,
not 26.05 C++. It does the same traversal and the same BLAKE3 over prim path,
points (f32) and faceVertexIndices (i32) (first 12 hex digits; FNV-1a 64
until 2026-09-25, when the repo moved to one hash, briefly SHA-256, then
BLAKE3 for speed), and it writes the `.usdc` inputs. It
is **not the same probe TU built natively**. That build, llvm-mingw or MSVC
against a host OpenUSD 26.05, was skipped to stay inside the budget. The
control therefore separates "the guest computed it" from "nothing was there"
and pins exact values, but a guest/host disagreement would not tell a 26.05
vs 26.08 difference apart from a guest defect. None occurred.

Inputs: `inputs/skel_quad.usda` was written for this gate (2 quads, a
Skeleton, a SkelRoot, SkelBindingAPI). `inputs/blendshape_test.usda` is a copy
from `datasource-flow-project`. `morph_stress_test.usda` is read in place from
`C:/contract-manifest/3-interactor/datasource-flow-project/`, and its `.usdc`
lives in `C:/b/g0g-inputs`, uncommitted. The `.usdc` files in `inputs/` are
host exports.

Not tested: textures and other external asset references inside a stage.
They resolve through the same resolver and would need the host to push their
bytes first. usdz (its file format is linked) was not tested, and `pxr/exec` was not built.

## Porting estimate for idtx-flow's converter

idtx-flow at `75544a01` is about 7.8k lines:
- `shared/src/idtxflow` (4.2k) is OpenUSD-only with no godot-cpp, and would
  compile against these libraries as is. The exception is
  `exec/ExecBridgeManager`, which needs `pxr/exec`. That is not built here:
  `PXR_BUILD_EXEC=OFF` gives one more library set to cross-compile, with the
  same arch port.
- idtx-flow's own schema extension (`idtxflow_ext`, `openusdextension`) adds
  its plugInfo.json and generatedSchema.usda to `gen_resources.py`, the same
  mechanism.
- `source/resolver/UsdGodotAssetResolver` maps onto `UsdProbe_MemResolver`,
  fed by the host.
- `source/nodes` + `converter` (about 3.1k lines, 14 of their files on godot-cpp)
  cannot run in the guest as they are. Under rule 6 the guest emits packed
  arrays (mesh, skin, blend shapes, hierarchy) and a host GDScript builds the
  nodes.

Estimate: about 1 day for the core onto this build (exec excluded), 2–3 days
for the packed-array boundary and the host-side node builder, and exec is
unknown until its gate. ELF size is about 32 MB before any strip or
dead-stripping work.

## Reproduce

```
git -C C:/b clone -b v26.05 --depth 1 https://github.com/PixarAnimationStudios/OpenUSD.git usd2605
python gates/0g-openusd/patch_usd.py C:/b/usd2605
# oneTBB v2021.12.0 zip unpacked to C:/b/oneTBB-2021.12.0
bash gates/0g-openusd/build_usd_rv64.sh                 # -> C:/b/g0g
BUILD_DIR=C:/b/g0g-elf BUILD_FIT=0 BUILD_TARGETS=usd_probe ./build.sh   # -> project/usd_probe.elf (gitignored)
python gates/0g-openusd/host_control.py                 # -> native-control.log
godot --path project --headless --import
godot --path project --script gate_usd.gd --rendering-driver vulkan --xr-mode off > gates/0g-openusd/run.log 2>&1
```

Rule 8: `project/main.gd` has `usd_init()` and `usd_load(path = skel_quad.usda,
path_mode = 0)`. They were not exercised over MCP in this gate.

## The org fork (2026-09-23)

The user approved an org fork: `V-Sekai-fire/OpenUSD`, branch
`riscv64-sandbox` @ `df3f6b4`, is tag `v26.05` (2095fafa) plus exactly the
four files of `openusd-26.05-rv64.patch` (+28/−3). Build from that branch
instead of applying the patch to an upstream checkout; the patch file stays
here as the record of what the branch changes.
