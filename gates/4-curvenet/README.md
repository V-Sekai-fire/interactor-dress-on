# Gate 4-curvenet

Two parts. **Gate 4 (the stage)**: `curvenet.elf` runs Cassie's pen → curvenet → mesh and mesh → curvenet paths, and its output is bit-identical to a native build of the same code. **Gate 4 kernels** (below): the Cassie kernels come from `lean/`.

## Gate 4: curvenet.elf against its flat control

**Result: PASS** (`results.txt`, `run.log`). The guest ran all 9 checks in `guest/curvenet/checks.cpp` and every one passed. The native `cassie_checks.exe` runs the same two TUs (`native-checks.log`, `native-build.log`). Guest and native agree on all 9: same verdicts, same integer outputs, same float signatures. A second pass in the same guest produced byte-identical lines. `llvm-nm -C curvenet.elf` finds 13,521 symbols and 0 `Eigen::`.

How the gate was run:

```
tests/native/curvenet/build.sh                     # native; writes native-checks.log
BUILD_DIR=build/rv64 ./build.sh                    # riscv64; installs project/curvenet.elf
godot --path project --headless --import           # first run after adding the ELF
godot --path project --script gate_curvenet.gd --rendering-driver vulkan --xr-mode off > ../gates/4-curvenet/run.log 2>&1
godot --path project --script probe_curvenet_wrappers.gd --rendering-driver vulkan --xr-mode off > ../gates/4-curvenet/wrappers.log 2>&1
```

Every check has a negative control, and each control must come out differently from the check (Godot 4.7.2, riscv64 guest vs llvm-mingw clang 23 native):

| check | ints (guest = native) | guest ms | what it shows (control) |
|---|---|---|---|
| beautify_determinism | 18,1,1 | 3.4 | identical strokes give the same 18 floats (control: a 0.05 bump differs) |
| curvenet_extract | 1,0 | 0.6 | triangle → 1 cycle (control: open path → 0) |
| patch_pipeline | 3596,0 | 1023 | triangle cycle → 3596 patch triangles (control: one-edge graph → 0) |
| crossing_split | 9,9,1,9474,1,1,0,3 | 942 | 3 overshooting strokes → 9 edges, 9 nodes, 1 cycle, 9474 tris; edge 0 is untouched and keeps its id, crossed edge 1 is replaced (control: `add_stroke` → 0 cycles over 3 edges) |
| constraint_solver | 3,1,1 | 1.2 | mirror pins take 0.05 → 0.00000 (control: the raw input fails the 5e-3 tolerance) |
| pen_sphere | 1,65,65,1,3954,1,3944,2,2,0,0 | 971 | one closed stroke 1 cm off an r=0.5 icosphere(3): all 65 samples snapped to [0.499, 0.5025] → 1 patch → `mesh_build` 1 loop (3954 tris; PMP remesh 0.02: 3944 tris, 1 loop) → `curvenet_build` 2 curves, 2 knots (controls: 270° arc → 0 patches; split_closed=0 → 0) |
| extractor_cube | 12,8,8,0 | 13.8 | cube → 12 curves, 8 knots, all 8 of degree 3 (control: icosphere(3) → 0 curves) |
| delaunay_small_scale | 1,12,12,0 | 0.3 | a 1 mm polygon 1 m from the origin (n=11, h=8) → 12 = 2n−2−h triangles (control: collinear points refused) |
| mesh_weld | 4,5,2,1,1,4,1,2,2 | 0.1 | two triangles in two parts weld into V=4, E=5, F=2: 1 component, 1 loop of 4, Euler 1 (control: weld_eps=0 → 2 components, 2 loops) |

- **Speed.** The 9 checks take 444 ms natively and 2.96 s in the guest, about 6.7×.
- **The pen, driven the way a host drives it** (Godot `SphereMesh` body rewound to wire winding by `util/mesh_wire.gd`, signed volume +0.52):
  - `cn_set_body`: 79 ms.
  - 65 samples: 3.4 ms, 52 µs per `pen_point`.
  - `pen_end` (host-timed over 3 strokes): min 456 ms, median 469 ms, max 572 ms. The first stroke in a fresh process took 1.23 s (`wrappers.log`); the steady state is about 470 ms.
  - `mesh_build(0, 1e-5)`: 19 ms. `mesh_build(0.02, 1e-5)` with the PMP remesh: 334 ms.
  - `curvenet_build`: 0.9 ms. `curvenet_extract` on the cube: 1.8 ms.
  - Every buffer decodes on the host: 1 loop of 134 vertices, an area-weighted normal of +Y (0.999999), so the mesh is CCW-outward, 2 curves with both ends on knots, and the cube gives 12 curves and 8 knots.
- **Budget.** No call came near `execution_timeout` (8000 × 2^20 instructions). The heaviest vmcall was `check_all` at 2.1 s. So `pen_end` stays a single call: it needs no job and no state machine (`guest/jobs.h` is unused here). At about 0.5 s it is a hitch if it runs on the frame thread; a `WorkerThreadPool` host call is the next step if that hitch matters.
- **Heap** (`get_heap_usage`, `memory_max` 512 MB): 75 KB after load, 392 KB after the checks, 1.64 MB after the body, 3 strokes, 2 mesh builds and a curvenet.
- **FAIL paths answer instead of unwinding.** An unknown param gives `FAIL: unknown param ...`. An out-of-range index gives `FAIL: triangles[1] = 1 out of range [0, 1)`.
- **Rule 8.** `project/main.gd` has `pen_demo_circle`, `curvenet_checks`, `curvenet_build`, `mesh_build` and `curvenet_extract_demo`, and `wrappers.log` calls each of them. Each wrapper has only default arguments.

### The API and the godot-lite split

- `guest/curvenet/main.cpp` is the only TU that sees the sandbox's `api.hpp`. It talks to Cassie only through `guest/curvenet/curvenet_api.h` (std types).
- `curvenet_api.cpp` and `checks.cpp` see Cassie on godot-lite and never `api.hpp`. They form `curvenet_core`, which links into both the ELF and `cassie_checks.exe`.
- The wire format is `guest/common/mesh_wire.h` ↔ `project/util/mesh_wire.gd`: body-local Godot frame, metres, CCW-outward triangles. Godot's own front faces are clockwise, so the `.gd` side rewinds them.
- Entry points: `cn_reset/cn_set_param/cn_get_param/cn_set_body`, `pen_begin/pen_point/pen_end/pen_stroke`, `patch_count/patch_vertices/patch_indices`, `curvenet_build/curvenet_extract/curvenet_curves/curvenet_knots`, `mesh_build/mesh_vertices/mesh_indices/mesh_boundary_loops/mesh_patch_ids`, and `check/check_all/check_names`.
- `mesh_build` does four things:
  - merges the active patches;
  - orients each patch away from the body, using the body normal at the point nearest the patch centroid;
  - welds on a grid within `weld_eps` (≤ 0: no weld);
  - with `target_edge_length > 0`, runs PMP `uniform_remeshing` with the boundary marked `e:feature`/`v:feature`.

### What it took: five bugs, each fixed in its vendored subset

Each fix is recorded in that subset's `CITATION.cff` and `tools/vendor/patches/*.patch`. The vendoring scripts, run from pristine at c165a519d2 with the new patches, reproduce the edited tree with 0 files differing. None of the five was in godot-lite.

1. **A closed stroke never became a patch** (Cassie; adaptation 5).
   - `CassieSketcher::split_closed_strokes` splits a closed stroke into two edges, and `find_cycles` reports the resulting two-edge cycle.
   - But `sample_cycle_boundary` refused any cycle with fewer than 3 edges, so it handed the triangulator an empty boundary. pen_sphere read 1 cycle, 0 patches.
   - Now it samples two-edge cycles, and continues each edge from where the previous one left off.
2. **Every planar patch aborted** (PMP).
   - The org fork's `-fno-exceptions` patch turned `inverse()` of a singular 3x3 into `abort()`.
   - `minimize_squared_areas` in tangential smoothing reaches that on any flat one-ring, whose squared-area matrix has rank 2. An 8-point planar circle aborted `CassieTriangulator` (`0xc0000409`; the backtrace was `abort ← Remeshing::tangential_smoothing ← refine_patch`).
   - Upstream's `catch` fell back to `weighted_centroid`. That fallback is restored, without exceptions.
3. **The triangulation changed on every call** (Geogram).
   - `GEO::random_shuffle` (the BRIO insertion order) seeded from `std::random_device`.
   - A closed stroke on a sphere gives a cospherical boundary. On one boundary, pen_sphere's patch gave 3938–4014 triangles across 15 runs, and different counts within one process too.
   - It now uses a fixed seed. The pen_sphere output has been one value ever since.
4. **Guest ≠ native** (Geogram and MWT).
   - `std::shuffle` and `std::uniform_*_distribution` are implemented differently in libc++ (the llvm-mingw native control) and libstdc++ (the guest). mt19937's raw output is fixed by the standard; those algorithms are not.
   - After fix 3, patch_pipeline read 3622 in the guest against 3606 natively. crossing_split read 9462 against 9390, and pen_sphere 3992 against 3990.
   - Fisher-Yates and raw-output mapping replaced them in Geogram's `random_shuffle` and `random_*`, and in MWT's `Point3::pertube`. After that, all 9 checks match bit for bit.
5. **Double free in `~Delaunay3d`** (Geogram, guest only).
   - godot-sandbox's guest heap wraps only `malloc/calloc/realloc/free`. Its `memalign` fallback (`vendor/sandbox-api/docker/api/native.cpp:197`) retries `malloc` for a 64-byte-aligned block.
   - When 16 tries miss, it returns the last block, which it has already freed. The sandbox reported "Possible double-free for freed pointer" and then glibc's `pthread_mutex_lock` assertion. crossing_split, pen_sphere and delaunay_small_scale lost their results, and delaunay_small_scale took 16.7 s.
   - Geogram's `aligned_malloc/aligned_free` now carve aligned blocks out of plain `malloc`, with the base pointer stored below the block.
   - **Not fixed:** the `memalign` fallback itself. It belongs upstream in godot-sandbox, and any other guest that asks for more than 16-byte alignment will hit it.

### Reference numbers (informational)

These are the macOS reference numbers from the entities-godot module. It uses Godot's `Delaunay2D`; this tree uses Geogram BDEL2d, and its RNG is now fixed.

- **Matches the reference:** beautify (18 floats); crossing_split's 9 edges and 1 cycle.
- **Differs:** patch_pipeline gives 3596 triangles here against the reference's 3606; crossing_split gives 9474 against 9424.
- **curvenet_extract:** 1 cycle here, 2 in the reference. The check asks for ≥ 1, and this graph's `find_cycles` reports each face once.

## Gate 4 kernels: the Cassie kernels come from lean/, match the module's, and link

**Result: PASS.** CASSIE's four editing-pipeline kernels (`lean/Cassie`,
copied from entities-godot c165a519d2 and renamespaced) plus DiffCloth's
`SpmvDf32` build on this tree's LeanSlang `emit-fp` / Lean 4.30.0 with no
source change. `kernels/cassie/gen.sh` lowers them to the five
`kernels/cassie/cpp/*_emit.cpp` the vendored dispatchers include. Those are
line-for-line the prebuilt `thirdparty/avbd/*.cpu.cpp` that entities-godot
ships, and they compile, link and run natively and for riscv64.

| check | log | result |
|---|---|---|
| `lake build Cassie emit_cassie` | — | 28 jobs, exit 0, 5 s; the six copied `native_decide` examples (two each in CurveGenerateBezier, CurveNewton, CurveRdp) and the three new ones in `Cassie.lean` hold |
| `kernels/cassie/gen.sh` (emit), run twice | `gen.log` | 5 kernels to slang/, cpp and SPIR-V; 5/5 `spirv-val` clean (2056–6904 bytes); the second run leaves `git diff -- kernels/cassie` empty |
| `kernel-parity.sh`: each emit against `modules/cassie/thirdparty/avbd/<k>.cpu.cpp` | `kernel-parity.log` | 5 kernels, 755 lines compared, **0 differing** |
| negative control: one `3.0f` changed to `3.5f` in a copy of the curve_generate_bezier emit | `kernel-parity.log` | 2 differing lines (the one line, both sides) |
| `CURVENET_KERNELS_PENDING=OFF tests/native/curvenet/build.sh` (llvm-mingw, clean) | `native-kernels.log` | 141 TUs (136 + the 5 dispatchers), 0 unresolved, curvenet_smoke 7/7 and curvenet_kernels_smoke 10/10 PASS |
| `riscv64-kernels.sh` (org sysroot, `curvenet_compile_check`, PENDING=OFF) | `riscv64-kernels.log` | 0 warnings; cassie_core needs 5 `cassie_slang_dispatch::` symbols, cassie_kernels defines those 5, **0 unresolved**; 5 `main_0_Thread`s, one per kernel namespace; 0 `Eigen::` |

### What the parity diff strips, and why

Both sides lose the Slang prelude, `#line` directives, CRs and blank lines.
The reference's post-processing (the relative prelude `#include`, the
`SLANG_PRELUDE_EXPORT` neutering and the `namespace cassie_slang_<k>` wrap
that entities-godot's `avbd-codegen` adds) is stripped too. Current slangc
(2026.13.1) inlines the 4.2k-line prelude, guarded by `SLANG_CPP_PRELUDE_H`,
where the reference `#include`d it. The dispatchers include the prelude first
and so skip the inlined copy (the `guest/avbd/avbd_cpu.cpp` pattern). The
task expected some slangc-version naming drift. There is none: the struct,
field, temporary (`_S<n>`) and entry names match, so the dispatchers needed
no change beyond the spmv file name (`spmv_df32_emit.cpp`, recorded in
`vendor/cassie/CITATION.cff` and in `tools/vendor/patches/cassie.patch`).

### Found on the way (not fixed: bug-for-bug with the module)

- **Two epsilons print as zero.** LeanSlang's `litFloat` prints six decimal
  places. `1.0e-12` (curve_generate_bezier's singular-determinant guard) and
  `1.0e-9` (curve_newton's `|den|` guard) therefore emit as `0.000000`, and
  the guards read `abs(x) < 0.0f`, which is never true. The reference
  `.cpu.cpp` has the same `< 0.0f`, so this is how the module ships, not
  something the emit-fp bump introduced. The hand-written C++ these kernels
  replaced used the real epsilons. A fix belongs upstream (a `litFloat` that
  round-trips, or `litHalf`/`cast`-style exact literals) and then here as a
  re-pin.
- **slangc warns E41035 on curve_rdp's local stack.** The warning is
  "possibly uninitialized". The stack is only read below `stack_top`, which
  every push has written, so it is a false positive.

### Pins

entities-godot c165a519d2836f3ded600948abb9d8f799cd4c5f (Lean sources and
the reference `.cpu.cpp`); LeanSlang contract-lean-slang `emit-fp` e0e96da;
Lean 4.30.0; slangc 2026.13.1-1-g84792eb15 (scoop Vulkan SDK).
