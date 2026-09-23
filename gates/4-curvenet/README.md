# Gate 4-curvenet

Two parts. **Gate 4 (the stage)**: `curvenet.elf` runs Cassie's pen → curvenet → mesh and mesh → curvenet paths, and its output is bit-identical to a native build of the same code. **Gate 4 kernels** (below): the Cassie kernels come from `lean/`.

## Gate 4: curvenet.elf against its flat control

**Result: PASS** (`results.txt`, `run.log`). The guest ran all 10 checks in `guest/curvenet/checks.cpp` and every one passed. The native `cassie_checks.exe` runs the same two TUs (`native-checks.log`, `native-build.log`). Guest and native agree on all 10: same verdicts, same integer outputs, same float signatures. A second pass in the same guest produced byte-identical lines. `llvm-nm -C curvenet.elf` finds 13,587 symbols and 0 `Eigen::`.

How the gate was run:

```
tests/native/curvenet/build.sh                     # native; writes native-checks.log
BUILD_DIR=build/rv64 BUILD_FIT=0 BUILD_TARGETS=curvenet ./build.sh   # riscv64; installs project/curvenet.elf
godot --path project --headless --import           # first run after adding the ELF
godot --path project --script gate_curvenet.gd --rendering-driver vulkan --xr-mode off > ../gates/4-curvenet/run.log 2>&1
godot --path project --script probe_curvenet_wrappers.gd --rendering-driver vulkan --xr-mode off > ../gates/4-curvenet/wrappers.log 2>&1
```

Every check has a negative control, and each control must come out differently from the check (Godot 4.7.2, riscv64 guest vs llvm-mingw clang 23 native):

| check | ints (guest = native) | guest ms | what it shows (control) |
|---|---|---|---|
| beautify_determinism | 18,1,1 | 3.3 | identical strokes give the same 18 floats (control: a 0.05 bump differs) |
| curvenet_extract | 1,0 | 0.7 | triangle → 1 cycle (control: open path → 0) |
| patch_pipeline | 3596,0 | 1074 | triangle cycle → 3596 patch triangles (control: one-edge graph → 0) |
| crossing_split | 9,9,1,9474,1,1,0,3 | 995 | 3 overshooting strokes → 9 edges, 9 nodes, 1 cycle, 9474 tris; edge 0 is untouched and keeps its id, crossed edge 1 is replaced (control: `add_stroke` → 0 cycles over 3 edges) |
| constraint_solver | 3,1,1 | 1.2 | mirror pins take 0.05 → 0.00000 (control: the raw input fails the 5e-3 tolerance) |
| pen_sphere | 1,65,65,1,3954,1,3944,2,2,0,0 | 894 | one closed stroke 1 cm off an r=0.5 icosphere(3): all 65 samples snapped to [0.499, 0.5025] → 1 patch → `mesh_build` 1 loop (3954 tris; PMP remesh 0.02: 3944 tris, 1 loop) → `curvenet_build` 2 curves, 2 knots (controls: 270° arc → 0 patches; split_closed=0 → 0) |
| extractor_cube | 12,8,8,0 | 14.0 | cube → 12 curves, 8 knots, all 8 of degree 3 (control: icosphere(3) → at most 1 curve; it gives 0) |
| delaunay_small_scale | 1,12,12,0 | 0.4 | a 1 mm polygon 1 m from the origin (n=11, h=8) → 12 = 2n−2−h triangles (control: collinear points refused) |
| mesh_weld | 4,5,2,1,1,4,1,2,2 | 0.1 | two triangles in two parts weld into V=4, E=5, F=2: 1 component, 1 loop of 4, Euler 1 (control: weld_eps=0 → 2 components, 2 loops) |
| skirt_tube | 4,6,4,4,6,2,2,2,2,0,1,2,0,2,0,0,2,0,0,1,2,0,2,0,0,2,0,0,0,0,4,2,2 | 1208 | Cut 8's scripted skirt on a capped cylinder (r 0.15; rings of r 0.16 at y 0.9 and 0.5 snap onto it), the four half rings drawn as boundary strokes: 4 knots, 6 edges; 4 knots of degree 3, 6 curves; 2 cycles and 2 openings, 2 patches, both panels (each touches both rings and both seams), one per side. `mesh_build(0, 1e-5)` and `(0.02, 1e-5)` each: 1 component, 2 loops, Euler 0, both loops at a ring, no directed edge used twice, 0 inward triangles, patch ids 0 and 1, none unassigned, none on the wrong side of the seams (controls: back seam dropped → 0 patches, 0 panels; rings drawn as ordinary strokes → 4 patches, 2 of them caps). Ints: nodes, edges, knots, degree-3 knots, curves, cycles, openings, patches, panels, sum of sides; per mesh_build: components, loops, Euler, loops at a ring, twice-used directed edges, inward, ids, unassigned, wrong side; controls: patches and panels, then patches, panels and caps |

- **Speed.** The 10 checks take 634 ms natively and 4.19 s in the guest, about 6.6×.
- **The pen, driven the way a host drives it** (Godot `SphereMesh` body rewound to wire winding by `util/mesh_wire.gd`, signed volume +0.52):
  - `cn_set_body`: 90 ms.
  - 65 samples: 3.4–5.0 ms, 53–78 µs per `pen_point`.
  - `pen_end` (host-timed over 3 strokes): min 492 ms, median 493 ms, max 503 ms.
  - `pen_demo_circle`'s `pen_stroke`, the first stroke in a fresh process, took 1.36 s (`wrappers.log`). That is one vmcall for the whole stroke (`pen_begin`, 64 `pen_point`s and `pen_end`) plus the first call's warm-up, not a `pen_end` alone. The scripted pen's `pen_end` in the same process took 487 ms.
  - The patch: 2062 vertices, 3988 triangles (3998 before cut-4-tube; see (c) below). `mesh_build(0, 1e-5)`: 24.5 ms. `mesh_build(0.02, 1e-5)` with the PMP remesh: 357 ms, 3980 triangles, every one patch 0.
  - `curvenet_build`: 1.0 ms. `curvenet_extract` on the cube: 1.8 ms.
  - Every buffer decodes on the host: 1 loop of 134 vertices, an area-weighted normal of +Y (0.999999), so the mesh is CCW-outward, 2 curves with both ends on knots, and the cube gives 12 curves and 8 knots.
- **The skirt, driven the way a host drives it** (step 4b; Godot `CylinderMesh` r 0.15, y 0.3 to 1.1, rewound; signed volume 0.0565 = πr²h):
  - the four half rings with `cn_set_param("boundary", 1)`, the two seams with 0. `pen_end` takes 0.7–5.1 ms for the first five strokes (no patch yet: the half rings close only openings) and 220.5 ms for the back seam, which closes both panels.
  - 4 knots, 6 edges, 2 cycles, 2 openings, 2 patches (+x, y 0.500 to 0.900; −x, y 0.499 to 0.901).
  - `mesh_build(0, 1e-5)`: 13.3 ms, 1332 vertices, 2568 triangles, 1 component, 2 boundary loops at mean y 0.9 and 0.5, Euler 0, patch ids {0: 1274, 1: 1294}. `mesh_build(0.02, 1e-5)`: 477 ms, 1315 vertices, 2534 triangles, the same topology, ids {0: 1262, 1: 1272}.
  - `curvenet_build`: 1.5 ms, 6 curves, knots of degree [3, 3, 3, 3].
- **Budget.** No call came near `execution_timeout` (8000 × 2^20 instructions). The heaviest vmcall was `curvenet_checks` (`check_all`, `wrappers.log`) at 3.37 s. So `pen_end` stays a single call: it needs no job and no state machine (`guest/jobs.h` is unused here). At 0.2 to 1 s it is a hitch if it runs on the frame thread; a `WorkerThreadPool` host call is the next step if that hitch matters.
- **Heap** (`get_heap_usage`, `memory_max` 512 MB): 75 KB after load, 82 KB after the checks, 1.64 MB after the body, 3 strokes, 2 mesh builds and a curvenet.
- **FAIL paths answer instead of unwinding.** An unknown param gives `FAIL: unknown param ...`. An out-of-range index gives `FAIL: triangles[1] = 1 out of range [0, 1)`.
- **Rule 8, enforced.** Step 7 of `gate_curvenet.gd` reads every `ADD_API_FUNCTION` in `guest/curvenet/main.cpp` (23) and fails unless each one's name is passed as a string literal by a public `project/main.gd` function, and every such wrapper (19) has only default arguments. Two controls run the same audit on an edited copy of `main.gd`: with the `cn_get_param` wrapper removed it reports exactly that one missing, and with `patch_vertices(i: int)` it reports exactly that one argument without a default. The compiled script must also expose all 19 with no required argument.
  - `pen_begin`, `pen_point` and `pen_end` with no arguments replay a scripted stroke (`pen_demo_circle`'s circle, 65 samples) one sample at a time, as a tracked pen would deliver it. `pen_point(count = 0)` sends the rest of the stroke; `pen_end(id = -1)` ends the scripted stroke.
  - `wrappers.log` calls 17 of the wrappers once each, the scripted pen included: `cn_reset`, `pen_begin` (id 1), `pen_point` (64 samples, 3.5 ms), `pen_end` (1 patch), then `patch_count` reads 1.

### The API and the godot-lite split

- `guest/curvenet/main.cpp` is the only TU that sees the sandbox's `api.hpp`. It talks to Cassie only through `guest/curvenet/curvenet_api.h` (std types).
- `curvenet_api.cpp` and `checks.cpp` see Cassie on godot-lite and never `api.hpp`. They form `curvenet_core`, which links into both the ELF and `cassie_checks.exe`.
- The wire format is `guest/common/mesh_wire.h` ↔ `project/util/mesh_wire.gd`: body-local Godot frame, metres, CCW-outward triangles. Godot's own front faces are clockwise, so the `.gd` side rewinds them.
- Entry points: `cn_reset/cn_set_param/cn_get_param/cn_set_body`, `pen_begin/pen_point/pen_end/pen_stroke`, `patch_count/patch_vertices/patch_indices`, `curvenet_build/curvenet_extract/curvenet_curves/curvenet_knots`, `mesh_build/mesh_vertices/mesh_indices/mesh_boundary_loops/mesh_patch_ids`, and `check/check_all/check_names`.
- Params (`cn_set_param`): `snap_radius`, `surface_offset`, `target_edge_length`, `split_closed`, `merge_eps`, `mirror`, and `boundary`, a pen mode: a stroke begun while it is 1 is a boundary stroke, the edge of an opening such as a skirt's waist or hem, and a cycle made only of boundary strokes gets no patch. `pen_end` answers `cycles=` (the face cycles that bound surface) and `openings=` (those made only of boundary strokes).
- `mesh_build` does four things:
  - merges the active patches;
  - orients each patch away from the body, using the body normal at the point nearest the patch centroid;
  - welds on a grid within `weld_eps` (≤ 0: no weld);
  - with `target_edge_length > 0`, runs PMP `uniform_remeshing` with the boundary and the seams between patches marked `e:feature`/`v:feature`. `mesh_patch_ids` names each triangle's patch, after a remesh too: the patch nearest the triangle's centroid.

### The skirt is an open tube (cut-4-tube)

Cut 8's scripted skirt (`project/xr/pen_source_scripted.gd` on cut-8) draws a waist ring and a hem ring, each as two half rings from the front (+z) to the back (−z), then a front and a back seam. curvenet.elf surfaced the caps over the waist and the hem instead of the two panels, and the stroke ends did not always meet in shared knots (cut-8's `orders.txt`: 5 knots of degree 3, 3, 2, 3, 1; meshes of 2 components, or 17 loops). There were two causes, one per symptom, and the new check found a third.

**(a) Stroke ends that meet did not merge into one knot.** The fault was endpoint-to-edge, not endpoint-to-endpoint. `add_stroke`'s endpoint merge (`_find_or_create_node`, within `merge_eps`) works. The crossing detector that runs before it did the damage.
- `_crossings` treats a shared endpoint as a merge, not a crossing, only when the closest pair lies on both polylines' first or last segment (s < 0.05 or > 0.95). That rule holds while the proximity is well below the sample spacing.
- The sketcher bakes every stroke to 33 samples: 2 cm segments on a 0.64 m half ring, 1.3 cm on a 0.43 m seam. It passes proximity = max(snap 0.01, `merge_eps` 0.02 to 0.05).
- So a stroke that only starts at an existing knot also came within the proximity of the other edge's second or third segment, and that counted as a T-junction. Its split point, the midpoint of the closest pair, lay 1 to 6.5 cm from the knot.
- Within `merge_eps` the split point merged back into the knot and left the edge's polyline trimmed short of its node. Beyond it, it became a node of its own, of degree 1. On FoxGirl, rings then seams at `merge_eps` 0.05 left two extra knots, 6.55 cm from the back hem knot and 5.3 cm above the front one; seams first at the default 0.02 left a fifth 2.14 cm from the back waist knot.
- **Fix (Cassie adaptation 6).** The stretch beside a shared knot counts as the endpoint merge. From two ends that meet within `merge_eps`, `_knot_zone` grows along both polylines while each next segment stays within the proximity of the other's. Hits inside that zone are not crossings. A stroke that leaves a knot and crosses the other curve further on still splits there.
- Every order now gives 4 knots of degree 3 and 6 curves: the four orders × `merge_eps` 0.02 and 0.05 on FoxGirl, the check's cylinder, and the host step. The other 9 checks read the same numbers.

**(b) Why the caps and not the panels.** With the knots merged, `find_cycles` finds all four faces of the network (V − E + F = 4 − 6 + 4 = 2, a tube closed at both ends): the 2 panels and 2 two-edge caps.
- **Cassie surfaces every face its walk closes.** Unity CASSIE's `CycleDetection.DetectCycle` closes a cycle when it gets back to its start segment with `cycle.Count > 1`, so two-segment cycles count. It refuses only a segment already in two cycles (`g.ExistingCyclesCount(currentSegment) >= 2`, the manifold guard); every edge here borders exactly two faces. `SurfaceManager.AddPatch` surfaces whatever cycle it is handed. An unwanted patch is the user's to delete (`Graph.ManualDeletePatch`). This port does the same, and adaptation 5 surfaces two-edge cycles for closed strokes.
- **Before the fix, only the caps survived.** A ring's two half rings meet end to end and merge, so its cap always closed, while every panel ran through a broken knot. After (a), all four faces were surfaced, a closed bolster: the check's control reads 4 patches.
- **Whether a ring bounds a cap or an opening is the author's intent; the network cannot say it.** The same 6 curves bound the open tube, a sack with one cap, and the closed bolster.
- **Fix (Cassie adaptation 7): boundary strokes.**
  - `cn_set_param("boundary", 1)` is a pen mode, `CassieSketcher::boundary_strokes`, captured at `pen_begin`. A stroke begun in it marks its graph edges as boundary edges, and the slices of a split edge keep the mark.
  - A cycle made only of boundary edges is an opening (`CassieSketchGraph::is_opening`), and `CassieSurfaceManager::update` gives it no patch.
  - It is the declarative form of CASSIE's delete-patch: stated when the ring is drawn, the same in every stroke order, and not undone when the manager re-surfaces after the next stroke.
- **Rejected.** A rule that a cycle needs 3 or more edges (the Lean model of the walk has it; Unity closes at 2) hides a cap only while its ring is drawn as exactly two halves. A ring split at three knots gives a three-edge cap, and a ring drawn as one closed stroke becomes a two-edge cycle by design (pen_sphere). Choosing faces automatically, such as "the basis that uses every seam" or the fewest faces that cover every curve, would also drop the lid of a wire cube or the disk of a closed stroke.

**(c) Guest ≠ native on the skirt (Geogram).** skirt_tube first read the same ints in the guest and natively but different floats: 1304 welded vertices against 1307.
- **Where it diverged.** A temporary stage-fingerprint check (not committed) found these bit-identical in both: the strokes, the curves, the graph polylines, the cycle boundaries and MingCurve's edge-protected points. The first difference was the order of the Delaunay faces MingCurve hands DMWT, 1263 of them both ways. DMWT then tiled one panel differently: 84 faces both ways, but different ones.
- **Why.** The Delaunay insertion order is BRIO plus a Hilbert sort, whose splits are `std::nth_element` on one coordinate. The skirt's boundary points tie on coordinates exactly: ring samples snap to one height, seam samples to one plane. The standard leaves the order of tied elements to the implementation, and libc++ and libstdc++ order them differently.
- **Fix (Geogram, `mesh/mesh_reorder.cpp` `Hilbert_vcmp`).** Ties are broken on the vertex index, a strict total order. Inputs without ties sort as before.
- **Effect.** skirt_tube now matches bit for bit, and the other 9 checks read the same numbers. The host pen step's `SphereMesh` stroke, whose 64 samples snap to one height, went from 3998 to 3988 triangles. An A/B build of the same ELF with only the tie-break reverted gives 3998 again.

**`mesh_patch_ids` after a remesh answered −1 for every triangle** (the verifier's 3986/3986; the header even documented it). The remesh now holds the seams between patches as features, like the boundary, and gives each remeshed triangle the patch nearest its centroid (`CassieSurfacePatch::project` over each input patch's triangles; with one patch, that patch). skirt_tube checks the result: 0 unassigned and 0 on the wrong side of the seams after the 0.02 remesh. `wrappers.log` now reads `{ 0: 3980 }` where it read `{ -1: 3986 }`.

**FoxGirl** (`skirt-foxgirl.txt`, `skirt_foxgirl_probe.gd`). The scripted skirt on the Cut 8 fixture (waist r 0.2047 at y 0.9528, hem r 0.2305 at y 0.5249) ran through this curvenet.elf in the four stroke orders cut-8's probe tried, at `merge_eps` 0.02 and 0.05, with the rings as boundary strokes.
- **All 8 runs:** 4 knots of degree 3, 6 curves, 2 cycles and 2 openings, 2 panels (+x and −x, y 0.52 to 0.95). `mesh_build(0.03, 0.005)` and `(0, 1e-5)` each give 1 component, 2 loops at mean y 0.525 and 0.952–0.953, Euler 0, and patch ids 0 and 1.
- **Controls:** rings drawn as ordinary strokes → 4 patches (the caps are back); the back seam dropped → 0 patches.
- **Guest `pen_end`:** 1.03–1.04 s for the stroke that closes both panels, 0.79–0.87 s for one panel.
- **How it was run.** The probe needs cut-8's fixture and pen source. They were copied in untracked from cut-8 at df66f0ca2 and removed afterwards.

**For Cut 8.** `pen_source_scripted.gd`, or the pen bridge, sets `cn_set_param("boundary", 1)` for the four half rings and 0 for the seams. Gate 8's "2 cycles and 2 patches" holds as written, because openings are counted separately. Its dropped-seam control reads 0 cycles and 0 patches.

**Vendored record.** `CITATION.cff` gains Cassie adaptations 6 and 7 and the Geogram tie-break, and `tools/vendor/patches/{cassie,geogram}.patch` are regenerated. Both vendoring scripts, run from pristine c165a519d2 with the new patches, reproduce the edited trees: 318 files compared, 0 differing. With the previous patches, exactly the 6 edited files differ.

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
   - **Fixed where it lives**, in `vendor/sandbox-api/docker/api/native.cpp` (its `CITATION.cff` lists the adaptation). `memalign` now over-allocates by alignment − 1 and returns the aligned address inside the block. A side table maps that address back to the host block, and the wrapped `free`/`realloc` look it up first. A header below the block would not work, because the host heap only frees the pointer it handed out.
   - Gate 0F probe 17 (`gates/0f-runtime`) allocates 1000 blocks at each of 64, 128 and 4096 alignment through all four aligned entry points, with frees and reallocs interleaved. None is misaligned, overlapping or corrupted, and the heap ends where it started. Its control, a copy of the old fallback on the same sequence, returns freed blocks and overlaps.
   - Geogram's `basic/memory.h` is upstream's again. The first fix here, carving aligned blocks out of plain `malloc` inside Geogram, is gone from `tools/vendor/patches/geogram.patch`. The vendoring script run from pristine c165a519d2 reproduces the tree with 0 files differing. Natively (llvm-mingw, `GEO_COMPILER_MINGW`) upstream's `aligned_malloc` is plain `malloc`. In the guest it is `posix_memalign`, now the fixed path. All 9 checks read the same ints and float signatures as before, guest and native.

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
| `lake build Cassie emit_cassie` | — | 30 jobs, exit 0; the six copied `native_decide` examples (two each in CurveGenerateBezier, CurveNewton, CurveRdp), the three in `Cassie.lean`, and five added with the epsilon fix (each guard's whole emitted line, the two exact literals, and `litFloat 1.0e-12` = `0.000000`) hold |
| `kernels/cassie/gen.sh` (emit), run twice | `gen.log` | 5 kernels to slang/, cpp and SPIR-V; 5/5 `spirv-val` clean (2056–6904 bytes); the second run leaves `git diff -- kernels/cassie` empty |
| `kernel-parity.sh`: each emit against `modules/cassie/thirdparty/avbd/<k>.cpu.cpp` | `kernel-parity.log` | 5 kernels, 755 lines compared, **2 differing lines on each side, both intended**: the two epsilon guards below, `< 0.0f` in the reference and `< 9.999999960041972e-13f` / `< 9.99999971718068537e-10f` in the emit. With LeanSlang e0e96da the count was 0. |
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

### Found on the way

- **Two epsilons printed as zero (fixed; a behaviour change against the
  module).** LeanSlang's `litFloat` prints six decimal places. `1.0e-12`
  (curve_generate_bezier's singular-determinant guard) and `1.0e-9`
  (curve_newton's `|den|` guard) therefore emitted as `0.000000`, and the
  guards read `abs(x) < 0.0f`, which is never true. The module's prebuilt
  `curve_generate_bezier.cpu.cpp` and `curve_newton.cpu.cpp` at c165a519d2
  have the same `< 0.0f`, so the CASSIE module as shipped never fires them
  either. The hand-written C++ these kernels replaced used the real epsilons.
  LeanSlang `emit-fp` 60532ae added the exact forms `litFloatExact` and
  `litDoubleExact`, and emitted every consumer's kernels (main's lean/ and
  cuts 4, 5 and 6) byte-identically. Both guards are now `litFloatExact`.
  They emit `1.0e-12f` and `1.0e-9f`, and `native_decide` pins each whole
  emitted line. **No check output changed:** all 9 read the same ints and
  float signatures, guest and native, so no check's data came near either
  threshold; the guards now matter only where the kernels used to divide by
  (nearly) zero.
- **slangc warns E41035 on curve_rdp's local stack.** The warning is
  "possibly uninitialized". The stack is only read below `stack_top`, which
  every push has written, so it is a false positive.

### Pins

entities-godot c165a519d2836f3ded600948abb9d8f799cd4c5f (Lean sources and
the reference `.cpu.cpp`); LeanSlang contract-lean-slang `emit-fp` 60532ae;
Lean 4.30.0; slangc 2026.13.1-1-g84792eb15 (scoop Vulkan SDK).
