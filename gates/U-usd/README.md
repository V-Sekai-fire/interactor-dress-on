# Gate U: does a Pixal3D `.usdz` open from bytes inside the sandbox, and do its arrays equal the host's?

**State (2026-09-23, WIP, parked on the user's budget):** usd.elf opens the
real Pixal3D package from bytes and every array it hands over equals the
host's OpenUSD reading of the same file. The gate is **not PASS** yet: a
package that fails to open (a truncated zip, garbage) leaves the Godot process
in a state that segfaults later (finding 1, below), so the run dies at the
fourth case. What works, what is not finished and the exact next commands are
at the end.

## Question

Cut U: `usd.elf`, a godot-sandbox guest over the org's OpenUSD
(`V-Sekai-fire/OpenUSD` branch `riscv64-sandbox` @ df3f6b4 = v26.05 + Gate
0G's four-file arch port; oneTBB 2021.12.0 static; the build of
`gates/0g-openusd/build_usd_rv64.sh` at `C:/b/g0g`, whose source checkout
`C:/b/usd2605` is identical to the org branch: `git diff df3f6b4` is empty),
that takes one `.usdz` as one PackedByteArray and answers with mesh and
material arrays, so the loop's INFER state can read Pixal3D's answer
(AGENTS.md, the Stage 7 exception) without a filesystem.

## Design as built

- `guest/usd/mem_resolver.{h,cpp}`: Gate 0G's `UsdProbe_MemResolver` and the
  embedded plugInfo/generatedSchema resources, moved out of
  `guest/usd_probe/usd_probe_core.cpp` so both ELFs share them (usd_probe.elf
  is unchanged in behaviour; `cmake/usd.cmake` builds both).
- `guest/usd/usd_core.{h,cpp}` (gnu++17, the pxr-facing TU): the package
  crosses as **one asset**, `/mem/pkgN.usdz`; `SdfLayer::FindOrOpen` picks
  `SdfUsdzFileFormat` by the extension, and `Sdf_UsdzResolver` (in sdf's
  plugInfo.json, embedded) reads the members (`asset.usdc`, the textures) as
  sub-ranges of that buffer. Nothing is unpacked. USDA and USDC layers open
  the same way, bare or inside the package (`stub-text.usdz` has an
  `asset.usda` root layer). After the traversal the stage is dropped and only
  **flat tables** remain (one float vector for all points, one for normals,
  one for st, one int vector for the fan-triangulated corners, one byte
  vector for every texture, one string blob; POD records with offsets):
  AGENTS.md's "keep long-lived guest data flat".
- Meshes: `UsdGeomMesh` prims in `Traverse()` order; `faceVertexCounts`
  honoured with a fan per face (corner 0, i, i+1), holes skipped, `leftHanded`
  reversed; normals from `primvars:normals` else the `normals` attribute, st
  from `primvars:st` else the first `texCoord2f[]`/`float2[]` primvar. With
  per-point interpolation the mesh stays indexed on the USD points; a
  faceVarying normal or st makes one vertex per corner (`indexed=false`).
  The world transform is `UsdGeomXformCache::GetLocalToWorldTransform`, 16
  floats row-major (row i = image of axis i, row 3 = origin); the 3x3 part
  crosses as a matrix (rule 11) and `util/usd_nodes.gd` makes Godot's `Basis`
  of its rows.
- Materials: `UsdShadeMaterialBindingAPI::ComputeBoundMaterial` ->
  `ComputeSurfaceSource`; `diffuseColor`, `metallic`, `roughness`, `opacity`,
  `normal` each a constant or a connected `UsdUVTexture` (its `file` resolved
  through the package, the connected output as the channel). Texture bytes
  are read once per file through `ArGetResolver().OpenAsset` and handed over
  as they are in the package (PNG here).
- `guest/usd/main.cpp` (the sandbox-API TU): `usd_init`, `usd_open(bytes)`,
  `usd_push(chunk)` + `usd_open_staged()` for packages past the 16 MiB
  syscall view, `usd_close`, `usd_mesh_count`, `usd_material_count`,
  `usd_texture_count`, `usd_mesh_info(i)`, `usd_mesh_points/normals/uvs/
  indices(i)` and their `_slice(i, from, count)` forms, `usd_mesh_transform(i)`,
  `usd_material(i)`, `usd_texture_info(i)`, `usd_texture(i)`,
  `usd_texture_slice(i, from, count)`. Whole-array answers past 15 MiB are
  refused with a `FAIL: ... use slices` string. Every open line ends in
  `instr=<rdinstret>`, the call's instructions.
- `project/stages/usd_stage.gd` (Sandbox via `stages/sandbox_util.gd`;
  `open_package` pushes 8 MiB pieces past one chunk, `mesh(i)` / `material(i)`
  / `document()` fetch in 1 M-item and 8 MiB slices), `project/util/usd_nodes.gd`
  (ArrayMesh with the corner swap for Godot's clockwise front faces and
  v -> 1 - v for Godot's top-left UV origin, StandardMaterial3D with albedo /
  metallic / roughness textures and channels, a Node3D tree, Z-up and
  metersPerUnit at the root), the rule-8 wrappers in the stage with same-named
  `main.gd` delegates (`tests/wrapper_audit.gd` lists `guest/usd/main.cpp` ->
  `usd`); Gate 0G's `usd_init` wrapper became `usd_probe_init`.
- Not ported: the org's idtx-flow converter core
  (`shared/include/idtxflow/converter/MeshConverter.h`, `MaterialConverter.h`)
  reads the same UsdGeom / UsdShade data but is a C++20 concept-templated
  header library over a `TargetEngine` type set (TargetTypes.h,
  TypeConverter.h, MdlMaterialConverter.h, usdSkel queries) and densifies
  every mesh to one vertex per corner; the pxr-facing TU has to be gnu++17
  (Gate 0G), so the reads are redone on the USD API with the same fan
  triangulation and the same UsdPreviewSurface inputs. Skeletons, blend
  shapes, GeomSubsets and emissive/normal maps beyond the file reference are
  not read.

## Inputs

- The real Pixal3D answers (the `pixal3d-usdz` branch's
  `tools/services/pixal3d/runs` outputs, not committed; copied to
  `C:/b/gu-inputs/`): `real-asset.usdz` 11,138,085 B sha256 `0aad742f1fde…`,
  185,772 points, 197,448 triangles, textures 1536²; `real-t2048.usdz`
  14,856,491 B sha256 `d9c4211b8a21…`, 198,661 points, 203,652 triangles,
  textures 2048². Derived by `host_oracle.py`: `truncated-30.usdz` (cuts into
  the crate), `truncated-60.usdz` (the whole crate, no textures, no central
  directory), `bare-asset.usdc`, `bare-asset.usda`, `nomesh.usdz`,
  `garbage.bin`.
- Committed in `inputs/`: the service's stub quad as `stub-quad.usdz` (USDC
  root layer), `stub-text.usdz` (USDA root layer), `stub-unbound.usdz` (no
  material binding).

## Host oracle

`host_oracle.py` (usd-core 26.8 through Python, `timeout 300`): the same
traversal, fan triangulation, BLAKE3 prefixes over points (f32) and
corners (i32) (the guest reports the same two; until 2026-09-25 it reported
FNV-1a 64), texture bytes straight out of the zip
with their BLAKE3, the UsdPreviewSurface wiring. -> `host-oracle.log`.

## Numbers (`results.txt`, `ladder/*.txt`, host-timed, RTX 4090 box, Godot 4.7.2)

| step | measured |
|---|---|
| ELF load (`program=`), 31,895,272 B | 370-470 ms (0G's probe: 743 ms) |
| `usd_init` | 10-14 ms |
| `real-asset.usdz` open (11.1 MB, push 8 MiB + 2.7 MB, open) | 333-423 ms, 280 M instructions = 267 units; heap 36.2 MB |
| `real-t2048.usdz` open (14.9 MB) | 464 ms, 321 M instructions = 306 units; heap 45.9 MB |
| extract (`document()`: info, points, normals, uvs, indices, material, 2 textures) | 7-13 ms |
| stub quad (3.6 KB) | 6-9 ms after the first, 90-113 ms the first (schema registry) |
| guest == host | points 185,772 / 198,661, triangles 197,448 / 203,652, FNV of points and corners, sha256 of what crossed, texture sizes + sha256, wiring (diffuse rgb, metallic b, roughness g): **equal**, both real packages |
| textures decode in Godot | 1536x1536 / 2048x2048 RGBA8, both |
| usd_nodes | ArrayMesh 185,772 vertices 592,344 indices, aabb 0.66 x 0.88 x 0.68 m; a MeshInstance3D per mesh |
| second open in the same Sandbox | 285-320 ms (no stale state) |

Gate 0G's 6.7 ms USDC baseline was a 282 KB crate; this crate is 6.6 MB and
the open also reads two PNGs into the texture table (4.6 MB), so 0.33 s is
about 30 MB/s of crate + texture handling on the one guest thread.

Memory ladder (`ladder/<MiB>.txt`, one fresh Sandbox per process, the largest
package `real-t2048.usdz`, open + `document()`):

| memory_max MiB | 64 | 96 | 128 | 160 | 192 | 256 | 320 | 384 | 512 |
|---|---|---|---|---|---|---|---|---|---|
| result | push faults (`<null>`) | open dies | opens, arrays fault (`Out of memory`) | opens, material fault | **PASS** | PASS | PASS | PASS | PASS |

Floor 192 MiB; `usd_stage.gd` `MEM_MB` = 240 (x1.25). Largest call 306
units of 2^20 instructions; `TIMEOUT_UNITS` = 400 (x1.25; the addon's 8000
default was above it anyway). `allocations_max` 1,000,000 (sandbox_util's
default; the open holds ~18.8 k chunks).

Rule 8: `tests/probe_main_wrappers.gd` **PASS** (`wrappers.txt`, 44 checks)
with the usd wrappers, `usd_open` of the stub answering
`ok layer=usdz prims=9 meshes=1 materials=1 textures=2` and `usd_mesh_info`
the quad, over `main.tscn`'s stage.

Controls seen so far: `truncated-30.usdz` answers a clean
`ERR: usdz open failed: unknown` in 31 ms and the guest still answers
(`count()` 0); `garbage.bin` answers `ERR: usda open failed: </mem/pkg8.usda>
is not a valid usda layer`; `nomesh.usdz` opens with `meshes=0`;
`bare-asset.usda` / `.usdc` open as bare layers with the same checksums;
`truncated-60.usdz` opens (the crate is whole) with `textures=0` and a warning
naming the texture that does not resolve, as the host does (the ladder and
the seq probes; the gate itself has not reached them, finding 1).

## Finding 1 (open, blocks the gate): a failed open poisons the Godot process

Reproducer: `project/probe_usd_push.gd --seq=<a>,<b>,...` (one process, the
stage's `open_package` + `document()` + `close()` per name, since the fresh-
Sandbox change one Sandbox per document; `ladder/probe-seq.txt`):

| sequence | memory_max | result |
|---|---|---|
| real-asset x 2, real-asset + real-t2048, stub x 8 | 320 | fine |
| real-asset x 6 (one Sandbox) | 320, 160 | **Godot segfaults at the 5th open** (signal 11 inside the vmcall, no guest exception, heap 2.0 MB after every close) |
| bare-asset.usdc x 6 (one Sandbox) | 320 | segfault at the 5th |
| real-asset x 20, x 40 (one Sandbox) | 1024, 2048 | fine (40 x ~60 MB of allocations, so the arena does reuse freed memory) |
| real-asset x 12 (one Sandbox) | 512 | fine |
| truncated-30 (ERR), truncated-60 | 320, fresh Sandboxes | fine |
| truncated-30 (ERR), stub | 320, fresh | both open, **segfault at process exit** |
| stub, truncated-30 (ERR), stub | 320, fresh | segfault at the third (freeing the ERR Sandbox / loading the next) |
| garbage.bin (ERR), stub | 320, fresh | both open, segfault at exit |
| the gate (real, t2048, truncated-30, ...) | 320, 512, 1024, fresh; also with `binary_translation_bg_compilation` off | 33 PASS, then segfault after the ERR case, before the 4th open |

So two shapes: (a) in one Sandbox the fifth big document at memory_max 160
or 320 (not 512+) kills Godot; (b) a Sandbox whose open **failed** kills Godot
when it is freed (or at exit), whatever memory_max. The crash is a hard
signal 11 in `godot.console.exe` (`main+40d4d61` when the handler ran), not a
guest exception, so it is host-side memory corruption or a host resource the
guest left behind. Not caused by: the push path (whole `usd_open` of 14.9 MB
crashes the same), the host-side node/texture work (skipping it moves the
crash, does not remove it), the addon's background binary translation (off,
same crash). Nothing the guest does on the ERR path is known to differ from
Gate 0G's clean ERRs except that 0G never freed its Sandbox and never opened
a truncated **zip**. Candidates, none tested: pxr's diagnostic path on a
zip/usda failure (`ArchLogStackTrace` -> the arch port's raw `execve`
ecall?), a TfError/TfErrorMark left across the vmcall, TBB's
`cache_aligned_allocator` through `vendor/sandbox-api`'s memalign side
table, `clear_tables()`'s swaps, `erase_asset` while a `Sdf_UsdzResolver`
cache still holds the package.

Mitigations in the stage today: one document per Sandbox (`open_package`
frees and reloads usd.elf, ~1 s, for every document after the first; the
loop's INFER opens one a run) and `binary_translation_bg_compilation`
off. Neither is enough for the gate's ERR cases.

## Resume (exact next steps, nothing else was started)

1. Bisect the guest's ERR path with the prepared flags. The script
   `bisect_debug_flags.py` (copied here from the session's scratchpad) edits
   `guest/usd/usd_core.{h,cpp}`, `guest/usd/main.cpp`, the stage, main.gd and
   the probe to add `usd_debug(flags)`: 1 = no `close()` on an open error,
   2 = never `erase_asset` on close, 4 = `clear()` without the swaps, 8 = no
   `TfErrorMark`. Then:
   ```
   python gates/U-usd/bisect_debug_flags.py
   BUILD_DIR=C:/b/gu-elf BUILD_FIT=0 BUILD_TARGETS=usd ./build.sh
   for d in 0 1 2 4 8 15; do godot --path project --script probe_usd_push.gd --rendering-driver vulkan --xr-mode off -- --seq=garbage.bin,stub-quad.usdz --debug=$d; echo "debug=$d exit=$?"; sleep 3; done
   ```
   A flag whose run exits 0 names the guest-side cause. If none does, the
   cause is in pxr's failure path: pre-validate the bytes on the host
   (`util/usd_nodes.gd` or the stage: a zip needs its end-of-central-directory
   record and stored members; a crate its `PXR-USDC` magic and a table of
   contents inside the file; `#usda` for text) and refuse without calling the
   guest, then keep one ERR case in the gate as the recorded hazard.
2. Shape (a): `--seq` of real-asset x 6 at memory_max 320 with `--debug=4`
   (no swaps) and `--debug=2`; if neither helps, keep one document per
   Sandbox (already in the stage) and record it.
3. Then `bash gates/U-usd/run.sh` end to end (oracle, ladder, gate, rule 8)
   and fill in the negatives table above; `git add gates/U-usd` and commit.
4. Report the reproducer upstream (`V-Sekai-fire/godot-sandbox`): `usd.elf`
   sha256 `006d25338ef6…`, `probe_usd_push.gd --seq=garbage.bin,stub-quad.usdz`.

Not done yet, in the order they should follow: the gate's PASS line; the
pipeline's INFER state calling `usd_stage.open_package` on the service's
`usd_b64` (the stage and `usd_nodes` are ready for it: `document()` ->
`to_node()`); a native flat control of `usd_core.cpp` (the host oracle is
usd-core 26.8 through Python, not the same TU, as in Gate 0G).

## Reproduce

```
# the OpenUSD riscv64 build: gates/0g-openusd/README.md (C:/b/g0g, source C:/b/usd2605 == org riscv64-sandbox)
BUILD_DIR=C:/b/gu-elf BUILD_FIT=0 BUILD_TARGETS="usd usd_probe" ./build.sh   # -> project/usd.elf (gitignored, sha256 006d25338ef6...)
godot --path project --headless --import --xr-mode off
bash gates/U-usd/run.sh        # oracle, ladder (one process a rung), gate, rule 8
```

### 2026-09-25: BLAKE3, and wiring for unresolved textures

Every checksum Gate U compares is BLAKE3 (`guest/common/blake3.h`, first 12
hex digits). Godot's `HashingContext` has no BLAKE3, so `gate_usdz.gd` sums
what crossed by handing the arrays back to `usd_blake3`: a byte lost on
either trip changes the sum. `host-oracle.log` was regenerated on Linux
(usd-core 26.8, Python 3.11) from the committed inputs only; the real packages
and the inputs derived from them (`real-*.usdz`, `truncated-*.usdz`,
`bare-asset.usdc`) read `missing` until `host_oracle.py` runs where
`C:/b/gu-inputs/` is.

`usd_material` now answers `<input>_file` with the connected texture's
authored path even when it does not resolve, and the gate records every
connected input as wiring. Before, `bare-asset.usda` (a loose layer whose
textures are not beside it) read wiring `-` while the host listed three
connections. Linux headless: Gate U PASS on the committed cases.
