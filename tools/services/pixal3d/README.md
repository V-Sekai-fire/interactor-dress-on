# Pixal3D service, native win-64

Stage 7's image -> mesh step (AGENTS.md: the loop calls this HTTP service; USD is
the interchange) running natively on Windows 11 under pixi. No Docker, no WSL.

```
pixi install            # torch 2.8.0+cu128, triton-windows, the Windows wheels below
pixi run fetch          # clone the pinned sources into src/, apply patches/, fetch + verify weights
pixi run check          # imports + one real kernel from each CUDA extension
pixi run smoke-stub     # WEFTSPUN_STUB=1: the HTTP + USDZ contract, no GPU, no weights
pixi run smoke          # real: POST /predict then /extract, one .usdz checked by check_usdz
pixi run serve          # the service on HOST:PORT (default 127.0.0.1:8000)
python smoke.py --texture-sizes 2048,1536,1024 --name real-sweep   # one extract per size
python smoke.py --usdz FILE [--glb FILE]                            # check a package on disk
```

API: `POST /predict {image, seed, fov, resolution, nviews, view_resolution}` ->
`{state, views, cameras, camera_params, seconds, vram_peak_mib}` (unchanged from the
service repo). `POST /extract {state, decimation_target, texture_size}` -> **one
.usdz**: `{usd_b64, format: "usdz", usd_bytes, usd_files, texture_size, glb_path,
usdz_seconds, stub, seconds, vram_peak_mib}`. `texture_size` defaults to **1536**
(upstream 2048; below). The GLB is no longer in the response: the server keeps it on
its own disk at `glb_path`, for debugging only. The `glb` and `layer` keys are gone, so
a client still reading them fails with a KeyError rather than taking a zip for text.

The package (`patches/service-win.patch`, `_to_usdz`), stored uncompressed and
64-byte aligned as usdz requires (`Sdf.ZipFileWriter`):

| file | holds |
|---|---|
| `asset.usdc` (first, the root layer; binary crate) | `/Asset` (default prim; upAxis Y, metersPerUnit 1); `/Asset/Geometry/Mesh_0`, a UsdGeom.Mesh in the GLB's frame: points, faceVertexCounts/Indices, vertex normals, `primvars:st` (vertex); `/Asset/Materials/Material_0`, a UsdPreviewSurface bound with MaterialBindingAPI: `diffuseColor` <- base color `.rgb` (sRGB), `roughness` <- metallicRoughness `.g`, `metallic` <- `.b` (raw), glTF factors as the UsdUVTexture `scale`, `st` from a UsdPrimvarReader_float2; a normal, occlusion or emissive map would be wired the same way (o_voxel writes none) |
| `textures/material0_base_color.png` | the GLB's base color texels, RGB: the material is glTF `OPAQUE`, whose alpha "is ignored", and the alpha channel was 18% of that PNG (9.03 -> 7.40 MB at 2048) |
| `textures/material0_metallic_roughness.png` | the GLB's metallicRoughness texels (R 0, G roughness, B metallic) |

`smoke.py`'s `check_usdz` (also used by tools/runpod/pixal3d/local_test.py) passes a
package when: the first file is a `.usdc` crate and every file is stored and 64-byte
aligned; the stage has upAxis Y, metersPerUnit 1, a default prim and exactly one
UsdGeom.Mesh whose counts agree (3 indices a face, every index below the point count,
one normal and one st a point); the mesh is bound to a UsdPreviewSurface whose
diffuseColor, metallic and roughness come from UsdUVTexture nodes; every texture's file
resolves inside the package (`asset.usdz[textures/...]`) and opens as a PNG; every
registered UsdValidation validator reports no error; and the base64 is at most 18 MB
(RunPod's result cap is 20 MB). On the desk it also compares the package with the GLB
at `glb_path` (`compare_glb`): points, faces, st and the RGB texels of each texture must
be equal, and st must be (u, 1 - v) of the GLB's raw TEXCOORD_0 (glTF's uv origin is
the image's top left, USD's st origin its bottom left). Each check is shown to fire on
a package broken for it in gates/7-pixal3d/usdz (controls.log).

## What had to change for Windows

| Linux recipe | win-64 here | why |
|---|---|---|
| torch 2.6.0 cu124 | torch 2.8.0 cu128 | the prebuilt Windows wheels of the CUDA extensions exist for 2.7.0, 2.8.0, 2.10.0; 2.7.0's triton (3.3) was measured to fail FlexGEMM 2.0 (below) |
| triton | triton-windows 3.4.0 | flex_gemm's Windows wheel names it as its dependency |
| cumesh, flex_gemm, o_voxel, nvdiffrast, nvdiffrec_render (source / Space wheels) | V-Sekai-fire/ComfyUI-Trellis2-visualbruno (fork of visualbruno/ComfyUI-Trellis2) @14597418, `wheels/Windows/Torch280`, cp311 | no build: the desk's nvcc is 12.4, torch is cu128 |
| flash_attn | `ATTN_BACKEND=sdpa` | upstream's own backend switch; no Windows flash-attn build |
| natten (cutlass-fna) | na2d_block, `patches/naf-na2d-block.patch` | no NATTEN win-64 build; the pure-torch form is exact (gates/7-pixal3d/aux-models/naf_ref.py) |
| MoGe-2 | MoGe-3 (`Ruicheng/moge-3-vitl`), user's choice | its refiner needs FlexGEMM 2.0, exposed as `flex_gemm2` beside Pixal3D's 0.0.1 |
| RMBG-2.0 (gated) | `ZhengPeng7/BiRefNet_HR-matting`, user's choice | via `PIXAL3D_REMBG_MODEL` |
| preview views via `render_multiview` | `render_frames(..., envmap=forest.exr)`, `shaded` | a service bug, not a Windows one: `PbrMeshRenderer.render` requires an envmap `render_multiview` never passes, and there is no `color` key |

Measured on the way (negative results, kept because they cost time):

- torch 2.7.0 + triton-windows 3.3.1: every import and kernel in `pixi run check`
  passed, then MoGe-3's refiner failed in FlexGEMM 2.0's Triton hashmap:
  `AttributeError: 'dtype' object has no attribute 'itemsize'`. Rewriting that as
  `primitive_bitwidth` got one step further, to `KeyError: 'ier_type'` (triton 3.3
  does not parse `tl.pointer_type` annotations). triton 3.4 has both; so torch 2.8.
- pixi does not expand `$PIXI_PROJECT_ROOT` inside `[activation.env]` on win-64 (the
  literal string reached torch.hub), so `serve.py`/`smoke.py` set the paths.
- No card is pinned. Each launch takes the next GPU in PCI bus order
  (`../svc_common.py` `round_robin_gpu`, a counter beside the weights shared by
  every service and worktree), so Pixal3D and VoxHammer launched one after the
  other land on different cards: each needs most of a 24 GB card. A caller's
  `CUDA_VISIBLE_DEVICES` wins. Measured: four launches went 3090, 4090, 3090,
  4090, and torch saw only the card it was given.

Wheel sha256 (as downloaded from raw.githubusercontent.com at that commit):

| wheel | sha256 |
|---|---|
| cumesh-1.0-cp311-cp311-win_amd64.whl | d1ed72dfb2815776812bd634c175885bf373ae585306a5b819dc191041c1f024 |
| flex_gemm-0.0.1-cp311-cp311-win_amd64.whl | 71658a676bfc8e7d7b15550a4a502c759e6f315a064cb990a0235d1d1dd2d9ad |
| nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl | 7362a491d8012f3cd4b9097b029fc97161dbdf965a0658be313fbb8a64dfa3f0 |
| nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl | 2bc74423d15cce8dc7ee2db3741e6cc71e700f485cb51fd86f26c53b2755cab1 |
| o_voxel-0.0.1-cp311-cp311-win_amd64.whl | 531e370e7768d32a438ca7ebbdc5d7644cb3b9c4838f2539c6768dc464cb3145 |

Other sources, every one from an org fork (AGENTS.md rule 1; forked
2026-09-23 on the user's go): utils3d 0.0.2 from V-Sekai-fire/Storages release
20260430, a byte-for-byte mirror of LDYang694/Storages' asset (Pixal3D's pin;
sha256 ff63440827d6933807dd06c8a5a2db7e51fd5f33c7f3dddcc766a80e0f419252);
utils3d-moge @62f09d5 from V-Sekai-fire/utils3d-moge (EasternJournalist's, MoGe's
pin); FlexGEMM @b2fadb2 from V-Sekai-fire/FlexGEMM (JeffreyXiang's `dev/all_triton`,
MoGe's pin). The wheels come from V-Sekai-fire/ComfyUI-Trellis2-visualbruno at the
same commit (the org already had a different ComfyUI-TRELLIS2, PozzettiAndrea's,
hence the suffix).

## linux-64 (the RunPod worker image)

The same pixi.toml carries a linux-64 half in `[target.linux-64.*]` tables, used by
`tools/runpod/pixal3d/Dockerfile` (`pixi install --locked`, then `fetch.py --src`);
the win-64 half and its lock entries are unchanged by it (84 packages, the same
versions; the seven wheel/git URLs differ only by upstream -> org fork, and the five
wheels and utils3d hash to the sha256 values in the tables here). Linux takes python
3.13, torch 2.11.0 + cu130 (triton 3.6.0) and the `wheels/Linux/Torch2110` set of the
same fork commit, not the Torch291 set the plan named: Torch291's flex_gemm,
nvdiffrast, nvdiffrec_render and o_voxel carry sm_120 code only and failed on the
3090 ("no kernel image is available"); see gates/7-pixal3d/linux-wheels. Torch2110
covers sm_80, sm_86, sm_89 (via sm_86 SASS), sm_100 and sm_120, not sm_90 (H100).

Linux wheel sha256 (raw.githubusercontent.com, V-Sekai-fire/ComfyUI-Trellis2-visualbruno @14597418, `wheels/Linux/Torch2110`):

| wheel | sha256 |
|---|---|
| cumesh-1.0-cp313-cp313-linux_x86_64.whl | 60bf02cb02241ba3c942842b18494a43c9337b1bdf46def125b8c11670febadb |
| flex_gemm-1.0.0-cp313-cp313-linux_x86_64.whl | 088acf0a7d6207eedfc60afa5a887ce4eebe1af71cba97320686f575fd150cd6 |
| nvdiffrast-0.4.0-cp313-cp313-linux_x86_64.whl | 7432379e67596c45850394b416d7fd9d851dfeccd15336a6f8fd72c7beb5a131 |
| nvdiffrec_render-0.0.0-cp313-cp313-linux_x86_64.whl | aecccdea1f59acccf4936da9079bb54be7be0d70225e742eeb2914ec12610635 |
| o_voxel-0.0.1-cp313-cp313-linux_x86_64.whl | a6b17a70a63cdb3cb851df3189b58305763ac3f0bd74d5e9c8b3c9789dc2585a |

## Weights

All local and pinned; `serve.py` sets `HF_HUB_OFFLINE=1`. `fetch` downloads what is
missing into the main checkout's `models/` and checks every file against
`tools/models/manifest.tsv` (appending rows for new files):
`Pixal3D-src/` (TencentARC/Pixal3D @b0cb2e1, single-view files), `dinov3-vitl16/`,
`moge-3-vitl/model.pt`, `BiRefNet_HR-matting/`, `NAF/naf_release.pth`.
The chibifire mirrors were checked first: `chibifire/Pixal3D-Comfy` holds ComfyUI
repacks and `chibifire/Pixal3D-GGUF` holds GGUF, neither in the layout
`Pixal3DImageTo3DPipeline.from_pretrained` reads, and `chibifire/TRELLIS.2-4B` is
TRELLIS.2's flow models, not Pixal3D's.

## Results (2026-09-23, RTX 3090 24 GB, driver 616.92)

`runs/` keeps the logs and JSON; the GLB/USD/USDZ/PNG outputs are gitignored. The
`smoke` and `smoke-stub` rows here are the GLB + .usda contract; their `runs/real.*` and
`runs/stub.*` are in commit ba2c3b8 (the USDZ runs below replaced them).

| run | result |
|---|---|
| `check` | PASS: torch 2.8.0+cu128, triton 3.4.0; nvdiffrast rasterizes, flex_gemm SparseConv3d runs through Triton, CuMesh initializes, na2d_block matches a brute-force neighbourhood attention to 1.3e-15 (float64) |
| `smoke-stub` | PASS in 2.6 s: /health, /predict, /extract; the layer opens in usd-core, upAxis Y, metersPerUnit 1 |
| `smoke` (Pixal3D `assets/images/17_img.png`, RGBA 779x1081, seed 42, 1024 cascade, 4 views) | PASS in 279.7 s |
| `check` + `smoke-stub` again after the lock moved to the org forks and gained linux-64 (`runs/check.log`, `runs/stub.*`, 3090 via `CUDA_VISIBLE_DEVICES=0`) | PASS: the same torch 2.8.0+cu128 / triton 3.4.0, every import and kernel (na2d_block 1.6e-15); smoke-stub PASS in 3.2 s |

Real run: model load 130.1 s (to /health ready). /predict 103.2 s wall, peak
torch allocation 13,233 MiB; MoGe-3 camera_angle_x 0.727 rad, distance 1.314.
/extract 43.3 s wall, peak 5,966 MiB; GLB 18.4 MB. USD layer 25.2 MB, one
UsdGeom.Mesh: 195,434 points, 198,046 triangles, vertex normals, primvars:st.
nvidia-smi peak on the GPU: 18,285 MiB used (3 MiB before). The 4090 was taken by
the VoxHammer service during this run, so the 3090 carried it.

## USDZ result (2026-09-23, RTX 4090 24 GB, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`)

Why the change: one /extract answered with the GLB (18.2 MB, 24.31 MB as base64)
beside a text `.usda` layer (26.18 MB, 34.90 MB as base64), 59.21 MB of JSON, three
times RunPod's 20 MB result cap (tools/runpod/pixal3d/README.md). Written as a USDC
crate the same geometry is 6.6-7.0 MB (points, normals and st are float arrays the
crate stores raw; the index arrays it compresses), and one package carries the
textures the .usda never had.

**Texture sweep** (`runs/real-sweep.*`: one predict, then one extract per size on the
same state; same image, seed 42):

| texture_size | .usdz | base64 | asset.usdc (share) | base color PNG | metallicRoughness PNG | textures (share) | points / faces |
|---|---|---|---|---|---|---|---|
| 2048 (upstream) | 14,856,491 B | **19,808,656 B: over 18 MB** | 6,990,358 (47.1%) | 7,177,571 | 687,966 | 7,865,537 (52.9%) | 198,661 / 203,652 |
| **1536** | 11,703,016 B | 15,604,024 B | 7,000,038 (59.8%) | 4,302,632 | 399,792 | 4,702,424 (40.2%) | 198,887 / 204,294 |
| 1024 | 9,237,038 B | 12,316,052 B | 6,991,166 (75.7%) | 2,063,541 | 181,723 | 2,245,264 (24.3%) | 198,672 / 203,688 |

The zip adds 40-66 B a file. So `texture_size` defaults to 1536, the largest of the
three that fits: 56% of 2048's texels (2.36 M vs 4.19 M), base color PNG 7.18 -> 4.30
MB, base64 19.81 -> 15.60 MB, 2.4 MB under the budget. The mesh is not touched
(`decimation_target` stays 210,000); the point counts above differ by < 0.2% only
because extract is not deterministic on one state. A request may still name 2048;
through RunPod that result is over budget, and the worker's handler fails a result over
20 MB loudly instead of letting the gateway drop it. Other sizes of the base color at
2048, measured on the desk's earlier GLB: RGBA as the GLB has it 9,029,105 B, RGB
7,400,611 B, RGB with PIL `optimize` 7,173,372 B (1.2 s more), so neither the alpha
drop nor harder PNG compression alone brings 2048 under 18 MB.

**Default run** (`pixi run smoke`, `runs/real.*`): PASS in 258.4 s. Load 147.7 s,
/predict 60.3 s (the sweep had just filled Triton's cache; the sweep's first predict
took 189.7 s), torch peak 13,233 MiB. /extract 43.7 s wall (41.9 s decode + bake,
peak 6,377 MiB; `_to_usdz` 1.53 s of it). The package: **11,138,085 B, base64
14,850,780 B, the whole JSON response 14,851,171 B** (was 59.21 MB): `asset.usdc`
6,564,022 B (58.9%), textures 4,573,519 B (41.1%: base color 4,203,037, metallicRoughness
370,482), 1536 x 1536 each. One UsdGeom.Mesh, 185,772 points, 197,448 triangles,
vertex normals, st; bound to a UsdPreviewSurface reading both textures, which resolve
inside the package; UsdValidation reports nothing; `compare_glb` against the kept GLB
(13,830,080 B): points, faces and st identical (max |diff| 0), base color and
metallicRoughness RGB texels identical; st is (u, 1 - v) of the GLB's raw TEXCOORD_0
(that check came after this run: gates/7-pixal3d/usdz/recheck.log). nvidia-smi peak on
the 4090 20,779 MiB (2,184 MiB before, the desktop). `smoke-stub` (`runs/stub.*`, a
textured quad through the same writer, compare_glb included) PASS in 6.4 s.
