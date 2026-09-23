# voxhammer-image-mesh-editing (native win-64)

Stage 7's image-conditioned mesh edit, served over HTTP on this machine with
no Docker and no WSL: a pixi env (`platforms = ["win-64"]`), the org's service
and VoxHammer checkouts at pinned commits, one patch.

```
pixi run fetch     # clone src/service + src/VoxHammer at pinned commits, apply patches/, place + verify weights
pixi run serve     # http://127.0.0.1:8765  GET /health, POST /predict
pixi run smoke     # serve, one real edit on VoxHammer's example, check the USD, -> runs/smoke.json
```

Pin the card with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<4090 index>`
(the smoke does); the Step-1 render subprocess inherits it.

## Contract (OpenUSD in, OpenUSD out)

`POST /predict` JSON, every blob base64 (or a URL / data URI):

| field | content |
|---|---|
| `mesh` | USD layer (usdc crate or usda), one triangle `UsdGeom.Mesh`. GLB is refused (400). |
| `region` | USD layer, one `UsdGeom.Cube` in the mesh's frame: the editable box. |
| `reference` | PNG, the edited view (VoxHammer's `2d_edit.png`). |
| `source_view` | PNG, the source render of that view (`2d_render.png`). |
| `view_mask` | PNG, the 2D mask of that view (`2d_mask.png`). |
| `seed`, `views` | seed (numpy/torch and the Hammersley offset), Step-1 view count (150). |

Response: `mesh` = `edited.usdc`, a crate holding one `UsdGeom.Mesh` with
vertex `displayColor`, in the caller's frame, upAxis and metersPerUnit;
`layer` = `edit.usda`, whose `/Asset/Edit` references `./edited.usdc`
(write both side by side and the layer composes the geometry; mute it and the
source is back, RFD 0053); plus `plan`, `times`, `faces`, `peak_vram_mib`.

`WEFTSPUN_STUB=1` keeps the upstream stub behaviour (no model, empty crate).

## How it runs on Windows

The path is V-Sekai-fire/VoxHammer's own win-64 route (its `run_edit_test.py`
driver), loaded in-process by the patched server:

- torch 2.4.0+cu118, xformers 0.0.27.post2, spconv-cu118: prebuilt win_amd64
  wheels (sources in `pixi.toml`'s header). Nothing CUDA is compiled here, so
  the system nvcc (12.4) is not used.
- `ATTN_BACKEND=xformers` (trellis's sparse modules ignore `sdpa` and fall back
  to flash-attn, which has no torch 2.4/cu118 Windows wheel).
- Blender is off the path: `bpy` is stubbed, Step 1 renders with Mitsuba
  (`cuda_ad_rgb`, out of process), and every PLY boundary is a USD file
  (`mesh.usda`, `voxels*.usda`, the region Cube).
- nvdiffrast and kaolin are import shims; the only nvdiffrast caller
  (`to_glb`'s texture bake) is replaced by a Gaussian-to-vertex-colour export.
- pysdf builds from its sdist with MSVC (VS2022) against conda-forge eigen.

## Weights

`microsoft/TRELLIS-image-large` @ `25e0d31f`, 17 rows in
`tools/models/manifest.tsv`, placed in `<main checkout>/models/TRELLIS-image-large`
(hard links into the HF cache when it already holds them). No chibifire
mirror holds TRELLIS (v1) image-large: `chibifire/TRELLIS.2-4B` is the v2
model and `chibifire/Pixal3D-GGUF` carries only `ss_dec_conv3d_16l8_fp16`
(same sha256). `pipeline.json` is adapted to name the two encoders
(manifest revision `…+encoders`). DINOv2 `dinov2_vitl14_reg` comes through
`torch.hub` unpinned, as upstream loads it.

## Smoke result

PASS, 2026-09-23, RTX 4090 alone (`runs/smoke.json`, `runs/smoke_server.log`).
VoxHammer's example (model.glb as a USD Mesh, 5243 vertices / 8050 triangles;
mask.glb's bounding box as the region Cube; its three 2D images), seed 0:

| stage | seconds |
|---|---|
| server up (model load 35.9) | 48.2 |
| Step-1 Mitsuba render, 150 views, cuda_ad_rgb | 8.1 |
| DINOv2 features + voxelize (7125 voxels) | 18.0 |
| run_edit: invert, masked re-denoise, splice, decode | 123.8 |
| whole request (mask, export and USD write included) | 227.5 |

Peak VRAM 13.0 GB by torch, 18.6 GB on the card by nvidia-smi (1.2 GB idle,
display). The answer is a `PXR-USDC` crate, one Mesh of 212 812 vertices /
425 628 triangles; the returned layer composes it (1 Mesh prim, same face
count); its extent is the input's to within 0.02 on x (the edited region) and
0.001 on y and z, so the unit-frame round trip holds.

One fix the first run needed: VoxHammer's `extract_features` runs DINOv2 with
autograd on and keeps all 15 batches' tokens; the 4090 filled to 24.1 GB by
batch 3, WDDM paged it (24 s -> 76 s a batch), it failed, and its bare
`except:` reported success. The patch runs Steps 2 and 4 under `no_grad` and
checks for `features.npz`: 15 batches in 5 s.
