# Pixal3D service, native win-64

Stage 7's image -> mesh step (AGENTS.md: the loop calls this HTTP service; USD is
the interchange) running natively on Windows 11 under pixi. No Docker, no WSL.

```
pixi install            # torch 2.8.0+cu128, triton-windows, the Windows wheels below
pixi run fetch          # clone the pinned sources into src/, apply patches/, fetch + verify weights
pixi run check          # imports + one real kernel from each CUDA extension
pixi run smoke-stub     # WEFTSPUN_STUB=1: the HTTP + USD contract, no GPU, no weights
pixi run smoke          # real: POST /predict then /extract, USD layer with a UsdGeom.Mesh
pixi run serve          # the service on HOST:PORT (default 127.0.0.1:8000)
```

API (unchanged from the service repo): `POST /predict {image, seed, fov, resolution,
nviews, view_resolution}` -> `{state, views, cameras, camera_params, seconds,
vram_peak_mib}`; `POST /extract {state, decimation_target, texture_size}` -> `{glb,
layer, seconds, vram_peak_mib}`, `layer` being a base64 `.usda` whose
`/Asset/Geometry/Mesh_*` prims carry the mesh (points, faceVertexCounts/Indices,
vertex normals, primvars:st; upAxis Y, metersPerUnit 1).

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

`runs/` keeps the logs and JSON; the GLB/USD/PNG outputs are gitignored.

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
the VoxHammer service during this run, so the 3090 carried it; the 4090 number
is not measured.
