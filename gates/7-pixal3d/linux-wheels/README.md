# Linux CUDA extension wheels: which GPUs can run them

Question: the RunPod Pixal3D image (tools/runpod/pixal3d) takes its five CUDA
extensions (cumesh, flex_gemm, nvdiffrast, nvdiffrec_render, o_voxel) prebuilt from
`V-Sekai-fire/ComfyUI-Trellis2-visualbruno` @14597418, because RunPod's build has no
GPU and 30 minutes. Which of the Linux wheel sets there run on which GPUs?

Plan of record assumed `wheels/Linux/Torch291` (torch 2.9.1, cp312) carries sm_86
cubins + sm_86 PTX. **It does not**: four of its five extensions carry sm_120 code
only. The image built from it failed on the desk's RTX 3090.

```
python gates/7-pixal3d/linux-wheels/fatbin_archs.py > fatbins.log
```

lists every (PTX|SASS, sm) entry of the CUDA fatbinaries inside each wheel's
extension modules (parsed directly, no CUDA toolkit). The flat control is the
Windows Torch280 set that runs on the desk's 3090 (sm_86) under
tools/services/pixal3d: the parser reports sm_86 for all of it, so a missing sm_86
elsewhere is the wheel, not the parser.

## Result (2026-09-23), `fatbins.log`

| set | cumesh | flex_gemm | nvdiffrast | nvdiffrec_render | o_voxel | runs on |
|---|---|---|---|---|---|---|
| Linux/Torch291, cp312, cu12 | SASS+PTX sm_86 | sm_120 | sm_120 | sm_120 | sm_120 | sm_120 only (RTX 5090, RTX PRO 6000) |
| Linux/Torch270, cp312, cu12 | SASS+PTX sm_86 | SASS+PTX sm_86 | SASS+PTX sm_86 | **missing** | SASS 80/86/89/90/120 | no nvdiffrec_render; torch 2.7's triton 3.3 fails MoGe-3's FlexGEMM 2.0 (tools/services/pixal3d/README.md) |
| Linux/Torch2110, cp313, cu13 | SASS 80/86/100/120 | SASS 80/86/100/120 (1.0.0) | SASS 80/86/100/120 + PTX 120 | SASS 80/86/100/120 + PTX 120 | SASS 80/86/100/120 | sm_80, sm_86, sm_89 (runs sm_86 SASS), sm_100, sm_120; **not sm_90** (H100/H200) |
| CONTROL Windows/Torch280, cp311 | sm_86 | sm_86 | sm_86 | sm_86 | sm_86 | the desk's 3090 (measured, tools/services/pixal3d) |

A SASS cubin for sm_X.y runs on sm_X.z, z >= y; PTX for sm_N JIT-compiles for sm >= N.

Measured in the image on the RTX 3090 (sm_86), `python smoke.py --imports-only`:

- Torch291 (`torch291-check-3090.log`): every import passes (imports load no kernel),
  then `nvdiffrast: Cuda error: 209 [cudaFuncGetAttributes(fineRasterKernel)]` and
  `flex_gemm conv: CUDA error: no kernel image is available for execution on the
  device`. cumesh and NAF pass. FAIL.
- Torch2110 (`torch2110-check-3090.log`): torch 2.11.0+cu130, triton 3.6.0; all ten
  imports and all five kernels pass (nvdiffrast rasterizes 1352 px, the flex_gemm
  Triton conv runs, CuMesh initializes, na2d_block matches brute force to 8.9e-16).
  PASS.

So the linux-64 target of tools/services/pixal3d/pixi.toml takes Torch2110: python
3.13, torch 2.11.0 + cu130 (the wheels link libcudart.so.13; host driver >= 580), and
flex_gemm 1.0.0, which is 0.0.1's module tree plus `ops/serialize` (the API Pixal3D
calls is unchanged; the 2.0 rewrite MoGe-3 needs stays the separate flex_gemm2). The
Torch2110 o_voxel also drops the Torch291 wheel's `cumesh @ git+JeffreyXiang/CuMesh`
and `flex_gemm @ git+JeffreyXiang/FlexGEMM` Requires-Dist, which uv refused beside
the org URLs ("conflicting URLs") and which point outside the org (rule 1).

Not measured: sm_89 (the 4090 was another agent's), sm_100, sm_120. RunPod's GPU list
for this endpoint must leave out H100/H200 (sm_90).
