# Pixal3D `proj` attention: math, ggml op map, selection

Source: `V-Sekai-fire/Pixal3D` @ `cdbb2bb` (the org fork's HEAD; the local checkout
`C:/contract-manifest/3-interactor/pixal3d-upstream` is the same commit). Files read:

- `pixal3d/modules/attention/proj_attention.py` (dense `ProjectAttention`)
- `pixal3d/modules/sparse/attention/proj_attention.py` (`SparseProjectAttention`)
- `pixal3d/modules/transformer/modulated.py` and `pixal3d/modules/sparse/transformer/modulated.py` (the block wiring)
- `pixal3d/models/sparse_structure_flow.py`, `pixal3d/models/structured_latent_flow.py`
- `pixal3d/pipelines/pixal3d_image_to_3d.py`
- `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`: `ProjGrid`,
  `project_points_to_image_batch`, `sample_features`, `DinoV3ProjFeatureExtractor`.
  The back-projection lives here, not in the attention module.
- `inference.py`: `IMAGE_COND_CONFIGS` and the camera estimate.

Copies of the five files named in the task are in `pixal3d_src/`.

## 1. What "proj attention" is

It is **not** a new attention kernel. The block keeps TRELLIS.2's cross-attention
unchanged. The change is in the context it attends to, plus one additive per-token term:

```
ProjectAttention.forward(x, (global, proj)):
    global_out = CrossAttn(x, global)            # the TRELLIS.2 MultiHeadAttention(type="cross")
    proj_out   = proj_linear(proj)               # nn.Linear(proj_in -> C), bias=True, per block
    return proj_out + global_out
```

The block (`ModulatedTransformerCrossBlock`, dense and sparse) is otherwise TRELLIS.2 unchanged:

```
h = LN1(x) * (1 + scale_msa) + shift_msa ; x += gate_msa * SelfAttn(h, rope)
h = LN2_affine(x)                        ; x += proj_linear(proj) + CrossAttn(h, global)   <- only change
h = LN3(x) * (1 + scale_mlp) + shift_mlp ; x += gate_mlp * MLP(h)
```

- **Dense (SS stage):** `proj` is `[B, N=G^3, P]`. Token `n` is aligned one-to-one with
  flow token `n`: the flow flattens `x.view(B,C,-1)` over (i, j, k) in "ij" order, and
  ProjGrid flattens `meshgrid(indexing='ij').reshape(-1,3)` the same way, so
  `n = i*G^2 + j*G + k`. The alignment requires G = the SS flow resolution (16).
- **Sparse (shape/tex SLAT stages):** `proj` is a SparseTensor whose feats are
  `z_proj_grid[b, x, y, z]` gathered at the active voxel coords
  (`get_proj_cond_shape`: `z_proj.reshape(B,G,G,G,-1)[b, x, y, z]`). The block adds
  `proj_linear(proj.feats)` row-wise to `global_out.feats`, so the voxel order must match
  the flow's voxel order. The feats are gathered with the flow's own coords, so it does.
- **`global` context:** `z_global = cat(CLS, 4 register tokens)` gives **5 tokens** x 1024,
  not TRELLIS.2's 1029 (CLS + registers + 32x32 patches). Cross-attention therefore runs over
  Lk = 5. The patch tokens reach the DiT only through `proj`.
- **CFG negative branch:** `neg_cond = {global: zeros, proj: zeros}`, so
  `proj_out_neg = bias` (a constant row). The matmul can be skipped there.
- The DINOv3 output used for both is `F.layer_norm(hidden, no affine)` over the last
  layer. That is `extract_features`, which is not the HF model's own affine final norm.

## 2. The back-projection and gather (ProjGrid), in closed form

Inputs per image: `camera_angle_x` (fov, rad), `distance` d, `mesh_scale` s, the
DINO image size R (512 for SS and shape_512, 1024 for shape_1024 and tex_1024), the
grid resolution G (16 SS, 32 shape_512, 64 shape_1024/tex), and the feature map
`F[H, W, C]`. F is the DINO patch grid `z[:, 5:].reshape(h, w, C)` with h = w = R/16,
or the NAF-upsampled map (section 4).

```
grid:   (x, y, z) = linspace(-1, 1, G) over (i, j, k)       # endpoints included, not voxel centres
world:  p_w = R_x(x, y, z) / (2 s) = (x, -z, y) / (2 s)      # rotation [[1,0,0],[0,0,-1],[0,1,0]]
camera: c2w = [[1,0,0,0],[0,0,-1,-d],[0,1,0,0],[0,0,0,1]]    # camera at (0,-d,0), looking +y_w, up +z_w
        p_c = inv(c2w) p_w = (x/2s, y/2s, z/2s - d)          # Blender camera looks down -z_c
depth = d - z/(2s)
f_px  = (16 / tan(fov/2)) * R / 32 = R / (2 tan(fov/2))       # 32 mm sensor, focal in pixels
u = R/2 + f_px * (x/2s) / (depth + 1e-8)
v = R/2 - f_px * (y/2s) / (depth + 1e-8)                      # y flipped
grid_sample(F, ((u,v)+0.5)/R*2-1, bilinear, align_corners=False, padding_mode='border'):
  ix = clamp((u + 0.5) * W / R - 0.5, 0, W-1);  iy = clamp((v + 0.5) * H / R - 0.5, 0, H-1)
  x0 = floor(ix), y0 = floor(iy), x1 = min(x0+1, W-1), y1 = min(y0+1, H-1), wx = ix-x0, wy = iy-y0
  proj[n, :] = (1-wx)(1-wy) F[y0,x0] + wx(1-wy) F[y0,x1] + (1-wx)wy F[y1,x0] + wx wy F[y1,x1]
```

- `valid_mask` (in view, depth > 0) is computed and then **unused**. Out-of-view points
  get border-clamped features.
- The `transform_matrix` argument is asserted None, so the only camera is the front view
  with a variable distance.
- **Checked:** `check_proj_gather.py` compares the verbatim upstream torch path against
  this closed form (4 row gathers + weights) on 4 configurations (G 16/32/64, fmap
  32/64/512, several fov/d/s). Max relative error is 6.1e-5, which is float32 coordinate
  rounding in the upstream path (`check_proj_gather.log`, PASS).
- **Camera parameters** come from outside the model:
  - `inference.py` gets fov from MoGe-2 (`Ruicheng/moge-2-vitl`, `fx`), or `--manual_fov`.
  - It solves d so that grid point (-1,0,0) lands at pixel column `-extend_pixel`:
    `d = f_px * (1/(2s)) / (R/2 + extend_pixel)`. With `extend_pixel = 0` this is
    `d = 1 / (2 s tan(fov/2))`.
  - The pipeline defaults (no estimate) are fov = 0.8575560450553894, d = 2.0, s = 1.0.

Everything above the `grid_sample` line depends only on (fov, d, s, R, G, H, W). The
index/weight table `idx[4][N], w[4][N]` is fixed per image and camera. It is built once on
the host (or by a trivial Lean kernel) and never inside the denoising loop.

## 3. ggml op mapping

| step | when | ggml ops | family |
|---|---|---|---|
| index/weight table (sec. 2) | once per image+camera, host | none (driver math); in-graph would need DIV, FLOOR, CLAMP | outside the graph |
| gather `proj = Σ_k w_k * F[idx_k]` | once per stage | 4x `GET_ROWS(F[C, H*W], idx_k[N])` + 4x `MUL(g_k[C,N], w_k[1,N])` (bcast `[b...]`) + 3x `ADD` | K2 |
| sparse stages: rows at active voxels | once per stage | build the table directly for the active voxels' grid points; this equals `GET_ROWS(z_proj[C, G^3], x*G^2+y*G+z)` | K2 |
| NAF branch concat (shape/tex only) | once per stage | `CONCAT(lr, hr, dim 0)` gives P = 2048 | K2 |
| `proj_linear` | per block, per forward | `MUL_MAT(W[P,1536], proj[P,N])` + `ADD(bias)` bcast `[.b..]` | K6, K2 |
| cross-attn over `global` (Lk = 5) | per block | `MUL_MAT` (to_q, to_kv), `RMS_NORM` + `MUL` gamma (qk_rms_norm_cross), `FLASH_ATTN_EXT` D=128 H=12 Lk=5 (or `MUL_MAT`+`SOFT_MAX`), `MUL_MAT` to_out + `ADD` | K6, K3, K2, K8 |
| combine | per block | `ADD(proj_out, global_out)`, `ADD` residual | K2 |

**Ops outside the K1–K8 union: none**, provided the table is built on the host.
- A single fused "bilinear gather" (out[n,c] = Σ_k w[k,n] F[idx[k,n], c]) is the only
  K9 candidate. It is an efficiency fusion of 11 dispatches, not a correctness need.
- `proj_out` does not depend on x or t, so it can be hoisted out of the sampling loop.
  - Hoisted, SS costs 30 blocks x 1536 x 4096 floats = 755 MB f32 (377 MB f16).
  - Recomputed, it costs 30 x 2x1024x1536x4096 = 0.39 TFLOP per forward, about 4% of the
    DiT forward (~10 TFLOP at N = 4096).
  - Recompute unless G3.cost says otherwise.
- **Census confirmation:** the TRELLIS.2 graph census (`census_union.csv`, p3_ss_flow and
  p3_slat512 columns) already covers every op in this table.
  - FLASH_ATTN_EXT: D=128, H=12, f32 Q/K/V.
  - MUL_MAT: bf16 x f32 on the Aero-Ex weights.
  - GET_ROWS: f32, from shape_dec.
  - The pixal3d-ggml graphs do RoPE as rotate-half, `NEG` + `CONCAT` + `MUL`, not `GGML_OP_ROPE`.

## 4. What is *not* covered: NAF (a blocker for 7b)

`IMAGE_COND_CONFIGS` in `inference.py`:

| stage | image_size R | grid G | NAF | proj_in_channels |
|---|---|---|---|---|
| ss (`image_cond_model_ss`) | 512 | 16 | no | 1024 (ckpt json omits `proj_in_channels`; defaults to `cond_channels`) |
| shape_512 | 512 | 32 | yes, target 512 | 2048 |
| shape_1024 | 1024 | 64 | yes, target 512 | 2048 |
| tex_1024 | 1024 | 64 | yes, target 1024 | 2048 |

- With NAF, `proj = cat(gather(F_lr), gather(NAF(image, F_lr, target)))`, which is 2048
  channels. The GGUFs confirm it:
  - `blocks.*.cross_attn.proj_linear.weight` is `[1024, 1536]` in `ss_flow_img_dit_1_3B_64_bf16`;
  - it is `[2048, 1536]` in both shape SLAT flows (512, 1024) and in the tex flow.
- NAF is `torch.hub.load("valeoai/NAF", "naf", pretrained=True)`: a separate network with
  its own weights, outside V-Sekai-fire.
  - Rule 1 applies: the user must decide whether to fork it into the org.
  - Its ops still have to be censused.
  - Milestone 7a (DINO → SS flow → SS dec) needs only the lr gather and no NAF.
  - 7b (512 SLAT) and 7c need NAF.
- MoGe-2 (camera fov) is a second external network. A fixed or manual fov avoids it.

## 5. How the checkpoints select `proj`

- **Per-model ckpt json** (`Sparse/ss_flow_img_dit_1_3B_64_bf16.json`,
  `shape/slat_flow_img2shape_dit_1_3B_{512,1024}_bf16.json`,
  `texture/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json`): `"image_attn_mode": "proj"`.
  - The block constructor switches on it: `"cross"` gives `MultiHeadAttention`,
    `"proj"` gives `ProjectAttention(cross_attn_block, C, proj_in)`, and `"gated_proj"`
    gives the DINO+VAE-colour variant (not used by these checkpoints).
  - `proj_in_channels` is present (2048) only in the SLAT jsons.
  - The SS json has `resolution: 16`, although the file name says 64.
- **Tensor names** carry the mode:
  - `blocks.N.cross_attn.cross_attn_block.{to_q,to_kv,to_out,q_rms_norm,k_rms_norm}` and
    `blocks.N.cross_attn.proj_linear.{weight,bias}`;
  - TRELLIS.2 cross mode uses `blocks.N.cross_attn.{to_q,...}`.
  - A loader can detect proj mode from the presence of `blocks.0.cross_attn.proj_linear.weight`,
    and read P = 1024 or 2048 (NAF needed) from its `ne[0]`.
- **`pipeline.json`** (identical in `Pixal3D-GGUF` and `Pixal3D-src`) does **not** select it:
  - it still says `"name": "Trellis2ImageTo3DPipeline"` and
    `"image_cond_model": DinoV3FeatureExtractor(facebook/…)`;
  - `Pixal3DImageTo3DPipeline.from_pretrained` ignores that entry and sets every
    `image_cond_model_*` to None;
  - `inference.py` attaches four `DinoV3ProjFeatureExtractor`s built from
    `IMAGE_COND_CONFIGS`, hard-coded in Python.
  - `pipeline.json` still supplies the samplers (SS: 12 steps, cfg 7.5, rescale 0.7,
    interval [0.6, 1.0], rescale_t 5.0), the SLAT normalisation and `default_pipeline_type`
    ("1536_cascade").
- **The Aero-Ex GGUF** carries no mode KV at all:
  - it has `general.architecture = "flux"`, `general.file_type = 32`, and
    `comfy.gguf.orig_shape.*` for the reshaped tensors (SS: `rope_phases`,
    `*.q/k_rms_norm.gamma`, `input_layer.weight`).
  - A trellis2.cpp port therefore reads the mode from the sidecar json, or from the
    `proj_linear` tensor.
  - pixal3d-ggml @1c22f5e has no proj support anywhere: no `proj_linear` or `image_attn`
    in `trellis2.cpp` or in its converters.
