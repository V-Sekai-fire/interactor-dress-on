# NAF (Neighborhood Attention Filtering): architecture and op census for Pixal3D's feature upsampler

Sources:
- Code: `V-Sekai-fire/NAF` @37f2dfc ("Add eval depth probing"), in `naf-src/`: `hubconf.py`, `src/model/naf.py`, `src/layers/{attentions,convolutions,rope}.py`, `config/model/naf.yaml`.
- Consumer: `V-Sekai-fire/Pixal3D` @cdbb2bb, in `pixal3d-src/`: `inference.py` (`IMAGE_COND_CONFIGS`), `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py` (`DinoV3ProjFeatureExtractor`, `ProjGrid`, `sample_features`), `pixal3d/pipelines/pixal3d_image_to_3d.py` (`get_proj_cond_shape`), `pixal3d/modules/{,sparse/}attention/proj_attention.py` (`proj_linear`).
- Weights: `models/NAF/naf_release.pth`. It is byte-identical to the upstream release asset: sha256 c096c1ab…847c98f, 2,664,431 B.
- DINOv3 feeding it: `models/dinov3-vitl16/model.safetensors` (camenduru/dinov3-vitl16-pretrain-lvd1689m).
- Reference: `naf_ref.py`. Its output is in `naf_oracle/{shape_512,shape_1024,tex_1024}/` and its logs are `naf_oracle/naf_ref.*.log`.
- NATTEN semantics: NATTEN 0.21.7, installed as a pure-python target in `natten_pkg/`. On CPU it runs the `flex-fna` backend, which is torch flex_attention, so no compiled extension is needed.

## 0. Result: the new-op list

**No new Lean kernel is required.** Every op in NAF, and every op in the Pixal3D code that consumes NAF's output, decomposes exactly into the 22-op set plus host-built int32/float tables. `naf_ref.py` checks each decomposition against torch or against real NATTEN, and every result is at fp32 rounding (≤ 3.4e-6 relative; the 512–1024 end-to-end runs are ≤ 8.6e-7).

Upstream `natten.na2d(backend="cutlass-fna")` needs CUDA. NAF only ever calls it on K/V that were nearest-exact upsampled by an integer factor d. In that case the dilated neighbourhood attention collapses exactly into a **block attention**: every hi-res query in low-res block (bi,bj) attends to the same 9×9 low-res window, which starts at `clamp(bi-4, 0, hk-9)`. That window is a GET_ROWS gather over a host table. A literal port of NATTEN's dilated window rule and the block form were both checked against real NATTEN flex-fna, and the block form was also checked against the literal rule at full Pixal3D resolution.

| op outside the 22 | where in NAF | exact decomposition | check (max rel err) |
|---|---|---|---|
| **natten `na2d`** (2-D neighbourhood attention, k = 9×9, dilation d = T/hk, no RPB, scale 64^-0.5) | `CrossAttention`, the only attention | K/V never upsampled. Pixels are reordered block-major by one host `GET_ROWS` permutation. Then per 256-block chunk: `GET_ROWS`(K_lr, win81) → `CONT(PERMUTE)` → `MUL_MAT`(Kw, Qb) → `SOFT_MAX_EXT`(scale 0.125) → `GET_ROWS`(V_lr, win81) → `CONT(PERMUTE)` → `MUL_MAT`(Vwᵀ, P). The alternative is `FLASH_ATTN_EXT`(q [64,d²,4,Bk], k [64,81,4,Bk], v [256,81,4,Bk]) with Dk = 64 and Dv = 256; if the Lean FA wants Dv = Dk, run it 4× on 64-wide V slices, which is exact because attention is linear in V. | block vs NATTEN 4.6e-7; literal rule vs NATTEN ≤ 5.3e-7 (8 cases incl. non-square, k ≠ square, Dv ≠ D); FA-split vs chain 3.5e-7; full-res vs module 7.0e-7 (512) |
| `F.interpolate(mode="nearest-exact")` of K and V to T×T | `CrossAttention._resize` | **never materialised**: it is absorbed into the block attention. The general form is `GET_ROWS` with host `src = min(floor((i+.5)·in/out), in-1)`. | 0.0 |
| Conv2d 3×3, `padding_mode="reflect"` | `sem_encoder` (5 convs) | one host table `idx[p·9+t]` = reflect-padded source pixel of tap t for output pixel p. Then `GET_ROWS`(x channels-last [Ci,P], idx) → `RESHAPE` [9·Ci, P] → `MUL_MAT`(cols, W[9·Ci, O]) → channel-major [P, O] → `ADD` bias. Banded over P. Alternative: row-reflect by `GET_ROWS`, column-reflect by `GET_ROWS` after `CONT(TRANSPOSE)`, then `IM2COL`+`MUL_MAT`. | 1.1e-6 (both forms) |
| Conv2d 1×1 (reflect is a no-op at k = 1) | `encoder` (5 convs) | `CONT(TRANSPOSE)` → `MUL_MAT` → `ADD` bias | in the end-to-end check |
| GroupNorm(8 groups, 128 ch, eps 1e-5, affine) | 8 per branch, 16 total | on channel-major [P, C] each group is contiguous: `RESHAPE` [16·P, 8] → `NORM`(1e-5) → `RESHAPE` → `MUL` γ[1,C] → `ADD` β[1,C] | 1.6e-7 |
| SiLU | after every GN | `SILU` | – |
| channel concat 128 ‖ 128 | `ImageEncoder.forward_encoder` | `CONCAT` (dim 1 of channel-major) | – |
| `adaptive_avg_pool2d`, integer factor | encoder → T (factor r = S/T ∈ {1,2}); KeyEncoder T → hk (factor d ∈ {8,16}) | encoder: `RESHAPE` [r,W/r,H,C] → `MEAN` → `RESHAPE` [W/r,r,H/r,C] → `CONT(PERMUTE)` → `MEAN`. keys, in block-major order: view [256, d², Bk] → `CONT(PERMUTE)` [d²,256,Bk] → `MEAN` | in the end-to-end check |
| `adaptive_avg_pool2d`, non-integer factor (Pixal3D never hits this) | same | two `MUL_MAT`s with host bin-average matrices (separable) | 1.8e-7 |
| bilinear `interpolate`, align_corners = False, downscale (only when S > 4T, which **never fires in Pixal3D**, since S ≤ 2T) | `ImageEncoder.forward` pre-resize | two `MUL_MAT`s with host 2-tap matrices (separable), or per-axis 2 × `GET_ROWS` + `MUL` + `ADD` | 2.4e-6 |
| 2-D axial RoPE (DINOv3 form, 4 heads × 64, base 100, 16 periods per axis, coords `2(i+.5)/n-1`) | `ImageEncoder.rope` on the queries | two `ROPE` passes, `GGML_ROPE_TYPE_NEOX`, n_dims 64, freq_base 1, freq_scale 1, ext_factor 0, attn_factor 1. Host int32 positions: pos_h = 2y+1-T, pos_w = 2x+1-T. Host freq_factors [32]: pass 1 is `[T·period/2π ×16, 1e30 ×16]`, pass 2 is `[1e30 ×16, T·period/2π ×16]`. This works because theta_i = pos/ff_i = 2π·coord/period, and 1e30 makes the rotation the identity to below fp32 epsilon. Bit-level alternative: host cos/sin tables with `MUL`, `ADD`, `NEG` and `CONCAT` (rotate-half). | 4.0e-7 |
| **grid_sample**, bilinear, align_corners = False, padding border (Pixal3D `ProjGrid`, on lr and hr) | consumer | host: project the R³ grid, or only the active sparse voxels, to pixels. Tap table `x = clamp(((g+1)W-1)/2, 0, W-1)`, i0 = ⌊x⌋, i1 = min(i0+1, W-1), same for y, giving 4 indices and 4 weights per point. Device: 4 × `GET_ROWS`(feat [C, H·W]) → `MUL` w[1,K] → `ADD`. For hr the host re-indexes taps into block-major slots, so NAF's output is never un-permuted. | lr **0.0**; hr on the block-major ggml output 5.4e-7 |
| lr ‖ hr → 2048 | `DinoV3ProjFeatureExtractor.forward` | `CONCAT` on ne0, or skip it: split `proj_linear` W[1536, 2048] into two halves and `ADD` two `MUL_MAT`s | – |

The following do **not occur** in NAF or on its Pixal3D path: deformable conv, window attention, relative-position bias (the NATTEN ≥0.20 `na2d` path has none, and the legacy `na2d_qk/av` path is called without one), bicubic, and align_corners = True. GELU, LayerNorm and FLASH_ATTN_EXT come from DINOv3 (census'd separately), not from NAF.

View ops (`RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`) are metadata-only in ggml. They are materialised by `CPY/CONT` where noted.

**Optional, not required: a fused Lean `na2d_block` kernel** (spec in §6). It removes the 81× K/V window gathers: unchunked at tex_1024, the V window alone is 1024·81·4096·4 B = 1.36 GB and the K window 0.34 GB. Chunking the block axis (the reference uses 512 blocks per chunk, 170 MB of V) keeps the decomposed form practical. The fused kernel is a bandwidth optimisation, like MoGe's optional fused ReLU.

## 1. How Pixal3D uses NAF

- `inference.py` `IMAGE_COND_CONFIGS` has four DINOv3 feature extractors (`DinoV3ProjFeatureExtractor`). Three of them use NAF:

  | stage | image_size S | DINOv3 grid hk = S/16 | `naf_target_size` T | d = T/hk | encoder pool r = S/T | ProjGrid R (default; pipeline overrides for HR) |
  |---|---|---|---|---|---|---|
  | `ss` | 512 | 32 | no NAF | – | – | 16 |
  | `shape_512` | 512 | 32 | **512** | 16 | 1 | 32 |
  | `shape_1024` | 1024 | 64 | **512** | 8 | 2 | 64 (`actual_hr_resolution//16`, ≤ 64) |
  | `tex_1024` | 1024 | 64 | **1024** | 16 | 1 | 64 (same override) |

  The config-file comments say "DINOv3 32x32" for the 1024 stages. The code decides: `patch_number = image_size // 16`, so it is 64×64 at 1024, and the reference confirms this.
- **DINOv3 features in.** `extract_features` runs all 24 layers of ViT-L/16. It then applies `F.layer_norm(h, (1024,))` **without affine**; this is *not* the model's final `norm`, and no intermediate layers are used. Tokens are `[cls, 4 registers, hk·hk patches]`. `lr_features = z[:, 5:]` is reshaped to [1, hk, hk, 1024] and permuted to [1, 1024, hk, hk]. In ggml the patch tokens are already [1024, hk·hk], which is exactly the V_lr layout NAF's attention and the lr grid_sample want, so no transpose is needed.
- **Guide image.** `image_for_naf` is the same S×S LANCZOS-resized RGB, in [0,1] and *not* ImageNet-normalised. It is a clone taken before `transforms.Normalize`.
- **Call.** `hr = naf_model(image_for_naf, lr_features, (T, T))` gives [1, 1024, T, T]. NAF is loaded by `torch.hub.load("valeoai/NAF", "naf", pretrained=True)`, which is fp32 with the default config (dim 256, 4 attention heads, 4 RoPE heads, kernel 9, img_layers 2, rope_base 100).
- **Consumption.** `ProjGrid` projects a dense R³ grid through a fixed front camera (`camera_angle_x`, `distance` = 2, `mesh_scale` = 1; the FoV comes from MoGe) and samples lr and hr with `grid_sample(bilinear, align_corners=False, border)`. The result is `z_proj = cat([z_lr, z_hr], -1)` of shape [1, R³, 2048]. `get_proj_cond_shape` then keeps only the rows at the sparse voxel coords, so a ggml port should project **only the active voxels**. The 2048 channels feed `proj_linear = nn.Linear(2048, 1536)` in every one of the 30 DiT blocks (`ProjectAttention` / `SparseProjectAttention`), where the result is added to the cross-attention output. The unconditional branch uses zeros.
- **Sharing.** `shape_1024` and `tex_1024` receive the *same* 1024² image, so their DINOv3 tokens and their NAF encoder output (before pooling, RoPE and attention) are identical. The encoder, which is 86–96% of NAF FLOPs, can run **once** for both. `shape_1024` then 2×-pools it before RoPE, and `tex_1024` uses it at full size. See §5 and the check in §7.

## 2. Forward graph at Pixal3D's resolution (B = 1)

Notation: S is the image size, T the output size, hk = S/16, d = T/hk, r = S/T, P_S = S², P = T², Bk = hk². Torch shapes are NCHW.

| # | block | op | shape shape_512 (S 512, T 512, hk 32, d 16) | shape_1024 (S 1024, T 512, hk 64, d 8) | tex_1024 (S 1024, T 1024, hk 64, d 16) |
|---|---|---|---|---|---|
| 0 | inputs | image [0,1] / DINOv3 lr | [1,3,512,512] / [1,1024,32,32] | [1,3,1024,1024] / [1,1024,64,64] | same as shape_1024 |
| 1 | pre-resize | `if S > 4T: bilinear` | skipped | skipped | skipped |
| 2 | `encoder` (1×1 branch) | Conv1×1 3→128, then 2 × EncBlock[GN8 → SiLU → Conv1×1 128→128 → GN8 → SiLU → Conv1×1 128→128] (no residual, no shortcut) | [1,128,512,512] | [1,128,1024,1024] | [1,128,1024,1024] |
| 3 | `sem_encoder` (3×3 branch) | Conv3×3-reflect 3→128, then 2 × EncBlock[GN8 → SiLU → Conv3×3-reflect 128→128 → GN8 → SiLU → Conv3×3-reflect] | [1,128,512,512] | [1,128,1024,1024] | [1,128,1024,1024] |
| 4 | concat | cat(dim 1) | [1,256,512,512] | [1,256,1024,1024] | [1,256,1024,1024] |
| 5 | pool | adaptive_avg_pool2d → T | identity (r = 1) | 2×2 mean → [1,256,512,512] | identity |
| 6 | RoPE | 4 heads × 64, axial (16 h-freqs + 16 w-freqs per head) | [1,256,512,512] = queries | same | [1,256,1024,1024] |
| 7 | KeyEncoder | adaptive_avg_pool2d(q) → hk | [1,256,32,32] | [1,256,64,64] | [1,256,64,64] |
| 8 | values | lr features, untouched | [1,1024,32,32] | [1,1024,64,64] | [1,1024,64,64] |
| 9 | `_resize` | nearest-exact K, V → T (block-constant) | K [1,512,512,4,64], V [1,512,512,4,256] | same | [1,1024,1024,4,64/256] |
| 10 | `na2d` | 9×9 window, dilation d, scale 1/8, softmax over 81 | [1,512,512,4,256] | same | [1,1024,1024,4,256] |
| 11 | output | rearrange → NCHW | **[1,1024,512,512]** | **[1,1024,512,512]** | **[1,1024,1024,1024]** |
| 12 | Pixal3D | grid_sample lr and hr at R³ (or the sparse) points, then concat | [1,32768,2048] | [1,262144,2048] (R = 64) | [1,262144,2048] |

Head split: channel c belongs to head c // 64 for Q/K and c // 256 for V (`"b (n d) h w"`, n outer). The attention output is concatenated per head, so output channel = 256·head + j, which is DINOv3 channel order. NAF is a per-head convex combination of DINOv3 features: every hr feature is a softmax-weighted average of ≤81 lr features from the same head slice.

## 3. The ggml op chain (executed op-for-op by `naf_ggml` in `naf_ref.py`)

Layouts use ggml `ne` order, fastest first. "Channel-major" means [P, C] (ne0 = pixel), which is torch NCHW memory. "Channels-last" means [C, P].

Host tables, built once per (S, T, hk) and dumped for S, T as `tbl_*.npy`:
- `im2col_reflect[9·P_S]` (int32): for pixel p = y·S+x and tap t = 3ky+kx, the source is `rf(y+ky-1)·S + rf(x+kx-1)` with `rf(-1) = 1`, `rf(S) = S-2`. That is 37.7 MB at S = 1024; use it in bands.
- `block_perm[P]` (int32): for slot q = ((bi·hk+bj)·d+a)·d+b, the raster pixel is (bi·d+a)·T + bj·d+b. `inv` is its inverse.
- `pos_h[P]`, `pos_w[P]` (int32, block-major): 2y+1-T and 2x+1-T. `ff_h[32]` and `ff_w[32]` (f32) as in §0.
- `win81[Bk·81]` (int32): `(sh(bi)+u)·hk + sw(bj)+v`, with `sh(i) = clamp(i-4, 0, hk-9)`.
- grid_sample taps: `idx[4,K]` and `w[4,K]` for lr (raster over hk²) and for hr (raster over T², then mapped through `inv` to block-major).

Chain:
```
img_cm [P_S, 3]                                     (NCHW image, [0,1])
for branch in (encoder k=1, sem_encoder k=3):
  x = conv(img_cm, W0, b0)                          # conv: see below
  repeat 2 (EncBlock):
    x = SILU(ADD(MUL(RESHAPE(NORM(RESHAPE(x,[16P_S,8]),1e-5),[P_S,128]), g1), b1))
    x = conv(x, W1, bias1)
    x = SILU(ADD(MUL(RESHAPE(NORM(RESHAPE(x,[16P_S,8]),1e-5),[P_S,128]), g2), b2))
    x = conv(x, W2, bias2)
x = CONCAT(enc, sem, dim=1)                         # [P_S, 256]
if r == 2: x = MEAN(CONT(PERMUTE(RESHAPE(MEAN(RESHAPE(x,[2,S/2,S,256])),[S/2,2,S/2,256]))))   # [P, 256]
xb = GET_ROWS(CONT(TRANSPOSE(x)), block_perm)       # [256, P] channels-last, block-major
q  = ROPE(ROPE(RESHAPE(xb,[64,4,P]), pos_h, ff_h), pos_w, ff_w)   # NEOX, base 1
k_lr = RESHAPE(MEAN(CONT(PERMUTE(RESHAPE(q,[256,d²,Bk])))), [256,Bk])  # KeyEncoder
qb = CONT(PERMUTE(RESHAPE(q,[64,4,d²,Bk]) -> [64,d²,4,Bk]))
for each chunk of blocks:
  Kw = CONT(PERMUTE(RESHAPE(GET_ROWS(k_lr, win81),[64,4,81,b]) -> [64,81,4,b]))
  Pr = SOFT_MAX_EXT(MUL_MAT(Kw, qb), mask=NULL, scale=0.125, max_bias=0)       # [81,d²,4,b]
  Vw = CONT(PERMUTE(RESHAPE(GET_ROWS(V_lr, win81),[256,4,81,b]) -> [81,256,4,b]))
  O  = CONT(PERMUTE(MUL_MAT(Vw, Pr) -> [256,4,d²,b]))                          # [1024, d²·b]
hr_blk [1024, P]                                    # block-major pixel order
z_hr = ADD(ADD(ADD(MUL(GET_ROWS(hr_blk,i0),w0), MUL(GET_ROWS(hr_blk,i1),w1)), ...))   # [1024, K]
z_lr = same on V_lr [1024, Bk] with the lr taps
conv 1x1: MUL_MAT(CONT(TRANSPOSE(x)) [Ci,P], W[Ci,O]) -> [P,O]; ADD bias
conv 3x3: MUL_MAT(RESHAPE(GET_ROWS(CONT(TRANSPOSE(x)), im2col_reflect), [9Ci,P]), W[9Ci,O]) -> [P,O]; ADD bias
          (W stored with ne0 = (ci fastest, then t = 3ky+kx): torch w.permute(0,2,3,1).reshape(O, 9Ci))
```
`MUL_MAT(a, b)` is ggml's `res[i,j] = Σ_k a[k,i]·b[k,j]`, and `MUL_MAT(Vw, Pr)` contracts over the 81 window slots. With `FLASH_ATTN_EXT(qb, Kw, V [256,81,4,b])` the result arrives as [256, 4, d², b], already in the per-pixel layout.

Op counts per NAF call: 10 convs (5 GET_ROWS-im2col + 5 MUL_MAT for 3×3; 5 MUL_MAT for 1×1), 16 GN (16 NORM + 32 MUL/ADD), 16 SILU, 1 CONCAT, 0–1 pool pair, 2 ROPE, 1 key pool, plus per chunk 2 GET_ROWS + 2 MUL_MAT + 1 SOFT_MAX_EXT + 3 CONT, and 4 GET_ROWS + 4 MUL + 3 ADD per grid_sample.

## 4. Weights

- 37 tensors, all **float32**: 36 parameters plus the persistent buffer `image_encoder.rope.periods` [16]. The buffer equals `100^(2i/32)` exactly and can be host-generated.
- **662,528 parameters** (662,544 floats incl. periods), 2.65 MB:
  - `encoder` (1×1): conv0 3·128+128 = 512; 2 blocks × (4 GN·128 + 2 × (128·128+128)) = 67,072; total 67,584.
  - `sem_encoder` (3×3): conv0 27·128+128 = 3,584; 2 blocks × (512 + 2 × (1152·128+128)) = 591,360; total 594,944.
  - `upsampler` (CrossAttention): **0 parameters**. Q, K and V are the RoPE'd guide features, their pooled copy, and the raw DINOv3 features; there are no projections.
- Keep fp32. The whole model is 2.6 MB, and f16 would save nothing that matters.

## 5. FLOPs (2 per MAC, B = 1)

Per input pixel, the encoder costs 659,200 MAC: the 1×1 branch is 3·128 + 4·128² = 65,920 and the 3×3 branch is 27·128 + 4·1152·128 = 593,280. Per output pixel, the attention costs 103,680 MAC: QK is 4·81·64 = 20,736 and AV is 81·1024 = 82,944. GN, SiLU, RoPE and softmax add less than 2%.

| stage | encoder (at S²) | attention (at T²) | total | encoder share |
|---|---|---|---|---|
| shape_512 | 345.6 GFLOP | 54.4 GFLOP | **≈ 0.40 TFLOP** | 86% |
| shape_1024 | 1382.4 GFLOP | 54.4 GFLOP | **≈ 1.44 TFLOP** | 96% |
| tex_1024 | 1382.4 GFLOP | 217.4 GFLOP | **≈ 1.60 TFLOP** | 86% |
| all three, encoder shared at 1024 (§1) | 1728.1 | 326.2 | **≈ 2.05 TFLOP** (vs 3.44 unshared) | |

89.5% of the encoder cost is the four 128→128 3×3 convs at full image resolution, i.e. 4 × 1152 × 128 MAC per pixel. The grid_sample and the lr ‖ hr concat are negligible (4 taps × 1024 ch × K points). The consumer's `proj_linear` costs 2·2048·1536 FLOP per token per DiT block per step; that belongs to the DiT, not to NAF.

Sparsity: grid_sample only reads hr at the tapped pixels. Over the dense R³ grid that is 27.8% of pixels at shape_512, 47.7% at shape_1024 and 36.0% at tex_1024; restricting to the active voxels gives fewer. Evaluating NAF's attention only there matches the dense output (§7). The encoder cannot be sparsified, because five chained 3×3 convs give it an 11×11 receptive field before pooling, so this saves attention FLOPs only.

## 6. Optional fused kernel spec: `na2d_block` (not required)

- Inputs:
  - `Q` f32 [64, 4, P], block-major, RoPE'd (P = T², slot q = ((bi·hk+bj)·d+a)·d+b).
  - `K_lr` f32 [256, Bk] and `V_lr` f32 [1024, Bk] (Bk = hk², raster).
  - int params hk, d, ks = 9, and `scale` = 0.125.
- Output: `O` f32 [1024, P], block-major.
- Math: for slot q in block (bi,bj) and head n ∈ 0..3:
  - window j = 9u+v, u,v ∈ 0..8, at w_j = (clamp(bi-4,0,hk-9)+u)·hk + clamp(bj-4,0,hk-9)+v;
  - s_j = scale · Σ_{c<64} Q[c,n,q]·K_lr[64n+c, w_j];
  - p = softmax_j(s);
  - O[256n+c', q] = Σ_j p_j·V_lr[256n+c', w_j] for c' < 256.
- Dispatch: one workgroup per (block, head). K/V windows are loaded once into shared memory and reused by all d² ∈ {64, 256} queries of the block. That is 81·(64+256)·4 B = 104 KB per (block, head) when fully staged, so stage V in 64-channel slices.
- Oracle: `hr_features.npy` after `tbl_block_perm`, or `naf_ggml` in `naf_ref.py`.

## 7. Reference run (`naf_ref.py`, CPU, fp32, torch 2.14 CPU, 6 threads)

- The repo's `NAF` module runs unmodified from `naf-src` with the checkpoint (`load_state_dict(strict=True)`). Only `natten.na2d` is substituted, by `na2d_block_from_hires`, which asserts that K/V really are block-constant and then runs the block attention.
- DINOv3 is a pure-torch port of the HF `DINOv3ViTModel` eager path, checked line-by-line against `modeling_dinov3_vit.py`: key_bias off, exact GELU, RoPE on the patch tokens only, base 100, and the non-affine final layer_norm Pixal3D uses. It was not run against `transformers` here.
- Example: `pixal3d-src/assets/images/0_img.png`, run through Pixal3D's `preprocess_image` (alpha branch, 1.1× bbox crop, black bg) to 1038², then LANCZOS to S. Camera at pipeline defaults.

Equivalence tests (log `naf_oracle/naf_ref.shape_512.log`):
- **T1, NATTEN semantics.** Real NATTEN 0.21.7 flex-fna against the literal dilated-window port, on 5 random cases (non-square, k 3/5/9, d 1–5, Dv ≠ D): ≤ 3.7e-7. On 3 NAF-shaped cases (nearest-exact upsampled K/V, d 6/8/16), block form vs NATTEN ≤ 4.6e-7 and literal vs NATTEN ≤ 5.3e-7.
- **T4, decompositions.**
  - bilinear (ac = False) 2.4e-6
  - non-integer adaptive pool 1.8e-7
  - nearest-exact 0.0
  - GroupNorm via NORM 1.6e-7
  - reflect conv3x3: GET_ROWS + IM2COL 1.2e-6; GET_ROWS-im2col 1.1e-6
  - FA with 4×64 V split vs MUL_MAT chain 3.5e-7
  - axial RoPE as 2 × NEOX ROPE with freq_factors 4.0e-7
- **T2, whole model vs real NATTEN.** 160² image, 10×10×256 features → 80² (d 8, pool r 2). The block form gives 5.8e-7 and the ggml-literal chain 3.2e-6.

Pixal3D-resolution runs:

| config | NAF out | ggml chain vs module | literal NATTEN rule vs module (8 rows: edges, block borders, centre) | grid_sample lr / hr (block-major) | sparse NAF (tapped px) | DINOv3 / NAF module time |
|---|---|---|---|---|---|---|
| shape_512 | [1,1024,512,512] | 7.8e-7 | 7.0e-7 | 0.0 / 5.4e-7 | 72,876 px (27.8%): 7.8e-7 | 6.5 s / 10.4 s |
| shape_1024 | [1,1024,512,512] | 7.0e-7 | 8.6e-7 | 0.0 / 7.0e-7 | 124,976 px (47.7%): 7.0e-7 | 51.9 s / 18.7 s |
| tex_1024 | [1,1024,1024,1024] | 8.6e-7 | 8.6e-7 | 0.0 / 7.8e-7 | 377,076 px (36.0%): 8.6e-7 | 39.2 s / 36.7 s |

Sharing check (`naf_share_check.py`): the `dino_tokens` of shape_1024 and tex_1024 are identical (max abs **0.0**), and shape_1024 `enc_pooled` equals the 2×2 mean of tex_1024 `enc_pooled` to **9.3e-8** relative. One 1024² encoder therefore serves both stages.

Run history: an earlier attempt died in tex_1024. The sparse-NAF check gathered its V windows unchunked, which is U·81·1024 floats, about 166 GB at U = 377k, and the process reached a 109.5 GB working set before it was killed. That check is now chunked at 4096 points, and tex_1024 was rerun alone, which is why `naf_ref.shape_1024.log` and `naf_ref.tex_1024.log` are separate.

Oracle files per config, in `naf_oracle/<config>/` (all float32 `.npy`, NCHW unless noted; `meta.json` has shapes, sha256 prefixes, errors and FLOPs):
- `image_naf` [1,3,S,S]: the NAF guide image.
- `dino_tokens` [1,5+hk²,1024]: Pixal3D's `z`.
- `lr_features` [1,1024,hk,hk].
- `enc_pooled` [1,256,T,T]: after the concat and the pool.
- `queries_rope` [1,256,T,T].
- `keys_lr` [1,256,hk,hk].
- `hr_features` [1,1024,T,T]: NAF output, raster.
- `proj_points_ndc` [1,R³,2]: grid_sample coords.
- `z_proj` [1,R³,2048]: the `proj_linear` input.
- Host tables: `tbl_block_perm` [T²] i32, `tbl_window81` [hk²·81] i32, `tbl_gs_{lr,hr}_idx*` [4,R³] i32, `tbl_gs_{lr,hr}_w` [4,R³] f32.

Rerun: `.venv-convert/Scripts/python.exe naf_ref.py --configs shape_512,shape_1024,tex_1024 [--skip-tests] [--threads N]`. It needs `natten_pkg/` for T1/T2 only; the configs run without it. Peak RAM at tex_1024 was not measured after the chunking fix: the largest live tensors are 4 GB each (hr, hr_blk, rows_full) plus a 2 GB z_proj, and the run completed on a 128 GB machine. The dumps total 14 GB for the three configs.
