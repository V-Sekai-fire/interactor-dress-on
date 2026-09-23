# BiRefNet (HR-matting): architecture and op census

Source: `models/BiRefNet_HR-matting/birefnet.py` and `BiRefNet_config.py` (HF
`ZhengPeng7/BiRefNet_HR-matting` @5d6b6f8). Reference and proofs:
`birefnet_ref.py` (CPU torch 2.14, fp32) with output in `birefnet_ref_out/`, and
`birefnet_flops_meta.py`.

## Verdict: the new-op list

BiRefNet has **no** neighborhood attention (natten), **no** grid_sample, **no**
bicubic interpolation in the graph, and **no** relative-position interpolation. The
Swin window attention, cyclic shift, padding, patch merging, every
`F.interpolate(bilinear, align_corners=True)`, the patch rearranges, GAP and the
1×1→H×W broadcast all decompose into the 22-op set plus host index tables. The
script checks each decomposition on real weights and activations (numbers below).

Two ops are missing:

| # | op | why the 22-op set can't do it | where | size |
|---|----|-------------------------------|-------|------|
| 1 | **`bilinear_sample_zeros`** (the deformable-conv sampler; spec in section 6) | Sample coordinates are data-dependent: offsets come out of a conv in the same graph. The set has no floor, no float→int, and no data-indexed gather with fractional weights, and GET_ROWS needs host-built int32 indices. | 20 `DeformableConv2d` calls: 5 `ASPPDeformable` × taps k ∈ {1,1,3,7}, C=64 → 256 | 338 M bilinear samples at 1024² (1.35 G at 2048²) |
| 2 | **`relu`** y = max(x,0) (trivial elementwise) | RELU is not in the 22-op set. | 38 sites: 7 per `BasicDecBlk` × 5, plus 3 `gdt_convs` | elementwise |

If adding a kernel for `relu` has to wait, `MUL(x, SIGMOID(SCALE(x, 1e30)))` is exact
for |x| > ~1e-28. Measured on 2^20 samples: 4 mismatches, max error 2.7e-31. It
needs a sigmoid that returns 0/1 (not NaN) for ±inf. ggml-cpu's
`1/(1+expf(-x))` does; a GPU fast-math sigmoid has to be checked.

A fallback exists for #1 without a new kernel, but it breaks the pipeline into
pieces. Submit the offset/modulator convs, read the offsets back once the fence
is done, and have the host build four GET_ROWS index tables plus weights. The
sample is then 4×GET_ROWS + MUL + ADD, followed by MUL(mask). That means 5
readback round-trips per image (one per `ASPPDeformable`) and about 25 MB of offsets per
level-1 block. It stays within Rule 4, but a kernel is the right answer.

Host-side (C++, not graph) additions for Pixal3D parity: a PIL-compatible
**BILINEAR** resample to 1024² (with antialias) and a PIL **BICUBIC** resample for the
mask back to image size. pixal3d-ggml's `pil_coeffs/pil_resize` only has lanczos3,
and the filter is pluggable.

## 1. Configuration and how Pixal3D uses it

`Config` (hardcoded in birefnet.py): `bb=swin_v1_l` (embed 192, depths 2/2/18/2, heads
6/12/24/48, window 12, mlp 4, qkv bias, patch 4, LN, out norms), `mul_scl_ipt='cat'`
(the backbone runs twice: at full resolution and on a bilinear-ac ½-res copy, then
channel-concat), `cxt_num=3` (x1..x3 downsampled into x4), `dec_ipt=True`,
`dec_ipt_split=True` (image patches fed into every decoder level),
`dec_att='ASPPDeformable'` (parallel sizes 1,3,7), `squeeze_block='BasicDecBlk_x1'`,
`dec_blk='BasicDecBlk'`, `lat_blk='BasicLatBlk'` (1×1), `ms_supervision=out_ref=True`
(the gdt gating is live at inference), `batch_size=4` (so BatchNorm2d exists and
is folded at inference), `refine=''`, `SDPA_enabled=False`. Channels
`lateral_channels_in_collection` = [1536,768,384,192]×2 = **[3072,1536,768,384]**,
and `cxt` = [384,768,1536].

**Pixal3D** (`V-Sekai-fire/Pixal3D` `pixal3d/pipelines/rembg/BiRefNet.py`, same file
as TRELLIS.2):
1. `preprocess_image`: if the input is RGBA with any alpha≠255, **BiRefNet is
   skipped**. This covers 13 of the 19 bundled examples; the 6 `s_*` scene photos are RGB.
   Otherwise the image is downscaled so max side ≤ 1024 (PIL LANCZOS) and converted to RGB.
2. `BiRefNet.__call__`: `Resize((1024,1024))` (PIL BILINEAR, aspect **not**
   preserved), `ToTensor`, `Normalize(ImageNet mean/std)`, `model(x)[-1].sigmoid()`,
   then `ToPILImage` (`mul(255).byte()`, truncation) and `.resize(image.size)` (PIL
   default **BICUBIC**), then `putalpha`.
3. Post: bbox of `alpha > 0.8*255` (>204), square side `int(max(w,h)*1.1)` centred
   on it, `crop` (PIL rounds half-even and zero-fills outside), then `rgb*a + bg*(1-a)` with
   `bg_color=(0,0,0)` and uint8 truncation.
4. **Resolution is 1024², not HR-matting's native 2048²** (the HF `handler.py` uses
   2048 for `Matting-HR`).
5. **Model identity:** Pixal3D's `pipeline.json` (HF TencentARC/Pixal3D) names
   `rembg_model: {name: BiRefNet, args: {model_name: "briaai/RMBG-2.0"}}`. It does not
   name BiRefNet_HR-matting. RMBG-2.0 is a BiRefNet (Swin-L) retrain, but it is
   gated and CC BY-NC 4.0, so it was not verified locally. HR-matting is MIT.
   Same code, different weights.
6. transformers is pinned to 4.57.3, so `from_pretrained` loads the F16 file as
   **fp32**. On CUDA, cuDNN convs default to TF32. The upstream GPU output is
   therefore itself about 1e-3 from an fp32 CPU run.

**pixal3d-ggml** (`trellis2_remove_solid_background_rgba` and `trellis2_preprocess_rgba`)
has **no neural background removal**; `docs/PLAN.md` puts BiRefNet/RMBG-2.0 out
of scope. What it does:
- It uses existing alpha if more than max(4, 1%) of pixels have alpha < 250.
- AUTO mode flood-fills from the border through "near-black" (max(RGB) ≤ 80)
  or "near-white" (min(RGB) ≥ 175 → 255−min ≤ 80) pixels. It picks black or white
  when ≥55% of border pixels, or ≥3 corners, qualify. The alpha feather is a
  smoothstep on distance 12→72.
- `trellis2_preprocess_rgba` then matches the reference path (LANCZOS ≤1024, bbox
  >204, half-even crop, composite onto black) with **one difference: margin ×1.0
  (TRELLIS.2) versus Pixal3D's ×1.1**. It also takes no `bg_color`. A BiRefNet port
  needs the ×1.1 margin to match Pixal3D.

## 2. Forward graph at 1024² (Pixal3D), batch 1, NCHW

Legend: `C3` = Conv3×3 s1 p1, `C1` = Conv1×1, `BN` = folded, `R` = ReLU.

**Input** x (1,3,1024,1024).

**Backbone, full pass** (Swin-L). Tokens are [L,C]; every block is W-MSA or SW-MSA (shift 6, even/odd):

| stage | tokens H×W → padded (windows) | dim / heads / hd | blocks | out (after norm_i) |
|---|---|---|---|---|
| patch_embed | Conv4×4 s4 3→192 + LN | | | (192,256,256) |
| 0 | 256² → 264² (22²=484 win × 144) | 192/6/32 | 2 | x1 (192,256,256) |
| merge | 256²×192 → gather 2×2 → LN(768) → Linear 768→384 | | | |
| 1 | 128² → 132² (11²=121) | 384/12/32 | 2 | x2 (384,128,128) |
| merge | → LN(1536) → 1536→768 | | | |
| 2 | 64² → 72² (6²=36) | 768/24/32 | 18 | x3 (768,64,64) |
| merge | → LN(3072) → 3072→1536 | | | |
| 3 | 32² → 36² (3²=9) | 1536/48/32 | 2 | x4 (1536,32,32) |

Swin block: `x + proj(WMSA(LN1(x)))`, then `+ fc2(GELU_erf(fc1(LN2(x))))`, with MLP 4×.
Padding is applied after LN1 as zero rows, and those rows are **not masked**. They
become keys and values carrying only the qkv bias. Shifted windows use a −100
additive mask computed on the padded grid.

**Backbone, half pass**: `interp_ac(x, 512²)` (2-tap, no antialias), then the same Swin:
x1_ (192,128²) with 132² pad, x2_ (384,64²) with 72², x3_ (768,32²) with 36², and
x4_ (1536,16²) with **24²** (2²=4 windows; 8 of 24 rows are padding).

**Multi-scale cat**: x_i = cat(x_i, interp_ac(x_i_, size(x_i))), giving x1 (384,256²),
x2 (768,128²), x3 (1536,64²), x4 (3072,32²).
**cxt**: x4 = cat(interp_ac(x1,32²), interp_ac(x2,32²), interp_ac(x3,32²), x4), giving
**(5760,32,32)**. These are 2-tap downsamples 256→32, 128→32 and 64→32.

**ASPPDeformable(64)** (inside every BasicDecBlk, at the block's H×W):
`b1 = R(BN(DCN_k1(x)))`, `b2..b4 = R(BN(DCN_k{1,3,7}(x)))` (each 64→256, no bias), and
`b5 = repeat(R(BN(C1_64→256(mean_HW(x)))))`, then `R(BN(C1_1280→64(cat(b1..b5))))`.
In `DCN_k(x)`: `off = Conv_k(x) 64→2k²` (+bias, pad k//2),
`m = 2·sigmoid(Conv_k(x) 64→k²)`, then `deform_conv2d(x, off, W_k, mask=m, pad=k//2)`.

**BasicDecBlk(in→out)** = `BN(C3_64→out(ASPPDeformable(R(BN(C3_in→64(x))))))`.
There is no ReLU on its output.

**Squeeze**: BasicDecBlk(5760→3072) at 32², giving (3072,32,32).

**Decoder** (SimpleConvs = C3 in→64 then C3 64→out, with no activation;
`patches_g(x)` = rearrange `b c (hg h)(wg w) → b (c hg wg) h w`; each interp to the
same size is an identity):

| step | op | shape |
|---|---|---|
| ipt5 | SimpleConvs(patches_32(x): 3072ch@32²) → 384 | (384,32²) |
| d4 | BasicDecBlk(cat(x4,ipt5)=3456 → 1536) | p4 (1536,32²) |
| gdt4 | p4 ·= σ(C1_16→1(R(BN(C3_1536→16(p4))))) | |
| up | interp_ac(p4, 64²) + C1_1536→1536(x3) | (1536,64²) |
| ipt4 | SimpleConvs(patches_16: 768ch@64²) → 384; cat | (1920,64²) |
| d3 | BasicDecBlk(1920 → 768), gdt3 (768→16→1) | (768,64²) |
| up | interp_ac → 128² + C1_768→768(x2) | (768,128²) |
| ipt3 | SimpleConvs(patches_8: 192ch@128²) → 192; cat | (960,128²) |
| d2 | BasicDecBlk(960 → 384), gdt2 (384→16→1) | (384,128²) |
| up | interp_ac → 256² + C1_384→384(x1) | (384,256²) |
| ipt2 | SimpleConvs(patches_4: 48ch@256²) → 96; cat | (480,256²) |
| d1 | BasicDecBlk(480 → 192), no gdt | (192,256²) |
| up | interp_ac → **1024²** | (192,1024²) |
| ipt1 | SimpleConvs(x: 3ch@1024²) → 48; cat | (240,1024²) |
| out | C1_240→1 (+bias) | logits (1,1,1024,1024) |

Output: sigmoid(logits), which is the alpha. Only `outs[-1]` exists in eval.

## 3. Op mapping to ggml (22-op set + host tables)

Layout: activations are token-major `[C, W·H]` (ne0 = C) for LN, 1×1 conv and
Linear, and channel-planar `[W, H, C]` for IM2COL. The two are switched with
PERMUTE+CONT (CPY/CONT).

| torch | ggml |
|---|---|
| Conv2d k×k (3×3 p1; 4×4 s4 patch embed; 7×7 p3 offsets) | IM2COL (2D) → MUL_MAT(W[C·k², Cout]) → ADD(bias) |
| Conv 1×1 / Linear | MUL_MAT (+ADD bias) |
| BatchNorm2d (eval) | folded into the preceding conv at convert time (eps 1e-5), or MUL+ADD per channel |
| LayerNorm (eps 1e-5) | NORM → MUL(γ) → ADD(β) |
| nn.GELU (erf) | GELU_ERF |
| ReLU | **new `relu`**, or the MUL/SIGMOID/SCALE fallback |
| sigmoid; `2·sigmoid` | SIGMOID; SIGMOID+SCALE(2) |
| gdt gating p·σ(a) | MUL (broadcast [W,H,1] over C) |
| F.pad + torch.roll(−s) + window_partition | CONCAT(one zero row) → **GET_ROWS(fwd table, nW·144 int32)** |
| window_reverse + roll(+s) + crop | **GET_ROWS(inv table, H·W)** |
| qkv / heads split | MUL_MAT + ADD, then view+PERMUTE+CONT to [hd, N, nH, nW] |
| q·scale, q·kᵀ | SCALE, MUL_MAT |
| relative position bias `table[index]` | GET_ROWS(table[529,nH], host idx[144²]) → PERMUTE+CONT → [144,144,nH,1] (constant per block; precompute at load) |
| + bias, + shift mask, softmax | ADD(bias, broadcast over nW) → SOFT_MAX(ext, mask [144,144,1,nW] broadcast over heads, scale 1) |
| attn·v, proj | MUL_MAT (vᵀ via PERMUTE+CONT), MUL_MAT+ADD |
| PatchMerging x0..x3 cat | **GET_ROWS(table: t·4+k)** → view [4C, T] → NORM/MUL/ADD → MUL_MAT (no bias) |
| out norm_i + to NCHW | NORM/MUL/ADD → PERMUTE+CONT |
| `F.interpolate(bilinear, align_corners=True)` any size | separable, per axis: **GET_ROWS(i0) · MUL(λ0) + GET_ROWS(i1) · MUL(λ1)** with host tables `s=f32(in-1)/(out-1)`, `src=s·j`, `i0=int(src)`, `λ1=clamp(src−i0,0,1)`, `λ0=1−λ1`, `i1=i0+(i0<in−1)`. Do the W axis first for bit parity. The gathered axis must be ne1, so PERMUTE+CONT between axes. |
| interpolate 1×1 → H×W (ASPP GAP branch) | REPEAT (exact: scale 0 means every tap reads index 0) |
| AdaptiveAvgPool2d(1) | reshape [W·H, C] → MEAN |
| torch.cat(dim=1) | CONCAT |
| rearrange `b c (hg h)(wg w) → b (c hg wg) h w` | per channel (C=3): view [w, wg, h, hg] → PERMUTE → [w, h, wg, hg] → CONT; CONCAT the 3 channels. Or one GET_ROWS with rows of length w. |
| deform_conv2d (modulated) | offset/modulator convs as above, then **new `bilinear_sample_zeros`** → MUL(mask) → MUL_MAT(W[C·K, 256]) |
| Dropout / DropPath (eval) | identity |

On FLASH_ATTN_EXT: it can serve unshifted windows with the per-head bias as the
F16 mask [144, pad(144), nH, 1]. Shifted windows need bias+mask combined per
(head, window): [144²·nH·nW], which is 120 MB F16 for stage 0. The explicit
MUL_MAT/SOFT_MAX path is the oracle-exact choice. The mask is nonzero only in the
last window row and column (2·nWh−1 windows), so it can be restricted to those.

Verification on the real model (`birefnet_ref_out/ref_log.txt`):
- The Swin block through the gather tables vs `SwinTransformerBlock` gave **max error
  0.000** for L0B0 (256→264 pad, unshifted), L2B0 (72, unshifted), L2B1 (72, shift 6)
  and L3B1 (36, shift 6).
- PatchMerging via GET_ROWS: **0.000**.
- Bilinear ac gather (W then H) vs `F.interpolate`: **0.000** for 256→1024, 1024→512
  and 1→32. The other cases are ≤ 4.8e-7 (H-first ≤ 7.2e-7).
- deform im2col + MUL_MAT vs the module, on real offsets (|off| up to 6.18, mask
  0.02–1.97): 3.3e-6 (k7 @32²) and 1.7e-6 (k3 @256²).

## 4. Weights and parameters

- File: `model.safetensors`, 444.5 MB, 754 tensors. **F16**: 687 tensors,
  220,202,578 elements. **I64**: 67 tensors, 497,707 elements. The I64 tensors are 24
  `relative_position_index` [144,144] buffers and 43 `num_batches_tracked`.
- **Learned params: 220,176,498.** The backbone is 195,201,204 (88.7%): Swin-L
  layers 195.19 M including the relative-position bias tables, patch_embed 9.8 k and
  out norms 5.8 k. The rest is the squeeze module
  (6.67 M), decoder blocks 4/3/2/1 (4.45/3.12/2.34/1.95 M), lateral 1×1 (2.36/0.59/0.15 M),
  ipt_blk5..1 (1.99/0.66/0.22/0.08/0.03 M) and gdt (0.39 M).
  Buffers are 523,787, which includes 26,080 F16 BN running stats.
- Do not ship the I64 index buffers: the tables are generated at load. The
  relative-position bias tables are learned (529×nH per block). GGUF f16 is about
  441 MB; q8_0 is about 235 MB.

## 5. FLOPs (2·MAC; conv/mm/bmm via `torch.utils.flop_counter`)

| input | total | backbone full | backbone ½ | (attention bmm) | squeeze + decoder |
|---|---|---|---|---|---|
| **1024² (Pixal3D)** | **2,535 GFLOP** | 1,596 | 411 | 84 | 528 |
| 2048² (HR native) | 9,773 GFLOP | 6,066 | 1,596 | 304 | 2,110 |

At 1024: decoder_block1 is 253 GFLOP. The largest single piece is its k7 deform,
64→256 at 256²: 105 GFLOP for the GEMM plus 60 GFLOP for the offset/modulator
convs. Then come decoder_block2 (76), ipt_blk1 (62, two 3×3 convs at 1024²),
lateral 1×1 (19 each), decoder_block3 (25), squeeze (14) and d4 (9).
The non-GEMM work is 338 M deform samples, LN and softmax over 144² per head-window.

IM2COL peak buffers at 1024 need row-band tiling:
- ipt_blk1 `C3 64→48` @1024²: 576×1 M = 2.4 GB f32
- d1 `conv_in 480→64` @256²: 1.1 GB
- k7 offset / modulator / deform columns 3136×65536: 0.82 GB each

At 2048 all of these grow ×4.

CPU timing (torch fp32, 8 threads): 41.6 s per 1024² forward on an idle-ish
host, and 148 s on the second run while CPU was shared.

## 6. New kernel specs

**`bilinear_sample_zeros`** (Lean → Slang; inference only, no backward):
- inputs: `X` f32 [C, H, W] (NCHW plane-major, B=1); `P` f32 [2, N], the sample
  coordinates in *pixel units* (row y, col x).
- output: `Y` f32 [C, N].
- math, per (c, n), a transcription of torchvision `bilinear_interpolate`:
  ```
  y=P[0,n]; x=P[1,n]
  if (y <= -1 || y >= H || x <= -1 || x >= W) → 0
  yl=floor(y); xl=floor(x); yh=yl+1; xh=xl+1
  ly=y-yl; lx=x-xl; hy=1-ly; hx=1-lx
  v1 = (yl>=0 && xl>=0)     ? X[c,yl,xl] : 0
  v2 = (yl>=0 && xh<=W-1)   ? X[c,yl,xh] : 0
  v3 = (yh<=H-1 && xl>=0)   ? X[c,yh,xl] : 0
  v4 = (yh<=H-1 && xh<=W-1) ? X[c,yh,xh] : 0
  Y[c,n] = (hy*hx)*v1 + (hy*lx)*v2 + (ly*hx)*v3 + (ly*lx)*v4     (this order)
  ```
- use for DCNv2, with K=kh·kw, k=i·kw+j and N = K·Ho·Wo:
  - `P[0] = ADD(base_y, OFF[2k])` and `P[1] = ADD(base_x, OFF[2k+1])`, where
    `base_y[k,oy,ox] = oy·s − p + i·d` and `base_x = ox·s − p + j·d` are host tables.
    The integer part is added first, then the offset, matching torchvision.
    OFF is the offset-conv output [2K, Ho, Wo], with Δy on even channels.
  - `COLS = MUL(Y viewed [C, K, Ho·Wo], mask [1, K, Ho·Wo])` (mask = 2·σ(·)).
  - `out = MUL_MAT(W[Cout, C·K], COLS)`, where COLS row = c·K + k is torchvision's
    column layout, then BN-fold → relu.
- The same kernel is `grid_sample(mode=bilinear, padding_mode=zeros)` once the
  coordinates are mapped in-graph with SCALE/ADD. The zero-padding semantics are
  identical; the corner-weight rounding can differ by 1 ulp. That makes it reusable
  beyond BiRefNet.
- Sizes in BiRefNet at 1024: C=64; K ∈ {1,1,9,49}; Ho=Wo ∈ {32 (×2 blocks), 64, 128,
  256}. The largest single call is Y [64, 49·65536] = 205 M floats, so tile over
  output rows. A fused variant that accumulates `W[:,:,k] @ (m_k ⊙ sample_k)` per tap
  avoids the C·K column buffer. `deform_conv2d_torch` in the script does exactly this.
- Oracle: `deform_sq_k7_{in,offset,mask,out}.npy` and `deform_d1_k3_*.npy`. Proof
  against the documented scalar semantics (float64 loop): max error ≤ 6.2e-6 over
  k ∈ {1,3,7}, including exact-integer and exactly −1 offsets. Zero offset with unit
  mask vs F.conv2d: 3.8e-6.

**`relu`**: y = max(x, 0), elementwise f32, any shape.

## 7. Reference run (`birefnet_ref.py`)

- The unmodified upstream `birefnet.py` is exec'd with stubbed
  transformers/timm/torchvision/kornia, and `deform_conv2d` is the proved
  pure-torch version. No compiled extension was needed and nothing was
  installed (einops and pillow were already present). The strict `load_state_dict`
  loaded all 754 tensors.
- **Ran:** `04_BunnyCake.jpg` (moge examples, 547×800 RGB). Logits span
  [−15.2, 14.9]; 28.6% of pixels have alpha > 0.8 and 3.2% are soft
  (0.05–0.95). Pixal3D crop box is (−95.5, 44.5, 648.5, 788.5), final RGB 744×744
  (`pixal3d_final.png`, which looks correct).
- **Pixal3D scene example `s_14_img.jpg`** (1080×1350 → 819×1024) with HR-matting
  weights: logits span [−12.6, −2.73], so max alpha is 0.061 (15/255). No pixel
  passes >204, and **upstream `preprocess_image` would raise ValueError**. A
  matting model finds no subject in a kitchen scene. Pixal3D's actual RMBG-2.0 is a
  general-salient model and may behave differently; this is a model-choice risk.
  The logits are kept in `birefnet_ref_out_s14_partial/` and the run log in
  `birefnet_ref_run_s14.log`.
- Oracle arrays in `birefnet_ref_out/` (35 .npy; shapes in `manifest.json`):
  - input stages: `in_rgb_u8` (after Pixal3D downscale), `in_resized_u8` (PIL
    bilinear 1024²), `in_tensor` (normalized)
  - `bb_{full,half}_x{1..4}`
  - `squeeze_{in,out}`
  - `dec_block{4..1}_{in,out}`
  - the two deform taps
  - output stages: `logits`, `pred`, `mask_u8` (truncated), `mask_fullres_u8` (PIL
    bicubic), `rgba_u8`, `pixal3d_final_rgb_u8`
- Usage: `python birefnet_ref.py [image] [--res 1024|2048] [--out DIR]`. The FLOP
  split is `birefnet_flops_meta.py` (meta device, no compute).
