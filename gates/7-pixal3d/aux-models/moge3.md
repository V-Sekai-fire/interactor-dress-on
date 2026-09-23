# MoGe-3 (moge-3-vitl): architecture and op census for the Pixal3D camera FoV

Sources:
- Code: `V-Sekai-fire/MoGe` @74fbce0 "V3": `moge/model/v3.py`, `v2.py`, `modules/{dinov2_encoder,conv_stack,mlp}.py`, `modules/dinov2/*`, `utils/geometry_{torch,numpy}.py`.
- Consumer: `V-Sekai-fire/Pixal3D` @cdbb2bb: `inference.py` and `app.py` (`get_camera_params_wild_moge`), `pixal3d/pipelines/pixal3d_image_to_3d.py` (`preprocess_image`).
- Weights: `Ruicheng/moge-3-vitl` @184008f, `models/moge-3-vitl/model.pt` (sha256 9b41b7b9…7925, 1,481,333,394 B).
- Reference: `moge3_ref.py`, output in `moge3_ref_out/` (with `report.json`) and the log in `moge3_ref_run.log`.

## 0. Result

**The FoV path needs no new Lean kernel.** Every op on the path maps to the 22-op set, or to host work on at most 4096 values. The table below lists the ops outside the 22 and how each is decomposed. Each decomposition was checked in `moge3_ref.py`.

| op outside the 22 | where it is used | decomposition | check vs torch |
|---|---|---|---|
| **ReLU** | every ResidualConvBlock (2 per block), scale-head MLP | `MUL(x, SIGMOID(SCALE(SCALE(x, 2^127), 2^127)))`. Every nonzero finite x maps to \|t\| ≥ 2^105 or ±inf, so the sigmoid returns exactly 0 or 1; x = 0 gives 0·0.5 = 0 and NaN propagates. | **bit-exact** on 1M values from 2^-150 to 2^60, plus ±0, denormals and ±FLT_MAX |
| replicate padding (all 3×3 convs use `padding_mode='replicate'`) | 3×3 convs | 4 × `CONCAT` of 1-wide edge views (W then H), then conv with p = 0 | bit-exact; conv result maxabs 0.0 |
| Conv2d 3×3 | neck, heads | `CONV_3D` (direct, KD = 1, p = 0) on the replicate-padded input, or `IM2COL`+`MUL_MAT`; bias `ADD` | 0.0 (pad identity) |
| Conv2d 1×1 | input and output blocks, the 4 encoder projections | `MUL_MAT` (+`ADD` bias) | n/a |
| ConvTranspose2d k2 s2 | resamplers 0–2 | `MUL_MAT` W[Cout·4, Cin] × x, then pixel shuffle as `CONT(PERMUTE)`, then `ADD` bias | 2.6e-5 abs at \|y\|max 13.7 (1.9e-6 rel; K = 1024 summation order) |
| bilinear ×2 upsample, align_corners = False | resampler 3 (480→960) | per axis: `left/right` = `CONCAT` of clamped shifted views; even = `ADD(SCALE(left, .25), SCALE(x, .75))`, odd = `ADD(SCALE(x, .75), SCALE(right, .25))`; interleave = `CONCAT` on a unit inner dim, then reshape | **bit-exact (0.0)** |
| bilinear resize to S×S, align_corners = False, no AA | head outputs → image size | per axis: `GET_ROWS`(i0), `GET_ROWS`(i1), `MUL` by w0/w1, `ADD`; tables built on the host in float32 | **bit-exact (0.0)**; not needed for the FoV (§1) |
| bilinear resize **with antialias** (S → 840) | encoder input | separable banded matrices built on the host (1–2 taps when S < 840, 3–4 when S > 840): `MUL_MAT` Ry·X·Rxᵀ, or K × (`GET_ROWS`+`MUL`+`ADD`). Tables **must be float32**: float64 tables are 2e-5 off because torch's own source coordinate carries the float32 rounding of S/840. | 1.8e-7 |
| bicubic interp of pos_embed (37² → 60², `scale_factor = 60.1/37`, A = −0.75) | ViT token prep | weights-only: **bake the 3601×1024 table at conversion** (Pixal3D input is always square, so the grid is always 60×60). Or `MUL_MAT` By·P·Bxᵀ with host tables. | 1.4e-7 |
| image normalize (x−m)/s | encoder | `ADD`(−m) then `MUL`(1/s), broadcast [3] | ≤1 ulp (reciprocal) |
| LayerNorm eps 1e-6 | 24×2 plus 4 taps | `NORM` + `MUL` γ + `ADD` β | – |
| exact GELU | ViT MLP | `GELU_ERF` (K1 uses A&S erf, about 1.5e-7) | – |
| SDPA 16 heads × 64, N = 3601, no mask | ViT | `FLASH_ATTN_EXT` (scale 0.125, mask NULL, D = 64) | – |
| EXP | `remap_output='exp'` (z = e^logz), metric scale | **host**: only the 64×64 FoV samples need it (§1) | – |
| sigmoid > 0.5 on the mask | mask head | host, on the same 4096 samples (`SIGMOID` if done on device) | – |
| nearest 64×64 subsample, `recover_focal_shift` (scipy LM) | `infer()` | host: index tables `floor(i·S/64)`, then a 1-D Levenberg–Marquardt solve over ≤4096 points (in `moge3_ref.py`, scipy-free) | sparse vs dense samples: **0.0**, masks equal |

The following never occur on this path: natten or neighbourhood attention, deformable conv, grid_sample, window attention or relative-position bias, GroupNorm (`res_block_*_norm: none`) and registers (plain `dinov2_vitl14`). The **only** part that would need new kernels is the sparse refiner `Sparse3DUNet` (FlexGEMM submanifold sparse conv on Triton, 39.4M params), and it is not used (§1).

Optional, not required: a fused Lean `relu` in K1 would replace 4 dispatches per activation with 1. The ReLU tensors are large (64×480² and 32×960²), so this is a bandwidth saving and nothing more.

## 1. How Pixal3D uses it, and why refine_steps = 0 suffices

- **Call site.** `inference.py` → `get_camera_params_wild_moge(tmp_png, moge_model)`:
  - it re-reads the preprocessed PNG as RGB/255 and calls `moge_model.infer(image_tensor)` with all defaults;
  - upstream loads `moge.model.v2`, `Ruicheng/moge-2-vitl`, `use_fp16=True`; plan 4b swaps in v3 and moge-3-vitl.
- **Input.** `preprocess_image` does the following:
  - resize with LANCZOS only if max(W, H) > 1024;
  - take the alpha bbox (α > 0.8·255), side = `int(1.1·max extent)`, crop a square with PIL rounding, pad transparent;
  - composite on black and quantise to uint8.
  - So the MoGe input is **square, S×S with S ≤ ~1126** (808×808 for `assets/images/21_img.png`). Aspect is 1, so the token grid is always **60×60 at num_tokens = 3600** (`resolution_level=9` → 1200 + 9/9·2400), and the ViT input is always **840×840**.
- **Output used.** Only `intrinsics[0,0]` (fx, normalised):
  - `camera_angle_x = 2·atan(W/(2·fx·W)) = 2·atan(1/(2·fx))`;
  - with fx = focal/2·√(1+a²)/a and a = 1, `camera_angle_x = 2·atan(1/(√2·focal))`, where focal is relative to the half-diagonal.
  - Pixal3D's `distance_from_fov` with its defaults (mesh_scale 1, extend_pixel 0, 512 px) reduces to **`distance = 0.5/tan(camera_angle_x/2)` = fx**.
  - Depth, points, normal, metric scale and the solved shift are all discarded.
- **FoV subgraph.** focal comes from `recover_focal_shift(points, mask>0.5)`:
  - it nearest-samples a **64×64** grid of the full-resolution affine points, uv and mask;
  - it then solves `min_s Σ‖f(s)·xy/(z+s) − uv‖²` with `f(s) = ⟨p,uv⟩/⟨p,p⟩`, `p = xy/(z+s)`.
  - Each full-resolution sample is a bilinear 2×2 tap of the 960×960 head output. So the FoV needs:
    - ViT + neck (dense) + **points_head** + **mask_head**;
    - 64·64 bilinear samples of [3 + 1, 960, 960], then exp/sigmoid on the host.
  - It does **not** need normal_head, scale_head, the full-resolution resize, force_projection or the depth.
  - Checked: sparse samples = dense `infer()` samples, maxabs **0.0**, masks identical. The tap tables are saved as `fov_sample_{rows,cols,wrows,wcols}.npy`.
- **refine_steps = 0.** In `v3.forward` the refiner is reached only inside `if refine_steps > 0:`. With 0, `coords = raw_coord` (points_head output), which is exactly the v2 dense graph with `remap_output='exp'`.
  - The only FlexGEMM contact is v3.py's **module-scope import** of `sparse_unet` → `flex_gemm`. The reference stubs it with a class that raises if it is ever touched, and it never was.
  - Call `infer(img, refine_steps=0)`: v3's default is 3.
  - Caveat: refinement changes only log z, but focal is re-solved on the refined map. The steps = 3 FoV is therefore slightly different, and it cannot be measured here because it needs Triton/CUDA.
- **Token count matters.** Measured FoV for this image (fp32 CPU):

  | tokens | FoV |
  |---|---|
  | 1200 | 29.04° |
  | 2400 | 32.09° |
  | 3600 | **35.57°** |

  The port must use 3600 to match Pixal3D's call.

## 2. Forward graph at Pixal3D's resolution (S×S input, 60×60 grid; B = 1)

Shapes are PyTorch [C, H, W]. The ggml layout is [W, H, C] for the conv stack (conv ops are W-innermost) and [1024, N] for tokens.

**Encoder** (DINOv2 ViT-L/14; `dinov2_vitl14`, init_values = 1 → LayerScale, qkv/proj/ffn bias, no registers, no mask token used)

| # | op | out shape |
|---|---|---|
| E0 | bilinear resize, align_corners = False, **antialias = True** | [3, 840, 840] |
| E1 | (x − mean)/std, ImageNet | [3, 840, 840] |
| E2 | patch_embed Conv 14×14 s14, 3→1024; flatten | [3600, 1024] |
| E3 | prepend cls; + pos_embed (37² bicubic → 60², offset 0.1) | [3601, 1024] |
| E4 | 24 × Block: `x += γ1⊙proj(SDPA(qkv(LN1 x)))`; `x += γ2⊙fc2(GELU(fc1(LN2 x)))`. qkv 1024→3072, 16 heads × 64, proj 1024→1024, fc1 1024→4096, fc2 4096→1024 | [3601, 1024] |
| E5 | taps after blocks 5, 11, 17, 23 → shared final LN → drop cls → [1024, 60, 60] → four separate 1×1 1024→1024 → **sum** | F0 [1024, 60, 60] |
| E6 | cls of the block-23 tap (post-LN) → scale head | [1024] |

**UV planes.** `normalized_view_plane_uv` at 60·2^l for l = 0..4 gives [2, 60·2^l, 60·2^l]. These are host constants: u spans ±(1/√2)(n−1)/n and v likewise, since a = 1.

**Neck** (ConvStack: dims [1024, 256, 128, 64, 32], res blocks [0, 2, 2, 2, 0], no norms, ReLU. ResBlock = `x + conv3(relu(conv3(relu(x))))`, both convs C→C with replicate padding.)

| level | ops | out |
|---|---|---|
| L0 | 1×1 1026→1024 on concat(F0, uv0) | N0 [1024, 60, 60] |
| R0 | ConvT 1024→256 k2s2 → conv3 256→256 | [256, 120, 120] |
| L1 | + 1×1 2→256 (uv1); 2 ResBlocks | N1 [256, 120, 120] |
| R1 | ConvT 256→128 → conv3 128→128 | [128, 240, 240] |
| L2 | + 1×1 2→128 (uv2); 2 ResBlocks | N2 [128, 240, 240] |
| R2 | ConvT 128→64 → conv3 64→64 | [64, 480, 480] |
| L3 | + 1×1 2→64 (uv3); 2 ResBlocks | N3 [64, 480, 480] |
| R3 | **bilinear ×2** → conv3 64→32 | [32, 960, 960] |
| L4 | + 1×1 2→32 (uv4) | N4 [32, 960, 960] |

**Heads.** points (out 3), mask (out 1) and normal (out 3) are identical stacks with res blocks [0, 1, 1, 1, 0]. At each level: `x (+)= 1×1 C→C (N_l)`, then ResBlocks, then the same resamplers R0–R3. At the end, a 1×1 32→{3|1|3} at level 4 produces:

| raw output | shape | contents |
|---|---|---|
| points | [3, 960, 960] | u = x/z, v = y/z, log z |
| mask | [1, 960, 960] | mask logits |
| normal | [3, 960, 960] | normals |

**Scale head.** MLP 1024→1024→ReLU→1024→ReLU→1, then exp (unused by Pixal3D).

**Post (`infer`).**
- Bilinear resize to S×S, then `xyz = (u·e^z, v·e^z, e^z)`.
- The mask goes through sigmoid, then > 0.5.
- The 64×64 nearest subsample feeds the LM → focal and shift → K.

## 3. Weights

All **fp32** in the checkpoint: 607 tensors, 370,284,399 params.

| part | params |
|---|---|
| encoder.backbone (ViT-L/14; pos_embed [1, 1370, 1024], cls, mask_token) | 304,368,640 |
| encoder.output_projections (4 × 1×1) | 4,198,400 |
| neck | 6,157,376 |
| points_head / mask_head / normal_head | 4,692,323 / 4,692,257 / 4,692,323 |
| scale_head | 2,100,225 |
| refiner (Sparse3DUNet, **unused**) | 39,382,849 |
| **dense, no refiner** | **330,901,544** (+6 buffer values: image mean/std) |
| **FoV path** (no normal or scale head) | **324,108,996** |

GGUF suggestion:
- f16 for the 329.1M matrix and conv weights: about 0.65 GB on the FoV path.
- f32 for LN, bias, LayerScale and the baked 3601×1024 pos-embed.
- Upstream Pixal3D ran MoGe under fp16 autocast on CUDA, so f16 weights stay inside upstream's own precision regime.
- Gate vs the fp32 oracle on the FoV angle, not bitwise.

## 4. FLOPs (2 per MAC; 60×60 grid; independent of S apart from E0 and the final resize)

| block | GFLOP |
|---|---|
| ViT linear (qkv + proj + MLP), 24 layers, N = 3601 | 2174.9 |
| ViT attention QKᵀ + PV (4·N²·d·24) | 1274.7 |
| patch embed + 4 output projections | 4.3 + 30.2 |
| neck | 311.7 |
| points_head | 217.2 |
| mask_head | 217.1 |
| normal_head (not needed) | 217.2 |
| **total, all heads** | **4447** |
| **FoV path** (ViT + neck + points + mask) | **4230** |
| FoV path, level-4 stage of points and mask evaluated only at the 16,384 tapped pixels (`GET_ROWS` im2col of 3×3 patches → `MUL_MAT`) | 4163 (saves 67.0 of 68.2) |

- Cross-check against `torch.utils.flop_counter`: conv 997.78 G and addmm 2174.94 G match the analytic numbers to the FLOP. The counter does not count SDPA on CPU, which is why attention was added analytically.
- The ViT is 82% of the FoV path.
- CPU torch fp32 on this machine: forward 25.8 s at 3600 tokens (12.9 s at 1200, 17.0 s at 2400).
- Memory note: IM2COL of the level-4 conv (64·9 × 960²) is 2.1 GB in f32. Use `CONV_3D` direct, or tile by rows.

## 5. Reference run (done, CPU, fp32)

`moge3_ref.py` ran successfully in `.venv-convert` (torch 2.14 CPU).
- No new deps: it has a pure-Python PNG decoder, stubs for flex_gemm, utils3d, cv2, scipy and huggingface_hub, and a scipy-free LM.
- Input: Pixal3D `assets/images/21_img.png` (810×798 RGBA), preprocessed exactly as `preprocess_image` does → crop box (2, 0, 810, 808) → 808×808.

Results:
- focal (half-diagonal) **2.204564**, shift 0.58397, 1494/4096 valid samples.
- fx = fy = **1.558862**.
- **camera_angle_x = 0.620760 rad = 35.567°**, Pixal3D distance = **1.558862**.
- metric_scale 1.7657.

Saved `.npy` files (f32 unless noted; 389 MB):

| group | files |
|---|---|
| inputs | `input_rgb_u8` (u8), `input_image_f32_chw` |
| encoder | `image14_norm` [3, 840, 840], `patch_embed_tokens` [3600, 1024], `pos_embed_interp` [3601, 1024], `vit_layer{5,11,17,23}_normed` [3601, 1024], `encoder_features` [1024, 60, 60], `cls_token` |
| neck | `neck_level0..4` |
| heads | `{points,mask,normal}_head_raw` [·, 960, 960], `scale_head_raw` |
| forward outputs | `forward_{points,mask,normal,metric_scale}` at 808² |
| FoV samples | `fov_{uv,points,mask}_lr` [64, 64, ·], `fov_sample_*` tap tables |
| camera and tables | `intrinsics`, `tables_aa_resize_{rows,cols}` [840, 808] |

Numbers are in `moge3_ref_out/report.json`.

Oracle caveat: upstream solves with `scipy.optimize.least_squares(method='lm', ftol=1e-3)`, which stops early. Our solver runs to convergence from the same x0 = 0, so upstream's FoV can differ in the last digits. The oracle defines the converged value.

Not measured: MoGe-3 (refine 0) FoV vs upstream Pixal3D's MoGe-2 FoV on the same image. That needs `Ruicheng/moge-2-vitl`, which is not downloaded.
