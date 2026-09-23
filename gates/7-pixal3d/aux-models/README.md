# Stage 7 auxiliary networks — census and CPU references

Pixal3D's pipeline calls three networks besides DINOv3 and the DiTs. User
decisions (2026-09-23): NAF is forked into the org (`V-Sekai-fire/NAF`),
camera fov comes from **MoGe-3** (upstream Pixal3D uses MoGe-2), background
removal is **BiRefNet_HR-matting**. Weights are pinned in
`tools/models/manifest.tsv`; each `*.md` here is the architecture, the op
mapping onto the 22-op ggml-rd set, and the FLOP count; each `*_ref.py` is a
CPU-torch reference that saves inputs/outputs as oracles.

| network | new Lean kernels | notes |
|---|---|---|
| NAF (`naf.md`, 662k params fp32) | **none** | `natten na2d` on nearest-upsampled K/V is exactly a block attention (GET_ROWS with a host window table + MUL_MAT + SOFT_MAX + MUL_MAT; ≤4.6e-7 vs real NATTEN 0.21.7). ROPE needs freq factors (src2), base 1, negative positions — or the cos/sin-table fallback. The tex_1024 output (4 GiB f32) and the 3×3 im2col at 1024 (4.8 GB) must be chunked; the HR ProjGrid R starts at 96 in 1536_cascade (not ≤64). |
| MoGe-3 fov (`moge3.md`) | **none** (ReLU recommended) | Only `intrinsics[0,0]` is used: camera_angle_x = 2·atan(1/(2·fx)) = 35.567° on example 21_img.png. Subgraph: ViT (dinov2 L/14, 60×60 tokens at 840²), neck, points + mask heads sampled at 64×64 — equal to dense `infer()`. Sparse refinement (FlexGEMM/Triton) unused. The x·sigmoid(2^254·x) ReLU gives −0 for negatives and NaN at −inf, so a real `relu` kernel goes into K1. |
| BiRefNet_HR-matting (`birefnet.md`, fp16) | **`bilinear_sample_zeros`** (deformable-conv sampler, torchvision OOB rules) + **`relu`** | Only the 2.4 GB im2col at 1024 exceeds a 2 GiB buffer. Oracle checked against a float64 transcription of torchvision's kernel: 2.8e-6 / 1.1e-5. On scene images the matting model gives poor masks for 2–3 of 6 examples (portrait/object model); Pixal3D raises on 1. |

`naf_release_notes.md`: the notes for mirroring `naf_release.pth`
(2,664,431 B, sha256 c096c1ab…c98f, Apache-2.0) as a release on
`V-Sekai-fire/NAF` (tag `model`, which upstream points at f63aadf). The
release was not created: the auto-mode classifier refused `gh release create`
as a new public surface; it needs the user's go-ahead.
