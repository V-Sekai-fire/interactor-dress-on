# G3.census — which ggml ops the two apps actually run

Host-native census, before any guest work: a scratch patch on a copy of the org
ggml (`patch_census.py`, `patch_census_time.py`; never committed to ggml) logs
every node that reaches the CPU `ggml_graph_compute`: op, unary op, types,
contiguity (C contiguous, R rows contiguous, N neither), `ne`, `nb`,
`op_params` and per-op wall time. `aggregate.py` folds the raw logs into the
CSVs here.

**Result: 22 compute ops, none outside the K1–K8 kernel families** of the plan
(`census_union.csv`, `census_summary.txt`).

| app | nodes | graphs run |
|---|---|---|
| skin-tokens (V-Sekai-fire/skin-tokens-ggml 097a0cc, against org ggml 0.19 — 0 API-drift errors) | 1,883,592 | full rig + bind on the official giraffe |
| Pixal3D (V-Sekai-fire/pixal3d-ggml 1c22f5e) | 15,813 | DINOv3 @512, one ss_flow and one slat512 forward on remapped Aero-Ex GGUFs, ss_dec, shape_dec |

Type rows the plan's first K specs did not list, required by the census:
MUL_MAT bf16×f32 (Pixal3D DiT weights), MUL_MAT f16×f16 (DINO patch-embed
im2col), CONT f16→f16 with row-strided sources, MUL_MAT with non-contiguous
src (RC/RR).

Not censused (a static grep finds no op outside K1–K8, but type/shape classes
are unknown): tex_dec, shape_dec cascade upsample, shape_enc (the only MEAN),
and the `TRELLIS2_SDPA_EXACT` path.

**G7.schema evidence** (`g7schema_*`): the chibifire/Pixal3D-GGUF DiTs
(Aero-Ex ComfyUI conversions) fail pixal3d-ggml's loader (`unexpected
architecture 'flux'`; note `ss_flow_info` still exits 0 — gate on the text).
A scratch remap (architecture/KVs, `comfy.gguf.orig_shape` reshapes, strip
`.cross_attn_block.`, F16 norms→F32) makes the TRELLIS.2 graphs load and run;
they ignore `proj_linear`. pixal3d-ggml has no `proj` support yet.

`proj_attention.md`: Pixal3D's proj attention is TRELLIS.2 cross-attention
(over CLS + 4 registers, Lk=5) plus a per-block `proj_linear` of
back-projected DINO features; the camera math and the bilinear
`grid_sample` (align_corners=False, border) reduce to a host-built tap table
(`check_proj_gather.py`; `control_proj_gather.py` is its negative control:
five perturbations all fail, 0.72–1.58 error; float64 run ≤1e-13).

Verifier corrections: skin-tokens decode is 4.2 s per logits call at 8 threads
(`run_st_rig.log`); the earlier "60 s/token at 16 threads" had no log. The
KV cache grows by CONCAT (36% of op time in the rig run) — make K2's concat
fast. Upstream Pixal3D also uses BiRefNet (background), MoGe (fov) and NAF
(feature upsampling); see `gates/7-pixal3d/aux-models/`.
