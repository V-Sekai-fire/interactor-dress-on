#!/usr/bin/env python3
"""Convert MoGe-3 (Ruicheng/moge-3-vitl, models/moge-3-vitl/model.pt) to a GGUF
for vendor/moge3 (the Pixal3D camera-FoV subgraph on ggml).

Only the FoV path is written (gates/7-pixal3d/aux-models/moge3.md section 1):
encoder (DINOv2 ViT-L/14 + the four output projections), neck, points_head,
mask_head. normal_head, scale_head, mask_token and the sparse refiner are
dropped (refine_steps = 0).

Conversion-time transforms (everything else keeps its torch name minus the
leading "model." and its torch memory order, so ggml ne is the reversed torch
shape):
  * encoder.backbone.pos_embed [1, 1370, 1024] is replaced by
    encoder.backbone.pos_embed_60x60 [3601, 1024]: cls position + the 37x37
    table bicubic-interpolated to 60x60 exactly as DINOv2's
    interpolate_pos_encoding does it (scale_factor (60 + 0.1)/37, A = -0.75,
    antialias False). Pixal3D's MoGe input is always square, so the grid is
    always 60x60 at 3600 tokens.
  * ConvTranspose2d k2 s2 weights [Cin, Cout, 2, 2] become
    <name>.weight_abco [2*2*Cout, Cin] with rows ordered (a, b, co): a 1x1
    conv to 4*Cout channels followed by a pixel shuffle.
  * encoder.image_mean / image_std travel as KV (moge3.image_mean/std).

--dtype f16 stores every weight with >= 65536 elements and ndim >= 2 as f16
(the ViT matrices, output projections, conv kernels, patch embed); biases,
norms, LayerScale, small convs and the baked pos-embed stay f32.

Run with C:/interactor-dress-on/.venv-convert (torch CPU + gguf 0.19):
  python tools/models/convert_moge3_to_gguf.py \
      --model models/moge-3-vitl/model.pt --output C:/b/moge3/moge3_f32.gguf --dtype f32
"""
import argparse
import hashlib
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import gguf

ARCH = "moge3"
GRID = 60                 # base_h = base_w = round(sqrt(3600)) for aspect 1
INTERP_OFFSET = 0.1       # dinov2_vitl14 hub default
KEEP_PREFIXES = ("encoder.", "neck.", "points_head.", "mask_head.")


def bake_pos_embed(pos_embed: torch.Tensor, grid: int) -> torch.Tensor:
    """DinoVisionTransformer.interpolate_pos_encoding for an (grid*14)^2 input."""
    pe = pos_embed.float()
    cls_pos, patch_pos = pe[:, 0, :], pe[:, 1:, :]
    n = patch_pos.shape[1]
    m = int(math.sqrt(n))
    assert m * m == n
    dim = pe.shape[-1]
    sx = float(grid + INTERP_OFFSET) / m
    sy = float(grid + INTERP_OFFSET) / m
    patch = F.interpolate(patch_pos.reshape(1, m, m, dim).permute(0, 3, 1, 2),
                          mode="bicubic", antialias=False, scale_factor=(sy, sx))
    assert tuple(patch.shape[-2:]) == (grid, grid)
    patch = patch.permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat([cls_pos[:, None, :], patch], dim=1)[0]  # [1 + grid^2, dim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moge-3-vitl/model.pt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f16")
    ap.add_argument("--oracle", default=None,
                    help="moge3_ref_out dir: check the baked pos-embed against pos_embed_interp.npy")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=True)
    cfg = ck["model_config"]
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in ck["model"].items()}

    enc = cfg["encoder"]
    assert enc["backbone"] == "dinov2_vitl14", enc
    assert list(enc["intermediate_layers"]) == [5, 11, 17, 23], enc
    for head in ("neck", "points_head", "mask_head"):
        c = cfg[head]
        assert c["res_block_in_norm"] == "none" and c["res_block_hidden_norm"] == "none", c
        assert list(c["resamplers"]) == ["conv_transpose"] * 3 + ["bilinear"], c
        assert c.get("activation", "relu") == "relu", c

    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_name("MoGe-3 ViT-L (FoV path: encoder, neck, points_head, mask_head)")
    w.add_string("moge3.source", "Ruicheng/moge-3-vitl@184008f model.pt; code V-Sekai-fire/MoGe@74fbce0 (v3)")
    w.add_uint32("moge3.grid", GRID)
    w.add_uint32("moge3.patch_size", 14)
    w.add_uint32("moge3.embed_dim", 1024)
    w.add_uint32("moge3.n_layers", 24)
    w.add_uint32("moge3.n_heads", 16)
    w.add_float32("moge3.ln_eps", 1e-6)
    w.add_array("moge3.intermediate_layers", [5, 11, 17, 23])
    for head in ("neck", "points_head", "mask_head"):
        c = cfg[head]
        w.add_array(f"moge3.{head}.dim_res_blocks", [int(x) for x in c["dim_res_blocks"]])
        w.add_array(f"moge3.{head}.num_res_blocks", [int(x) for x in c["num_res_blocks"]])
    w.add_array("moge3.image_mean", [float(x) for x in sd["encoder.image_mean"].flatten()])
    w.add_array("moge3.image_std", [float(x) for x in sd["encoder.image_std"].flatten()])
    w.add_string("moge3.dtype", args.dtype)

    n_written, n_params, n_f16 = 0, 0, 0

    def put(name, t: torch.Tensor):
        nonlocal n_written, n_params, n_f16
        a = t.detach().float().contiguous().numpy()
        if args.dtype == "f16" and a.ndim >= 2 and a.size >= 65536 and "pos_embed" not in name:
            a = a.astype(np.float16)
            n_f16 += a.size
        w.add_tensor(name, a)
        n_written += 1
        n_params += a.size

    for name in sorted(sd):
        t = sd[name]
        if not name.startswith(KEEP_PREFIXES):
            continue
        if name in ("encoder.image_mean", "encoder.image_std", "encoder.backbone.mask_token"):
            continue
        if name == "encoder.backbone.pos_embed":
            pe = bake_pos_embed(t, GRID)
            if args.oracle:
                ref = np.load(os.path.join(args.oracle, "pos_embed_interp.npy"))
                d = float(np.abs(pe.numpy() - ref).max())
                print(f"pos_embed_60x60 vs oracle pos_embed_interp: maxabs {d:.3g}")
                assert d == 0.0, d
            put("encoder.backbone.pos_embed_60x60", pe)
            continue
        if name.endswith(".weight") and ".resamplers." in name and name.split(".")[-2] == "0" \
                and t.ndim == 4 and t.shape[2:] == (2, 2):
            cin, cout = t.shape[:2]
            put(name[:-len(".weight")] + ".weight_abco", t.permute(2, 3, 1, 0).reshape(4 * cout, cin))
            continue
        if name.startswith("encoder.backbone.cls_token"):
            put(name, t.reshape(-1))
            continue
        put(name, t)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    h = hashlib.sha256()
    with open(args.output, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    print(f"wrote {args.output}: {n_written} tensors, {n_params:,} params ({n_f16:,} f16), "
          f"{os.path.getsize(args.output):,} B, sha256 {h.hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())
