"""NAF CPU-torch reference + oracle dump, at the resolutions Pixal3D uses.

Sources (read, not modified):
  NAF      V-Sekai-fire/NAF @37f2dfc180f2de53d98bd601109c0da0dd6b0f43   (./naf-src)
  Pixal3D  V-Sekai-fire/Pixal3D @cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af (./pixal3d-src)
  weights  models/NAF/naf_release.pth  sha256 c096c1ab...847c98f
           models/dinov3-vitl16/model.safetensors (camenduru/dinov3-vitl16-pretrain-lvd1689m)

The repo's NAF module runs unmodified except for one function: its CrossAttention
calls natten.na2d(backend="cutlass-fna"), which needs CUDA. We substitute
`na2d_block`, an exact pure-torch neighbourhood attention specialised to NAF's
case (K/V are nearest-exact upsampled by an integer factor). Equivalence is
proven here against (a) the real NATTEN 0.21.7 flex-fna backend on small cases
and (b) a literal port of NATTEN's dilated window rule at full resolution.

A second, independent implementation `naf_ggml` executes the exact ggml op chain
written up in naf.md section 2, one g_* helper per ggml op, tensors held in ggml
ne order (reflect-pad + im2col as one GET_ROWS host table, MUL_MAT, GroupNorm as
RESHAPE+NORM, pooling as RESHAPE+MEAN, RoPE as two NEOX ROPE passes with
freq_factors, block-major pixel order by a GET_ROWS permutation, attention as
GET_ROWS window gather + batched MUL_MAT + SOFT_MAX_EXT + MUL_MAT, grid_sample as
a 4-tap GET_ROWS with host tables). It is compared against the repo module at
every Pixal3D config. (`naf_plan` is the earlier torch-level version, kept for T2.)

Usage: python naf_ref.py [--configs shape_512,shape_1024,tex_1024] [--threads 8]
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NAF_SRC = os.path.join(HERE, "naf-src")
NATTEN_PKG = os.path.join(HERE, "natten_pkg")
MODELS = "C:/interactor-dress-on/models"
NAF_W = os.path.join(MODELS, "NAF", "naf_release.pth")
DINO_W = os.path.join(MODELS, "dinov3-vitl16", "model.safetensors")
EXAMPLE = os.path.join(HERE, "pixal3d-src", "assets", "images", "0_img.png")
OUT = os.path.join(HERE, "naf_oracle")

# Pixal3D inference.py IMAGE_COND_CONFIGS (NAF-using stages) + ProjGrid grid res
CONFIGS = {
    "shape_512": dict(image_size=512, grid_resolution=32, naf_target_size=512),
    "shape_1024": dict(image_size=1024, grid_resolution=64, naf_target_size=512),
    "tex_1024": dict(image_size=1024, grid_resolution=64, naf_target_size=1024),
}
CAM = dict(camera_angle_x=0.8575560450553894, distance=2.0, mesh_scale=1.0)  # pipeline defaults

LOG = []


LOGFILE = [None]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)
    if LOGFILE[0]:
        with open(LOGFILE[0], "a", encoding="utf-8") as f:
            f.write(s + "\n")


def maxrel(a, b):
    a = a.double(); b = b.double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item(), (a - b).abs().max().item()


# ----------------------------------------------------------------------------
# NATTEN semantics, pure torch
# ----------------------------------------------------------------------------
def natten_window_1d(L, k, d):
    """For every index i in [0, L): the k neighbour indices NATTEN uses (non-causal,
    stride 1, odd k). Documented rule (natten/backends/flex.py get_na_flex_mask):
    split the axis into d dilation groups g = i % d of length Lg = ceil((L-g)/d);
    inside the group the query sits at m = i // d; window centre
    c = clamp(m, k//2, Lg-1-k//2); window = group positions c-k//2 .. c+k//2,
    i.e. hi-res indices g + d*(c-k//2+t)."""
    i = torch.arange(L)
    g = i % d
    m = i // d
    Lg = (L - g + d - 1) // d
    assert int(Lg.min()) >= k, "NATTEN requires every dilation group to hold a full window"
    c = torch.maximum(torch.minimum(m, Lg - 1 - k // 2), torch.full_like(m, k // 2))
    return g[:, None] + d * (c[:, None] - k // 2 + torch.arange(k)[None, :])  # [L, k]


def na2d_literal(q, k, v, kernel_size, dilation, scale=None, rows=None, row_chunk=2):
    """Literal dilated 2-D neighbourhood attention on [B,H,W,N,D] tensors, any H/W/d.
    rows: optional list of query rows to evaluate (returns [B,len(rows),W,N,Dv])."""
    B, H, W, N, D = q.shape
    Dv = v.shape[-1]
    scale = D ** -0.5 if scale is None else scale
    ih = natten_window_1d(H, kernel_size[0], dilation[0])
    iw = natten_window_1d(W, kernel_size[1], dilation[1])
    rows = list(range(H)) if rows is None else list(rows)
    kf = k.reshape(B, H * W, N, D)
    vf = v.reshape(B, H * W, N, Dv)
    outs = []
    for s in range(0, len(rows), row_chunk):
        rr = torch.tensor(rows[s:s + row_chunk])
        nb = (ih[rr][:, None, :, None] * W + iw[None, :, None, :]).reshape(len(rr), W, -1)  # [R,W,kk]
        kg = kf[:, nb]  # [B,R,W,kk,N,D]
        vg = vf[:, nb]
        qs = q[:, rr]  # [B,R,W,N,D]
        sc = torch.einsum("brwnd,brwknd->brwnk", qs, kg) * scale
        a = sc.softmax(-1)
        outs.append(torch.einsum("brwnk,brwknd->brwnd", a, vg))
    return torch.cat(outs, 1)


def na2d_block(q, k_lr, v_lr, kernel_size, scale=None, block_chunk=256):
    """NAF's case, exact: K/V are the low-res maps nearest-exact upsampled by the
    integer factor d = H/hk. Every hi-res query in low-res block (bi,bj) attends to
    the same k x k low-res window starting at clamp(bi-k//2, 0, hk-k). Returns
    [B,H,W,N,Dv]. This is the formulation the ggml graph implements."""
    B, H, W, N, D = q.shape
    _, hk, wk, _, Dv = v_lr.shape
    dh, dw = H // hk, W // wk
    assert dh * hk == H and dw * wk == W
    kh, kw = kernel_size
    scale = D ** -0.5 if scale is None else scale
    sh = (torch.arange(hk) - kh // 2).clamp(0, hk - kh)
    sw = (torch.arange(wk) - kw // 2).clamp(0, wk - kw)
    wh = sh[:, None] + torch.arange(kh)  # [hk,kh]
    ww = sw[:, None] + torch.arange(kw)  # [wk,kw]
    idx = (wh[:, None, :, None] * wk + ww[None, :, None, :]).reshape(hk * wk, kh * kw)  # host table [Bk, 81]
    qb = q.reshape(B, hk, dh, wk, dw, N, D).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, hk * wk, N, dh * dw, D)
    kf = k_lr.reshape(B, hk * wk, N, D)
    vf = v_lr.reshape(B, hk * wk, N, Dv)
    out = torch.empty(B, hk * wk, N, dh * dw, Dv, dtype=q.dtype)
    for s in range(0, hk * wk, block_chunk):
        ii = idx[s:s + block_chunk]
        kg = kf[:, ii].permute(0, 1, 3, 2, 4)  # [B,b,N,81,D]
        vg = vf[:, ii].permute(0, 1, 3, 2, 4)  # [B,b,N,81,Dv]
        a = (qb[:, s:s + block_chunk] @ kg.transpose(-1, -2) * scale).softmax(-1)  # [B,b,N,d2,81]
        out[:, s:s + block_chunk] = a @ vg
    return out.reshape(B, hk, wk, N, dh, dw, Dv).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, H, W, N, Dv)


def na2d_block_from_hires(q, k, v, kernel_size, dilation, stride=1, backend=None, scale=None):
    """Drop-in for natten.na2d as NAF calls it: checks K/V really are nearest-exact
    integer upsamples, then runs na2d_block on the low-res grids."""
    assert stride == 1 or stride == (1, 1)
    dh, dw = dilation
    k_lr = k[:, ::dh, ::dw]
    v_lr = v[:, ::dh, ::dw]
    assert torch.equal(k, k_lr.repeat_interleave(dh, 1).repeat_interleave(dw, 2)), "K not block-constant"
    assert torch.equal(v[:, :, :: max(1, v.shape[2] // 64)],
                       v_lr.repeat_interleave(dh, 1).repeat_interleave(dw, 2)[:, :, :: max(1, v.shape[2] // 64)])
    return na2d_block(q, k_lr, v_lr, kernel_size, scale=scale)


# ----------------------------------------------------------------------------
# Load the repo's NAF module (unmodified) with the attention substituted
# ----------------------------------------------------------------------------
REAL_NATTEN = None


def load_naf():
    global REAL_NATTEN
    if os.path.isdir(NATTEN_PKG):
        sys.path.insert(0, NATTEN_PKG)
        try:
            import natten as _n
            REAL_NATTEN = _n
        except Exception as e:  # pragma: no cover
            log("natten import failed:", e)
    if REAL_NATTEN is None:
        import types
        stub = types.ModuleType("natten")
        stub.na2d = na2d_block_from_hires
        sys.modules["natten"] = stub
    sys.path.insert(0, NAF_SRC)
    import src.layers.attentions as A  # noqa: E402
    A.NATTEN_RECENT = True
    A.na2d = na2d_block_from_hires
    spec = importlib.util.spec_from_file_location("naf_model", os.path.join(NAF_SRC, "src", "model", "naf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.NAF()
    sd = torch.load(NAF_W, map_location="cpu", weights_only=True)
    missing = model.load_state_dict(sd, strict=True)
    model.eval()
    return model, sd, A


# ----------------------------------------------------------------------------
# DINOv3 ViT-L/16, pure torch port of transformers DINOv3ViTModel (eager path)
# ----------------------------------------------------------------------------
def dinov3_tokens(pixel, W):
    """pixel [1,3,S,S] ImageNet-normalised. Returns Pixal3D's `z`: all 24 layers,
    then F.layer_norm without affine (NOT the model's final `norm`). [1, 5+hw, 1024]"""
    nh, hd = 16, 64
    x = F.conv2d(pixel, W["embeddings.patch_embeddings.weight"], W["embeddings.patch_embeddings.bias"], stride=16)
    B, C, h, w = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = torch.cat([W["embeddings.cls_token"], W["embeddings.register_tokens"], x], 1)
    ch = torch.arange(0.5, h) / h
    cw = torch.arange(0.5, w) / w
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), -1).flatten(0, 1) * 2.0 - 1.0
    inv_freq = 1 / 100.0 ** torch.arange(0, 1, 4 / hd, dtype=torch.float32)
    ang = (2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)
    cos, sin = ang.cos(), ang.sin()

    def rope(t):
        pre, p = t[:, :, :5], t[:, :, 5:]
        p1, p2 = p.chunk(2, -1)
        return torch.cat([pre, p * cos + torch.cat([-p2, p1], -1) * sin], 2)

    for i in range(24):
        L = lambda n: W[f"layer.{i}.{n}"]  # noqa: E731
        r = x
        y = F.layer_norm(x, (1024,), L("norm1.weight"), L("norm1.bias"), 1e-5)
        q = F.linear(y, L("attention.q_proj.weight"), L("attention.q_proj.bias")).view(B, -1, nh, hd).transpose(1, 2)
        k = F.linear(y, L("attention.k_proj.weight")).view(B, -1, nh, hd).transpose(1, 2)
        v = F.linear(y, L("attention.v_proj.weight"), L("attention.v_proj.bias")).view(B, -1, nh, hd).transpose(1, 2)
        q, k = rope(q), rope(k)
        a = (q @ k.transpose(-1, -2) * hd ** -0.5).softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(B, -1, 1024)
        o = F.linear(o, L("attention.o_proj.weight"), L("attention.o_proj.bias"))
        x = o * L("layer_scale1.lambda1") + r
        r = x
        y = F.layer_norm(x, (1024,), L("norm2.weight"), L("norm2.bias"), 1e-5)
        y = F.linear(F.gelu(F.linear(y, L("mlp.up_proj.weight"), L("mlp.up_proj.bias"))),
                     L("mlp.down_proj.weight"), L("mlp.down_proj.bias"))
        x = y * L("layer_scale2.lambda1") + r
    return F.layer_norm(x, x.shape[-1:])


# ----------------------------------------------------------------------------
# Pixal3D pieces (copied from pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
# and pixal3d/pipelines/pixal3d_image_to_3d.py; transformers/dist imports avoided)
# ----------------------------------------------------------------------------
def preprocess_image(input_img, bg_color=(0, 0, 0)):
    """Pixal3DImageTo3DPipeline.preprocess_image, has-alpha branch (no BiRefNet)."""
    alpha = np.array(input_img)[:, :, 3]
    assert input_img.mode == "RGBA" and not np.all(alpha == 255)
    max_size = max(input_img.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input_img = input_img.resize((int(input_img.width * scale), int(input_img.height * scale)), Image.Resampling.LANCZOS)
    output = input_img
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    rgb = output[:, :, :3]
    a = output[:, :, 3:4]
    bg = np.array(bg_color, dtype=np.float32) / 255.0
    output = rgb * a + bg * (1.0 - a)
    return Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))


def proj_grid_points(grid_resolution, image_resolution, camera_angle_x, distance, mesh_scale):
    one = torch.linspace(-1, 1, grid_resolution)
    x, y, z = torch.meshgrid(one, one, one, indexing="ij")
    gp = torch.stack((x, y, z), -1)
    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = torch.matmul(gp, R.T).reshape(-1, 3)[None]
    gp = gp / torch.tensor([mesh_scale])[:, None, None] / 2
    T = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])[None].clone()
    T[:, 1, 3] = -distance
    N = gp.shape[1]
    ph = torch.cat([gp, torch.ones(1, N, 1)], -1)
    w2c = torch.linalg.inv(T.float())
    pc = torch.bmm(ph, w2c.transpose(-2, -1))[..., :3]
    xc, yc, zc = pc[..., 0], pc[..., 1], pc[..., 2]
    fl = 16.0 / torch.tan(torch.tensor([camera_angle_x]) / 2.0)
    flp = (fl * image_resolution / 32.0).unsqueeze(1)
    xp = flp * xc / (-zc + 1e-8) + image_resolution / 2.0
    yp = -(flp * yc / (-zc + 1e-8)) + image_resolution / 2.0
    pts = torch.stack([xp, yp], -1)
    return (pts + 0.5) / image_resolution * 2 - 1  # [1,K,2] grid_sample coords


def sample_features(fmap, qn):
    B, C, H, W = fmap.shape
    return F.grid_sample(fmap, qn.view(B, -1, 1, 2), mode="bilinear", align_corners=False,
                         padding_mode="border").squeeze(-1)  # [B,C,K]


# ----------------------------------------------------------------------------
# The ggml decomposition, executed in torch op-for-op (see naf.md section 2)
# ----------------------------------------------------------------------------
def grid_sample_taps(qn, H, W):
    """Host table for grid_sample(bilinear, align_corners=False, border):
    4 pixel indices + 4 weights per point. GET_ROWS + MUL + ADD on device."""
    gx, gy = qn[0, :, 0].double(), qn[0, :, 1].double()
    x = (((gx + 1) * W - 1) / 2).clamp(0, W - 1)
    y = (((gy + 1) * H - 1) / 2).clamp(0, H - 1)
    x0, y0 = x.floor(), y.floor()
    fx, fy = x - x0, y - y0
    x0, y0 = x0.long(), y0.long()
    x1, y1 = (x0 + 1).clamp(max=W - 1), (y0 + 1).clamp(max=H - 1)
    idx = torch.stack([y0 * W + x0, y0 * W + x1, y1 * W + x0, y1 * W + x1], 0)  # [4,K]
    wt = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 0).float()
    return idx, wt


def gs_apply(feat_rows, idx, wt):
    """feat_rows [P, C] (ggml [C,P]); GET_ROWS x4, MUL by weight, ADD."""
    return sum(feat_rows[idx[t]] * wt[t][:, None] for t in range(4))  # [K,C]


def reflect_index(n):
    """1-px reflect padding index table (PyTorch 'reflect', no edge repeat)."""
    i = torch.arange(-1, n + 1)
    i = i.abs()
    return torch.where(i > n - 1, 2 * (n - 1) - i, i)


def group_norm_as_norm(x, gamma, beta, groups=8, eps=1e-5):
    """x [C,H,W] spatial-major ggml [W,H,C]: group g is contiguous (C/G)*H*W floats.
    RESHAPE [C/G*H*W, G] -> NORM(eps) -> RESHAPE -> MUL gamma -> ADD beta."""
    C, H, W = x.shape
    y = x.reshape(groups, -1)
    mu = y.mean(-1, keepdim=True)
    var = ((y - mu) ** 2).mean(-1, keepdim=True)
    y = ((y - mu) / torch.sqrt(var + eps)).reshape(C, H, W)
    return y * gamma[:, None, None] + beta[:, None, None]


def conv_im2col(x, w, b, reflect, band=64):
    """Conv2d(stride 1, pad k//2 reflect) = reflect-pad GET_ROWS tables + IM2COL + MUL_MAT + ADD."""
    C, H, W = x.shape
    O, Ci, kh, kw = w.shape
    if kh == 1:
        return (w.reshape(O, Ci) @ x.reshape(C, -1)).reshape(O, H, W) + b[:, None, None]
    ih, iw = reflect_index(H), reflect_index(W)
    xp = x[:, ih][:, :, iw]  # GET_ROWS along H then W
    wm = w.reshape(O, Ci * kh * kw)
    out = torch.empty(O, H, W)
    for r in range(0, H, band):
        r1 = min(H, r + band)
        cols = F.unfold(xp[None, :, r:r1 + 2], (kh, kw))[0]  # IM2COL [Ci*9, band*W]
        out[:, r:r1] = (wm @ cols).reshape(O, r1 - r, W)
    return out + b[:, None, None]


def encoder_plan(img, sd, pre):
    x = conv_im2col(img, sd[pre + "0.weight"], sd[pre + "0.bias"], True)
    for blk in (1, 2):
        p = f"{pre}{blk}."
        x = group_norm_as_norm(x, sd[p + "norm1.weight"], sd[p + "norm1.bias"])
        x = F.silu(x)
        x = conv_im2col(x, sd[p + "conv1.weight"], sd[p + "conv1.bias"], True)
        x = group_norm_as_norm(x, sd[p + "norm2.weight"], sd[p + "norm2.bias"])
        x = F.silu(x)
        x = conv_im2col(x, sd[p + "conv2.weight"], sd[p + "conv2.bias"], True)
    return x


def pool_reshape_mean(x, oh, ow):
    """adaptive_avg_pool2d for divisible sizes: RESHAPE + MEAN (ne0) + PERMUTE/CONT + MEAN."""
    C, H, W = x.shape
    rh, rw = H // oh, W // ow
    assert rh * oh == H and rw * ow == W
    return x.reshape(C, oh, rh, ow, rw).mean(4).mean(2)


def rope_ggml_neox(x, periods, heads=4):
    """ggml ROPE (NEOX, freq_base=1, freq_scale=1, ext_factor=0, attn_factor=1) applied twice
    with int32 positions pos_h = 2h+1-H / pos_w = 2w+1-W and freq_factors
    ff = [H*period/(2pi) x16, BIG x16] then [BIG x16, W*period/(2pi) x16].
    theta_i = pos / ff[i]; pair (i, i+32) rotated. x [256,H,W] -> [256,H,W]."""
    C, H, W = x.shape
    hd = C // heads
    BIG = 1e30
    per = periods.float()
    ffh = torch.cat([H * per / (2 * math.pi), torch.full((16,), BIG)])
    ffw = torch.cat([torch.full((16,), BIG), W * per / (2 * math.pi)])
    ph = (2 * torch.arange(H) + 1 - H).float()[:, None].expand(H, W).reshape(-1)
    pw = (2 * torch.arange(W) + 1 - W).float()[None, :].expand(H, W).reshape(-1)
    t = x.reshape(heads, hd, H * W)
    for pos, ff in ((ph, ffh), (pw, ffw)):
        th = pos[None, :] / ff[:, None]  # [32, P]
        c, s = torch.cos(th), torch.sin(th)
        a, b = t[:, :32], t[:, 32:]
        t = torch.cat([a * c - b * s, a * s + b * c], 1)
    return t.reshape(C, H, W)


# ----------------------------------------------------------------------------
# ggml-literal simulation (naf.md section 2). Convention: a ggml tensor with
# ne=[ne0,ne1,ne2,ne3] is held as a contiguous torch tensor of shape
# [ne3,ne2,ne1,ne0] (reversed). Every g_* helper is one ggml op.
# ----------------------------------------------------------------------------
def g_cont_transpose(a):              # CONT(TRANSPOSE(a)) on the two lowest dims
    return a.transpose(-1, -2).contiguous()


def g_get_rows(a, idx):               # GET_ROWS(a [ne0, R], idx int32 [n]) -> [ne0, n]
    return a[idx]


def g_mul_mat(a, b):                  # MUL_MAT(a [K,M,..], b [K,N,..]) -> [M,N,..]: res[i,j]=sum_k a[k,i] b[k,j]
    return b @ a.transpose(-1, -2)


def g_norm(a, eps):                   # NORM over ne0 (no affine)
    mu = a.mean(-1, keepdim=True)
    var = (a - mu).pow(2).mean(-1, keepdim=True)
    return (a - mu) / torch.sqrt(var + eps)


def g_mean(a):                        # MEAN over ne0 -> ne0 = 1
    return a.mean(-1, keepdim=True)


def g_soft_max_ext(a, scale):         # SOFT_MAX_EXT(a, mask=NULL, scale, max_bias=0) over ne0
    return (a * scale).softmax(-1)


def g_rope_neox(x, pos, ff):
    """ROPE(x [n_dims, n_head, n_tok], pos int32 [n_tok], freq_factors ff [n_dims/2]),
    mode=NEOX, freq_base=1, freq_scale=1, ext_factor=0, attn_factor=1 (ggml CPU rule:
    theta_i = pos * base^(-2i/n_dims) / ff[i]; pairs (i, i+n_dims/2))."""
    n = x.shape[-1]
    th = pos.float()[:, None, None] / ff[None, None, :]      # [tok,1,n/2]
    c, s = torch.cos(th), torch.sin(th)
    a, b = x[..., : n // 2], x[..., n // 2:]
    return torch.cat([a * c - b * s, a * s + b * c], -1)


def tbl_reflect_im2col(H, W):
    """Host table [9*H*W] int32: for output pixel p (raster) and tap t=ky*3+kx, the
    source pixel of a 3x3 conv with 1-px 'reflect' padding. idx[p*9+t]."""
    rh, rw = reflect_index(H), reflect_index(W)          # padded coord -> source coord
    oy = torch.arange(H)[:, None, None, None]
    ox = torch.arange(W)[None, :, None, None]
    ky = torch.arange(3)[None, None, :, None]
    kx = torch.arange(3)[None, None, None, :]
    return (rh[oy + ky] * W + rw[ox + kx]).reshape(-1)


def tbl_block_major(T, d):
    """Host permutation [T*T]: block-major slot q -> raster pixel. Slot order is
    (bi, bj, a, b) with pixel (bi*d+a, bj*d+b); inverse maps raster -> slot."""
    hk = T // d
    bi = torch.arange(hk)[:, None, None, None]
    bj = torch.arange(hk)[None, :, None, None]
    a = torch.arange(d)[None, None, :, None]
    b = torch.arange(d)[None, None, None, :]
    perm = ((bi * d + a) * T + bj * d + b).reshape(-1)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    return perm, inv


def tbl_window81(hk, wk, ks=9):
    """Host table [hk*wk*81] int32: the 9x9 low-res window of every low-res block."""
    sh = (torch.arange(hk) - ks // 2).clamp(0, hk - ks)
    sw = (torch.arange(wk) - ks // 2).clamp(0, wk - ks)
    return ((sh[:, None, None, None] + torch.arange(ks)[None, None, :, None]) * wk
            + (sw[None, :, None, None] + torch.arange(ks)[None, None, None, :])).reshape(-1)


def g_conv(x_cm, w, b, H, W, band_px=1 << 16):
    """Conv2d stride 1 (1x1, or 3x3 reflect) in ggml ops. x_cm: channel-major ggml
    [P, Ci] (torch [Ci, P]). Returns channel-major [P, O] (torch [O, P])."""
    O, Ci, kh, kw = w.shape
    P = H * W
    x_cl = g_cont_transpose(x_cm)                       # CONT(TRANSPOSE) -> [Ci, P]
    wg = w.permute(0, 2, 3, 1).reshape(O, kh * kw * Ci)  # weight stored ggml [9*Ci, O], ne0 = (c, t)
    out = torch.empty(O, P)
    if kh == 1:
        out = g_mul_mat(x_cl, wg)                        # MUL_MAT -> [P, O]
    else:
        idx = tbl_reflect_im2col(H, W)
        for s in range(0, P, band_px):                   # banding = graph chunking
            e = min(P, s + band_px)
            cols = g_get_rows(x_cl, idx[s * 9:e * 9])    # GET_ROWS -> [Ci, 9*(e-s)]
            cols = cols.reshape(e - s, 9 * Ci)           # RESHAPE -> [9*Ci, e-s]
            out[:, s:e] = g_mul_mat(cols, wg)            # MUL_MAT -> [e-s, O]
    return out + b[:, None]                              # ADD bias [1, O]


def g_group_norm(x_cm, gamma, beta, groups=8, eps=1e-5):
    C, P = x_cm.shape
    y = g_norm(x_cm.reshape(groups, (C // groups) * P), eps)   # RESHAPE [C/G*P, G] -> NORM
    return y.reshape(C, P) * gamma[:, None] + beta[:, None]    # RESHAPE -> MUL -> ADD


def g_encoder(img_cm, sd, pre, H, W):
    x = g_conv(img_cm, sd[pre + "0.weight"], sd[pre + "0.bias"], H, W)
    for blk in (1, 2):
        p = f"{pre}{blk}."
        x = F.silu(g_group_norm(x, sd[p + "norm1.weight"], sd[p + "norm1.bias"]))   # SILU
        x = g_conv(x, sd[p + "conv1.weight"], sd[p + "conv1.bias"], H, W)
        x = F.silu(g_group_norm(x, sd[p + "norm2.weight"], sd[p + "norm2.bias"]))
        x = g_conv(x, sd[p + "conv2.weight"], sd[p + "conv2.bias"], H, W)
    return x


def g_pool(x_cm, H, W, r):
    """adaptive_avg_pool2d by integer factor r on channel-major ggml [W,H,C]:
    RESHAPE [r,W/r,H,C] -> MEAN -> RESHAPE [W/r,r,H/r,C] -> PERMUTE [r,W/r,H/r,C] -> CONT -> MEAN."""
    C = x_cm.shape[0]
    if r == 1:
        return x_cm
    y = g_mean(x_cm.reshape(C, H, W // r, r))[..., 0]
    y = y.reshape(C, H // r, r, W // r).permute(0, 1, 3, 2).contiguous()
    return g_mean(y)[..., 0].reshape(C, -1)


def naf_ggml(img, feats_tok, S, T, sd, heads=4, ks=9, block_chunk=512, return_parts=False):
    """img [3,S,S] in [0,1] (channel-major ggml [S*S,3]); feats_tok: DINOv3 patch
    tokens ggml [1024, hk*wk] (torch [hk*wk,1024]) = Pixal3D's z[5:], no transpose.
    Returns hr features ggml [1024, T*T] in BLOCK-MAJOR pixel order (torch [T*T,1024])."""
    Bk, C = feats_tok.shape
    hk = int(round(math.sqrt(Bk)))
    d = T // hk
    r = S // T
    P = T * T
    img_cm = img.reshape(3, S * S)
    x = torch.cat([g_encoder(img_cm, sd, "image_encoder.encoder.", S, S),
                   g_encoder(img_cm, sd, "image_encoder.sem_encoder.", S, S)], 0)   # CONCAT (ggml dim 1)
    x = g_pool(x, S, S, r)                                   # [T*T, 256] channel-major
    x_pool = x
    perm, inv = tbl_block_major(T, d)
    xb = g_get_rows(g_cont_transpose(x), perm)               # CONT(TRANSPOSE) + GET_ROWS -> [256, P] block-major
    xb = xb.reshape(P, heads, 64)                            # RESHAPE [64, 4, P]
    per = sd["image_encoder.rope.periods"].float()
    BIG = torch.full((16,), 1e30)
    ffh = torch.cat([T * per / (2 * math.pi), BIG])          # host freq_factors [32]
    ffw = torch.cat([BIG, T * per / (2 * math.pi)])
    py, px = perm // T, perm % T
    posh, posw = 2 * py + 1 - T, 2 * px + 1 - T               # host int32 positions
    q = g_rope_neox(g_rope_neox(xb, posh, ffh), posw, ffw)    # ROPE x2 -> [64, 4, P]
    # KeyEncoder: mean over each d x d block (contiguous d^2 slots in block-major order)
    qv = q.reshape(Bk, d * d, heads * 64)                    # view [256, d^2, Bk]
    k_lr = g_mean(g_cont_transpose(qv))[..., 0]              # PERMUTE+CONT [d^2,256,Bk] -> MEAN -> [256, Bk]
    idx81 = tbl_window81(hk, hk, ks)
    out = torch.empty(Bk, d * d, heads, C // heads)
    qb = q.reshape(Bk, d * d, heads, 64).permute(0, 2, 1, 3).contiguous()   # PERMUTE+CONT [64, d^2, 4, Bk]
    scale = 64 ** -0.5
    for s in range(0, Bk, block_chunk):
        e = min(Bk, s + block_chunk)
        ii = idx81[s * 81:e * 81]
        kw_ = g_get_rows(k_lr, ii).reshape(e - s, 81, heads, 64).permute(0, 2, 1, 3).contiguous()  # [64,81,4,b]
        sc = g_mul_mat(kw_, qb[s:e])                          # [81, d^2, 4, b]
        pr = g_soft_max_ext(sc, scale)
        vw = g_get_rows(feats_tok, ii).reshape(e - s, 81, heads, C // heads).permute(0, 2, 3, 1).contiguous()  # [81,256,4,b]
        o = g_mul_mat(vw, pr)                                 # [256, d^2, 4, b]
        out[s:e] = o.permute(0, 2, 1, 3)                      # PERMUTE+CONT -> [256, 4, d^2, b]
    hr_blk = out.reshape(P, C)                                # RESHAPE [1024, P] block-major
    if return_parts:
        return hr_blk, dict(perm=perm, inv=inv, x_pool=x_pool.reshape(256, T, T),
                            q=q.reshape(P, 256)[inv].T.reshape(256, T, T), k_lr=k_lr.T.reshape(256, hk, hk))
    return hr_blk


def naf_plan(img, feats, out_hw, sd, heads=4, ks=9, return_parts=False):
    """img [3,S,S] in [0,1]; feats [C,hk,wk]; returns hr [C,oh,ow]."""
    oh, ow = out_hw
    assert img.shape[1] <= 4 * oh and img.shape[2] <= 4 * ow  # the bilinear pre-resize never fires in Pixal3D
    x = torch.cat([encoder_plan(img, sd, "image_encoder.encoder."),
                   encoder_plan(img, sd, "image_encoder.sem_encoder.")], 0)  # CONCAT -> [256,S,S]
    x = pool_reshape_mean(x, oh, ow)
    q = rope_ggml_neox(x, sd["image_encoder.rope.periods"])  # [256,oh,ow]
    C, hk, wk = feats.shape
    k_lr = pool_reshape_mean(q, hk, wk)  # KeyEncoder
    Q = q.permute(1, 2, 0).reshape(1, oh, ow, heads, -1)
    K = k_lr.permute(1, 2, 0).reshape(1, hk, wk, heads, -1)
    V = feats.permute(1, 2, 0).reshape(1, hk, wk, heads, -1)
    o = na2d_block(Q, K, V, (ks, ks))
    hr = o.reshape(oh, ow, -1).permute(2, 0, 1)
    if return_parts:
        return hr, dict(x_pool=x, q=q, k_lr=k_lr)
    return hr


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_natten_small():
    """Real NATTEN (flex-fna, the only CPU backend) vs na2d_literal vs na2d_block."""
    if REAL_NATTEN is None:
        log("[T1] natten not importable; skipped")
        return
    g = torch.Generator().manual_seed(0)
    cases = [((20, 20), (5, 5), (2, 2), 8, 8), ((23, 19), (5, 3), (3, 4), 8, 8), ((13, 17), (3, 3), (4, 5), 8, 16),
             ((11, 11), (9, 9), (1, 1), 8, 8), ((36, 36), (9, 9), (4, 4), 8, 32)]
    worst = 0.0
    for (H, W), kk, dd, D, Dv in cases:
        q = torch.randn(1, H, W, 2, D, generator=g)
        k = torch.randn(1, H, W, 2, D, generator=g)
        v = torch.randn(1, H, W, 2, Dv, generator=g)
        # flex-fna needs Dv == D: attention is linear in V, so run it per D-wide V slice
        ref = torch.cat([REAL_NATTEN.na2d(q, k, v[..., s:s + D].contiguous(), kernel_size=kk, dilation=dd, stride=1,
                                          backend="flex-fna") for s in range(0, Dv, D)], -1)
        lit = na2d_literal(q, k, v, kk, dd)
        e = maxrel(lit, ref)[0]
        worst = max(worst, e)
        log(f"[T1] natten flex-fna vs literal H,W={H},{W} k={kk} d={dd} D={D} Dv={Dv}: max rel err {e:.2e}")
    # NAF-shaped: nearest-exact integer upsample -> block formulation
    for (hk, wk), d in (((10, 10), 8), ((9, 12), 16), ((12, 12), 6)):
        H, W = hk * d, wk * d
        q = torch.randn(1, H, W, 2, 8, generator=g)
        kl = torch.randn(1, hk, wk, 2, 8, generator=g)
        vl = torch.randn(1, hk, wk, 2, 16, generator=g)
        up = lambda t: t.repeat_interleave(d, 1).repeat_interleave(d, 2)  # == F.interpolate nearest-exact
        chk = F.interpolate(kl.permute(0, 3, 4, 1, 2).reshape(1, 16, hk, wk), size=(H, W), mode="nearest-exact")
        assert torch.equal(chk.reshape(1, 2, 8, H, W).permute(0, 3, 4, 1, 2), up(kl))
        ref = torch.cat([REAL_NATTEN.na2d(q, up(kl), up(vl)[..., s:s + 8].contiguous(), kernel_size=(9, 9),
                                          dilation=(d, d), stride=1, backend="flex-fna") for s in (0, 8)], -1)
        blk = na2d_block(q, kl, vl, (9, 9))
        lit = na2d_literal(q, up(kl), up(vl), (9, 9), (d, d))
        e1, e2 = maxrel(blk, ref)[0], maxrel(lit, ref)[0]
        worst = max(worst, e1, e2)
        log(f"[T1] NAF-shaped hk,wk={hk},{wk} d={d}: block vs natten {e1:.2e}, literal vs natten {e2:.2e}")
    log(f"[T1] worst {worst:.2e}")
    return worst


def test_resample_decomps():
    """Decompositions for ops Pixal3D's NAF path does not hit but the module has."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 5, 37, 29, generator=g)
    # bilinear, align_corners=False, downscale: separable 2-tap matrices (MUL_MAT)
    def interp_mat(n_in, n_out):
        sc = n_in / n_out
        src = ((torch.arange(n_out, dtype=torch.float64) + 0.5) * sc - 0.5).clamp(min=0)
        i0 = src.floor().long().clamp(max=n_in - 1)
        i1 = (i0 + 1).clamp(max=n_in - 1)
        l1 = src - i0
        M = torch.zeros(n_out, n_in, dtype=torch.float64)
        M[torch.arange(n_out), i0] += 1 - l1
        M[torch.arange(n_out), i1] += l1
        return M.float()
    ref = F.interpolate(x, size=(16, 11), mode="bilinear", align_corners=False)
    got = interp_mat(37, 16) @ x @ interp_mat(29, 11).T
    e1 = maxrel(got, ref)[0]
    # adaptive_avg_pool2d, non-divisible: separable bin-average matrices
    def pool_mat(n_in, n_out):
        M = torch.zeros(n_out, n_in)
        for i in range(n_out):
            a, b = (i * n_in) // n_out, -((-(i + 1) * n_in) // n_out)
            M[i, a:b] = 1.0 / (b - a)
        return M
    ref2 = F.adaptive_avg_pool2d(x, (7, 5))
    got2 = pool_mat(37, 7) @ x @ pool_mat(29, 5).T
    e2 = maxrel(got2, ref2)[0]
    # nearest-exact general: GET_ROWS with src = min(floor((i+.5)*in/out), in-1)
    ref3 = F.interpolate(x, size=(50, 71), mode="nearest-exact")
    ih = ((torch.arange(50) + 0.5) * 37 / 50).floor().long().clamp(max=36)
    iw = ((torch.arange(71) + 0.5) * 29 / 71).floor().long().clamp(max=28)
    e3 = maxrel(x[:, :, ih][:, :, :, iw], ref3)[0]
    # GroupNorm as NORM on a [C/G*H*W, G] reshape
    gn = torch.nn.GroupNorm(8, 128)
    torch.nn.init.normal_(gn.weight); torch.nn.init.normal_(gn.bias)
    xx = torch.randn(1, 128, 9, 13, generator=g)
    e4 = maxrel(group_norm_as_norm(xx[0], gn.weight.detach(), gn.bias.detach()), gn(xx)[0].detach())[0]
    # reflect conv via index table + im2col
    cv = torch.nn.Conv2d(128, 64, 3, padding=1, padding_mode="reflect")
    e5 = maxrel(conv_im2col(xx[0], cv.weight.detach(), cv.bias.detach(), True, band=4), cv(xx)[0].detach())[0]
    # reflect conv3x3 as ONE GET_ROWS (host im2col+reflect table) + ONE MUL_MAT, channel-major in/out
    e6 = maxrel(g_conv(xx[0].reshape(128, -1), cv.weight.detach(), cv.bias.detach(), 9, 13, band_px=50),
                cv(xx)[0].detach().reshape(64, -1))[0]
    # GroupNorm on channel-major via RESHAPE+NORM (ggml-literal helper)
    e7 = maxrel(g_group_norm(xx[0].reshape(128, -1), gn.weight.detach(), gn.bias.detach()),
                gn(xx)[0].detach().reshape(128, -1))[0]
    # FLASH_ATTN_EXT stand-in (SDPA) with Dk=64, Dv=256 run as 4 x Dv=64 slices == MUL_MAT/SOFT_MAX chain
    q = torch.randn(7, 4, 64, 64, generator=g); k = torch.randn(7, 4, 81, 64, generator=g); v = torch.randn(7, 4, 81, 256, generator=g)
    chain = g_mul_mat(v.transpose(-1, -2).contiguous(), g_soft_max_ext(g_mul_mat(k, q), 0.125))
    fa = torch.cat([F.scaled_dot_product_attention(q, k, v[..., s:s + 64], scale=0.125) for s in range(0, 256, 64)], -1)
    e8 = maxrel(fa, chain)[0]
    # 2-D axial RoPE (NAF/DINOv3 form) == two ggml NEOX ROPE passes with freq_factors
    per = 100.0 ** (2 * torch.arange(16, dtype=torch.float32) / 32)
    H_, W_ = 12, 20
    xr = torch.randn(1, 256, H_, W_, generator=g)
    sys.path.insert(0, NAF_SRC)
    from src.layers.rope import RoPE as _R
    rp = _R(embed_dim=256, num_heads=4, base=100.0, rescale_coords=2.0).eval()
    ref_r = rp(xr)[0].reshape(256, -1).T.reshape(-1, 4, 64)
    big = torch.full((16,), 1e30)
    py = torch.arange(H_)[:, None].expand(H_, W_).reshape(-1); px = torch.arange(W_)[None, :].expand(H_, W_).reshape(-1)
    got_r = g_rope_neox(g_rope_neox(xr[0].reshape(256, -1).T.reshape(-1, 4, 64), 2 * py + 1 - H_,
                                    torch.cat([H_ * per / (2 * math.pi), big])), 2 * px + 1 - W_, torch.cat([big, W_ * per / (2 * math.pi)]))
    e9 = maxrel(got_r, ref_r)[0]
    log(f"[T4] bilinear(ac=False) as 2 MUL_MAT {e1:.2e}; adaptive pool non-divisible as 2 MUL_MAT {e2:.2e}; "
        f"nearest-exact as GET_ROWS {e3:.2e}; GroupNorm as NORM {e4:.2e}; reflect conv3x3 as GET_ROWS+IM2COL+MUL_MAT {e5:.2e}; "
        f"reflect conv3x3 as GET_ROWS(im2col table)+MUL_MAT {e6:.2e}; GroupNorm channel-major RESHAPE+NORM {e7:.2e}; "
        f"FA(Dv=4x64 split) vs MUL_MAT+SOFT_MAX chain {e8:.2e}; axial RoPE as 2x ROPE_NEOX+freq_factors {e9:.2e}")
    return max(e1, e2, e3, e4, e5, e6, e7, e8, e9)


def test_naf_medium(model):
    """Repo NAF (weights) end to end at a size real NATTEN flex can hold, C=256 so Dv==D."""
    if REAL_NATTEN is None:
        return None
    import src.layers.attentions as A
    g = torch.Generator().manual_seed(2)
    img = torch.rand(1, 3, 160, 160, generator=g)
    feats = torch.randn(1, 256, 10, 10, generator=g)
    with torch.no_grad():
        A.na2d = lambda q, k, v, kernel_size, dilation, stride=1, backend=None: REAL_NATTEN.na2d(
            q, k, v, kernel_size=kernel_size, dilation=dilation, stride=stride, backend="flex-fna")
        ref = model(img, feats, (80, 80))
        A.na2d = na2d_block_from_hires
        got = model(img, feats, (80, 80))
        plan = naf_plan(img[0], feats[0], (80, 80), model.state_dict())
        sdf = {k: v.float() for k, v in model.state_dict().items()}
        hb, pp = naf_ggml(img[0], feats[0].reshape(256, -1).T.contiguous(), 160, 80, sdf, return_parts=True)
        gg = hb[pp["inv"]].T.reshape(1, 256, 80, 80)
    e1, e2, e3 = maxrel(got, ref)[0], maxrel(plan[None], ref)[0], maxrel(gg, ref)[0]
    log(f"[T2] repo NAF 160px image, 10x10x256 feats -> 80x80 (d=8, pool r=2): na2d_block vs real natten {e1:.2e}; "
        f"torch plan vs real natten {e2:.2e}; ggml-literal chain vs real natten {e3:.2e}")
    return max(e1, e2, e3)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def flops(S_in, out, hk, C=1024, heads=4, dqk=64, kk=81):
    P = S_in * S_in
    enc1 = P * (2 * 3 * 128 + 4 * 2 * 128 * 128)
    enc3 = P * (2 * 27 * 128 + 4 * 2 * 1152 * 128)
    norm_act = P * 256 * 2 * 2 * 10  # 4 GN + 4 SiLU per branch, ~10 flop/elt, rough
    Q = out * out
    rope = Q * 256 * 6
    attn = Q * (heads * kk * dqk * 2 + kk * C * 2 + heads * kk * 5)
    return dict(enc_1x1=enc1, enc_3x3=enc3, norm_act=norm_act, rope=rope, attn=attn,
                total=enc1 + enc3 + norm_act + rope + attn)


def run_config(name, cfg, model, sd, dino_w, pil):
    S, R, T = cfg["image_size"], cfg["grid_resolution"], cfg["naf_target_size"]
    od = os.path.join(OUT, name)
    os.makedirs(od, exist_ok=True)
    img = pil.resize((S, S), Image.LANCZOS)
    img = torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None]
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    t0 = time.perf_counter()
    with torch.no_grad():
        z = dinov3_tokens((img - mean) / std, dino_w)
    t_dino = time.perf_counter() - t0
    hp = S // 16
    patches = z[:, 5:].reshape(1, hp, hp, -1)
    lr = patches.permute(0, 3, 1, 2).contiguous()  # [1,1024,hp,hp]
    t0 = time.perf_counter()
    with torch.no_grad():
        hr = model(img, lr, (T, T))  # repo module, attention = na2d_block
    t_naf = time.perf_counter() - t0
    log(f"[{name}] image {S}x{S} -> DINOv3 {hp}x{hp}x1024 ({t_dino:.1f}s) -> NAF {tuple(hr.shape)} ({t_naf:.1f}s)")
    # ggml-literal op chain (naf.md section 2), fed Pixal3D's token layout directly
    t0 = time.perf_counter()
    with torch.no_grad():
        sdf = {k: v.float() for k, v in sd.items()}
        hr_blk, parts = naf_ggml(img[0], z[0, 5:].contiguous(), S, T, sdf, return_parts=True)
    hr_rows = hr[0].reshape(1024, -1).T  # raster [T*T, 1024]
    e_plan = maxrel(hr_blk[parts["inv"]], hr_rows)
    log(f"[{name}] ggml-literal chain vs repo module: max rel {e_plan[0]:.2e} (abs {e_plan[1]:.2e}) ({time.perf_counter() - t0:.1f}s)")
    # literal NATTEN rule on hi-res upsampled K/V, at a few query rows (edges, centre, block borders)
    with torch.no_grad():
        x = model.image_encoder(img, output_size=(T, T))
        d = T // hp
        qh = x.reshape(1, 4, 64, T, T).permute(0, 3, 4, 1, 2)
        kh = F.interpolate(F.adaptive_avg_pool2d(x, (hp, hp)), size=(T, T), mode="nearest-exact").reshape(1, 4, 64, T, T).permute(0, 3, 4, 1, 2)
        rows = [0, 1, d - 1, d, T // 2 - 1, T // 2, T - d, T - 1]
        vh = F.interpolate(lr, size=(T, T), mode="nearest-exact")[:, :, :, :].reshape(1, 4, 256, T, T).permute(0, 3, 4, 1, 2)
        lit = na2d_literal(qh, kh, vh, (9, 9), (d, d), rows=rows, row_chunk=1)
        del vh
        e_lit = maxrel(lit.reshape(1, len(rows), T, 1024).permute(0, 3, 1, 2), hr[:, :, rows])
    log(f"[{name}] literal NATTEN rule (hi-res K/V, rows {rows}) vs module: max rel {e_lit[0]:.2e}")
    # Pixal3D ProjGrid: lr || hr -> 2048 channels
    qn = proj_grid_points(R, S, **CAM)
    with torch.no_grad():
        z_lr = sample_features(lr, qn).permute(0, 2, 1)  # [1,R^3,1024]
        z_hr = sample_features(hr, qn).permute(0, 2, 1)
        z_proj = torch.cat([z_lr, z_hr], -1)  # [1,R^3,2048] -> proj_linear in every proj block
    # ggml: host 4-tap tables + GET_ROWS; and NAF evaluated only at the tapped hr pixels
    idx_l, wt_l = grid_sample_taps(qn, hp, hp)
    idx_h, wt_h = grid_sample_taps(qn, T, T)
    g_lr = gs_apply(lr[0].reshape(1024, -1).T, idx_l, wt_l)
    e_gs = maxrel(g_lr, z_lr[0])
    g_hr_blk = gs_apply(hr_blk, parts["inv"][idx_h], wt_h)  # taps re-indexed to block-major slots on the host
    e_gs_blk = maxrel(g_hr_blk, z_hr[0])
    del hr_blk
    uniq = torch.unique(idx_h)
    # sparse NAF: each tapped pixel -> its block's 81-window, 1 query
    qv = parts["q"].reshape(4, 64, -1)[:, :, uniq].permute(2, 0, 1)  # [U,4,64]
    ys, xs = uniq // T, uniq % T
    bi, bj = ys // d, xs // d
    sh = (bi - 4).clamp(0, hp - 9); sw = (bj - 4).clamp(0, hp - 9)
    win = ((sh[:, None] + torch.arange(9))[:, :, None] * hp + (sw[:, None] + torch.arange(9))[:, None, :]).reshape(-1, 81)
    Kl = parts["k_lr"].reshape(4, 64, -1).permute(2, 0, 1)  # [hk*wk,4,64]
    Vl = lr[0].reshape(4, 256, -1).permute(2, 0, 1)
    sp = torch.empty(len(uniq), 1024)
    for s in range(0, len(uniq), 4096):  # chunked: an unchunked V window gather is U*81*1024 floats (166 GB at tex_1024)
        wi = win[s:s + 4096]
        a = torch.einsum("und,uknd->unk", qv[s:s + 4096], Kl[wi]).mul(0.125).softmax(-1)
        sp[s:s + 4096] = torch.einsum("unk,uknd->und", a, Vl[wi]).reshape(-1, 1024)
    e_sparse = maxrel(sp, hr_rows[uniq])
    rows_full = torch.zeros(T * T, 1024); rows_full[uniq] = sp
    g_hr = gs_apply(rows_full, idx_h, wt_h)
    e_gs2 = maxrel(g_hr, z_hr[0])
    log(f"[{name}] grid_sample as 4-tap GET_ROWS: lr {e_gs[0]:.2e}; hr on block-major ggml output {e_gs_blk[0]:.2e}; hr via sparse NAF at {len(uniq)} of {T*T} pixels "
        f"({100*len(uniq)/(T*T):.1f}%): pixels {e_sparse[0]:.2e}, sampled {e_gs2[0]:.2e}")
    # dump
    arrs = dict(image_naf=img.numpy(), dino_tokens=z.numpy(), lr_features=lr.numpy(),
                enc_pooled=parts["x_pool"][None].numpy(), queries_rope=parts["q"][None].numpy(),
                keys_lr=parts["k_lr"][None].numpy(), hr_features=hr.numpy(),
                proj_points_ndc=qn.numpy(), z_proj=z_proj.numpy())
    tables = dict(tbl_block_perm=parts["perm"], tbl_window81=tbl_window81(hp, hp),
                  tbl_gs_lr_idx=idx_l, tbl_gs_lr_w=wt_l, tbl_gs_hr_idx_raster=idx_h, tbl_gs_hr_w=wt_h)
    for k, v in tables.items():
        v = v.numpy().astype(np.int32 if v.dtype == torch.int64 else np.float32)
        np.save(os.path.join(od, k + ".npy"), v)
    meta = dict(config=name, **cfg, camera=CAM, output_size=[T, T], dilation=d, lr_grid=[hp, hp],
                example=os.path.relpath(EXAMPLE, HERE), naf_commit="37f2dfc180f2de53d98bd601109c0da0dd6b0f43",
                pixal3d_commit="cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af",
                errors=dict(ggml_chain_vs_module=e_plan[0], grid_sample_hr_blockmajor=e_gs_blk[0], literal_rows_vs_module=e_lit[0], grid_sample_lr=e_gs[0],
                            sparse_naf_pixels=e_sparse[0], sparse_naf_sampled=e_gs2[0]),
                timings_s=dict(dino=t_dino, naf=t_naf), flops=flops(S, T, hp), arrays={})
    for k, v in arrs.items():
        np.save(os.path.join(od, k + ".npy"), v.astype(np.float32))
        meta["arrays"][k] = dict(shape=list(v.shape), sha256_16=sha(v.astype(np.float32)))
    with open(os.path.join(od, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log(f"[{name}] saved {len(arrs)} arrays to {od}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="shape_512")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(OUT, exist_ok=True)
    LOGFILE[0] = os.path.join(OUT, "naf_ref." + args.configs.replace(",", "+") + ".log")
    open(LOGFILE[0], "w").close()
    model, sd, _ = load_naf()
    nparam = sum(p.numel() for p in model.parameters())
    nbuf = sum(b.numel() for b in model.buffers())
    log(f"NAF params {nparam} (+{nbuf} buffer floats: rope.periods), dtypes {sorted({str(v.dtype) for v in sd.values()})}; natten={'0.21.7 flex-fna' if REAL_NATTEN else 'absent'}")
    if not args.skip_tests:
        test_natten_small()
        test_resample_decomps()
        test_naf_medium(model)
    from safetensors.torch import load_file
    dino_w = load_file(DINO_W)
    pil = preprocess_image(Image.open(EXAMPLE))
    log(f"example {EXAMPLE}: preprocessed to {pil.size}")
    for name in args.configs.split(","):
        run_config(name, CONFIGS[name], model, sd, dino_w, pil)
    log("done")


if __name__ == "__main__":
    main()
