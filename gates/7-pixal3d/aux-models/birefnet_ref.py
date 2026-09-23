"""BiRefNet (HR-matting weights) CPU-torch reference + ggml-decomposition proofs.

Runs the unmodified upstream birefnet.py (HF ZhengPeng7/BiRefNet_HR-matting @5d6b6f8)
with the heavy imports stubbed out:
  transformers.PretrainedConfig/PreTrainedModel -> minimal nn.Module shims
  timm.layers (DropPath, to_2tuple, trunc_normal_) -> eval-mode equivalents
  torchvision.ops.deform_conv2d -> pure-torch deform_conv2d_torch (proved below)
  torchvision.models / kornia.filters.laplacian -> unused-at-inference stubs
and reproduces Pixal3D's background-removal path exactly
(pixal3d/pipelines/pixal3d_image_to_3d.py::preprocess_image +
 pixal3d/pipelines/rembg/BiRefNet.py::__call__), saving .npy oracles.

usage: python birefnet_ref.py [image] [--res 1024] [--out DIR] [--no-meta2048]
"""
import argparse, math, os, sys, time, types, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = r"C:\interactor-dress-on\models\BiRefNet_HR-matting"
DEFAULT_IMG = r"C:\contract-manifest\3-interactor\pixal3d-upstream\assets\images\s_14_img.jpg"


# ----------------------------------------------------------------------------
# 1. deform_conv2d (torchvision.ops, modulated DCNv2) in pure torch
# ----------------------------------------------------------------------------
def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def _bilinear_tv(flat, h, w, H, W):
    """torchvision deform_conv2d_kernel.cpp::bilinear_interpolate, vectorised.
    flat (B,C,H*W); h,w (B,N) float sample coords -> (B,C,N)."""
    valid = (h > -1) & (h < H) & (w > -1) & (w < W)
    hl = torch.floor(h); wl = torch.floor(w)
    hh_i = hl + 1; wh_i = wl + 1
    lh = h - hl; lw = w - wl
    hh = 1 - lh; hw = 1 - lw
    c1 = valid & (hl >= 0) & (wl >= 0)
    c2 = valid & (hl >= 0) & (wh_i <= W - 1)
    c3 = valid & (hh_i <= H - 1) & (wl >= 0)
    c4 = valid & (hh_i <= H - 1) & (wh_i <= W - 1)
    B, C, _ = flat.shape

    def tap(yy, xx, c):
        yi = yy.clamp(0, H - 1).long(); xi = xx.clamp(0, W - 1).long()
        idx = (yi * W + xi).unsqueeze(1).expand(B, C, yi.shape[-1])
        v = torch.gather(flat, 2, idx)
        return v * c.unsqueeze(1).to(flat.dtype)

    v1 = tap(hl, wl, c1); v2 = tap(hl, wh_i, c2); v3 = tap(hh_i, wl, c3); v4 = tap(hh_i, wh_i, c4)
    w1 = (hh * hw).unsqueeze(1); w2 = (hh * lw).unsqueeze(1)
    w3 = (lh * hw).unsqueeze(1); w4 = (lh * lw).unsqueeze(1)
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


def deform_im2col_torch(input, offset, mask, kh, kw, stride=1, padding=0, dilation=1):
    """columns (B, C*K, Ho*Wo), row = c*K + k  (torchvision layout)."""
    sh, sw = _pair(stride); ph, pw = _pair(padding); dh, dw = _pair(dilation)
    B, C, H, W = input.shape
    K = kh * kw
    Ho = (H + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    Wo = (W + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    assert offset.shape[1] == 2 * K, "only offset_groups == 1"
    off = offset.reshape(B, K, 2, Ho * Wo)
    oy = (torch.arange(Ho, dtype=input.dtype) * sh - ph).view(Ho, 1).expand(Ho, Wo).reshape(1, -1)
    ox = (torch.arange(Wo, dtype=input.dtype) * sw - pw).view(1, Wo).expand(Ho, Wo).reshape(1, -1)
    flat = input.reshape(B, C, H * W)
    cols = []
    for k in range(K):
        i, j = divmod(k, kw)
        h = (oy + i * dh) + off[:, k, 0]
        w = (ox + j * dw) + off[:, k, 1]
        val = _bilinear_tv(flat, h, w, H, W)
        if mask is not None:
            val = mask.reshape(B, K, Ho * Wo)[:, k].unsqueeze(1) * val
        cols.append(val)
    return torch.stack(cols, 2).reshape(B, C * K, Ho * Wo), Ho, Wo


def deform_conv2d_torch(input, offset, weight, bias=None, stride=(1, 1), padding=(0, 0),
                        dilation=(1, 1), mask=None):
    """Drop-in for torchvision.ops.deform_conv2d (groups=1, offset_groups=1).
    Per-tap accumulation keeps memory at (C, Ho*Wo) instead of (C*K, Ho*Wo)."""
    sh, sw = _pair(stride); ph, pw = _pair(padding); dh, dw = _pair(dilation)
    B, C, H, W = input.shape
    Cout, Cg, kh, kw = weight.shape
    assert Cg == C, "only weight groups == 1"
    K = kh * kw
    Ho = (H + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    Wo = (W + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    assert offset.shape[1] == 2 * K, "only offset_groups == 1"
    off = offset.reshape(B, K, 2, Ho * Wo)
    msk = mask.reshape(B, K, Ho * Wo) if mask is not None else None
    oy = (torch.arange(Ho, dtype=input.dtype, device=input.device) * sh - ph).view(Ho, 1).expand(Ho, Wo).reshape(1, -1)
    ox = (torch.arange(Wo, dtype=input.dtype, device=input.device) * sw - pw).view(1, Wo).expand(Ho, Wo).reshape(1, -1)
    flat = input.reshape(B, C, H * W)
    wk = weight.reshape(Cout, C, K)
    out = input.new_zeros(B, Cout, Ho * Wo)
    for k in range(K):
        i, j = divmod(k, kw)
        h = (oy + i * dh) + off[:, k, 0]
        w = (ox + j * dw) + off[:, k, 1]
        val = _bilinear_tv(flat, h, w, H, W)
        if msk is not None:
            val = msk[:, k].unsqueeze(1) * val
        out = out + torch.matmul(wk[:, :, k], val)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out.view(B, Cout, Ho, Wo)


def deform_conv2d_naive(input, offset, weight, bias, stride, padding, dilation, mask):
    """Literal scalar transcription of torchvision's documented kernel
    (deformable_im2col_kernel + bilinear_interpolate), float64, Python loops."""
    sh, sw = _pair(stride); ph, pw = _pair(padding); dh, dw = _pair(dilation)
    x = input.double().numpy(); o = offset.double().numpy(); m = mask.double().numpy()
    wgt = weight.double().numpy()
    B, C, H, W = x.shape; Cout, _, kh, kw = wgt.shape
    Ho = (H + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    Wo = (W + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1

    def bil(img, h, w):
        if h <= -1 or H <= h or w <= -1 or W <= w:
            return 0.0
        h_low = math.floor(h); w_low = math.floor(w); h_high = h_low + 1; w_high = w_low + 1
        lh = h - h_low; lw = w - w_low; hh = 1 - lh; hw = 1 - lw
        v1 = img[h_low, w_low] if (h_low >= 0 and w_low >= 0) else 0.0
        v2 = img[h_low, w_high] if (h_low >= 0 and w_high <= W - 1) else 0.0
        v3 = img[h_high, w_low] if (h_high <= H - 1 and w_low >= 0) else 0.0
        v4 = img[h_high, w_high] if (h_high <= H - 1 and w_high <= W - 1) else 0.0
        return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4

    cols = np.zeros((B, C * kh * kw, Ho * Wo))
    for b in range(B):
        for c in range(C):
            for i in range(kh):
                for j in range(kw):
                    k = i * kw + j
                    for oy in range(Ho):
                        for ox in range(Wo):
                            y = (oy * sh - ph) + i * dh + o[b, 2 * k, oy, ox]
                            xx = (ox * sw - pw) + j * dw + o[b, 2 * k + 1, oy, ox]
                            cols[b, c * kh * kw + k, oy * Wo + ox] = m[b, k, oy, ox] * bil(x[b, c], y, xx)
    out = np.einsum("ok,bkn->bon", wgt.reshape(Cout, -1), cols)
    if bias is not None:
        out = out + bias.double().numpy()[None, :, None]
    return torch.from_numpy(out.reshape(B, Cout, Ho, Wo))


def prove_deform(log):
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for (k, pad, H, W) in [(1, 0, 6, 5), (3, 1, 7, 6), (7, 3, 9, 8), (3, 1, 4, 4)]:
        B, C, Co = 2, 3, 4
        x = torch.randn(B, C, H, W, generator=g)
        off = torch.randn(B, 2 * k * k, H, W, generator=g) * 2.5   # many samples leave the image
        off[:, :, 0, 0] = 1.0                                       # exact-integer offsets
        off[:, :, 1, 1] = -1.0                                      # exactly on the -1 boundary
        msk = 2 * torch.sigmoid(torch.randn(B, k * k, H, W, generator=g))
        wt = torch.randn(Co, C, k, k, generator=g); bs = torch.randn(Co, generator=g)
        a = deform_conv2d_torch(x, off, wt, bs, (1, 1), pad, (1, 1), msk).double()
        r = deform_conv2d_naive(x, off, wt, bs, 1, pad, 1, msk)
        e = (a - r).abs().max().item(); worst = max(worst, e)
        log(f"  deform k={k} pad={pad} {H}x{W}: max|torch - naive(doc semantics, f64)| = {e:.3e}")
    # identity case: zero offset, unit mask == plain conv2d
    x = torch.randn(1, 5, 11, 10, generator=g); wt = torch.randn(6, 5, 3, 3, generator=g)
    a = deform_conv2d_torch(x, torch.zeros(1, 18, 11, 10), wt, None, 1, 1, 1, torch.ones(1, 9, 11, 10))
    e0 = (a - F.conv2d(x, wt, None, 1, 1)).abs().max().item()
    log(f"  deform zero-offset/unit-mask vs F.conv2d: {e0:.3e}")
    return max(worst, e0)


# ----------------------------------------------------------------------------
# 2. module stubs so birefnet.py imports without transformers/timm/torchvision/kornia
# ----------------------------------------------------------------------------
def install_stubs():
    tf = types.ModuleType("transformers")

    class PretrainedConfig:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class PreTrainedModel(nn.Module):
        def __init__(self, config=None, *a, **k):
            super().__init__()

        def post_init(self):
            pass

    tf.PretrainedConfig = PretrainedConfig
    tf.PreTrainedModel = PreTrainedModel
    sys.modules["transformers"] = tf

    timm = types.ModuleType("timm"); tl = types.ModuleType("timm.layers")

    class DropPath(nn.Identity):
        def __init__(self, *a, **k):
            super().__init__()

    tl.DropPath = DropPath
    tl.to_2tuple = lambda v: tuple(v) if isinstance(v, (tuple, list)) else (v, v)
    tl.trunc_normal_ = lambda t, std=1.0, **k: nn.init.trunc_normal_(t, std=std)
    timm.layers = tl
    sys.modules["timm"] = timm; sys.modules["timm.layers"] = tl

    tv = types.ModuleType("torchvision"); tvo = types.ModuleType("torchvision.ops")
    tvm = types.ModuleType("torchvision.models")
    tvo.deform_conv2d = deform_conv2d_torch

    def _unused(*a, **k):
        raise RuntimeError("torchvision backbone not used by this config")

    for n in ["vgg16", "vgg16_bn", "resnet50"]:
        setattr(tvm, n, _unused)
    for n in ["VGG16_Weights", "VGG16_BN_Weights", "ResNet50_Weights"]:
        setattr(tvm, n, types.SimpleNamespace(DEFAULT=None))
    tv.ops = tvo; tv.models = tvm
    sys.modules["torchvision"] = tv; sys.modules["torchvision.ops"] = tvo; sys.modules["torchvision.models"] = tvm

    ko = types.ModuleType("kornia"); kf = types.ModuleType("kornia.filters")
    kf.laplacian = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("training only"))
    ko.filters = kf
    sys.modules["kornia"] = ko; sys.modules["kornia.filters"] = kf


def load_birefnet(dtype=torch.float32, device="cpu"):
    install_stubs()
    src = open(os.path.join(MODEL_DIR, "birefnet.py"), encoding="utf-8").read()
    cfg = open(os.path.join(MODEL_DIR, "BiRefNet_config.py"), encoding="utf-8").read()
    src = src.replace("from .BiRefNet_config import BiRefNetConfig", cfg)
    mod = types.ModuleType("birefnet_upstream")
    mod.__file__ = os.path.join(MODEL_DIR, "birefnet.py")
    sys.modules["birefnet_upstream"] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    model = mod.BiRefNet(bb_pretrained=False, config=mod.BiRefNetConfig(bb_pretrained=False))
    from safetensors.torch import load_file
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model, mod, sd


# ----------------------------------------------------------------------------
# 3. ggml-decomposition proofs on the real weights
# ----------------------------------------------------------------------------
def bilinear_ac_tables(n_in, n_out):
    """torch upsample_bilinear2d(align_corners=True) index/weight tables, float32
    exactly as aten computes them (area_pixel_compute_scale + compute_source_index)."""
    scale = np.float32(n_in - 1) / np.float32(n_out - 1) if n_out > 1 else np.float32(0)
    scale = np.float32(scale)
    src = (scale * np.arange(n_out, dtype=np.float32)).astype(np.float32)
    i0 = src.astype(np.int64)
    l1 = np.clip(src - i0.astype(np.float32), 0, 1).astype(np.float32)
    l0 = (np.float32(1) - l1).astype(np.float32)
    i1 = i0 + (i0 < n_in - 1)
    return i0, i1, l0, l1


def interp_ac_gather(x, Ho, Wo):
    """bilinear align_corners=True as GET_ROWS x2 + MUL + ADD per axis (H then W)."""
    B, C, H, W = x.shape
    i0, i1, l0, l1 = bilinear_ac_tables(H, Ho)
    t = x.index_select(2, torch.from_numpy(i0)) * torch.from_numpy(l0).view(1, 1, -1, 1) \
        + x.index_select(2, torch.from_numpy(i1)) * torch.from_numpy(l1).view(1, 1, -1, 1)
    j0, j1, m0, m1 = bilinear_ac_tables(W, Wo)
    return t.index_select(3, torch.from_numpy(j0)) * torch.from_numpy(m0).view(1, 1, 1, -1) \
        + t.index_select(3, torch.from_numpy(j1)) * torch.from_numpy(m1).view(1, 1, 1, -1)


def interp_ac_gather_wfirst(x, Ho, Wo):
    B, C, H, W = x.shape
    j0, j1, m0, m1 = bilinear_ac_tables(W, Wo)
    t = x.index_select(3, torch.from_numpy(j0)) * torch.from_numpy(m0).view(1, 1, 1, -1) \
        + x.index_select(3, torch.from_numpy(j1)) * torch.from_numpy(m1).view(1, 1, 1, -1)
    i0, i1, l0, l1 = bilinear_ac_tables(H, Ho)
    return t.index_select(2, torch.from_numpy(i0)) * torch.from_numpy(l0).view(1, 1, -1, 1) \
        + t.index_select(2, torch.from_numpy(i1)) * torch.from_numpy(l1).view(1, 1, -1, 1)


def swin_tables(H, W, ws, s):
    """Host index tables: (pad + roll(-s) + window_partition) and its inverse, plus shift mask."""
    Hp = -(-H // ws) * ws; Wp = -(-W // ws) * ws
    nWh, nWw = Hp // ws, Wp // ws
    zero_row = H * W
    fwd = np.empty(nWh * nWw * ws * ws, np.int32)
    pos = np.empty((Hp, Wp), np.int64)   # shifted-frame (r,c) -> window-token slot
    for wy in range(nWh):
        for wx in range(nWw):
            for iy in range(ws):
                for ix in range(ws):
                    r = wy * ws + iy; c = wx * ws + ix
                    slot = ((wy * nWw + wx) * ws + iy) * ws + ix
                    pos[r, c] = slot
                    sy = (r + s) % Hp; sx = (c + s) % Wp
                    fwd[slot] = sy * W + sx if (sy < H and sx < W) else zero_row
    inv = np.empty(H * W, np.int32)
    for y in range(H):
        for x in range(W):
            inv[y * W + x] = pos[(y - s) % Hp, (x - s) % Wp]
    mask = None
    if s > 0:
        lab = np.zeros((Hp, Wp), np.int64)
        def region(v, n):
            return 0 if v < n - ws else (1 if v < n - s else 2)
        for r in range(Hp):
            for c in range(Wp):
                lab[r, c] = 3 * region(r, Hp) + region(c, Wp)
        wl = np.empty((nWh * nWw, ws * ws), np.int64)
        for r in range(Hp):
            for c in range(Wp):
                sl = pos[r, c]; wl[sl // (ws * ws), sl % (ws * ws)] = lab[r, c]
        mask = np.where(wl[:, None, :] != wl[:, :, None], -100.0, 0.0).astype(np.float32)  # (nW, Nq, Nk)
    return fwd, inv, mask


def swin_block_gather(blk, x, H, W):
    """SwinTransformerBlock via GET_ROWS tables + MUL_MAT + ADD + SOFT_MAX(mask)."""
    ws = blk.window_size; s = blk.shift_size; a = blk.attn
    C = x.shape[-1]; nH = a.num_heads; hd = C // nH; N = ws * ws
    fwd, inv, mask = swin_tables(H, W, ws, s)
    xn = F.layer_norm(x[0], (C,), blk.norm1.weight, blk.norm1.bias, blk.norm1.eps)
    xz = torch.cat([xn, xn.new_zeros(1, C)], 0)                       # CONCAT zero row
    xw = xz.index_select(0, torch.from_numpy(fwd).long())             # GET_ROWS
    nW = xw.shape[0] // N
    qkv = F.linear(xw, a.qkv.weight, a.qkv.bias).view(nW, N, 3, nH, hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * a.scale, qkv[1], qkv[2]
    bias = a.relative_position_bias_table.index_select(0, a.relative_position_index.view(-1)) \
        .view(N, N, nH).permute(2, 0, 1)                              # GET_ROWS + CONT (const)
    att = q @ k.transpose(-2, -1) + bias.unsqueeze(0)
    if mask is not None:
        att = att + torch.from_numpy(mask).unsqueeze(1)
    att = att.softmax(-1)
    o = (att @ v).transpose(1, 2).reshape(nW * N, C)
    o = F.linear(o, a.proj.weight, a.proj.bias)
    y = o.index_select(0, torch.from_numpy(inv).long())              # GET_ROWS (inverse)
    x2 = x[0] + y
    h = F.layer_norm(x2, (C,), blk.norm2.weight, blk.norm2.bias, blk.norm2.eps)
    h = F.linear(F.gelu(F.linear(h, blk.mlp.fc1.weight, blk.mlp.fc1.bias)), blk.mlp.fc2.weight, blk.mlp.fc2.bias)
    return (x2 + h).unsqueeze(0)


def patch_merge_gather(pm, x, H, W):
    C = x.shape[-1]; H2, W2 = H // 2, W // 2
    idx = np.empty(H2 * W2 * 4, np.int64)
    for y in range(H2):
        for xx in range(W2):
            t = y * W2 + xx
            idx[t * 4 + 0] = (2 * y) * W + 2 * xx
            idx[t * 4 + 1] = (2 * y + 1) * W + 2 * xx
            idx[t * 4 + 2] = (2 * y) * W + 2 * xx + 1
            idx[t * 4 + 3] = (2 * y + 1) * W + 2 * xx + 1
    g = x[0].index_select(0, torch.from_numpy(idx)).reshape(H2 * W2, 4 * C)
    g = F.layer_norm(g, (4 * C,), pm.norm.weight, pm.norm.bias, pm.norm.eps)
    return F.linear(g, pm.reduction.weight).unsqueeze(0)


def build_swin_mask_like_upstream(H, W, ws, s, mod):
    Hp = int(np.ceil(H / ws)) * ws; Wp = int(np.ceil(W / ws)) * ws
    img_mask = torch.zeros((1, Hp, Wp, 1))
    cnt = 0
    for h in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
        for w in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
            img_mask[:, h, w, :] = cnt; cnt += 1
    mw = mod.window_partition(img_mask, ws).view(-1, ws * ws)
    am = mw.unsqueeze(1) - mw.unsqueeze(2)
    return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)


def relu_via_sigmoid(x):
    return x * torch.sigmoid(x * 1e30)


# ----------------------------------------------------------------------------
# 4. Pixal3D preprocessing (pixal3d_image_to_3d.py::preprocess_image, rembg/BiRefNet.py)
# ----------------------------------------------------------------------------
def pixal3d_resize_input(img):
    from PIL import Image
    has_alpha = False
    if img.mode == "RGBA":
        a = np.array(img)[:, :, 3]
        has_alpha = not np.all(a == 255)
    max_size = max(img.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    return img, has_alpha


def birefnet_transform(img_rgb, res):
    from PIL import Image
    r = img_rgb.resize((res, res), Image.BILINEAR)            # transforms.Resize((res,res)) on PIL
    t = torch.from_numpy(np.array(r, dtype=np.uint8)).permute(2, 0, 1).float().div(255)   # ToTensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0), np.array(r)


def pixal3d_post(img_rgb, pred, bg_color=(0, 0, 0)):
    from PIL import Image
    mask_u8 = pred.mul(255).byte().numpy()                   # ToPILImage on float -> mul(255).byte()
    mask_pil = Image.fromarray(mask_u8, mode="L").resize(img_rgb.size)   # default resample (BICUBIC)
    rgba = img_rgb.copy(); rgba.putalpha(mask_pil)
    out_np = np.array(rgba)
    alpha = out_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if bbox.size == 0:   # upstream raises ValueError (np.min of empty) here
        return mask_u8, np.array(mask_pil), out_np, None, None
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.1)
    box = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    crop = np.array(rgba.crop(box)).astype(np.float32) / 255
    rgb, a = crop[:, :, :3], crop[:, :, 3:4]
    bg = np.array(bg_color, dtype=np.float32) / 255.0
    final = (np.clip(rgb * a + bg * (1.0 - a), 0, 1) * 255).astype(np.uint8)
    return mask_u8, np.array(mask_pil), out_np, final, [float(v) for v in box]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=DEFAULT_IMG)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--out", default=os.path.join(HERE, "birefnet_ref_out"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-meta2048", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    def save(name, t):
        a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
        np.save(os.path.join(args.out, name + ".npy"), a)

    log(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
    log("[1] deform_conv2d pure-torch vs documented torchvision semantics")
    dmax = prove_deform(log)

    t0 = time.time()
    model, mod, sd = load_birefnet()
    n_par = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    n_bb = sum(p.numel() for p in model.bb.parameters())
    log(f"[2] loaded (strict) in {time.time()-t0:.1f}s: params {n_par:,} (backbone {n_bb:,}), buffers {n_buf:,}; file tensors {len(sd)}")
    cfg = model.config
    log(f"    config: bb={cfg.bb} mul_scl_ipt={cfg.mul_scl_ipt} cxt={cfg.cxt} dec_att={cfg.dec_att} "
        f"squeeze={cfg.squeeze_block} dec_blk={cfg.dec_blk} dec_ipt_split={cfg.dec_ipt_split} out_ref={cfg.out_ref} "
        f"batch_size={cfg.batch_size}(BN present)")

    # --- decomposition proofs on real weights ---
    log("[3] ggml decomposition proofs")
    g = torch.Generator().manual_seed(1)
    for (hi, wi, ho, wo) in [(32, 32, 64, 64), (256, 256, 1024, 1024), (1024, 1024, 512, 512),
                             (256, 256, 32, 32), (1, 1, 32, 32), (64, 64, 32, 32)]:
        x = torch.randn(1, 3, hi, wi, generator=g)
        ref = F.interpolate(x, size=(ho, wo), mode="bilinear", align_corners=True)
        e1 = (interp_ac_gather(x, ho, wo) - ref).abs().max().item()
        e2 = (interp_ac_gather_wfirst(x, ho, wo) - ref).abs().max().item()
        log(f"  bilinear ac {hi}x{wi}->{ho}x{wo}: max|gather(H then W) - F.interpolate| = {e1:.3e}; (W then H) = {e2:.3e}")
    xr = torch.randn(1 << 20, generator=g) * 10
    xr[:16] = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 1e-20, -1e-20, 3e38, -3e38, 1e-3, -1e-3, 1.0, -1.0, 5.0, -5.0, 1e-38, -1e-38])
    er = (relu_via_sigmoid(xr) - torch.relu(xr)).abs()
    log(f"  relu == x*sigmoid(1e30*x): max abs err {er.max().item():.3e}, nonzero-err count {(er > 0).sum().item()} of {xr.numel()} (only |x|<~1e-28 can differ)")

    # capture a few Swin block inputs for the attention proof
    caps = {}

    def cap(name):
        def h(m, inp, out):
            if name not in caps:
                caps[name] = (inp[0].detach().clone(), m.H, m.W, out.detach().clone())
        return h

    hs = [model.bb.layers[3].blocks[1].register_forward_hook(cap("L3B1")),
          model.bb.layers[2].blocks[1].register_forward_hook(cap("L2B1")),
          model.bb.layers[2].blocks[0].register_forward_hook(cap("L2B0"))]
    pm_caps = {}

    def cap_pm(m, inp, out):
        if "pm" not in pm_caps:
            pm_caps["pm"] = (inp[0].detach().clone(), inp[1], inp[2], out.detach().clone())

    hs.append(model.bb.layers[2].downsample.register_forward_hook(cap_pm))

    # --- Pixal3D preprocessing and forward ---
    from PIL import Image
    img0 = Image.open(args.image)
    img, has_alpha = pixal3d_resize_input(img0)
    log(f"[4] image {os.path.basename(args.image)} {img0.mode} {img0.size} -> Pixal3D-resized {img.size}; has_alpha={has_alpha}"
        + (" (Pixal3D would SKIP BiRefNet)" if has_alpha else ""))
    img_rgb = img.convert("RGB")
    x, resized = birefnet_transform(img_rgb, args.res)
    save("in_rgb_u8", np.array(img_rgb)); save("in_resized_u8", resized); save("in_tensor", x)

    # hooks for oracle taps
    taps = {}
    bb_calls = []

    def bb_hook(m, inp, out):
        bb_calls.append([o.detach().clone() for o in out])

    def tap(name):
        def h(m, inp, out):
            if name in taps:
                return
            taps[name] = {"in": inp[0].detach().clone(), "out": out.detach().clone()}
        return h

    hs.append(model.bb.register_forward_hook(bb_hook))
    hs.append(model.squeeze_module.register_forward_hook(tap("squeeze")))
    for i in (4, 3, 2, 1):
        hs.append(getattr(model.decoder, f"decoder_block{i}").register_forward_hook(tap(f"dec_block{i}")))
    hs.append(model.squeeze_module[0].dec_att.aspp_deforms[2].atrous_conv.register_forward_hook(tap("deform_sq_k7")))
    hs.append(model.decoder.decoder_block1.dec_att.aspp_deforms[1].atrous_conv.register_forward_hook(tap("deform_d1_k3")))
    hs.append(model.bb.layers[0].blocks[0].register_forward_hook(cap("L0B0")))

    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False, depth=3)
    t0 = time.time()
    with torch.no_grad(), fc:
        outs = model(x)
    t_fwd = time.time() - t0
    logits = outs[-1]
    pred = logits.sigmoid()[0].squeeze()
    log(f"    forward {args.res}x{args.res} on CPU fp32: {t_fwd:.1f}s; outputs {len(outs)}: {[tuple(o.shape) for o in outs]}")
    log(f"    logits range [{logits.min().item():.3f}, {logits.max().item():.3f}], pred mean {pred.mean().item():.4f}, "
        f"frac>0.5 {(pred > 0.5).float().mean().item():.4f}")
    tot = fc.get_total_flops()
    log(f"    FLOPs (FlopCounterMode, conv/mm/bmm only, 2*MAC): {tot/1e9:.1f} GFLOP")
    counts = fc.get_flop_counts()
    rows = []
    for k, v in counts.items():
        if k.count(".") <= 2 and k != "Global":
            rows.append((k, sum(v.values())))
    for k, v in sorted(rows, key=lambda r: -r[1])[:40]:
        if v > 0:
            log(f"      {k:55s} {v/1e9:9.2f} GFLOP")
    for h in hs:
        h.remove()

    # deform taps: offset / modulator recomputed outside the FLOP counter
    with torch.no_grad():
        for nm, m in (("deform_sq_k7", model.squeeze_module[0].dec_att.aspp_deforms[2].atrous_conv),
                      ("deform_d1_k3", model.decoder.decoder_block1.dec_att.aspp_deforms[1].atrous_conv)):
            xin = taps[nm]["in"]
            taps[nm]["offset"] = m.offset_conv(xin)
            taps[nm]["mask"] = 2.0 * torch.sigmoid(m.modulator_conv(xin))
    # --- oracle saves ---
    assert len(bb_calls) == 2, len(bb_calls)
    for ci, tag in enumerate(["full", "half"]):
        for li, f in enumerate(bb_calls[ci]):
            save(f"bb_{tag}_x{li+1}", f)
    for k, d in taps.items():
        for kk, vv in d.items():
            save(f"{k}_{kk}", vv)
    save("logits", logits); save("pred", pred)
    mask_u8, mask_full, rgba, final, box = pixal3d_post(img_rgb, pred)
    save("mask_u8", mask_u8); save("mask_fullres_u8", mask_full); save("rgba_u8", rgba)
    Image.fromarray(mask_full).save(os.path.join(args.out, "mask.png"))
    if final is None:
        log("    Pixal3D post: NO pixel with alpha > 204 -> upstream preprocess_image raises ValueError on this input")
    else:
        save("pixal3d_final_rgb_u8", final)
        Image.fromarray(final).save(os.path.join(args.out, "pixal3d_final.png"))
        log(f"    Pixal3D crop box {box} -> final {final.shape}; alpha>204 frac {(mask_full > 204).mean():.4f}")

    # deform tap re-check through the standalone op (proves the in-model op == kernel spec)
    for nm in ("deform_sq_k7", "deform_d1_k3"):
        d = taps[nm]
        m = (model.squeeze_module[0].dec_att.aspp_deforms[2].atrous_conv if nm == "deform_sq_k7"
             else model.decoder.decoder_block1.dec_att.aspp_deforms[1].atrous_conv)
        cols, Ho, Wo = deform_im2col_torch(d["in"], d["offset"], d["mask"], m.regular_conv.kernel_size[0],
                                           m.regular_conv.kernel_size[1], 1, m.padding, 1)
        y = torch.matmul(m.regular_conv.weight.reshape(m.regular_conv.out_channels, -1), cols).view_as(d["out"])
        log(f"    {nm}: in {tuple(d['in'].shape)} offset |max| {d['offset'].abs().max().item():.3f} "
            f"mask [{d['mask'].min().item():.3f},{d['mask'].max().item():.3f}] "
            f"im2col+MUL_MAT vs module max err {(y - d['out']).abs().max().item():.3e}")

    # --- Swin attention via gather tables vs the module (real weights, real activations) ---
    with torch.no_grad():
        for nm in ("L0B0", "L2B0", "L2B1", "L3B1"):
            xin, H, W, yref = caps[nm]
            li, bi = int(nm[1]), int(nm[3])
            blk = model.bb.layers[li].blocks[bi]
            yg = swin_block_gather(blk, xin, H, W)
            e = (yg - yref).abs().max().item(); sc = yref.abs().max().item()
            log(f"  swin {nm} (H=W={H}, ws={blk.window_size}, shift={blk.shift_size}, pad->{-(-H//blk.window_size)*blk.window_size}): "
                f"gather-table block vs module max err {e:.3e} (|y|max {sc:.1f})")
        xin, H, W, yref = pm_caps["pm"]
        e = (patch_merge_gather(model.bb.layers[2].downsample, xin, H, W) - yref).abs().max().item()
        log(f"  patch-merging {H}x{W} via GET_ROWS table: max err {e:.3e}")

    # --- FLOPs at 2048 on the meta device (no compute) ---
    if not args.no_meta2048:
        try:
            model_meta = model.to("meta")
            xm = torch.empty(1, 3, 2048, 2048, device="meta")
            fc2 = FlopCounterMode(display=False)
            with torch.no_grad(), fc2:
                model_meta(xm)
            log(f"    FLOPs @2048 (meta device): {fc2.get_total_flops()/1e9:.1f} GFLOP")
        except Exception as ex:
            log(f"    FLOPs @2048 meta run failed: {type(ex).__name__}: {ex}")

    with open(os.path.join(args.out, "ref_log.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    manifest = {n: list(np.load(os.path.join(args.out, n), mmap_mode="r").shape)
                for n in sorted(os.listdir(args.out)) if n.endswith(".npy")}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    log(f"saved {len(manifest)} arrays to {args.out}")


if __name__ == "__main__":
    main()
