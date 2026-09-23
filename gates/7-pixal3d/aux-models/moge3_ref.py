"""MoGe-3 (Ruicheng/moge-3-vitl) CPU-torch reference for the camera-FoV path Pixal3D uses.

Runs the dense MoGe-3 graph with refine_steps=0 (no FlexGEMM/Triton sparse refiner), on a
Pixal3D example image preprocessed exactly as Pixal3DImageTo3DPipeline.preprocess_image does,
then the MoGe infer() post-processing (recover_focal_shift -> intrinsics) and Pixal3D's
get_camera_params_wild_moge formulas. Saves inputs, intermediates and outputs as .npy for
ggml oracle use, and checks every non-ggml op decomposition claimed in moge3.md.

No compiled extensions beyond torch are needed. Stubs stand in for modules that v3.py imports
at module scope but never calls on this path (flex_gemm, utils3d, cv2, scipy, huggingface_hub).

Run: C:/interactor-dress-on/.venv-convert/Scripts/python.exe moge3_ref.py
"""
import os, sys, types, struct, zlib, math, json, time

os.environ.setdefault("XFORMERS_DISABLED", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MOGE = r"C:/contract-manifest/3-interactor/moge-upstream"
CKPT = r"C:/interactor-dress-on/models/moge-3-vitl/model.pt"
IMG = r"C:/contract-manifest/3-interactor/pixal3d-upstream/assets/images/21_img.png"
OUT = os.path.join(HERE, "moge3_ref_out")
os.makedirs(OUT, exist_ok=True)
REPORT = {}


def save(name, t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    np.save(os.path.join(OUT, name + ".npy"), a)


# ----------------------------------------------------------------------------------------------
# Import-time stubs (never called with refine_steps=0 and our own post-processing)
# ----------------------------------------------------------------------------------------------
class _Unavailable:
    def __init__(self, *a, **k):
        raise RuntimeError("stub: sparse refiner / compiled dependency not available on this path")


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


_stub("flex_gemm")
_stub("flex_gemm.ops", NeighborCache=_Unavailable)
_fgnn = _stub("flex_gemm.nn")
def _fgnn_getattr(n):
    if n.startswith("__"):
        raise AttributeError(n)
    return _Unavailable


_fgnn.__getattr__ = _fgnn_getattr
_u3 = _stub("utils3d")
_u3.pt = _stub("utils3d.pt")
_u3.np = _stub("utils3d.np")
_stub("cv2")
_stub("huggingface_hub", hf_hub_download=None)  # local checkpoint only
_stub("scipy")
_stub("scipy.signal", fftconvolve=None)
_stub("scipy.optimize", least_squares=None)
sys.path.insert(0, MOGE)

from moge.model.v3 import MoGeModel  # noqa: E402
from moge.utils.geometry_torch import normalized_view_plane_uv  # noqa: E402


# ----------------------------------------------------------------------------------------------
# Minimal PNG decoder (8-bit RGB/RGBA, non-interlaced) -- no Pillow in .venv-convert
# ----------------------------------------------------------------------------------------------
def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if typ == b"IHDR":
            w, h, bd, ct, _, _, il = struct.unpack(">IIBBBBB", body)
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
    assert bd == 8 and il == 0 and ct in (2, 6), (bd, il, ct)
    bpp = 4 if ct == 6 else 3
    raw = zlib.decompress(idat)
    stride = w * bpp
    rows, prev = [], [0] * stride
    for y in range(h):
        base = y * (stride + 1)
        ft = raw[base]
        line = raw[base + 1:base + 1 + stride]
        cur = list(line)
        if ft == 1:
            for i in range(bpp, stride):
                cur[i] = (cur[i] + cur[i - bpp]) & 255
        elif ft == 2:
            cur = [(a + b) & 255 for a, b in zip(line, prev)]
        elif ft == 3:
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif ft == 4:
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (line[i] + pr) & 255
        else:
            assert ft == 0
        rows.append(cur)
        prev = cur
    return np.array(rows, dtype=np.uint8).reshape(h, w, bpp)


# ----------------------------------------------------------------------------------------------
# Pixal3D preprocess_image (RGBA input with real alpha -> no rembg), bg=(0,0,0)
# ----------------------------------------------------------------------------------------------
def pixal3d_preprocess(rgba):
    h, w = rgba.shape[:2]
    assert max(w, h) <= 1024, "example must not need the LANCZOS pre-resize (no Pillow here)"
    alpha = rgba[:, :, 3]
    assert not np.all(alpha == 255)
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.1)
    box = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    x0, y0, x1, y1 = map(int, map(round, box))  # PIL Image.crop rounds the box this way
    crop = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)  # out-of-image area is transparent black
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgba[sy0:sy1, sx0:sx1]
    out = crop.astype(np.float32) / 255
    rgb, a = out[:, :, :3], out[:, :, 3:4]
    bg = np.array((0, 0, 0), dtype=np.float32) / 255.0
    out = rgb * a + bg * (1.0 - a)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), (x0, y0, x1, y1)


# ----------------------------------------------------------------------------------------------
# Host-built index/weight tables (the decompositions used in moge3.md)
# ----------------------------------------------------------------------------------------------
def bilinear_taps(n_in, n_out):
    """torch bilinear, align_corners=False, antialias=False: 2 taps per output."""
    scale = np.float32(n_in) / np.float32(n_out)
    dst = np.arange(n_out, dtype=np.float32)
    src = scale * (dst + np.float32(0.5)) - np.float32(0.5)
    src = np.maximum(src, np.float32(0))
    i0 = src.astype(np.int64)
    i1 = i0 + (i0 < n_in - 1)
    l1 = (src - i0.astype(np.float32)).astype(np.float32)
    l0 = (np.float32(1) - l1).astype(np.float32)
    return i0, i1, l0, l1


def bilinear_resize_taps(x, H, W):
    """x (C,h,w) -> (C,H,W) via GET_ROWS(i0), GET_ROWS(i1), MUL by weights, ADD; W pass then H pass."""
    C, h, w = x.shape
    i0, i1, l0, l1 = bilinear_taps(w, W)
    t = x[:, :, i0] * torch.from_numpy(l0) + x[:, :, i1] * torch.from_numpy(l1)
    j0, j1, m0, m1 = bilinear_taps(h, H)
    return t[:, j0, :] * torch.from_numpy(m0)[:, None] + t[:, j1, :] * torch.from_numpy(m1)[:, None]


def aa_bilinear_matrix(n_in, n_out):
    """torch _upsample_bilinear2d_aa weights (align_corners=False), dense [n_out, n_in].
    Computed in float32 like torch's opmath: float64 tables drift by ~2e-5 at the far edge
    because torch's own source coordinate carries the float32 rounding of n_in/n_out."""
    f = np.float32
    scale = f(n_in) / f(n_out)
    support = scale if scale >= 1 else f(1.0)
    invscale = f(1.0) / scale if scale >= 1 else f(1.0)
    M = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        center = scale * (f(i) + f(0.5))
        xmin = max(int(center - support + f(0.5)), 0)
        xmax = min(int(center + support + f(0.5)), n_in)
        ws = []
        for j in range(xmax - xmin):
            x = abs((f(j) + f(xmin) - center + f(0.5)) * invscale)
            ws.append(f(1) - x if x < 1 else f(0))
        tot = f(0)
        for wv in ws:
            tot = f(tot + wv)
        for j, wv in enumerate(ws):
            M[i, xmin + j] = wv / tot if tot != 0 else wv
    return M


def flop_table(cfg, bh, bw):
    """Analytic FLOPs (2 per MAC) for the dense graph at base grid bh x bw."""
    N = bh * bw + 1
    d, L = 1024, 24
    vit = dict(
        patch_embed=2 * 3 * 14 * 14 * d * (N - 1),
        qkv_proj_mlp=L * 2 * N * (3 * d * d + d * d + 2 * 4 * d * d),
        attention_qk_pv=L * 4 * N * N * d,
        output_projections=4 * 2 * d * d * (N - 1),
    )
    def stack(c):
        D = c["dim_res_blocks"]; din = c["dim_in"]; dout = c["dim_out"] or [None] * 5; nrb = c["num_res_blocks"]; rs = c["resamplers"]
        hw = [bh * bw * 4 ** l for l in range(5)]
        t = dict(input_1x1=0, res3x3=0, resampler=0, output_1x1=0)
        for l in range(5):
            if din[l] is not None:
                t["input_1x1"] += 2 * din[l] * D[l] * hw[l]
            t["res3x3"] += nrb[l] * 2 * (2 * 9 * D[l] * D[l] * hw[l])
            if dout[l] is not None:
                t["output_1x1"] += 2 * D[l] * dout[l] * hw[l]
            if l < 4:
                if rs[l] == "conv_transpose":
                    t["resampler"] += 2 * D[l] * D[l + 1] * 4 * hw[l] + 2 * 9 * D[l + 1] ** 2 * hw[l + 1]
                else:  # bilinear x2 then 3x3 D[l] -> D[l+1]
                    t["resampler"] += 2 * 9 * D[l] * D[l + 1] * hw[l + 1]
        t["total"] = sum(t.values())
        return t
    out = dict(vit=vit, vit_total=sum(vit.values()))
    for k in ("neck", "points_head", "mask_head", "normal_head"):
        out[k] = stack(cfg[k])
    out["scale_head"] = 2 * (1024 * 1024 * 2 + 1024)
    out["total_all_heads"] = out["vit_total"] + sum(out[k]["total"] for k in ("neck", "points_head", "mask_head", "normal_head")) + out["scale_head"]
    out["total_fov_path"] = out["vit_total"] + sum(out[k]["total"] for k in ("neck", "points_head", "mask_head"))
    # sparse final stage: level-4 resampler conv + output 1x1 only at the 2x2 taps of 64x64 samples
    lvl4 = bh * bw * 256
    per_px = lambda c: 2 * 9 * c["dim_res_blocks"][3] * c["dim_res_blocks"][4] + 2 * c["dim_res_blocks"][4] * (c["dim_out"][4] or 0)
    out["fov_level4_dense"] = sum(per_px(cfg[k]) * lvl4 for k in ("points_head", "mask_head"))
    out["fov_level4_sparse_16384px"] = sum(per_px(cfg[k]) * 16384 for k in ("points_head", "mask_head"))
    return out


def bicubic_matrix(n_in, n_out, scale_factor):
    """torch upsample_bicubic2d, align_corners=False, given scale_factor, A=-0.75, clamped taps."""
    A = -0.75
    s = np.float32(1.0 / scale_factor)
    M = np.zeros((n_out, n_in), np.float64)
    for d in range(n_out):
        real = float(np.float32(s * np.float32(d + 0.5) - np.float32(0.5)))
        i = math.floor(real)
        t = real - i
        x1 = t + 1.0
        w0 = ((A * x1 - 5 * A) * x1 + 8 * A) * x1 - 4 * A
        w1 = ((A + 2) * t - (A + 3)) * t * t + 1
        x2 = 1.0 - t
        w2 = ((A + 2) * x2 - (A + 3)) * x2 * x2 + 1
        x3 = x2 + 1.0
        w3 = ((A * x3 - 5 * A) * x3 + 8 * A) * x3 - 4 * A
        for k, wv in zip(range(-1, 3), (w0, w1, w2, w3)):
            M[d, min(max(i + k, 0), n_in - 1)] += wv
    return M


def nearest_idx(n_in, n_out):
    if n_in == n_out:
        return np.arange(n_out)
    scale = np.float32(n_in) / np.float32(n_out)
    return np.minimum(np.floor(np.arange(n_out, dtype=np.float32) * scale).astype(np.int64), n_in - 1)


def relu_22op(x):
    """relu(x) = x * SIGMOID(SCALE(SCALE(x, 2^127), 2^127)): every nonzero finite x maps to |t|>=2^105 or inf."""
    t = x * float(2.0 ** 127)
    t = t * float(2.0 ** 127)
    return x * torch.sigmoid(t)


def up2_bilinear_stencil(x):
    """nn.Upsample(scale 2, bilinear, align_corners=False) via replicate views + SCALE/ADD + interleave CONCAT."""
    def pass_last(x):
        left = torch.cat([x[..., :1], x[..., :-1]], dim=-1)   # clamped x[i-1]
        right = torch.cat([x[..., 1:], x[..., -1:]], dim=-1)  # clamped x[i+1]
        even = left * 0.25 + x * 0.75
        odd = x * 0.75 + right * 0.25
        return torch.stack([even, odd], dim=-1).flatten(-2)   # CONCAT on a unit inner dim, reshape
    return pass_last(pass_last(x).transpose(-1, -2)).transpose(-1, -2)


# ----------------------------------------------------------------------------------------------
# Focal/shift recovery (MoGe recover_focal_shift, scipy-free 1-D Levenberg-Marquardt to convergence)
# ----------------------------------------------------------------------------------------------
def solve_focal_shift(uv, xyz):
    uv = uv.reshape(-1, 2).astype(np.float64)
    xy = xyz[..., :2].reshape(-1, 2).astype(np.float64)
    z = xyz[..., 2].reshape(-1).astype(np.float64)

    def resid(s):
        p = xy / (z + s)[:, None]
        f = (p * uv).sum() / np.square(p).sum()
        return (f * p - uv).ravel()

    s, lam = 0.0, 1e-3
    r = resid(s)
    E = r @ r
    for _ in range(500):
        h = 1e-7 * max(1.0, abs(s))
        J = (resid(s + h) - resid(s - h)) / (2 * h)
        step = -(J @ r) / (J @ J * (1 + lam))
        rn = resid(s + step)
        En = rn @ rn
        if En < E:
            s, r, E, lam = s + step, rn, En, lam * 0.3
            if abs(step) < 1e-13 * max(1.0, abs(s)):
                break
        else:
            lam *= 10
            if lam > 1e12:
                break
    shift = np.float32(s)
    # upstream: optim_focal from float32 xy/z and float32 shift
    xy32, z32, uv32 = xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1), uv.reshape(-1, 2).astype(np.float32)
    p = xy32 / (z32 + shift)[:, None]
    focal = (p * uv32).sum() / np.square(p).sum()
    return shift, np.float32(focal), float(E)


def main():
    torch.set_grad_enabled(False)
    t0 = time.time()
    # ---------------- model ----------------
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    cfg = dict(ck["model_config"])
    cfg.pop("refiner")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MoGeModel(**cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing, missing
    assert all(k.startswith("refiner.") for k in unexpected)
    model.eval()
    n_all = sum(v.numel() for v in ck["model"].values())
    n_dense = sum(p.numel() for p in model.parameters())
    n_fov = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("normal_head", "scale_head")))
    REPORT["params"] = dict(checkpoint_total=n_all, dense_no_refiner=n_dense,
                            refiner=n_all - n_dense, fov_path=n_fov,
                            dtypes=sorted({str(v.dtype) for v in ck["model"].values()}))
    del ck

    # ---------------- input (Pixal3D preprocess) ----------------
    rgba = read_png(IMG)
    rgb_u8, box = pixal3d_preprocess(rgba)
    S_h, S_w = rgb_u8.shape[:2]
    image = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    save("input_rgb_u8", rgb_u8)
    save("input_image_f32_chw", image)
    REPORT["input"] = dict(src=IMG, src_hw=list(rgba.shape[:2]), crop_box=box, moge_input_hw=[S_h, S_w])

    num_tokens = int(cfg["num_tokens_range"][0] + (9 / 9) * (cfg["num_tokens_range"][1] - cfg["num_tokens_range"][0]))
    ar = S_w / S_h
    base_h, base_w = round((num_tokens / ar) ** 0.5), round((num_tokens * ar) ** 0.5)
    REPORT["tokens"] = dict(num_tokens=num_tokens, base_hw=[base_h, base_w], vit_input_hw=[base_h * 14, base_w * 14])

    # ---------------- hooks ----------------
    cap = {}
    bb = model.encoder.backbone
    hooks = [
        bb.patch_embed.register_forward_pre_hook(lambda m, a: cap.__setitem__("image14_norm", a[0].clone())),
        bb.patch_embed.register_forward_hook(lambda m, a, o: cap.__setitem__("patch_embed", o.clone())),
        model.encoder.register_forward_hook(lambda m, a, o: cap.update(enc_feat=o[0].clone(), cls=o[1].clone())),
        model.neck.register_forward_hook(lambda m, a, o: cap.__setitem__("neck", [t.clone() for t in o])),
        model.points_head.register_forward_hook(lambda m, a, o: cap.__setitem__("points_raw", o[-1].clone())),
        model.mask_head.register_forward_hook(lambda m, a, o: cap.__setitem__("mask_raw", o[-1].clone())),
        model.normal_head.register_forward_hook(lambda m, a, o: cap.__setitem__("normal_raw", o[-1].clone())),
        model.scale_head.register_forward_hook(lambda m, a, o: cap.__setitem__("scale_raw", o.clone())),
    ]
    norm_outs = []
    hooks.append(bb.norm.register_forward_hook(lambda m, a, o: norm_outs.append(o.clone())))

    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False, depth=3)
    t1 = time.time()
    with fc:
        out = model.forward(image[None], num_tokens=num_tokens, refine_steps=0)
    t_fwd = time.time() - t1
    for h in hooks:
        h.remove()
    REPORT["cpu_forward_s"] = round(t_fwd, 2)
    flops = fc.get_flop_counts()
    REPORT["flops_counter"] = {k: int(sum(v.values())) for k, v in flops.items() if k.count(".") <= 2}
    REPORT["flops_counter_ops"] = {str(op): int(n) for op, n in flops.get("Global", {}).items()}

    # ---------------- save intermediates ----------------
    save("image14_norm", cap["image14_norm"][0])
    save("patch_embed_tokens", cap["patch_embed"][0])
    pos = bb.interpolate_pos_encoding(torch.zeros(1, base_h * base_w + 1, 1024), base_h * 14, base_w * 14)
    save("pos_embed_interp", pos[0])
    assert len(norm_outs) == 4
    for li, t in zip([5, 11, 17, 23], norm_outs):
        save(f"vit_layer{li}_normed", t[0])
    save("encoder_features", cap["enc_feat"][0])
    save("cls_token", cap["cls"][0])
    for i, t in enumerate(cap["neck"]):
        save(f"neck_level{i}", t[0])
    save("points_head_raw", cap["points_raw"][0])  # (3, 16h, 16w): x/z, y/z, log z
    save("mask_head_raw", cap["mask_raw"][0])
    save("normal_head_raw", cap["normal_raw"][0])
    save("scale_head_raw", cap["scale_raw"][0])
    for k in ("points", "mask", "normal", "metric_scale"):
        save("forward_" + k, out[k][0])

    # ---------------- infer() post-processing (v3, refine_steps=0, fov_x=None) ----------------
    points = out["points"].float()           # (1,H,W,3) affine
    mask_binary = out["mask"].float() > 0.5
    H, W = points.shape[1:3]
    uv = normalized_view_plane_uv(W, H, dtype=torch.float32)
    iy, ix = nearest_idx(H, 64), nearest_idx(W, 64)
    pts_lr = points[0][iy][:, ix].numpy()
    uv_lr = uv[iy][:, ix].numpy()
    m_lr = mask_binary[0][iy][:, ix].numpy()
    shift, focal, E = solve_focal_shift(uv_lr[m_lr], pts_lr[m_lr])
    fx = focal / 2 * (1 + ar ** 2) ** 0.5 / ar
    fy = focal / 2 * (1 + ar ** 2) ** 0.5
    K = np.array([[fx, 0, 0.5], [0, fy, 0.5], [0, 0, 1]], np.float32)
    # Pixal3D get_camera_params_wild_moge
    fx_normalized = K[0, 0]
    camera_angle_x = 2 * math.atan(W / (2 * (fx_normalized * W)))
    rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = torch.tensor([-1.0, 0.0, 0.0]) @ rot.T / 1.0 / 2
    f_pixels = float((16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))) * 512 / 32.0)
    xt = 0.0
    distance = f_pixels * gp[0].item() / (xt - 512 / 2.0) - gp[1].item()
    save("fov_uv_lr", uv_lr)
    save("fov_points_lr", pts_lr)
    save("fov_mask_lr", m_lr)
    save("intrinsics", K)
    REPORT["camera"] = dict(focal_rel_half_diag=float(focal), shift=float(shift), lm_cost=E,
                            n_valid_lr=int(m_lr.sum()), fx_norm=float(fx), fy_norm=float(fy),
                            camera_angle_x_rad=camera_angle_x, camera_angle_x_deg=math.degrees(camera_angle_x),
                            pixal3d_distance=distance, pixal3d_f_pixels_512=f_pixels,
                            metric_scale=float(out["metric_scale"][0]))

    # ---------------- sparse fov subgraph == dense (4096 bilinear samples of the head maps) -------
    ph, pw = cap["points_raw"].shape[-2:]
    j0, j1, m0, m1 = bilinear_taps(ph, H)
    i0, i1, l0, l1 = bilinear_taps(pw, W)
    def sample(raw):  # raw (C, ph, pw) -> (64, 64, C)
        r = raw[:, :, i0[ix]] * torch.from_numpy(l0[ix]) + raw[:, :, i1[ix]] * torch.from_numpy(l1[ix])
        r = r[:, j0[iy], :] * torch.from_numpy(m0[iy])[:, None] + r[:, j1[iy], :] * torch.from_numpy(m1[iy])[:, None]
        return r.permute(1, 2, 0)
    c = sample(cap["points_raw"][0])
    z = torch.exp(c[..., 2:])
    pts_sparse = torch.cat([c[..., :2] * z, z], dim=-1)
    msk_sparse = torch.sigmoid(sample(cap["mask_raw"][0])[..., 0]) > 0.5
    save("fov_sample_rows", np.stack([j0[iy], j1[iy]]))
    save("fov_sample_cols", np.stack([i0[ix], i1[ix]]))
    save("fov_sample_wrows", np.stack([m0[iy], m1[iy]]))
    save("fov_sample_wcols", np.stack([l0[ix], l1[ix]]))
    REPORT["check_sparse_fov_points_maxabs"] = float((pts_sparse - torch.from_numpy(pts_lr)).abs().max())
    REPORT["check_sparse_fov_mask_equal"] = bool((msk_sparse.numpy() == m_lr).all())

    # ---------------- decomposition checks ----------------
    chk = {}
    g = torch.Generator().manual_seed(0)
    # relu exact
    x = torch.randn(1 << 20, generator=g) * torch.exp2(torch.randint(-150, 60, (1 << 20,), generator=g).float())
    x = torch.cat([x, torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1.17549435e-38, -1.17549435e-38, 3.4e38, -3.4e38])])
    chk["relu_22op_bitexact"] = bool(torch.equal(relu_22op(x).abs(), torch.relu(x).abs()))
    # bilinear x2 stencil
    xs = cap["neck"][3][:, :, :64, :64].contiguous()
    ref = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)(xs)
    chk["up2_bilinear_stencil_maxabs"] = float((up2_bilinear_stencil(xs) - ref).abs().max())
    chk["up2_bilinear_ref_absmax"] = float(ref.abs().max())
    # conv_transpose k2 s2 == matmul + pixel shuffle
    ct = model.neck.resamplers[0][0]
    xi = cap["neck"][0][:, :, :8, :8]
    Cin, Cout = ct.weight.shape[:2]
    Wm = ct.weight.permute(1, 2, 3, 0).reshape(Cout * 4, Cin)          # [Cout*kh*kw, Cin]
    y = (Wm @ xi[0].reshape(Cin, -1)).reshape(Cout, 2, 2, 8, 8)       # MUL_MAT
    y = y.permute(0, 3, 1, 4, 2).reshape(Cout, 16, 16) + ct.bias[:, None, None]  # CONT(permute) + ADD
    chk["convT_k2s2_matmul_maxabs"] = float((y - ct(xi)[0]).abs().max())
    chk["convT_k2s2_ref_absmax"] = float(ct(xi).abs().max())
    # replicate pad via CONCAT of views
    xp = cap["neck"][1][:, :, :20, :20]
    padc = torch.cat([xp[..., :1], xp, xp[..., -1:]], -1)
    padc = torch.cat([padc[..., :1, :], padc, padc[..., -1:, :]], -2)
    chk["replicate_pad_concat_equal"] = bool(torch.equal(padc, F.pad(xp, (1, 1, 1, 1), mode="replicate")))
    conv = model.neck.res_blocks[1][0].layers[2]  # 3x3 replicate, 256->256
    chk["conv3x3_replicate_as_pad0_maxabs"] = float((F.conv2d(padc, conv.weight, conv.bias) - conv(xp)).abs().max())
    # bicubic pos-embed (weights-only; bake at conversion)
    M = int(math.sqrt(bb.pos_embed.shape[1] - 1))
    By = torch.from_numpy(bicubic_matrix(M, base_h, (base_h + 0.1) / M)).float()
    Bx = torch.from_numpy(bicubic_matrix(M, base_w, (base_w + 0.1) / M)).float()
    pe = bb.pos_embed[0, 1:].reshape(M, M, -1)
    pe2 = torch.einsum("ij,jkc->ikc", By, torch.einsum("lk,jkc->jlc", Bx, pe)).reshape(-1, pe.shape[-1])
    chk["bicubic_posembed_matrix_maxabs"] = float((pe2 - pos[0, 1:]).abs().max())
    # antialiased bilinear image resize to ViT input (separable host matrices)
    Ry = torch.from_numpy(aa_bilinear_matrix(H, base_h * 14)).float()
    Rx = torch.from_numpy(aa_bilinear_matrix(W, base_w * 14)).float()
    im14 = torch.einsum("ij,cjk,lk->cil", Ry, image, Rx)
    ref14 = F.interpolate(image[None], (base_h * 14, base_w * 14), mode="bilinear", align_corners=False, antialias=True)[0]
    chk["aa_bilinear_matrix_maxabs"] = float((im14 - ref14).abs().max())
    nz = (Ry != 0).sum(1)
    chk["aa_bilinear_taps_per_output"] = [int(nz.min()), int(nz.max())]
    mean = model.encoder.image_mean[0]; std = model.encoder.image_std[0]
    chk["image14_norm_vs_hook_maxabs"] = float(((ref14 - mean) / std - cap["image14_norm"][0]).abs().max())
    # final bilinear resize via 2-tap gathers
    rr = bilinear_resize_taps(cap["points_raw"][0], H, W)
    rref = F.interpolate(cap["points_raw"], (H, W), mode="bilinear", align_corners=False)[0]
    chk["bilinear_resize_taps_maxabs"] = float((rr - rref).abs().max())
    REPORT["decomposition_checks"] = chk
    save("tables_aa_resize_rows", Ry)
    save("tables_aa_resize_cols", Rx)

    REPORT["flops_analytic"] = flop_table(cfg, base_h, base_w)
    REPORT["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(REPORT, f, indent=1, default=str)
    print(json.dumps(REPORT, indent=1, default=str))


if __name__ == "__main__":
    main()
