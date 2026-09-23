# Check: Pixal3D ProjGrid (verbatim math from
# pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py @cdbb2bb) equals the
# closed-form host table "4 x GET_ROWS + 4 x MUL(bcast) + 3 x ADD" written in proj_attention.md.
import math, sys
import numpy as np
import torch
import torch.nn.functional as F


def upstream(fmap_bhwc, fov, dist, scale, G, R):
    one = torch.linspace(-1, 1, G)
    x, y, z = torch.meshgrid(one, one, one, indexing="ij")
    gp = torch.stack((x, y, z), -1)
    rot = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    gp = torch.matmul(gp, rot.T).reshape(-1, 3)[None]
    gp = gp / torch.tensor([scale])[:, None, None] / 2
    T = torch.tensor([[1.0, 0, 0, 0], [0, 0, -1.0, -2.0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])[None].clone()
    T[:, 1, 3] = -dist
    ph = torch.cat([gp, torch.ones(1, gp.shape[1], 1)], -1)
    w2c = torch.linalg.inv(T)
    pc = torch.bmm(ph, w2c.transpose(-2, -1))[..., :3]
    xc, yc, zc = pc[..., 0], pc[..., 1], pc[..., 2]
    fpx = (16.0 / torch.tan(torch.tensor([fov]) / 2.0)) * R / 32.0
    fpx = fpx[:, None]
    u = fpx * xc / (-zc + 1e-8) + R / 2.0
    v = -(fpx * yc / (-zc + 1e-8)) + R / 2.0
    pts = torch.stack([u, v], -1)
    ndc = (pts + 0.5) / R * 2 - 1
    fm = fmap_bhwc.permute(0, 3, 1, 2)
    out = F.grid_sample(fm, ndc.view(1, -1, 1, 2), mode="bilinear", align_corners=False, padding_mode="border")
    return out.squeeze(-1).permute(0, 2, 1)[0]  # [G^3, C]


def closed_form(fmap_hwc, fov, dist, scale, G, R):
    H, W, C = fmap_hwc.shape
    f = R / (2.0 * math.tan(fov / 2.0))
    lin = np.linspace(-1, 1, G)
    i, j, k = np.meshgrid(lin, lin, lin, indexing="ij")
    X, Y, Z = i.ravel() / (2 * scale), j.ravel() / (2 * scale), k.ravel() / (2 * scale)
    depth = dist - Z                                   # camera at (0,-d,0) world, looking +y
    u = R / 2 + f * X / (depth + 1e-8)
    v = R / 2 - f * Y / (depth + 1e-8)
    ix = np.clip((u + 0.5) * W / R - 0.5, 0, W - 1)   # grid_sample align_corners=False, border
    iy = np.clip((v + 0.5) * H / R - 0.5, 0, H - 1)
    x0, y0 = np.floor(ix).astype(int), np.floor(iy).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    wx, wy = ix - x0, iy - y0
    flat = fmap_hwc.reshape(H * W, C)
    rows = [(y0 * W + x0, (1 - wx) * (1 - wy)), (y0 * W + x1, wx * (1 - wy)),
            (y1 * W + x0, (1 - wx) * wy), (y1 * W + x1, wx * wy)]
    out = np.zeros((G ** 3, C))
    for idx, w in rows:                                  # GET_ROWS, MUL (bcast), ADD
        out += flat[idx] * w[:, None]
    return out


rng = np.random.default_rng(0)
worst = 0.0
for (G, R, Hf, fov, dist, scale) in [(16, 512, 32, 0.8575560450553894, 2.0, 1.0),
                                    (16, 512, 32, 0.5, 1.094, 1.0),
                                    (32, 512, 512, 0.7, 1.5, 1.2),
                                    (64, 1024, 64, 0.9, 1.2, 0.9)]:
    fm = rng.standard_normal((Hf, Hf, 24)).astype(np.float64)
    a = upstream(torch.from_numpy(fm)[None].float(), fov, dist, scale, G, R).double().numpy()
    b = closed_form(fm, fov, dist, scale, G, R)
    err = np.abs(a - b).max() / (np.abs(a).max() + 1e-12)
    worst = max(worst, err)
    print(f"G={G} R={R} fmap={Hf}x{Hf} fov={fov:.4f} d={dist} s={scale}: max rel err {err:.2e}")
print("PASS" if worst < 1e-4 else "FAIL", f"worst={worst:.2e}")
