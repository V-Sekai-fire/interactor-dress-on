# Negative controls for census/check_proj_gather.py: perturbed closed forms must FAIL
# the same comparator (worst rel err < 1e-4 means PASS).
import math, runpy, sys
import numpy as np
import torch

S = sys.argv[1]
g = runpy.run_path(S + "/check_proj_gather.py")   # runs the positive check too
upstream, closed_form = g["upstream"], g["closed_form"]


def variant(kind):
    def cf(fmap_hwc, fov, dist, scale, G, R):
        H, W, C = fmap_hwc.shape
        f = R / (2.0 * math.tan(fov / 2.0))
        lin = np.linspace(-1, 1, G)
        i, j, k = np.meshgrid(lin, lin, lin, indexing="ij")
        X, Y, Z = i.ravel() / (2 * scale), j.ravel() / (2 * scale), k.ravel() / (2 * scale)
        depth = (dist + Z) if kind == "depth_sign" else (dist - Z)
        u = R / 2 + f * X / (depth + 1e-8)
        v = (R / 2 + f * Y / (depth + 1e-8)) if kind == "no_yflip" else (R / 2 - f * Y / (depth + 1e-8))
        if kind == "align_corners":
            ix = np.clip((u + 0.5) / R * (W - 1), 0, W - 1)
            iy = np.clip((v + 0.5) / R * (H - 1), 0, H - 1)
        else:
            ix = np.clip((u + 0.5) * W / R - 0.5, 0, W - 1)
            iy = np.clip((v + 0.5) * H / R - 0.5, 0, H - 1)
        x0, y0 = np.floor(ix).astype(int), np.floor(iy).astype(int)
        x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
        wx, wy = ix - x0, iy - y0
        flat = fmap_hwc.reshape(H * W, C)
        if kind == "swap_xy":
            rows = [(x0 * W + y0, (1 - wx) * (1 - wy)), (x1 * W + y0, wx * (1 - wy)),
                    (x0 * W + y1, (1 - wx) * wy), (x1 * W + y1, wx * wy)]
        else:
            rows = [(y0 * W + x0, (1 - wx) * (1 - wy)), (y0 * W + x1, wx * (1 - wy)),
                    (y1 * W + x0, (1 - wx) * wy), (y1 * W + x1, wx * wy)]
        if kind == "nearest":
            rows = [(np.rint(iy).astype(int) * W + np.rint(ix).astype(int), np.ones_like(wx))]
        out = np.zeros((G ** 3, C))
        for idx, w in rows:
            out += flat[idx] * w[:, None]
        return out
    return cf


cfgs = [(16, 512, 32, 0.8575560450553894, 2.0, 1.0), (16, 512, 32, 0.5, 1.094, 1.0),
        (32, 512, 512, 0.7, 1.5, 1.2), (64, 1024, 64, 0.9, 1.2, 0.9)]
all_fail = True
for kind in ["identity", "align_corners", "no_yflip", "depth_sign", "swap_xy", "nearest"]:
    rng = np.random.default_rng(0)
    worst = 0.0
    for (G, R, Hf, fov, dist, scale) in cfgs:
        fm = rng.standard_normal((Hf, Hf, 24)).astype(np.float64)
        a = upstream(torch.from_numpy(fm)[None].float(), fov, dist, scale, G, R).double().numpy()
        b = variant(kind)(fm, fov, dist, scale, G, R)
        worst = max(worst, np.abs(a - b).max() / (np.abs(a).max() + 1e-12))
    verdict = "PASS" if worst < 1e-4 else "FAIL"
    print(f"control {kind:14s} worst={worst:.2e} -> {verdict}")
    if kind != "identity" and verdict == "PASS":
        all_fail = False
print("CONTROLS OK (every perturbation fails)" if all_fail else "CONTROLS BROKEN")
