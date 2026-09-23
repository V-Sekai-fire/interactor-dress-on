"""Fit quality independent of the solve path: how far the garment's fit
vertices (those not in no-fit.txt) sit from the avatar, in voxels.

    python gates/6-fit/fit_gap.py AVATAR.obj NO_FIT.txt VOXEL GARMENT.obj [GARMENT.obj ...]

A GARMENT may also be a .f64 file (row-major xyz doubles, solve frame: what
fit_native and gate_fit.gd write); only its vertices are used.

AVATAR.obj is the avatar in the solve frame (the oracle's step_avatar_<last>.obj;
every run here shares it). Distances are exact point-triangle (numpy, brute
force over all avatar triangles), unsigned. Prints per garment: mean, p50,
p95, max over fit vertices, and the mean over all vertices.
"""
import sys

import numpy as np


def load_obj(path):
    if path.endswith(".f64"):
        return np.fromfile(path, dtype="<f8").reshape(-1, 3), None
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(v), np.array(f, dtype=np.int64)


def point_tri_dist(p, a, b, c):
    """Distance from each point p (n,3) to each triangle (m,3,3); returns (n,) min."""
    # Ericson, Real-Time Collision Detection 5.1.5, vectorised over triangles.
    ab, ac = b - a, c - a
    out = np.empty(len(p))
    for i, q in enumerate(p):
        ap = q - a
        d1 = np.einsum("ij,ij->i", ab, ap)
        d2 = np.einsum("ij,ij->i", ac, ap)
        bp = q - b
        d3 = np.einsum("ij,ij->i", ab, bp)
        d4 = np.einsum("ij,ij->i", ac, bp)
        cp = q - c
        d5 = np.einsum("ij,ij->i", ab, cp)
        d6 = np.einsum("ij,ij->i", ac, cp)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = va + vb + vc
            v = vb / denom
            w = vc / denom
            closest = a + ab * v[:, None] + ac * w[:, None]
            # edge regions
            t_ab = d1 / (d1 - d3)
            e_ab = a + ab * t_ab[:, None]
            t_ac = d2 / (d2 - d6)
            e_ac = a + ac * t_ac[:, None]
            t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            e_bc = b + (c - b) * t_bc[:, None]
        m_a = (d1 <= 0) & (d2 <= 0)
        m_b = (d3 >= 0) & (d4 <= d3)
        m_c = (d6 >= 0) & (d5 <= d6)
        m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        cl = closest
        cl = np.where(m_bc[:, None], e_bc, cl)
        cl = np.where(m_ac[:, None], e_ac, cl)
        cl = np.where(m_ab[:, None], e_ab, cl)
        cl = np.where(m_c[:, None], c, cl)
        cl = np.where(m_b[:, None], b, cl)
        cl = np.where(m_a[:, None], a, cl)
        out[i] = np.sqrt(((cl - q) ** 2).sum(1).min())
    return out


def main():
    av, af = load_obj(sys.argv[1])
    no_fit = set(int(t) for t in open(sys.argv[2]).read().split())
    h = float(sys.argv[3])
    a, b, c = av[af[:, 0]], av[af[:, 1]], av[af[:, 2]]
    for g in sys.argv[4:]:
        gv, _ = load_obj(g)
        fit = np.array([i for i in range(len(gv)) if i not in no_fit])
        d = point_tri_dist(gv[fit], a, b, c) / h
        print(f"{g}: fit verts {len(fit)}  gap mean {d.mean():.4f}  p50 {np.median(d):.4f}  "
              f"p95 {np.percentile(d, 95):.4f}  max {d.max():.4f} voxels", flush=True)


if __name__ == "__main__":
    main()
