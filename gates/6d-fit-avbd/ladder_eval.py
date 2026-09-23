"""Gate 6d: shape and gap statistics of AVBD-fitted garments against the
PolyFEM fit and the body.

    python gates/6d-fit-avbd/ladder_eval.py AVATAR.obj REF.obj VOXEL_M GARMENT.obj [GARMENT.obj ...]

AVATAR.obj   the body in body space (project/fixtures/foxgirl/avatar.obj)
REF.obj      the PolyFEM fit of the same authored mesh (gates/8-loop/flat-psd.fitted.obj)
VOXEL_M      one fit voxel in metres: fit_config.json's voxel_size (0.01 solve
             units) over the fit's target_scale (1.5276 for this body, the
             FIT_BEGIN line), 0.0065462 m
GARMENT.obj  each AVBD result (body space), same vertex order and triangles as REF

Prints, per garment: the vertex count and whether the triangles equal REF's;
the per-vertex distance to REF (mean, p50, p95, max, in mm); the surface
distance to REF, both ways (each vertex to the other mesh's nearest
triangle: the shape difference, which a vertex sliding along the surface does
not change, where the per-vertex distance does); the unsigned
point-triangle gap to the body over every vertex (mean, p50, p95, max, in mm
and voxels) and how many vertices sit inside the body (signed by the nearest
triangle's normal, a hint only: fit.elf's fit_check_intersections is the
verdict). REF's own gap is printed first. Distances are exact point-triangle
(gates/6-fit/fit_gap.py's routine, brute force over all body triangles).
"""
import os
import sys

sys.dont_write_bytecode = True  # no __pycache__ under gates/6-fit
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "6-fit"))
from fit_gap import load_obj, point_tri_dist  # noqa: E402


def signed_hint(p, a, b, c):
    """Sign of the nearest triangle's plane at each point (negative = inside);
    a hint for a closed, outward-wound body."""
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1)[:, None]
    cen = (a + b + c) / 3.0
    out = np.empty(len(p))
    for i, q in enumerate(p):
        d2 = ((cen - q) ** 2).sum(1)
        k = int(np.argmin(d2))
        out[i] = np.dot(q - cen[k], n[k])
    return out


def stats(d):
    return "mean %.2f p50 %.2f p95 %.2f max %.2f" % (d.mean(), np.median(d), np.percentile(d, 95), d.max())


def kabsch(src, dst):
    """The best rigid transform src -> dst (rotation as a 3x3 matrix, Kabsch /
    Umeyama with the scale fixed at 1): R, t with dst ~ src @ R.T + t."""
    cs, ct = src.mean(0), dst.mean(0)
    a, b = src - cs, dst - ct
    U, _, Vt = np.linalg.svd(b.T @ a)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1.0
    R = U @ np.diag(d) @ Vt
    return R, ct - R @ cs


def rot_y(src, dst):
    """The best rotation about y (+ translation) src -> dst, as the 3x3
    matrix built from the least-squares 2x2 [[c, s], [-s, c]] in the x-z
    plane (c, s normalised from the sums, never an angle)."""
    cs, ct = src.mean(0), dst.mean(0)
    a, b = src - cs, dst - ct
    c = (a[:, 0] * b[:, 0] + a[:, 2] * b[:, 2]).sum()
    s = (a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]).sum()
    n = np.hypot(c, s)
    c, s = c / n, s / n
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return R, ct - R @ cs


def rot_summary(R):
    """The rotation's 6D truncation (its first two rows) and, as a readable
    summary only, the angle from the trace."""
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    return "rows [%s] [%s] angle %.1f deg" % (" ".join("%.4f" % x for x in R[0]), " ".join("%.4f" % x for x in R[1]), ang)


def main():
    av, af = load_obj(sys.argv[1])
    rv, rf = load_obj(sys.argv[2])
    h = float(sys.argv[3])
    a, b, c = av[af[:, 0]], av[af[:, 1]], av[af[:, 2]]
    ra, rb, rc = rv[rf[:, 0]], rv[rf[:, 1]], rv[rf[:, 2]]
    print("voxel %.7f m; REF %s: %d v %d f" % (h, os.path.basename(sys.argv[2]), len(rv), len(rf)))
    dref = point_tri_dist(rv, a, b, c) * 1000.0
    sref = signed_hint(rv, a, b, c)
    print("REF gap to body: %s mm | %s voxels | inside(hint) %d | y %.3f..%.3f" % (stats(dref), stats(dref / (1000.0 * h)),
          int((sref < 0).sum()), rv[:, 1].min(), rv[:, 1].max()))
    for g in sys.argv[4:]:
        gv, gf = load_obj(g)
        same_tris = gf.shape == rf.shape and bool((gf == rf).all())
        if len(gv) != len(rv):
            print("%s: %d v (REF %d): not comparable" % (g, len(gv), len(rv)))
            continue
        dv = np.linalg.norm(gv - rv, axis=1) * 1000.0
        dg = point_tri_dist(gv, a, b, c) * 1000.0
        sg = signed_hint(gv, a, b, c)
        # The surface distance, both ways: a vertex that slid along the
        # surface counts in the per-vertex distance and not here.
        ds1 = point_tri_dist(gv, ra, rb, rc) * 1000.0
        ds2 = point_tri_dist(rv, gv[gf[:, 0]], gv[gf[:, 1]], gv[gf[:, 2]]) * 1000.0
        ds = np.concatenate([ds1, ds2])
        print("%s: %d v, triangles %s REF | to REF: %s mm | surface to REF surface: %s mm (REF to it: mean %.2f p95 %.2f) | gap to body: %s mm | %s voxels | inside(hint) %d | y %.3f..%.3f" % (
            os.path.basename(g), len(gv), "==" if same_tris else "!=", stats(dv), stats(ds), ds2.mean(),
            np.percentile(ds2, 95), stats(dg), stats(dg / (1000.0 * h)), int((sg < 0).sum()), gv[:, 1].min(),
            gv[:, 1].max()), flush=True)
        # What the per-vertex distance is made of: the residual after the
        # best rotation about y, and after the best rigid transform.
        Ry, ty = rot_y(gv, rv)
        dy = np.linalg.norm(gv @ Ry.T + ty - rv, axis=1) * 1000.0
        Rk, tk = kabsch(gv, rv)
        dk = np.linalg.norm(gv @ Rk.T + tk - rv, axis=1) * 1000.0
        print("    after the best rotation about y (%s): to REF %s mm | after the best rigid transform (%s): %s mm" % (
            rot_summary(Ry), stats(dy), rot_summary(Rk), stats(dk)), flush=True)


if __name__ == "__main__":
    main()
