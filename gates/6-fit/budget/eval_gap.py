"""Fit gap of the loop's garment, the fit budget's quality guard.

    python gates/6-fit/budget/eval_gap.py GARMENT [GARMENT ...]

GARMENT: a fit_native output dir (garment_final.f64, solve frame; the
normalisation is read from its run.log) or a Gate 8 <out>.fitted.obj (body
space). Every garment is taken to the solve frame (x_solve = target_scale *
x_body + center), where cloth-fit's voxel_size (0.01) is defined, and its
vertices' unsigned distance to the FoxGirl fixture avatar (the target at
alpha = 1) is measured in voxels (gates/6-fit/fit_gap.py's point-triangle
distance). The loop's garment has no no-fit list, so every vertex counts.

Guard (the task's quality guard, against the loop's current run,
gates/8-loop/flat.fitted.obj): mean within 0.25 voxel and p95 within 0.5
voxel of the reference.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gates", "6-fit"))
from fit_gap import load_obj, point_tri_dist  # noqa: E402

VOXEL = 0.01
# fit_begin of Gate 8's flat run (the same for every mesh edge: the skeletons set it)
LOOP_NORM = (1.5276081186803154, np.array([-0.00084284095419066558, -1.5261948035249069, -0.0069841535617863326]))
REF = os.path.join(ROOT, "gates", "8-loop", "flat.fitted.obj")


def norm_from_log(path):
    txt = open(path).read()
    m = re.search(r"target_scale ([-0-9.e+]+) center \(([-0-9.e+]+), ([-0-9.e+]+), ([-0-9.e+]+)\)", txt)
    return float(m.group(1)), np.array([float(m.group(i)) for i in (2, 3, 4)])


def garment_solve(g):
    if os.path.isdir(g):
        return np.fromfile(os.path.join(g, "garment_final.f64"), dtype="<f8").reshape(-1, 3), norm_from_log(
            os.path.join(g, "run.log"))
    v, _ = load_obj(g)
    s, c = LOOP_NORM
    return v * s + c, LOOP_NORM


def gap(g, tris):
    v, _ = garment_solve(g)
    return point_tri_dist(v, *tris) / VOXEL


def main():
    av, af = load_obj(os.path.join(ROOT, "project", "fixtures", "foxgirl", "avatar.obj"))
    s, c = LOOP_NORM
    av = av * s + c
    tris = (av[af[:, 0]], av[af[:, 1]], av[af[:, 2]])
    ref = gap(REF, tris)
    rm, rp = ref.mean(), np.percentile(ref, 95)
    print(f"reference {os.path.relpath(REF, ROOT)}: n {len(ref)}  gap mean {rm:.4f}  p95 {rp:.4f}  max {ref.max():.4f} voxels")
    for g in sys.argv[1:]:
        d = gap(g, tris)
        m, p = d.mean(), np.percentile(d, 95)
        ok = abs(m - rm) <= 0.25 and abs(p - rp) <= 0.5
        print(f"{g}: n {len(d)}  gap mean {m:.4f} (d {m - rm:+.4f})  p95 {p:.4f} (d {p - rp:+.4f})  "
              f"max {d.max():.4f} voxels  guard {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
