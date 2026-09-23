"""Gate 6d, "Does the rotation generalise": every variant's PolyFEM fit
(fit_native, the solve frame mapped back to body space with the run's
normalisation line) and avbd fit against its input, and against each other.

    python gates/6d-fit-avbd/generalise/analyse.py > gates/6d-fit-avbd/generalise/results.txt

Per variant: PolyFEM's azimuth vs the input (the best rotation about y from
the org's fitter, overall and per 10 cm height band), the avbd fit's
azimuth vs the input, the per-vertex distance avbd -> PolyFEM raw / after
the best rotation about y / after the best rigid transform (Kabsch), the
surface distance both ways, and the walls. Angles are the trace's summary
of the matrix (rule 11). Also the LCL skirt on FoxGirl from Gate 6's native
outputs (C:/b, solve frame, z up), where they exist.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "gates", "6-fit"))
sys.path.insert(0, os.path.join(ROOT, "gates", "6d-fit-avbd"))
from fit_gap import load_obj, point_tri_dist  # noqa: E402
from ladder_eval import sinew_rotation, rot_summary  # noqa: E402

VARIANTS = ["base", "rot+30", "rot-30", "rad0.85", "rad1.15", "len0.85", "len1.15"]


def angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def signed_y_deg(R):
    """The signed angle about y of a rotation about y (its x-z 2x2 block)."""
    return np.degrees(np.arctan2(R[0, 2], R[0, 0]))


def solve_to_body(v, log_path):
    txt = open(log_path).read()
    m = re.search(r"normalisation: target_scale (\S+) center \((\S+), (\S+), (\S+)\)", txt)
    scale = float(m.group(1))
    c = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])
    return (v - c) / scale


def surface(a, af, b, bf):
    d1 = point_tri_dist(a, b[bf[:, 0]], b[bf[:, 1]], b[bf[:, 2]]) * 1000.0
    d2 = point_tri_dist(b, a[af[:, 0]], a[af[:, 1]], a[af[:, 2]]) * 1000.0
    d = np.concatenate([d1, d2])
    return d.mean(), np.percentile(d, 95)


def per_vertex(a, b):
    d = np.linalg.norm(a - b, axis=1) * 1000.0
    return d.mean(), np.percentile(d, 95), d.max()


def walls(path, key):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            t = line.split()
            if len(t) >= 4 and t[0] == key:
                out[t[1]] = int(t[3].split("=")[1])
    return out


def main():
    nw = walls(os.path.join(HERE, "native-runs.txt"), "native")
    print("| variant | PolyFEM azimuth vs input (deg, overall; per band low..high) | avbd azimuth vs input (deg) | avbd vs PolyFEM per vertex raw / after best y rotation (deg) / after Kabsch (mm mean, p95) | surface mean / p95 (mm) | PolyFEM wall (s, Newton) | avbd fit (s) |")
    print("|---|---|---|---|---|---|---|")
    for name in VARIANTS:
        inp, f = load_obj(os.path.join(HERE, "v-%s.obj" % name))
        nat_dir = os.path.join(HERE, "native-%s" % name)
        pv, _ = load_obj(os.path.join(nat_dir, "garment_final.obj"))
        pv = solve_to_body(pv, os.path.join(HERE, "native-%s.log" % name))
        av, _ = load_obj(os.path.join(HERE, "avbd-%s.fitted.obj" % name))
        Rp, _ = sinew_rotation(inp, pv, about_y=True)
        Ra, _ = sinew_rotation(inp, av, about_y=True)
        bands = []
        y = inp[:, 1]
        for lo in np.arange(np.floor(y.min() * 10) / 10, y.max(), 0.1):
            m = (y >= lo) & (y < lo + 0.1)
            if m.sum() >= 30:
                Rb, _ = sinew_rotation(inp[m], pv[m], about_y=True)
                bands.append("%+.1f" % signed_y_deg(Rb))
        raw = per_vertex(av, pv)
        Ry, ty = sinew_rotation(av, pv, about_y=True)
        ry = per_vertex(av @ Ry.T + ty, pv)
        Rk, tk = sinew_rotation(av, pv, about_y=False)
        rk = per_vertex(av @ Rk.T + tk, pv)
        sm, sp = surface(av, f, pv, f)
        newton = 0
        ph = os.path.join(nat_dir, "phases.tsv")
        if os.path.exists(ph):
            newton = sum(int(l.split("\t")[4]) for l in open(ph).read().splitlines()[1:] if l.strip())
        fit_s = ""
        at = os.path.join(HERE, "avbd-%s.txt" % name)
        if os.path.exists(at):
            m2 = re.search(r"wall_ms=([0-9.]+)", open(at).read())
            fit_s = "%.1f" % (float(m2.group(1)) / 1000.0) if m2 else ""
        print("| %s | %+.1f; %s | %+.1f | %.1f / %.1f; %+.1f deg: %.1f / %.1f; %.1f deg: %.1f / %.1f | %.1f / %.1f | %s, %d | %s |" % (
            name, signed_y_deg(Rp), " ".join(bands), signed_y_deg(Ra), raw[0], raw[1], signed_y_deg(Ry), ry[0], ry[1],
            angle_deg(Rk), rk[0], rk[1], sm, sp, str(nw.get(name, "?")), newton, fit_s))
        sys.stdout.flush()
    # The LCL skirt on FoxGirl (Gate 6's native outputs, the solve frame is z up).
    g0p = "C:/b/cf-up-out1/step_garment_0.obj"
    if os.path.exists(g0p):
        g0, _ = load_obj(g0p)
        nofit = set(int(t) for t in open(os.path.join(ROOT, "vendor/cloth-fit/garment-data/assets/garments/LCL_Skirt_DressEvening_003/no-fit.txt")).read().split())
        fit = np.array([i for i in range(len(g0)) if i not in nofit])
        for nm, p in [("LCL skirt, upstream cloth-fit 1 thread (cf-up-out1 step 252)", "C:/b/cf-up-out1/step_garment_252.obj"),
                      ("LCL skirt, fit_native sdf64b", "C:/b/fit-out-sdf64b/garment_final.obj")]:
            if not os.path.exists(p):
                continue
            g, _ = load_obj(p)
            # z up there: swap y and z into the y-up convention for the about-y fit.
            a = g0[:, [0, 2, 1]]
            b = g[:, [0, 2, 1]]
            R, _ = sinew_rotation(a, b, about_y=True)
            Rf, _ = sinew_rotation(a[fit], b[fit], about_y=True)
            print("| %s | %+.1f (fit vertices %+.1f) vs its retargeted start | - | - | - | - | - |" % (nm, signed_y_deg(R), signed_y_deg(Rf)))


if __name__ == "__main__":
    main()
