"""Bending constraints of the fitted skirt at a drape scale: nTarget (double, as
drape_scene.cpp) against the float |s| the kernel computes at rest."""
import math, sys
import numpy as np

obj = sys.argv[1]
V, F = [], []
for ln in open(obj):
    p = ln.split()
    if p and p[0] == "v":
        V.append([float(x) for x in p[1:4]])
    elif p and p[0] == "f":
        F.append([int(x) - 1 for x in p[1:4]])

for scale in [float(s) for s in sys.argv[2:]]:
    P32 = (np.array(V, dtype=np.float32) * np.float32(scale)).astype(np.float32)  # host float * s in float
    P = P32.astype(np.float64)
    edges = {}
    for t in F:
        for a in range(3):
            for b in range(a + 1, 3):
                lo, hi = min(t[a], t[b]), max(t[a], t[b])
                edges.setdefault((lo, hi), []).append(t[3 - (a + b)])
    rows = []
    for (lo, hi), opp in sorted(edges.items()):
        if len(opp) < 2:
            continue
        idx = [lo, hi, opp[0], opp[1]]
        pos = [P[i] for i in idx]
        n = lambda a: math.sqrt(float(np.dot(a, a)))
        l01, l02, l03 = n(pos[1] - pos[0]), n(pos[2] - pos[0]), n(pos[3] - pos[0])
        l12, l13 = n(pos[1] - pos[2]), n(pos[1] - pos[3])
        r0 = 0.5 * (l01 + l02 + l12)
        A0 = math.sqrt(max(0.0, r0 * (r0 - l01) * (r0 - l02) * (r0 - l12)))
        r1 = 0.5 * (l01 + l13 + l03)
        A1 = math.sqrt(max(0.0, r1 * (r1 - l01) * (r1 - l03) * (r1 - l13)))
        c02 = (l01**2 - l02**2 + l12**2) / (4 * A0)
        c12 = (l01**2 + l02**2 - l12**2) / (4 * A0)
        c03 = (l01**2 - l03**2 + l13**2) / (4 * A1)
        c13 = (l01**2 + l03**2 - l13**2) / (4 * A1)
        w = np.array([c02 + c03, c12 + c13, -(c02 + c12), -(c03 + c13)], dtype=np.float32)
        s = sum(pos[i] * float(w[i]) for i in range(4))
        nT = np.float32(n(s))
        # the kernel, float, no contraction
        q = [P32[i] for i in idx]
        sf = ((w[0] * q[0] + w[1] * q[1]) + (w[2] * q[2] + w[3] * q[3])).astype(np.float32)
        lenf = np.float32(np.sqrt(np.float32(np.dot(sf, sf))))
        rows.append((float(nT), float(lenf), float(np.max(np.abs(w))), idx))
    act = [r for r in rows if r[0] > 1e-6]
    small = [r for r in act if r[0] < 1e-4]
    zero = [r for r in act if r[1] == 0.0]
    print(f"scale {scale}: {len(rows)} bendings, {len(act)} active (nTarget > 1e-6), {len(small)} with nTarget < 1e-4, "
          f"float |s| == 0 at rest: {len(zero)}")
    worst = sorted(act, key=lambda r: r[0])[:8]
    for r in worst:
        print(f"   nTarget {r[0]:.3e} float|s| {r[1]:.3e} max|w| {r[2]:.2f} idx {r[3]}")
    ratio = sorted(act, key=lambda r: abs(r[1] - r[0]) / r[0], reverse=True)[:5]
    for r in ratio:
        print(f"   worst rel |s|/nTarget: nTarget {r[0]:.3e} float|s| {r[1]:.3e} idx {r[3]}")
