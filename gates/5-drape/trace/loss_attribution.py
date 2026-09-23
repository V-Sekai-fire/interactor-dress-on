#!/usr/bin/env python3
"""Gate 5 diagnostic: which trajectory carries the G5 loss gap at mu 0.01?

MATCH_TRAJECTORY (Simulation.cpp:3865-3903): L = sum_i |F_i - T_i|^2 / ((N+1) nV)
over frames 0..N. Three numbers from probe dumps (project/probe_drape_trace.gd)
and the native OBJ frames (the output directory in ../native/source_dir.txt):
  ours(0.01) against ours(0.3)          -- what G5 measures;
  native iter1 (0.01) against ours(0.3) -- native's trajectory, our target;
  and how far ours(0.01) is from native iter1, frame by frame.

    loss_attribution.py <dump mu 0.01> <dump mu 0.3>
"""
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
NAT = pathlib.Path(os.environ.get("NATIVE_OUT", "C:/cloth-dynamics-standalone/output")) / (
    (HERE.parent / "native" / "source_dir.txt").read_text().strip().splitlines()[0].strip())
N, NV = 350, 625


def frames(fn):
    t = open(fn).read()
    return {int(m.group(1)): [float(v) for v in m.group(2).split()] for m in re.finditer(r"^FRAME (\d+) (.*)$", t, re.M)}


def nat(it, k):
    v = []
    for l in open(NAT / it / ("%d.obj" % k)):
        if l.startswith("v "):
            v += [float(x) for x in l.split()[1:4]]
    return v


def loss(x, y):
    k = 1.0 / ((N + 1) * NV)
    return k * sum(sum((a - b) ** 2 for a, b in zip(x[i], y[i])) for i in range(N + 1))


a, t = frames(sys.argv[1]), frames(sys.argv[2])
n1 = {i: nat("iter1", i) for i in range(N + 1)}
print("L(ours mu 0.01, ours mu 0.3)          = %.9g" % loss(a, t))
print("L(native iter1 mu 0.01, ours mu 0.3)  = %.9g   (native's own: 1.65198194)" % loss(n1, t))
for i in (100, 121, 122, 150, 200, 350):
    print("frame %3d: max |ours(0.01) - native iter1| = %.3g" % (i, max(abs(p - q) for p, q in zip(a[i], n1[i]))))
