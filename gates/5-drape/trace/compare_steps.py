#!/usr/bin/env python3
"""Where does drape.elf's sphere demo leave the native run? (Gate 5 diagnostic)

Compares one probe dump (project/probe_drape_trace.gd: every step's statistics
and every frame) with the native reference:
  - the per-step [avbd-step] line of the matching native pass in
    gates/5-drape/native/stdout.log (|dx|_max, |dx|_mean, pred_max, drift_max
    and its vertex, all printed with %g, all computed in float as upstream's
    Simulation.cpp:1497-1545 does) and its [avbd-selfcoll] counts (upstream
    labels them with forwardRecords.size(), one ahead of the step that made
    them);
  - every frame of the native pass's OBJ export, beyond the OBJ's 6-digit print
    resolution (needs the native output directory named in
    gates/5-drape/native/source_dir.txt, under cloth-dynamics-standalone/output
    or $NATIVE_OUT; the repo keeps frames 0/1/10/50/100/350 only).

    compare_steps.py <probe dump> iter0|iter1|iter2|target
"""
import math
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
NATIVE = HERE.parent / "native"
STDOUT = NATIVE / "stdout.log"
PASSES = {"target": (39, 2279), "iter0": (2305, 4560), "iter1": (4568, 6850), "iter2": (6858, 9116)}


def native_dir():
    # The run's output directory under cloth-dynamics-standalone/output
    # (NATIVE_OUT overrides the root).
    name = (NATIVE / "source_dir.txt").read_text().strip().splitlines()[0].strip()
    return pathlib.Path(os.environ.get("NATIVE_OUT", "C:/cloth-dynamics-standalone/output")) / name


def main():
    dump, which = sys.argv[1], sys.argv[2]
    lo, hi = PASSES[which]
    lines = STDOUT.read_text(encoding="utf-8", errors="replace").split("\n")[lo - 1:hi]
    nat, selfc = {}, {}
    for l in lines:
        m = re.search(r"\[avbd-step\] step (\d+) .*\|Δx\|_max=(\S+)\s+\|Δx\|_mean=(\S+)\s+pred_max=(\S+)\s+"
                      r"drift_max=(\S+)\s+drift@(\S+)", l)
        if m:
            nat[int(m.group(1))] = m.groups()[1:]
        m = re.search(r"\[avbd-selfcoll\] step (\d+) resolved (\d+)", l)
        if m:
            selfc[int(m.group(1)) - 1] = int(m.group(2))  # upstream's label is one ahead
    txt = open(dump).read()
    ours = {}
    for m in re.finditer(r"step\s+(\d+) \|dx\|_max=(\S+) \|dx\|_mean=(\S+) pred_max=(\S+) drift_max=(\S+) "
                         r"drift@(\S+) .*self_pushes=(\d+)", txt):
        ours[int(m.group(1))] = m.groups()[1:]
    frames = {int(m.group(1)): [float(v) for v in m.group(2).split()]
              for m in re.finditer(r"^FRAME (\d+) (.*)$", txt, re.M)}
    g = lambda x: "%g" % float(x)
    first = None
    first_push = None
    for k in sorted(ours):
        a, n = ours[k], nat.get(k)
        diffs = []
        if n:
            for name, i, j in (("max", 0, 0), ("mean", 1, 1), ("pred", 2, 2), ("drift", 3, 3)):
                if g(a[i]) != n[j]:
                    diffs.append(name)
            if a[4] != n[4]:
                diffs.append("drift@")
        pushes, npush = int(a[5]), selfc.get(k, 0)
        if pushes != npush:
            diffs.append("pushes")
            if first_push is None:
                first_push = k
        if diffs and first is None:
            first = k
        if k <= 3 or k % 25 == 0 or (first is not None and k <= first + 5) or (pushes or npush) and k < 130:
            print("step %3d |dx|max %s/%s mean %s/%s pushes %d/%d %s" % (
                k, g(a[0]), n[0] if n else "-", g(a[1]), n[1] if n else "-", pushes, npush,
                ("DIFF " + ",".join(diffs)) if diffs else ""))
    print("first step with any printed statistic or push count different: %s (first push-count difference: %s)" % (
        first, first_push))
    nd = native_dir() / which
    if which == "target" or not nd.is_dir():
        print("frames: no native OBJ frames for %s" % which)
        return
    worst = []
    for k in sorted(frames):
        f = nd / ("%d.obj" % k)
        if not f.exists():
            continue
        v = []
        for l in f.read_text().split("\n"):
            if l.startswith("v "):
                v += [float(t) for t in l.split()[1:4]]
        # Beyond the print: |ours - native| minus half a unit of the 6th digit.
        excess = max(abs(a - b) - (0.5 * 10 ** (math.floor(math.log10(abs(b))) - 5) if b else 0.0)
                     for a, b in zip(frames[k], v))
        worst.append((k, max(abs(a - b) for a, b in zip(frames[k], v)), excess))
    first_frame = next((k for k, _, e in worst if e > 1e-6), None)
    for k, d, e in worst:
        if k in (1, 10, 20, 50, 60, 70, 71, 72, 75, 80, 100, 150, 200, 250, 300, 350):
            print("frame %3d max|dx| %.3g (beyond the 6-digit print %.3g)" % (k, d, e))
    print("first frame beyond the print by more than 1e-6: %s" % first_frame)


if __name__ == "__main__":
    main()
