#!/usr/bin/env python3
"""Gate 6g.0: compare fit_native arms (logs of run_arm.sh) against a reference.

    python analyze.py REF_LOG ARM_LOG [ARM_LOG ...]

Per arm: Newton iterations, final energy / |grad|, intersections, the
per-iteration energy and gradient-norm trajectories ([g6g-it] lines, 17
digits) against the reference, the line-search step sizes, the minimum
distance, and wall time.
"""
import math
import re
import sys

IT = re.compile(r"\[g6g-it\] iter=(\d+) strategy=(\S+) f=(\S+) grad_norm=(\S+)")
LS = re.compile(r"Line search finished \(nan_free_step_size=(\S+) collision_free_step_size=(\S+) descent_step_size=(\S+) final_step_size=(\S+)\)")
PH = re.compile(r"^phase (\d+) \(substep \d+, (\w+)\): (\S+)\s+newton (\d+)\s+minimize (\d+)\s+post_steps (\d+)\s+energy (\S+)\s+\|grad\| (\S+)\s+status (\S+)\s+wall (\S+) s")
MD = re.compile(r"Min distance: (\S+)")
FIN = re.compile(r"^final: .* intersections (\S+)")
EXITL = re.compile(r"^# exit (\d+)\s+wall (\S+) s(.*)")


def parse(path):
    d = {"it": [], "ls": [], "ph": [], "md": [], "final": None, "exit": None, "g6g": []}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = IT.search(line)
        if m:
            d["it"].append((int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))))
            continue
        m = LS.search(line)
        if m:
            d["ls"].append(tuple(float(x) for x in m.groups()))
            continue
        m = MD.search(line)
        if m:
            d["md"].append(float(m.group(1)))
            continue
        m = PH.search(line)
        if m:
            d["ph"].append(dict(phase=int(m.group(1)), kind=m.group(2), ok=m.group(3), newton=int(m.group(4)),
                                energy=float(m.group(7)), grad=float(m.group(8)), status=m.group(9), wall=float(m.group(10))))
            continue
        m = FIN.search(line)
        if m:
            d["final"] = m.group(1)
            continue
        m = EXITL.search(line)
        if m:
            d["exit"] = (int(m.group(1)), float(m.group(2)), m.group(3).strip())
            continue
        if line.startswith("[g6g]"):
            d["g6g"].append(line.rstrip())
    return d


def rel(a, b):
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    ref = parse(sys.argv[1])
    print(f"reference {sys.argv[1]}")
    for path in sys.argv[1:]:
        d = parse(path)
        name = path.replace("\\", "/").split("/")[-1].replace(".log", "")
        ph = d["ph"]
        tot_newton = sum(p["newton"] for p in ph)
        print(f"\n== {name}: exit {d['exit']}  phases {[ (p['phase'], p['newton'], p['energy'], p['grad'], p['status'], p['wall']) for p in ph]}  "
              f"newton {tot_newton}  intersections {d['final']}  iterations logged {len(d['it'])}")
        # trajectories against the reference, index by index
        n = min(len(d["it"]), len(ref["it"]))
        first_diff = None
        first_1e6 = None
        maxf = maxg = 0.0
        rows = []
        for i in range(n):
            a, b = d["it"][i], ref["it"][i]
            rf, rg = rel(a[2], b[2]), rel(a[3], b[3])
            if first_diff is None and (rf > 0 or rg > 0):
                first_diff = i
            if first_1e6 is None and (rf > 1e-6 or rg > 1e-6):
                first_1e6 = i
            maxf, maxg = max(maxf, rf), max(maxg, rg)
            rows.append((i, a[1], a[2], b[2], rf, a[3], b[3], rg))
        print(f"  trajectory vs reference over {n} common records: first differing record {first_diff}, "
              f"first > 1e-6 record {first_1e6}, max rel |f| diff {maxf:.3e}, max rel |grad| diff {maxg:.3e}")
        if path != sys.argv[1] and rows:
            show = [r for r in rows if r[0] < 12 or r[0] == n - 1 or (first_diff is not None and abs(r[0] - first_diff) <= 1)]
            for r in show:
                print(f"    rec {r[0]:3d} {r[1]:>22s}  f {r[2]:.17g} (ref {r[3]:.17g}, rel {r[4]:.2e})  |g| {r[5]:.10g} (ref {r[6]:.10g}, rel {r[7]:.2e})")
        # steps
        if d["ls"]:
            cf = [x[1] for x in d["ls"]]
            fs = [x[3] for x in d["ls"]]
            print(f"  line searches {len(d['ls'])}: collision-free step min {min(cf):.4g} median {sorted(cf)[len(cf)//2]:.4g}  "
                  f"final step min {min(fs):.4g} median {sorted(fs)[len(fs)//2]:.4g}  steps limited by CCD (cf<1) {sum(1 for x in cf if x < 1)}")
            m = min(len(d["ls"]), len(ref["ls"]))
            if path != sys.argv[1] and m:
                ratios = [d["ls"][i][1] / ref["ls"][i][1] for i in range(m) if ref["ls"][i][1] > 0]
                same = sum(1 for i in range(m) if d["ls"][i][1] == ref["ls"][i][1])
                print(f"  collision-free step vs reference, first {m} line searches: identical {same}, ratio min {min(ratios):.4g} max {max(ratios):.4g}")
        if d["md"]:
            print(f"  min distance over iterations: min {min(d['md']):.6g} last {d['md'][-1]:.6g}")
        for line in d["g6g"]:
            if "config" in line or "ccd" in line or "objective" in line:
                print("  " + line)


if __name__ == "__main__":
    main()
