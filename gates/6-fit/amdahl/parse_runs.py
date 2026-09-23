#!/usr/bin/env python3
"""Summarise runs/p0-*.log (run_p0.sh) into the tables of measure.md.

  python parse_runs.py runs/p0-a.log [runs/p0-b.log ...] [--md]

Per run: exit/timeout, Newton, minimizes, final energy, |grad|, the
intersection check, phase wall/cpu, polysolve's own timers summed over the
minimizes, the descent-strategy fallbacks, and the [amdahl] counters
(n, nnz(H), linear iterations per solve, CCD candidates and active
constraints per build).
"""
import re
import statistics
import sys

PH = re.compile(r"^phase 0 \(substep \d+, AL\): (\S+).*?newton (\d+)\s+minimize (\d+)\s+post_steps (\d+)\s+energy (\S+)\s+\|grad\| (\S+)\s+status (\S+)\s+wall ([\d.]+) s\s+cpu ([\d.]+) s")
FIN = re.compile(r"^final: .*intersections (\S+)")
EXIT = re.compile(r"^# exit (\d+)\s+wall ([\d.]+) s(.*)")
TIM = re.compile(r"\[timing\S*\] f: (\S+)s, grad_f: (\S+)s, update_direction: (\S+)s, line_search: (\S+)s, constraint_set_update: (\S+)s")
TIM2 = re.compile(r"\[timing\S*\]\[([^\]]+)\] assembly: (\S+)s; linear_solve: (\S+)s")
TIM3 = re.compile(r"\[timing\S*\]\[Backtracking\] constraint_set_update (\S+)s, checking_for_nan_inf (\S+)s, broad_phase_ccd (\S+)s, narrow_phase_ccd (\S+)s, classical_line_search (\S+)s")
LIN = re.compile(r"^\[amdahl\] linsolve (.+?) n (\d+) nnz (\d+) residual (\S+) info (.*)$")
CAND = re.compile(r"^\[amdahl\] candidates (\w+) vv (\d+) ev (\d+) ee (\d+) fv (\d+)")
COL = re.compile(r"^\[amdahl\] collisions vv (\d+) ev (\d+) ee (\d+) fv (\d+) pv (\d+)")
REVERT = re.compile(r"reverting to (.+)$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def stats(xs):
    if not xs:
        return "-"
    xs = sorted(xs)
    return f"{xs[0]} / {statistics.median(xs):g} / {xs[-1]} (mean {statistics.mean(xs):.0f}, n={len(xs)})"


def parse(path):
    r = {"path": path, "timing": [0.0] * 5, "asm": {}, "bt": [0.0] * 5, "lin": [], "cand": {"ccd": [], "static": []},
         "col": [], "reverts": {}, "phase": None, "final": None, "exit": None}
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw.rstrip("\n"))
        m = PH.search(line)
        if m:
            r["phase"] = m.groups()
        m = FIN.search(line)
        if m:
            r["final"] = m.group(1)
        m = EXIT.search(line)
        if m:
            r["exit"] = (int(m.group(1)), float(m.group(2)), m.group(3).strip())
        m = TIM.search(line)
        if m:
            r["timing"] = [a + float(b) for a, b in zip(r["timing"], m.groups())]
        m = TIM2.search(line)
        if m:
            k = m.group(1)
            a, l = r["asm"].get(k, (0.0, 0.0))
            r["asm"][k] = (a + float(m.group(2)), l + float(m.group(3)))
        m = TIM3.search(line)
        if m:
            r["bt"] = [a + float(b) for a, b in zip(r["bt"], m.groups())]
        m = LIN.search(line)
        if m:
            info = m.group(5)
            it = re.search(r'"solver_iter":(\d+)', info)
            r["lin"].append((m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), int(it.group(1)) if it else None))
        m = CAND.search(line)
        if m:
            r["cand"][m.group(1)].append(tuple(int(x) for x in m.groups()[1:]))
        m = COL.search(line)
        if m:
            r["col"].append(tuple(int(x) for x in m.groups()))
        m = REVERT.search(line)
        if m:
            k = m.group(1)
            r["reverts"][k] = r["reverts"].get(k, 0) + 1
    return r


def report(r):
    name = r["path"].split("/")[-1].split("\\")[-1]
    print(f"== {name}")
    if r["exit"]:
        print(f"exit {r['exit'][0]} wall {r['exit'][1]} s {r['exit'][2]}")
    if r["phase"]:
        st, nw, mi, ps, en, gn, status, wall, cpu = r["phase"]
        print(f"phase 0: {st} newton {nw} minimize {mi} energy {en} |grad| {gn} status {status} wall {wall} s cpu {cpu} s; intersections {r['final']}")
    else:
        print("phase 0: did not finish")
    t = r["timing"]
    print(f"polysolve timers (sum over minimizes): f {t[0]:.2f}  grad_f {t[1]:.2f}  update_direction {t[2]:.2f}  line_search {t[3]:.2f}  constraint_set_update {t[4]:.2f}")
    for k, (a, l) in r["asm"].items():
        print(f"  [{k}] assembly {a:.2f}  linear_solve {l:.2f}")
    b = r["bt"]
    print(f"  [Backtracking] constraint_set_update {b[0]:.2f}  nan_inf {b[1]:.2f}  broad_phase_ccd {b[2]:.2f}  narrow_phase_ccd {b[3]:.2f}  classical_line_search {b[4]:.2f}")
    if r["reverts"]:
        print("  fallbacks: " + ", ".join(f"-> {k} x{v}" for k, v in r["reverts"].items()))
    if r["lin"]:
        ns = sorted({x[1] for x in r["lin"]})
        print(f"linear solves {len(r['lin'])}: n {ns}; nnz(H) min/median/max {stats([x[2] for x in r['lin']])}")
        by = {}
        for s, n, nnz, res, it in r["lin"]:
            by.setdefault(s, []).append((it, res))
        for s, v in by.items():
            its = [i for i, _ in v if i is not None]
            capped = sum(1 for i in its if i >= 1000)
            print(f"  {s}: solves {len(v)}; iterations {stats(its)}; at max_iter 1000: {capped}")
    for tag in ("ccd", "static"):
        c = r["cand"][tag]
        if c:
            tot = [sum(x) for x in c]
            print(f"candidates {tag} builds {len(c)}: total {stats(tot)}; ee {stats([x[2] for x in c])}; fv {stats([x[3] for x in c])}; ev {stats([x[1] for x in c])}; vv {stats([x[0] for x in c])}")
    if r["col"]:
        tot = [sum(x) for x in r["col"]]
        print(f"active constraints (Collisions::build) {len(r['col'])}: total {stats(tot)}; vv {stats([x[0] for x in r['col']])}; ev {stats([x[1] for x in r['col']])}; ee {stats([x[2] for x in r['col']])}; fv {stats([x[3] for x in r['col']])}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if p.startswith("--"):
            continue
        report(parse(p))
        print()
