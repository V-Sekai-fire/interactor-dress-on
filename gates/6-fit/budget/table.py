"""The fit budget table from native runs (run_native.sh outputs).

    python gates/6-fit/budget/table.py NAME [NAME ...]   (default: every C:/b/budget/* dir)

Per candidate: fit_native wall and cpu seconds (begin + phases), Newton per
phase, final energy, intersections, the fit gap (eval_gap.py) and the guard
against the loop's current run: intersection-free, gap mean within 0.25 and
p95 within 0.5 voxel of gates/8-loop/flat.fitted.obj, final energy within 10%
of its 0.011207282055103298. Also the reduced phase's timing split from
polysolve's [timing] lines (assembly, linear solve, line search with its
broad-phase CCD, narrow-phase CCD and constraint-set update), cpu seconds.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_gap  # noqa: E402

OUT = os.environ.get("BUDGET_OUT", "C:/b/budget")
E_REF = 0.011207282055103298
BASE = os.environ.get("BUDGET_BASE", "base-e030")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse(d):
    txt = ANSI.sub("", open(os.path.join(d, "run.log"), errors="replace").read())
    r = {"phases": [], "ok": "final:" in txt}
    for m in re.finditer(r"^phase (\d) .*?: (\S+).*?newton (\d+)\s+minimize (\d+).*?energy (\S+).*?wall ([\d.]+) s\s+cpu ([\d.]+) s",
                         txt, re.M):
        r["phases"].append((int(m.group(3)), int(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7)), m.group(2)))
    m = re.search(r"time: begin\+phases wall ([\d.]+) s, cpu ([\d.]+) s", txt)
    r["wall"], r["cpu"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"^final: .*intersections (\S+)", txt, re.M)
    r["inter"] = m.group(1) if m else "?"
    m = re.search(r"garment (\d+) v", txt)
    r["nv"] = int(m.group(1)) if m else 0
    # timing split of the last [timing] block (the reduced phase; AL minimizes that
    # hit their cap throw before logging theirs)
    asm = lin = 0.0
    for m in re.finditer(r"\[timing\]\[\S+[^\]]*\] assembly: (\S+)s; linear_solve: (\S+)s", txt):
        asm += float(m.group(1))
        lin += float(m.group(2))
    r["asm"], r["lin"] = asm, lin
    m = re.findall(r"\[timing\] f: (\S+)s, grad_f: (\S+)s, update_direction: (\S+)s, line_search: (\S+)s", txt)
    r["ls"] = float(m[-1][3]) if m else 0.0
    m = re.findall(r"\[timing\]\[Backtracking\] constraint_set_update (\S+)s, checking_for_nan_inf (\S+)s, broad_phase_ccd (\S+)s, narrow_phase_ccd (\S+)s", txt)
    r["bp"], r["np"], r["csu"] = (float(m[-1][2]), float(m[-1][3]), float(m[-1][0])) if m else (0, 0, 0)
    return r


def main():
    names = sys.argv[1:] or sorted(os.listdir(OUT))
    av, af = eval_gap.load_obj(os.path.join(eval_gap.ROOT, "project", "fixtures", "foxgirl", "avatar.obj"))
    s, c = eval_gap.LOOP_NORM
    av = av * s + c
    tris = (av[af[:, 0]], av[af[:, 1]], av[af[:, 2]])
    ref = eval_gap.gap(eval_gap.REF, tris)
    rm, rp = ref.mean(), np.percentile(ref, 95)
    print(f"reference gates/8-loop/flat.fitted.obj (guest, 1393 s, 253 Newton): gap mean {rm:.3f} p95 {rp:.3f} voxels, energy {E_REF:.6g}")
    bd = os.path.join(OUT, BASE)
    bg = eval_gap.gap(bd, tris)
    bm, bp_, be = bg.mean(), np.percentile(bg, 95), parse(bd)["phases"][-1][2]
    print(f"native baseline {BASE}: gap mean {bm:.3f} p95 {bp_:.3f} voxels, energy {be:.6g}")
    print("| candidate | verts | native wall s | cpu s | Newton (AL / reduced) | energy (d% guest / native base) | gap mean / p95 (voxels) | intersections | guard guest / native base | reduced phase: assembly / LDLT / line search (broad CCD, narrow CCD, constraint set) s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        d = os.path.join(OUT, n)
        if not os.path.exists(os.path.join(d, "run.log")):
            continue
        r = parse(d)
        if not r["phases"] or not os.path.exists(os.path.join(d, "garment_final.f64")):
            print(f"| {n} | {r['nv']} | FAIL | | | | | | FAIL | |")
            continue
        g = eval_gap.gap(d, tris)
        m, p = g.mean(), np.percentile(g, 95)
        e = r["phases"][-1][2]
        de = (e / E_REF - 1) * 100
        ok = r["ok"] and r["inter"] == "none" and abs(m - rm) <= 0.25 and abs(p - rp) <= 0.5 and abs(de) <= 10
        dn = (e / be - 1) * 100
        okn = r["ok"] and r["inter"] == "none" and abs(m - bm) <= 0.25 and abs(p - bp_) <= 0.5 and abs(dn) <= 10
        nt = " / ".join(f"{ph[0]}" + (f" ({ph[1]} min)" if i == 0 else "") for i, ph in enumerate(r["phases"]))
        print(f"| {n} | {r['nv']} | {r['wall']:.1f} | {r['cpu']:.1f} | {nt} | {e:.6g} ({de:+.1f}% / {dn:+.1f}%) | {m:.3f} / {p:.3f} | "
              f"{r['inter']} | {'PASS' if ok else 'FAIL'} / {'PASS' if okn else 'FAIL'} | {r['asm']:.1f} / {r['lin']:.1f} / {r['ls']:.1f} ({r['bp']:.1f}, {r['np']:.1f}, {r['csu']:.1f}) |",
              flush=True)


if __name__ == "__main__":
    main()
