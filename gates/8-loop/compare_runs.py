#!/usr/bin/env python3
"""Gate 8: the VR run must equal the flat run.

    python gates/8-loop/compare_runs.py flat vr        # names under gates/8-loop/

Compares <a>.json with <b>.json and <a>.fitted.obj with <b>.fitted.obj:
- the STATE sequence and the terminal state;
- every integer the gate measures (cycles, openings, patches, curves, knots,
  knot degrees, edges, nodes, strokes; mesh vertices, triangles, loops,
  components; drape vertices and steps; pins);
- the fit_begin / fit phase / check answers with the timings and the
  instruction and heap counters taken out (those are host-timed or count
  host calls);
- the fitted vertices: max |a - b| <= 1e-6 (both are written with 9 decimals).
Prints one line per item and RESULT: PASS | FAIL. Wall times and frame counts
differ by design (the XR frame loop) and are printed, not compared.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TIMING = re.compile(r"(host_ms|wall_ms|ms/step|instructions|last_instructions|heap_used|heap_free|chunks)=?\s*[0-9.]+( MiB)?")


def load(name):
    with open(os.path.join(HERE, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def verts(name):
    out = []
    with open(os.path.join(HERE, name + ".fitted.obj"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                out.extend(float(x) for x in line.split()[1:4])
    return out


def strip(s):
    return TIMING.sub(r"\1=*", str(s))


def main():
    a, b = sys.argv[1], sys.argv[2]
    ja, jb = load(a), load(b)
    ok = True

    def item(name, va, vb):
        nonlocal ok
        same = va == vb
        ok = ok and same
        print("%s %s: %s%s" % ("PASS" if same else "FAIL", name, va, "" if same else " != %s" % (vb,)))

    item("terminal state", (ja["state"], ja["reason"]), (jb["state"], jb["reason"]))
    item("state sequence", [r["state"] for r in ja["records"]], [r["state"] for r in jb["records"]])
    ca, cb = ja.get("counts", {}), jb.get("counts", {})
    for k in ["strokes", "cycles", "openings", "patches", "curves", "knots", "knot_degrees", "edges", "nodes"]:
        item("counts." + k, ca.get(k), cb.get(k))
    ma, mb = ja.get("mesh", {}), jb.get("mesh", {})
    for k in ["vertices", "triangles", "loops", "components"]:
        item("mesh." + k, ma.get(k), mb.get(k))
    item("mesh_build", ma.get("mesh_build"), mb.get("mesh_build"))
    item("pins", ja.get("pins"), jb.get("pins"))
    item("fit_config", ja.get("fit_config"), jb.get("fit_config"))
    item("fit_begin", strip(ja.get("fit_begin")), strip(jb.get("fit_begin")))
    item("fit phases", [strip(p) for p in ja.get("fit_phases", [])], [strip(p) for p in jb.get("fit_phases", [])])
    item("check", ja.get("check"), jb.get("check"))
    item("check_control", ja.get("check_control"), jb.get("check_control"))
    da, db = ja.get("drape", {}), jb.get("drape", {})
    if isinstance(da, dict) and isinstance(db, dict):
        item("drape.vertices", da.get("vertices"), db.get("vertices"))
        item("drape.finite", da.get("finite"), db.get("finite"))
        sa = re.search(r"steps=(\d+)", str(da.get("status"))) if da else None
        sb = re.search(r"steps=(\d+)", str(db.get("status"))) if db else None
        item("drape.steps", sa and sa.group(1), sb and sb.group(1))
        print("INFO drape status: %s | %s" % (da.get("status"), db.get("status")))
    va, vb = verts(a), verts(b)
    if len(va) != len(vb) or not va:
        ok = False
        print("FAIL fitted vertices: %d vs %d floats" % (len(va), len(vb)))
    else:
        d = max(abs(x - y) for x, y in zip(va, vb))
        good = d <= 1e-6
        ok = ok and good
        print("%s fitted vertices: %d, max |a-b| = %.3g (limit 1e-6)" % ("PASS" if good else "FAIL", len(va) // 3, d))
    for r in zip(ja["records"], jb["records"]):
        print("INFO %-13s %9d ms %8d frames | %9d ms %8d frames" % (r[0]["state"], r[0]["ms"], r[0]["frames"],
                                                                    r[1]["ms"], r[1]["frames"]))
    print("INFO wall %.1f s | %.1f s" % (ja["wall_s"], jb["wall_s"]))
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
