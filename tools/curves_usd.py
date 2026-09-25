"""curves_usd -- CASSIE's .curves strokes <-> OpenUSD (.usda), through the host's OpenUSD.

  python tools/curves_usd.py to-usd <in.curves> <out.usda> [--to-body=tx,ty,tz,s] [--source-rev=SHA]
  python tools/curves_usd.py to-curves <in.usda> <out.curves>

.curves is CASSIE's plain-text stroke list: a "v <count>" line opens each stroke and
<count> "x y z" lines follow. It carries points only, in the Unity app's left-handed,
Y-up canvas space, in metres. The .usda is the stroke format the pen saves: one
BasisCurves prim per stroke (type linear) under /Creation, upAxis Y, metersPerUnit 1, in
the right-handed Body frame, so Z is negated on the way in. A --to-body transform
(translation, then uniform scale about the origin) is recorded in customLayerData with
the flip and the source's BLAKE3, and to-curves undoes both. .curves has no boundary
flag, so no stroke is marked: which cycles are openings is curvenet's decision.
"""
import argparse
import sys

import numpy as np
from blake3 import blake3
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, Vt

ROOT = "/Creation"


class CurvesError(ValueError):
    pass


def parse_curves(text):
    strokes, want, cur = [], 0, []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("v "):
            if len(cur) != want:
                raise CurvesError("line %d: stroke %d has %d of %d points" % (n, len(strokes), len(cur), want))
            if want:
                strokes.append(np.array(cur, dtype=np.float32))
            try:
                want = int(line[2:])
            except ValueError:
                raise CurvesError("line %d: bad count %r" % (n, line))
            if want < 2:
                raise CurvesError("line %d: a stroke needs 2 or more points, not %d" % (n, want))
            cur = []
            continue
        parts = line.split()
        if len(parts) != 3 or not want:
            raise CurvesError("line %d: expected 'x y z' inside a stroke, got %r" % (n, line))
        if len(cur) == want:
            raise CurvesError("line %d: stroke %d has more than %d points" % (n, len(strokes), want))
        cur.append([float(p) for p in parts])
    if len(cur) != want:
        raise CurvesError("stroke %d ends with %d of %d points" % (len(strokes), len(cur), want))
    if want:
        strokes.append(np.array(cur, dtype=np.float32))
    return strokes


def format_curves(strokes):
    # %.9g round-trips every float32 exactly.
    out = []
    for s in strokes:
        out.append("v %d" % len(s))
        out.extend("%.9g %.9g %.9g" % tuple(float(c) for c in p) for p in s)
    return "\n".join(out) + "\n"


def to_body(strokes, t=(0.0, 0.0, 0.0), scale=1.0):
    t = np.array(t, dtype=np.float32)
    return [((s * np.float32([1, 1, -1])) + t) * np.float32(scale) for s in strokes]


def from_body(strokes, t=(0.0, 0.0, 0.0), scale=1.0):
    t = np.array(t, dtype=np.float32)
    return [((s / np.float32(scale)) - t) * np.float32([1, 1, -1]) for s in strokes]


def write_usda(strokes, names=None, boundary=(), meta=None):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    if meta:
        stage.GetRootLayer().customLayerData = meta
    marked = set(boundary)
    for i, s in enumerate(strokes):
        if len(s) < 2:
            raise CurvesError("stroke %d has %d point(s); a stroke needs 2 or more" % (i, len(s)))
        name = names[i] if names else "stroke_%03d" % i
        if not Sdf.Path.IsValidIdentifier(name):
            raise CurvesError("stroke %d: %r is not a valid prim name" % (i, name))
        c = UsdGeom.BasisCurves.Define(stage, "%s/%s" % (ROOT, name))
        c.CreateTypeAttr(UsdGeom.Tokens.linear)
        c.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in s]))
        c.CreateCurveVertexCountsAttr(Vt.IntArray([len(s)]))
        if i in marked:
            pv = UsdGeom.PrimvarsAPI(c).CreatePrimvar("boundary", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.uniform)
            pv.Set(True)
    return stage.GetRootLayer().ExportToString()


def read_usda(text):
    layer = Sdf.Layer.CreateAnonymous(".usda")
    try:
        ok = layer.ImportFromString(text)
    except Tf.ErrorException as e:
        raise CurvesError("not a USD layer: %s" % str(e).strip().splitlines()[-1])
    if not ok:
        raise CurvesError("not a USD layer")
    stage = Usd.Stage.Open(layer)
    strokes, names, boundary = [], [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.BasisCurves):
            continue
        c = UsdGeom.BasisCurves(prim)
        pts = np.array(c.GetPointsAttr().Get() or [], dtype=np.float32).reshape(-1, 3)
        counts = list(c.GetCurveVertexCountsAttr().Get() or [])
        if sum(counts) != len(pts):
            raise CurvesError("%s: curveVertexCounts sum to %d, points hold %d" % (prim.GetPath(), sum(counts), len(pts)))
        pv = UsdGeom.PrimvarsAPI(c).GetPrimvar("boundary")
        mark = bool(pv.Get()) if pv and pv.HasAuthoredValue() else False
        at = 0
        for k, n in enumerate(counts):
            if n < 2:
                raise CurvesError("%s curve %d has %d point(s)" % (prim.GetPath(), k, n))
            if mark:
                boundary.append(len(strokes))
            strokes.append(pts[at:at + n])
            names.append(prim.GetName() if len(counts) == 1 else "%s_%d" % (prim.GetName(), k))
            at += n
    return strokes, names, boundary, dict(stage.GetRootLayer().customLayerData)


def main(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("to-usd")
    a.add_argument("src")
    a.add_argument("dst")
    a.add_argument("--to-body", default="0,0,0,1", help="tx,ty,tz,scale after the Z flip")
    a.add_argument("--source-rev", default="")
    b = sub.add_parser("to-curves")
    b.add_argument("src")
    b.add_argument("dst")
    o = ap.parse_args(argv)
    if o.cmd == "to-usd":
        raw = open(o.src, "rb").read()
        tx, ty, tz, s = (float(v) for v in o.to_body.split(","))
        strokes = to_body(parse_curves(raw.decode("utf-8")), (tx, ty, tz), s)
        meta = {"source": o.src.replace("\\", "/").split("/")[-1], "source_blake3": blake3(raw).hexdigest()[:12],
                "source_rev": o.source_rev, "converter": "dress-on tools/curves_usd.py",
                "from_frame": "unity left-handed y-up canvas, metres", "flip": "z",
                "to_body_translate": Gf.Vec3d(tx, ty, tz), "to_body_scale": s}
        open(o.dst, "w").write(write_usda(strokes, meta=meta))
        print("ok %d strokes, %d points -> %s" % (len(strokes), sum(len(x) for x in strokes), o.dst))
    else:
        strokes, _, _, meta = read_usda(open(o.src).read())
        t = tuple(meta.get("to_body_translate", (0.0, 0.0, 0.0)))
        s = float(meta.get("to_body_scale", 1.0))
        if meta.get("flip", "z") == "z":
            strokes = from_body(strokes, t, s)
        open(o.dst, "w").write(format_curves(strokes))
        print("ok %d strokes -> %s" % (len(strokes), o.dst))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except CurvesError as e:
        print("ERR: %s" % e)
        sys.exit(1)
