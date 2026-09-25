"""Gate 0G flat control: the host's OpenUSD (usd-core, Python) on the same
bytes, the same traversal and the same checksum as guest/usd_probe/usd_probe_core.cpp.
Also writes the USDC inputs (Sdf.Layer.Export) the guest then reads from bytes.

  python host_control.py            -> native-control.log (one line per case)

Checksum: BLAKE3 (first 12 hex digits) over, per UsdGeomMesh in Traverse()
order, the prim path string, points (float32 x3, default time) and
faceVertexIndices (int32).
"""
import os, sys
from blake3 import blake3
import numpy as np
from pxr import Usd, Sdf, UsdGeom, UsdSkel

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")
EXT = "C:/contract-manifest/3-interactor/datasource-flow-project/morph_stress_test.usda"
TMP = "C:/b/g0g-inputs"
os.makedirs(TMP, exist_ok=True)


def probe(stage, fmt):
    prims = meshes = skels = roots = npts = nfvi = 0
    h = blake3()
    for prim in stage.Traverse():
        prims += 1
        if prim.IsA(UsdSkel.Skeleton):
            skels += 1
        if prim.IsA(UsdSkel.Root):
            roots += 1
        if not prim.IsA(UsdGeom.Mesh):
            continue
        meshes += 1
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get(Usd.TimeCode.Default())
        fvi = m.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
        p = np.array(pts if pts is not None else [], dtype=np.float32).reshape(-1)
        f = np.array(fvi if fvi is not None else [], dtype=np.int32)
        npts += len(p) // 3
        nfvi += len(f)
        h.update(str(prim.GetPath()).encode())
        h.update(p.tobytes())
        h.update(f.tobytes())
    return (f"ok fmt={fmt} prims={prims} meshes={meshes} skels={skels} skelroots={roots} "
            f"points={npts} fvi={nfvi} mesh_blake3={h.hexdigest()[:12]}")


def run(name, data):
    crate = data[:8] == b"PXR-USDC"
    fmt = "usdc" if crate else "usda"
    try:
        if not crate:
            layer = Sdf.Layer.CreateAnonymous(".usda")
            if not layer.ImportFromString(data.decode("utf-8", "replace")):
                return "ERR: fmt=usda ImportFromString failed"
        else:
            path = os.path.join(TMP, "control_input.usdc")
            open(path, "wb").write(data)
            layer = Sdf.Layer.FindOrOpen(path)
            if layer:
                layer.Reload(True)
        if not layer:
            return f"ERR: fmt={fmt} open failed"
        stage = Usd.Stage.Open(layer, Usd.Stage.LoadAll)
        return probe(stage, fmt)
    except Exception as e:  # Tf errors surface as exceptions in Python
        return f"ERR: fmt={fmt} {str(e).splitlines()[0][:120]}"


def cases():
    out = []
    for n in ["skel_quad", "blendshape_test"]:
        a = os.path.join(INP, n + ".usda")
        c = os.path.join(INP, n + ".usdc")
        if not os.path.exists(c):
            Sdf.Layer.FindOrOpen(a).Export(c)
        out += [(n + ".usda", open(a, "rb").read()), (n + ".usdc", open(c, "rb").read())]
    if os.path.exists(EXT):
        c = os.path.join(TMP, "morph_stress_test.usdc")
        if not os.path.exists(c):
            Sdf.Layer.FindOrOpen(EXT).Export(c)
        out += [("morph_stress_test.usda", open(EXT, "rb").read()),
                ("morph_stress_test.usdc", open(c, "rb").read())]
    good = open(os.path.join(INP, "blendshape_test.usdc"), "rb").read()
    out += [("corrupt.usda", b"#usda 1.0\ndef Mesh \"M\" {\n  point3f[] points = [(1, 2,\n"),
            ("corrupt.usdc", good[: len(good) // 3]),
            ("garbage.bin", bytes(range(256)) * 4)]
    return out


if __name__ == "__main__":
    lines = [f"host OpenUSD {Usd.GetVersion()} (usd-core, Python {sys.version.split()[0]})"]
    for name, data in cases():
        lines.append(f"{name} bytes={len(data)} {run(name, data)}")
    open(os.path.join(HERE, "native-control.log"), "w", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))
