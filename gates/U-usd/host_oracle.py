"""Gate U flat control: the host's OpenUSD (usd-core through Python) on the same
packages usd.elf reads, the same traversal, triangulation and checksums as
guest/usd/usd_core.cpp, plus the derived inputs (a truncated package, a bare
layer, a package with no mesh, garbage).

  timeout 300 python gates/U-usd/host_oracle.py   -> host-oracle.log

Per package one line:
  <name> bytes=B blake3=<12> ok meshes=M points=P triangles=T material=I
  blake3_points=<12> blake3_indices=<12> materials=K textures=N
  tex=<file>:<size>:<b3_12>,... wired=<input>:<file>:<channel>,...
or "<name> bytes=B blake3=<12> ERR: <reason>". Checksums: BLAKE3 (first 12
hex digits) over the points as float32 xyz and over the triangle corners as
int32, per mesh in Traverse() order, each mesh on its own. The guest reports
the same two per mesh, and the gate re-sums the arrays that crossed: all three
must agree.
"""
import io, os, struct, sys, zipfile
from blake3 import blake3
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Sdf

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")
EXT = "C:/b/gu-inputs"  # the real Pixal3D packages (tools/services/pixal3d/runs outputs, not committed)
os.makedirs(EXT, exist_ok=True)


def probe(path):
    st = Usd.Stage.Open(path)
    if st is None:
        return "ERR: no stage"
    meshes = []
    tex = {}
    wired = []
    mats = {}
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts = np.array(m.GetPointsAttr().Get() or [], dtype=np.float32).reshape(-1, 3)
        fvc = list(m.GetFaceVertexCountsAttr().Get() or [])
        fvi = list(m.GetFaceVertexIndicesAttr().Get() or [])
        holes = set(m.GetHoleIndicesAttr().Get() or [])
        flip = m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
        pv = UsdGeom.PrimvarsAPI(prim)
        st_pv = pv.GetPrimvar("st")
        nrm_interp = m.GetNormalsInterpolation() if m.GetNormalsAttr().HasValue() else None
        st_interp = st_pv.GetInterpolation() if st_pv and st_pv.HasValue() else None
        if nrm_interp == "faceVarying" or st_interp == "faceVarying":
            return "ERR: oracle handles per-point attributes only (%s %s)" % (nrm_interp, st_interp)
        tris = []
        c0 = 0
        for f, n in enumerate(fvc):
            if f not in holes and n >= 3:
                for i in range(1, n - 1):
                    a, b, c = fvi[c0], fvi[c0 + i], fvi[c0 + i + 1]
                    if flip:
                        b, c = c, b
                    tris += [a, b, c]
            c0 += n
        tri = np.array(tris, dtype=np.int32)
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        mat_i = -1
        if bound:
            mp = str(bound.GetPath())
            if mp not in mats:
                mats[mp] = len(mats)
                sh = bound.ComputeSurfaceSource()[0]
                if sh:
                    for name in ("diffuseColor", "metallic", "roughness", "opacity", "normal"):
                        inp = sh.GetInput(name)
                        if not inp:
                            continue
                        src = inp.GetConnectedSource()
                        if src and src[0]:
                            f = UsdShade.Shader(src[0].GetPrim()).GetInput("file")
                            ap = f.Get() if f else None
                            file = ap.path if ap else ""
                            wired.append("%s:%s:%s" % (name, file, src[1]))
                            if file and file not in tex:
                                tex[file] = None
            mat_i = mats[mp]
        meshes.append((str(prim.GetPath()), len(pts), len(tri) // 3, mat_i,
                       blake3(pts.tobytes()).hexdigest()[:12], blake3(tri.tobytes()).hexdigest()[:12]))
    # texture bytes straight out of the zip (the guest hands them over as they are)
    texs = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for file in tex:
                try:
                    data = z.read(file)
                    texs.append("%s:%d:%s" % (file, len(data), blake3(data).hexdigest()[:12]))
                except KeyError:
                    texs.append("%s:missing" % file)
    out = "ok meshes=%d" % len(meshes)
    for i, (p, n, t, mi, sp, si) in enumerate(meshes):
        out += " mesh%d=%s points=%d triangles=%d material=%d blake3_points=%s blake3_indices=%s" % (i, p, n, t, mi, sp, si)
    out += " materials=%d textures=%d tex=%s wired=%s" % (len(mats), len(texs), ",".join(texs) or "-", ",".join(wired) or "-")
    return out


def stored_zip(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, data in members:
            z.writestr(name, data)


def derive():
    """The controls: written next to the real packages under EXT."""
    real = os.path.join(EXT, "real-asset.usdz")
    out = []
    if os.path.exists(real):
        data = open(real, "rb").read()
        # 30 % cuts into asset.usdc itself (6.56 MB at offset 64); 60 % keeps the
        # whole crate and loses the textures and the zip's central directory.
        open(os.path.join(EXT, "truncated-30.usdz"), "wb").write(data[: len(data) * 3 // 10])
        open(os.path.join(EXT, "truncated-60.usdz"), "wb").write(data[: len(data) * 6 // 10])
        out += ["truncated-30.usdz", "truncated-60.usdz"]
        with zipfile.ZipFile(real) as z:
            open(os.path.join(EXT, "bare-asset.usdc"), "wb").write(z.read("asset.usdc"))
        out.append("bare-asset.usdc")
    quad = os.path.join(INP, "stub-text.usdz")
    with zipfile.ZipFile(quad) as z:
        open(os.path.join(EXT, "bare-asset.usda"), "wb").write(z.read("asset.usda"))
    out.append("bare-asset.usda")
    nomesh = b'#usda 1.0\n(\n    defaultPrim = "Asset"\n    upAxis = "Y"\n)\ndef Xform "Asset" {\n    def Xform "Geometry" {}\n}\n'
    stored_zip(os.path.join(EXT, "nomesh.usdz"), [("asset.usda", nomesh)])
    out.append("nomesh.usdz")
    open(os.path.join(EXT, "garbage.bin"), "wb").write(bytes(range(256)) * 4)
    out.append("garbage.bin")
    return out


def main():
    print("host OpenUSD %s (usd-core, Python %s)" % (Usd.GetVersion(), sys.version.split()[0]))
    names = ["real-asset.usdz", "real-t2048.usdz"] + derive()
    cases = [(n, os.path.join(EXT, n)) for n in names] + [
        (n, os.path.join(INP, n)) for n in sorted(os.listdir(INP)) if n.endswith(".usdz")]
    for name, path in cases:
        if not os.path.exists(path):
            print("%s missing" % name)
            continue
        data = open(path, "rb").read()
        head = "%s bytes=%d blake3=%s" % (name, len(data), blake3(data).hexdigest()[:12])
        try:
            line = probe(path)
        except Exception as e:  # noqa: BLE001
            line = "ERR: %s" % str(e).splitlines()[0]
        print(head + " " + line)


if __name__ == "__main__":
    main()
