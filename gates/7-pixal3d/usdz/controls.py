"""Controls for smoke.check_usdz: does each check catch the fault it names?

  pixi run -m tools/services/pixal3d python gates/7-pixal3d/usdz/controls.py > gates/7-pixal3d/usdz/controls.log

Builds the stub asset with the server's own writer (_stub_glb -> _to_usdz: a textured
quad, no GPU), checks it (must pass), then breaks one thing at a time and checks that
check_usdz reports it. Needs `pixi run fetch --src` (the patched server under src/).
Exit 0 when the good package passes and every broken one fails for its own reason.
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[3] / "tools" / "services" / "pixal3d"
sys.path[:0] = [str(SERVICE), str(SERVICE / "src" / "service")]
import server  # noqa: E402
from smoke import check_usdz  # noqa: E402
from pxr import Sdf, Usd, UsdShade  # noqa: E402


def repack(out: Path, pkg: Path, names: list[str]) -> Path:
    w = Sdf.ZipFileWriter.CreateNew(str(out))
    for n in names:
        w.AddFile(str(pkg / n), n)
    w.Save()
    return out


def main() -> int:
    work = Path(tempfile.mkdtemp())
    glb = work / "output.glb"
    server._stub_glb(glb)
    good, sizes = server._to_usdz(glb, work)
    pkg = work / "usdz"
    names = list(sizes)  # asset.usdc first, then the textures
    cases = {"good (the server's package)": (good, None)}

    # zip it the ordinary way: deflated, so neither stored nor aligned
    z = work / "deflated.usdz"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for n in names:
            f.write(pkg / n, n)
    cases["deflated zip"] = (z, "compressed")

    # stored but not aligned: a 1-byte extra field shifts every local header's data
    z = work / "unaligned.usdz"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as f:
        for n in names:
            info = zipfile.ZipInfo(n)
            info.extra = b"\x00"
            f.writestr(info, (pkg / n).read_bytes())
    cases["stored, unaligned"] = (z, "not 64-byte aligned")

    # a texture left out of the package
    cases["texture missing"] = (repack(work / "missing.usdz", pkg, names[:-1]), "not a file in the package")

    # the binding removed
    unbound = work / "unbound"
    shutil.copytree(pkg, unbound)
    stage = Usd.Stage.Open(str(unbound / names[0]))
    for prim in stage.Traverse():
        UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
    stage.GetRootLayer().Save()
    cases["material unbound"] = (repack(work / "unbound.usdz", unbound, names), "no material bound")

    # a text layer first
    text = work / "text"
    shutil.copytree(pkg, text)
    Sdf.Layer.FindOrOpen(str(text / names[0])).Export(str(text / "asset.usda"))
    cases["usda root layer"] = (repack(work / "text.usdz", text, ["asset.usda"] + names[1:]),
                                "not a .usdc layer")

    # over budget (a tiny budget stands in for a 20 MB package)
    cases["over budget"] = (good, "budget")

    ok = True
    for label, (path, expect) in cases.items():
        rep = check_usdz(path, budget_b64=1000 if label == "over budget" else 18_000_000)
        caught = [p for p in rep["problems"] if expect and expect in p]
        passed = not rep["problems"] if expect is None else bool(caught)
        ok = ok and passed
        print(f"{'ok  ' if passed else 'FAIL'} {label}: {len(rep['problems'])} problem(s)")
        for p in rep["problems"]:
            print(f"       {p}")
        for v in rep.get("validation", []):
            print(f"       validation: {v}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
