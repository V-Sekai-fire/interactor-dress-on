"""Re-check the real packages under tools/services/pixal3d/runs (gitignored outputs of
`pixi run smoke` and the texture sweep) with the current check_usdz + compare_glb,
including the st convention against the GLB's raw TEXCOORD_0 bytes.

  pixi run -m tools/services/pixal3d python gates/7-pixal3d/usdz/recheck.py > gates/7-pixal3d/usdz/recheck.log
"""
import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[3] / "tools" / "services" / "pixal3d"
sys.path.insert(0, str(SERVICE))
from smoke import RUNS, check_usdz, compare_glb  # noqa: E402

PAIRS = [("real-asset.usdz", "real-output.glb")] + [
    (f"real-sweep-asset-t{t}.usdz", f"real-sweep-output-t{t}.glb") for t in (2048, 1536, 1024)]

for usdz, glb in PAIRS:
    rep = check_usdz(RUNS / usdz)
    cmp, more = compare_glb(RUNS / usdz, RUNS / glb)
    tex = {k.split("/")[-1]: (v.get("size"), v.get("bytes")) for k, v in rep["textures"].items()}
    print(f"{usdz}: {rep['bytes']} B, base64 {rep['b64_bytes']} B, layer {rep['share']['layer']:.3f} "
          f"textures {rep['share']['textures']:.3f}; {rep['points']} points {rep['faces']} faces; {tex}")
    print(f"  wired {rep.get('wired')}; validation {rep['validation']}")
    print(f"  compare_glb {cmp}")
    print(f"  problems {rep['problems'] + more}")
