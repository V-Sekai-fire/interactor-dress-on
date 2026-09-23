"""Which GPUs can run the prebuilt CUDA extension wheels? List the (PTX|SASS, sm_XX) entries
of every CUDA fatbinary embedded in each wheel's extension modules.

  python gates/7-pixal3d/linux-wheels/fatbin_archs.py [--cache DIR] > fatbins.log

Every wheel set V-Sekai-fire/ComfyUI-Trellis2-visualbruno holds at the pinned commit for
Linux, plus the Windows Torch280 set as the flat control (it runs on the desk's RTX 3090,
sm_86, so the parser must report sm_86 for it). Downloads go to --cache (default: a temp
dir). No CUDA toolkit needed: the fatbin container is parsed directly.

Fatbin container (as cuobjdump reads it): magic 0xBA55ED50 (u32), version (u16), header
size (u16), payload size (u64); then entries, each: kind (u16: 1 PTX, 2 SASS/ELF), u16,
entry header size (u32), payload size (u64), ..., sm arch (u32 at +28).

A SASS cubin for sm_X.y runs on sm_X.z with z >= y (same major); PTX for sm_N is
JIT-compiled by the driver for any sm >= N. So SASS sm_86 covers sm_86/sm_89, not sm_90.
"""
from __future__ import annotations

import argparse
import struct
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

REPO = "https://raw.githubusercontent.com/V-Sekai-fire/ComfyUI-Trellis2-visualbruno/14597418bbe33a440ead4667e2966408f0524a21/wheels"
SETS = {
    "Linux/Torch291 (cp312, torch 2.9.1)": ("Linux/Torch291", "cp312-cp312-linux_x86_64", [
        "cumesh-1.0", "flex_gemm-0.0.1", "nvdiffrast-0.4.0", "nvdiffrec_render-0.0.0", "o_voxel-0.0.1"]),
    "Linux/Torch270 (cp312, torch 2.7.0; no nvdiffrec_render)": ("Linux/Torch270", "cp312-cp312-linux_x86_64", [
        "cumesh-1.0", "flex_gemm-0.0.1", "nvdiffrast-0.4.0", "o_voxel-0.0.1"]),
    "Linux/Torch2110 (cp313, torch 2.11.0)": ("Linux/Torch2110", "cp313-cp313-linux_x86_64", [
        "cumesh-1.0", "flex_gemm-1.0.0", "nvdiffrast-0.4.0", "nvdiffrec_render-0.0.0", "o_voxel-0.0.1"]),
    "CONTROL Windows/Torch280 (cp311, torch 2.8.0; runs on the desk's sm_86)": ("Windows/Torch280", "cp311-cp311-win_amd64", [
        "cumesh-1.0", "flex_gemm-0.0.1", "nvdiffrast-0.4.0", "nvdiffrec_render-0.0.0", "o_voxel-0.0.1"]),
}
MAGIC = struct.pack("<I", 0xBA55ED50)


def scan(data: bytes) -> Counter:
    out: Counter = Counter()
    i = data.find(MAGIC)
    while i != -1:
        try:
            _, _, hsz, fsz = struct.unpack_from("<IHHQ", data, i)
            p, end = i + hsz, i + hsz + fsz
            while p < end:
                kind, _, ehsz, psz = struct.unpack_from("<HHIQ", data, p)
                if kind not in (1, 2) or ehsz == 0:
                    break
                arch = struct.unpack_from("<I", data, p + 28)[0]
                out[f"{'PTX' if kind == 1 else 'SASS'} sm_{arch}"] += 1
                p += ehsz + psz
        except struct.error:
            break
        i = data.find(MAGIC, max(end, i + 4))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    cache = Path(a.cache or tempfile.mkdtemp(prefix="fatbins-"))
    for title, (sub, tag, names) in SETS.items():
        print(f"## {title}: wheels/{sub}")
        for n in names:
            whl = f"{n}-{tag}.whl"
            dest = cache / sub / whl
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{REPO}/{sub}/{whl}", dest)
            with zipfile.ZipFile(dest) as z:
                for m in z.namelist():
                    if m.endswith((".so", ".pyd")):
                        c = scan(z.read(m))
                        archs = ", ".join(f"{k} x{v}" for k, v in sorted(c.items())) or "no fatbin (host code)"
                        print(f"  {whl}  {m.rsplit('/', 1)[-1]}: {archs}")
        print()


if __name__ == "__main__":
    main()
