"""Run the Pixal3D service (src/service/server.py) with every weight from the pinned
local models/ tree. `pixi run serve`; PORT (default 8000), HOST (default 127.0.0.1),
WEFTSPUN_STUB=1 for the no-GPU contract mode."""

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch import models_dir  # noqa: E402

M = models_dir()
env = {
    "PIXAL3D_WEIGHTS": M / "Pixal3D-src",
    "PIXAL3D_DINO": M / "dinov3-vitl16",
    "PIXAL3D_MOGE": M / "moge-3-vitl" / "model.pt",
    "PIXAL3D_REMBG_MODEL": M / "BiRefNet_HR-matting",
    "NAF_WEIGHTS": M / "NAF" / "naf_release.pth",
    "NAF_REPO": HERE / "src" / "naf",
}
for k, v in env.items():
    os.environ.setdefault(k, str(v))
os.environ.setdefault("HOST", "127.0.0.1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # every weight is local and pinned

service = HERE / "src" / "service"
for p in (service, HERE / "src" / "pixal3d", HERE / "src" / "moge", HERE / "src" / "flexgemm2-pkg"):
    sys.path.insert(0, str(p))
os.chdir(service)
runpy.run_path(str(service / "server.py"), run_name="__main__")
