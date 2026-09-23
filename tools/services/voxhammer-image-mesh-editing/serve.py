"""Run the VoxHammer service (src/service/server.py) on the pinned local weights.
`pixi run serve`; PORT (default 8765 from pixi.toml), HOST (default 127.0.0.1).

The TRELLIS weights come from models_dir("VOXHAMMER_MODELS") (svc_common: VOXHAMMER_MODELS,
MODELS_DIR, the main checkout's models/, or this tree's models/ without git), handed to the
server as VOXHAMMER_TRELLIS, so it starts outside a git checkout too. The GPU is the next
one in round robin, never a pinned card; a caller's CUDA_VISIBLE_DEVICES wins."""

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from svc_common import models_dir, round_robin_gpu  # noqa: E402

M = models_dir("VOXHAMMER_MODELS")
os.environ.setdefault("VOXHAMMER_TRELLIS", str(M / "TRELLIS-image-large"))
if os.environ.get("WEFTSPUN_STUB") != "1":
    round_robin_gpu(M, "voxhammer")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the weights are local and pinned

service = HERE / "src" / "service"
sys.argv[0] = str(service / "server.py")
runpy.run_path(str(service / "server.py"), run_name="__main__")
