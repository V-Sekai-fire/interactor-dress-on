"""Smoke: serve, one real image-conditioned edit, check the returned USD, record time and VRAM.

    pixi run smoke            # after `pixi run fetch`

Input is VoxHammer's own example (assets/example: model.glb, mask.glb, images/2d_*.png),
converted here to what the service accepts: the model as a USD Mesh layer, the mask as
a USD Cube layer (its bounding box, same frame). Result -> runs/smoke.json.
Wall clocks: server load 600 s, the request 1800 s; the server is killed in every branch.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VH = HERE / "src" / "VoxHammer"
RUNS = HERE / "runs"
sys.path.insert(0, str(HERE.parent))
from svc_common import models_dir, round_robin_gpu  # noqa: E402
WORK = RUNS / "work"
PORT = int(os.environ.get("SMOKE_PORT", "8766"))
LOAD_WALL, REQ_WALL = 600, 1800


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def build_request() -> dict:
    import numpy as np
    import trimesh
    sys.path.insert(0, str(VH / "tools"))
    import usd_io
    WORK.mkdir(parents=True, exist_ok=True)
    m = trimesh.load(VH / "assets/example/model.glb", force="mesh", process=False)
    usd_io.write_mesh(WORK / "request_mesh.usdc", np.asarray(m.vertices), np.asarray(m.faces), None,
                      up="Y", frame="source", source="VoxHammer assets/example/model.glb")
    k = trimesh.load(VH / "assets/example/mask.glb", force="mesh", process=False)
    usd_io.write_cube(WORK / "request_region.usda", k.vertices.min(0), k.vertices.max(0), up="Y",
                      frame="source", role="edit")
    img = VH / "assets/example/images"
    return {"mesh": b64(WORK / "request_mesh.usdc"), "region": b64(WORK / "request_region.usda"),
            "reference": b64(img / "2d_edit.png"), "source_view": b64(img / "2d_render.png"),
            "view_mask": b64(img / "2d_mask.png"), "seed": 0,
            "_faces_in": int(len(m.faces)), "_verts_in": int(len(m.vertices))}


def gpu_state(idx: str) -> tuple[str, int]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    for line in q.splitlines():
        i, name, used = [s.strip() for s in line.split(",")]
        if i == idx:
            return name, int(used)
    raise SystemExit(f"GPU {idx} not listed by nvidia-smi")


def main() -> int:
    req = build_request()
    meta = {k: req.pop(k) for k in [k for k in req if k.startswith("_")]}
    idx = round_robin_gpu(models_dir("VOXHAMMER_MODELS"), "voxhammer-smoke")  # the next card, not a pinned one
    if idx is None:
        raise SystemExit("no GPU listed by nvidia-smi")
    gpu_name, idle = gpu_state(idx)
    env = dict(os.environ, PORT=str(PORT))  # carries the pick; serve.py keeps a caller's CUDA_VISIBLE_DEVICES
    log = open(RUNS / "smoke_server.log", "w")
    t0 = time.time()
    srv = subprocess.Popen([sys.executable, "-u", str(HERE / "serve.py")], env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    peak = {"mib": idle}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                peak["mib"] = max(peak["mib"], gpu_state(idx)[1])
            except Exception:
                pass
            stop.wait(0.5)

    threading.Thread(target=sample, daemon=True).start()
    result = {"gpu": f"{gpu_name} (nvidia-smi index {idx}, round robin)", "idle_mib": idle, **meta}
    try:
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"server exited {srv.returncode} while loading; see runs/smoke_server.log")
            if time.time() - t0 > LOAD_WALL:
                raise RuntimeError("server not ready inside the load wall clock")
            try:
                h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read())
                if h.get("ready"):
                    result["server_ready_s"] = round(time.time() - t0, 1)
                    result["model_load_s"] = h.get("load_s")
                    break
            except OSError:
                pass
            time.sleep(1)
        t1 = time.time()
        r = urllib.request.Request(f"http://127.0.0.1:{PORT}/predict", data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(r, timeout=REQ_WALL).read())
        result["request_wall_s"] = round(time.time() - t1, 1)
        for k in ("times", "faces", "vertices", "peak_vram_mib", "plan", "stub"):
            result[k] = out.get(k)
        # The returned USD must carry real geometry, and the layer must compose it.
        from pxr import Usd, UsdGeom
        (WORK / out["mesh_name"]).write_bytes(base64.b64decode(out["mesh"]))
        (WORK / "edit.usda").write_bytes(base64.b64decode(out["layer"]))
        crate = (WORK / out["mesh_name"]).read_bytes()[:8]
        st = Usd.Stage.Open(str(WORK / "edit.usda"))
        meshes = [UsdGeom.Mesh(p) for p in st.Traverse() if p.IsA(UsdGeom.Mesh)]
        n = sum(len(m.GetFaceVertexCountsAttr().Get() or []) for m in meshes)
        ext = [list(map(float, e)) for e in meshes[0].GetExtentAttr().Get()] if meshes else None
        result.update(crate_magic=crate.decode(errors="replace"), layer_mesh_prims=len(meshes),
                      layer_faces=n, layer_extent=ext)
        result["pass"] = bool(crate == b"PXR-USDC" and meshes and n > 0 and not out.get("stub"))
    except Exception as e:
        result["pass"] = False
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        stop.set()
        srv.kill()
        srv.wait(30)
        log.close()
    result["nvidia_smi_peak_mib"] = peak["mib"]
    (RUNS / "smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
