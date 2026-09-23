"""Local test of the RunPod Pixal3D worker image, on this desk, before any endpoint exists.

  pixi run -m tools/services/pixal3d python tools/runpod/pixal3d/local_test.py [--stub]
      [--tag pixal3d-runpod:dev] [--gpu 0] [--models C:/interactor-dress-on/models]
      [--predicts N] [--name real-warm]

Runs the image the way RunPod runs it (CMD handler.py) but with RunPod's local API
server (--rp_serve_api), one GPU (docker --gpus device=N, nvidia-smi numbering) and the
weights mounted read-only at /runpod-volume/models (the network volume's place). Then,
through POST /runsync exactly as a client would send jobs: health, predict on the alpha
image smoke.py uses (seed 42, 4 views, 1024 cascade), extract on the returned state. The
USD layer is checked as smoke.py checks it (UsdGeom.Mesh, points, faces, normals, upAxis Y,
metersPerUnit 1). Wall times, the cold start (docker run -> the worker answering), the
request/response sizes RunPod's gateway would carry, and nvidia-smi's peak on that GPU go
to runs/<tag>.json and runs/<tag>.log; the container log to runs/<tag>-container.log.
--stub runs the server with WEFTSPUN_STUB=1 and no GPU (the contract only).
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import threading
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SERVICE = HERE.parents[1] / "services" / "pixal3d"
DEFAULT_IMAGE = SERVICE / "src" / "pixal3d" / "assets" / "images" / "17_img.png"
MB = 1e6


def gpu_poll(gpu: str, peak: dict, stop: threading.Event) -> None:
    while True:
        try:
            out = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=name,memory.used",
                                           "--format=csv,noheader,nounits"], text=True, timeout=10)
            name, used = [x.strip() for x in out.strip().rsplit(",", 1)]
            peak["name"] = name
            peak["mib"] = max(peak.get("mib", 0), int(used))
        except Exception:
            pass
        if stop.wait(0.5):
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pixal3d-runpod:dev")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--models", default="C:/interactor-dress-on/models")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--wall", type=float, default=2400.0)
    ap.add_argument("--predicts", type=int, default=1, help="predicts in a row on one worker")
    ap.add_argument("--name", default=None, help="evidence name under runs/ (default real|stub)")
    a = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    name = a.name or ("stub" if a.stub else "real")
    fh = open(RUNS / f"{name}.log", "w")

    def log(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    cname = f"pixal3d-rp-{name}-{int(time.time())}"
    cmd = ["docker", "run", "--rm", "--name", cname, "-p", f"127.0.0.1:{a.port}:8000"]
    if a.stub:
        cmd += ["-e", "WEFTSPUN_STUB=1"]
    else:
        cmd += ["--gpus", f"device={a.gpu}", "-v", f"{a.models}:/runpod-volume/models:ro"]
    cmd += [a.tag, "python", "-u", "handler.py", "--rp_serve_api", "--rp_api_host", "0.0.0.0",
            "--rp_api_port", "8000"]
    log("run:", " ".join(cmd))
    result: dict = {"mode": name, "tag": a.tag, "gpu": None if a.stub else a.gpu}
    peak, stop = {}, threading.Event()
    before: dict = {}
    if not a.stub:
        gpu_poll(a.gpu, before, _set())
        threading.Thread(target=gpu_poll, args=(a.gpu, peak, stop), daemon=True).start()
    clog = open(RUNS / f"{name}-container.log", "w")
    t_start = time.perf_counter()
    deadline = t_start + a.wall
    proc = subprocess.Popen(cmd, stdout=clog, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{a.port}"
    rc = 1

    def runsync(inp: dict) -> tuple[dict, float, int, int]:
        body = json.dumps({"input": inp})
        t0 = time.perf_counter()
        r = requests.post(base + "/runsync", data=body, headers={"Content-Type": "application/json"},
                          timeout=max(1.0, deadline - time.perf_counter()))
        dt = time.perf_counter() - t0
        r.raise_for_status()
        out = r.json()
        if out.get("status") not in (None, "COMPLETED") or "error" in out or "error" in (out.get("output") or {}):
            raise RuntimeError(f"job failed: {json.dumps(out)[:2000]}")
        return out.get("output", out), dt, len(body), len(r.content)

    try:
        # cold start: docker run -> RunPod's local API answers (it starts after the
        # handler module saw /health ready)
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"container exited {proc.returncode}; see runs/{name}-container.log")
            if time.perf_counter() > deadline:
                raise TimeoutError("wall clock expired before the worker answered")
            try:
                requests.get(base + "/", timeout=2)  # any HTTP answer: the API is up
                break
            except requests.RequestException:
                time.sleep(1)
        result["cold_start_s"] = time.perf_counter() - t_start
        log(f"worker up {result['cold_start_s']:.1f} s after docker run (model load included)")

        h, dt, _, _ = runsync({"route": "health"})
        log(f"health {h} in {dt:.2f} s")
        result["health"] = h

        img = base64.b64encode(Path(a.image).read_bytes()).decode()
        for n in range(a.predicts):  # the first pays the cold Triton/autotune caches
            pred, dt, req_b, resp_b = runsync({"route": "predict", "image": img, "seed": 42, "nviews": 4,
                                               "resolution": 1024})
            result["predict" if n == 0 else f"predict_{n + 1}"] = p = {
                "wall_s": dt, "request_bytes": req_b, "response_bytes": resp_b,
                "state_b64_bytes": len(pred["state"]), "views": len(pred["views"]),
                **{k: pred.get(k) for k in ("seconds", "vram_peak_mib", "camera_params")}}
            log(f"predict #{n + 1} {dt:.1f} s, request {req_b / MB:.2f} MB, response {resp_b / MB:.2f} MB, "
                f"state b64 {len(pred['state']) / MB:.2f} MB, server {p}")
        for i, v in enumerate(pred["views"][:4]):
            (RUNS / f"{name}-view{i}.png").write_bytes(base64.b64decode(v))

        ext, dt, req_b, resp_b = runsync({"route": "extract", "state": pred["state"]})
        result["extract"] = {"wall_s": dt, "request_bytes": req_b, "response_bytes": resp_b,
                             "glb_b64_bytes": len(ext["glb"]), "layer_b64_bytes": len(ext["layer"]),
                             **{k: ext.get(k) for k in ("seconds", "vram_peak_mib")}}
        log(f"extract {dt:.1f} s, request {req_b / MB:.2f} MB, response {resp_b / MB:.2f} MB "
            f"(glb b64 {len(ext['glb']) / MB:.2f} MB, USD layer b64 {len(ext['layer']) / MB:.2f} MB), "
            f"server {result['extract']}")
        usda = RUNS / f"{name}-layer.usda"
        usda.write_bytes(base64.b64decode(ext["layer"]))
        (RUNS / f"{name}-output.glb").write_bytes(base64.b64decode(ext["glb"]))

        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(usda))
        meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        pts = sum(len(m.GetPointsAttr().Get() or []) for m in meshes)
        faces = sum(len(m.GetFaceVertexCountsAttr().Get() or []) for m in meshes)
        result["usd"] = {"bytes": usda.stat().st_size, "upAxis": UsdGeom.GetStageUpAxis(stage),
                         "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage), "meshes": len(meshes),
                         "points": pts, "faces": faces,
                         "normals": all(m.GetNormalsAttr().HasValue() for m in meshes)}
        log(f"USD {result['usd']}")
        ok = result["usd"]["upAxis"] == "Y" and result["usd"]["metersPerUnit"] == 1.0
        if a.stub:
            ok = ok and pred.get("stub") is True and ext.get("stub") is True
        else:
            ok = ok and len(meshes) > 0 and pts > 0 and faces > 0
        rc = 0 if ok else 1
    except Exception as exc:
        log(f"FAIL {type(exc).__name__}: {exc}")
    finally:
        subprocess.run(["docker", "stop", "-t", "20", cname], capture_output=True)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        stop.set()
        clog.close()
    if not a.stub:
        result["nvidia_smi_used_mib"] = {"gpu": peak.get("name"), "peak": peak.get("mib"),
                                         "before": before.get("mib")}
    result["total_wall_s"] = time.perf_counter() - t_start
    result["pass"] = rc == 0
    (RUNS / f"{name}.json").write_text(json.dumps(result, indent=2))
    log(f"{'PASS' if rc == 0 else 'FAIL'} {name} in {result['total_wall_s']:.1f} s; "
        f"nvidia-smi {result.get('nvidia_smi_used_mib')}")
    return rc


def _set() -> threading.Event:
    e = threading.Event()
    e.set()
    return e


if __name__ == "__main__":
    raise SystemExit(main())
