"""RunPod queue worker for the Pixal3D image -> mesh service (Stage 7).

At worker start this launches the service's own server, tools/services/pixal3d/serve.py
(the code path `pixi run serve` takes on the desk), on 127.0.0.1:PIXAL3D_PORT and waits
for GET /health to report ready (the model load, ~2 min). Each job is then one HTTP call
to that server over loopback, and the job's result is the server's JSON body:

  {"input": {"route": "health"}}
  {"input": {"route": "predict", "image": "<base64 PNG>", "seed": 42, "nviews": 4, ...}}
  {"input": {"route": "extract", "state": "<state from predict>", "decimation_target": ...}}

The route is removed from the input; everything else is the request body of that route,
unchanged (tools/services/pixal3d/README.md). A non-200 answer becomes {"error": ...},
which RunPod reports as a failed job. If the server could not start, or has died, every
job fails with the reason and asks RunPod to replace the worker (refresh_worker).

One GPU per worker: the server gets the caller's CUDA_VISIBLE_DEVICES, else "0" (the one
card the container sees), so svc_common's round robin, which needs nvidia-smi and a
writable models dir, never runs here. Weights come from MODELS_DIR (the image sets
/runpod-volume/models, the network volume); nothing is downloaded (HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import runpod

SERVICE = Path(__file__).resolve().parents[2] / "services" / "pixal3d"
PORT = int(os.environ.get("PIXAL3D_PORT", "18000"))
BASE = f"http://127.0.0.1:{PORT}"
READY_TIMEOUT_S = float(os.environ.get("PIXAL3D_READY_TIMEOUT", "1200"))
ROUTES = {"health": ("GET", "/health"), "predict": ("POST", "/predict"), "extract": ("POST", "/extract")}

_server: subprocess.Popen | None = None
_startup_error: str | None = None


def log(msg: str) -> None:
    print(f"[pixal3d-worker] {msg}", flush=True)


def _stop_server() -> None:
    if _server is not None and _server.poll() is None:
        _server.terminate()
        try:
            _server.wait(20)
        except subprocess.TimeoutExpired:
            _server.kill()


def start_server() -> None:
    """Launch serve.py and block until /health says ready, the process exits, or the
    wall clock runs out. The server's output goes to this process's (the worker log)."""
    global _server, _startup_error
    env = dict(os.environ, HOST="127.0.0.1", PORT=str(PORT))
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("HF_HUB_OFFLINE", "1")
    log(f"starting {SERVICE / 'serve.py'} on {BASE}, CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}, "
        f"MODELS_DIR={env.get('MODELS_DIR')}")
    t0 = time.perf_counter()
    _server = subprocess.Popen([sys.executable, "-u", str(SERVICE / "serve.py")], env=env, cwd=str(SERVICE))
    atexit.register(_stop_server)
    while True:
        if _server.poll() is not None:
            _startup_error = f"serve.py exited with {_server.returncode} before /health was ready (see the worker log)"
            break
        if time.perf_counter() - t0 > READY_TIMEOUT_S:
            _stop_server()
            _startup_error = f"serve.py not ready after {READY_TIMEOUT_S:.0f} s (PIXAL3D_READY_TIMEOUT)"
            break
        try:
            health = requests.get(BASE + "/health", timeout=2).json()
            if health.get("ready"):
                log(f"server ready in {time.perf_counter() - t0:.1f} s: {health}")
                return
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1)
    log(f"FAIL {_startup_error}")


def handler(job: dict) -> dict:
    body = dict(job.get("input") or {})
    route = body.pop("route", None)
    if route not in ROUTES:
        return {"error": f"input.route must be one of {sorted(ROUTES)}, got {route!r}"}
    if _startup_error or _server is None or _server.poll() is not None:
        reason = _startup_error or f"serve.py exited with {_server.returncode if _server else None}"
        return {"error": f"pixal3d service unavailable: {reason}", "refresh_worker": True}
    method, path = ROUTES[route]
    t0 = time.perf_counter()
    try:
        r = requests.request(method, BASE + path, json=body if method == "POST" else None, timeout=None)
    except requests.RequestException as exc:
        return {"error": f"{path}: {type(exc).__name__}: {exc}", "refresh_worker": _server.poll() is not None}
    try:
        out = r.json()
    except ValueError:
        out = {"error": r.text[:2000]}
    log(f"{route} -> HTTP {r.status_code} in {time.perf_counter() - t0:.1f} s")
    if r.status_code != 200:
        detail = (out.get("error") or out.get("detail") or out) if isinstance(out, dict) else out
        return {"error": f"{path}: HTTP {r.status_code}: {detail}"}
    return out


if __name__ == "__main__":
    start_server()
    runpod.serverless.start({"handler": handler})
