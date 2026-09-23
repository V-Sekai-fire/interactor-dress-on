"""RunPod queue worker for the dress-on loop: one job = one Gate 8 run of the whole flow.

The flow runs exactly as on the desk: stock Godot 4.7.2 on the NVIDIA GPU (Vulkan
on an Xvfb display, Gate 0H) drives the guest ELFs (curvenet, fit, drape) through
project/gate_loop.gd. Stages that are not wired to a service yet stand in with
their fixture only when the job allows it (allow_fixture, default "infer,rig",
labelled FIXTURE in the result, as on the desk).

Input  {"allow_fixture": "infer,rig", "wallclock": 3000}
Output {"result": "RESULT: PASS FIXTURE:infer,rig", "seconds": ..., "gpu": "...",
        "results": [lines], "summary": {...}, "fitted_obj": "...", "screenshot_png": "<base64>"}
Progress: every new results line is sent with runpod.serverless.progress_update.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import runpod

GODOT = os.environ.get("GODOT", "/opt/godot/godot")
PROJECT = os.environ.get("DRESS_ON_PROJECT", "/app/project")
DISPLAY = ":99"
_xvfb: subprocess.Popen | None = None


def _nvidia_icd() -> str | None:
    """The container toolkit mounts the NVIDIA Vulkan driver when the graphics
    capability is granted. Some hosts mount the library without an ICD file, and
    /etc/vulkan/icd.d can be a read-only mount (Gate 0H), so a missing ICD file is
    written under /tmp and named through VK_ICD_FILENAMES."""
    for p in ("/etc/vulkan/icd.d/nvidia_icd.json", "/usr/share/vulkan/icd.d/nvidia_icd.json"):
        if os.path.exists(p):
            return p
    libs = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    if "libGLX_nvidia.so.0" not in libs:
        return None
    p = "/tmp/nvidia_icd.json"
    Path(p).write_text('{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}')
    return p


def _display() -> str:
    global _xvfb
    if _xvfb is None or _xvfb.poll() is not None:
        _xvfb = subprocess.Popen(["Xvfb", DISPLAY, "-screen", "0", "1280x720x24", "-nolisten", "tcp"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
    return DISPLAY


def handler(job):
    inp = job.get("input") or {}
    allow = str(inp.get("allow_fixture", "infer,rig"))
    wall = int(inp.get("wallclock", 3000))
    icd = _nvidia_icd()
    if icd is None:
        return {"error": "no NVIDIA Vulkan driver in the container (NVIDIA_DRIVER_CAPABILITIES must include graphics)"}
    work = Path(tempfile.mkdtemp(prefix="loop-"))
    out = work / "loop.txt"
    env = dict(os.environ, DISPLAY=_display(), VK_ICD_FILENAMES=icd)
    cmd = [GODOT, "--path", PROJECT, "--rendering-driver", "vulkan", "--xr-mode", "off", "--audio-driver", "Dummy",
           "--script", "gate_loop.gd", "--", "--gate=loop", f"--out={out}", f"--wallclock={wall}",
           f"--allow-fixture={allow}"]
    t0 = time.time()
    with open(work / "godot.log", "w") as log:
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        seen = 0
        while p.poll() is None:
            time.sleep(5)
            lines = out.read_text(errors="replace").splitlines() if out.exists() else []
            if len(lines) > seen:
                runpod.serverless.progress_update(job, {"t": round(time.time() - t0, 1), "lines": lines[seen:]})
                seen = len(lines)
            if time.time() - t0 > wall + 120:  # gate_loop quits on its own wall clock; this is the backstop
                p.kill()
    lines = out.read_text(errors="replace").splitlines() if out.exists() else []
    godot_log = (work / "godot.log").read_text(errors="replace")
    gpu = next((l.split("Using Device", 1)[1].strip(" :#") for l in godot_log.splitlines() if "Using Device" in l), None)
    res = {"result": next((l for l in reversed(lines) if l.startswith("RESULT")), "RESULT: FAIL (no result line)"),
           "exit_code": p.returncode, "seconds": round(time.time() - t0, 1), "gpu": gpu, "results": lines}
    base = out.with_suffix("")
    for key, path, kind in (("summary", base.with_suffix(".json"), "json"),
                            ("fitted_obj", Path(str(base) + ".fitted.obj"), "text"),
                            ("screenshot_png", base.with_suffix(".png"), "b64")):
        if path.exists():
            if kind == "json":
                res[key] = json.loads(path.read_text())
            elif kind == "text":
                res[key] = path.read_text()
            else:
                res[key] = base64.b64encode(path.read_bytes()).decode()
    if not res["result"].startswith("RESULT: PASS"):
        res["godot_log_tail"] = godot_log.splitlines()[-60:]
    return res


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
