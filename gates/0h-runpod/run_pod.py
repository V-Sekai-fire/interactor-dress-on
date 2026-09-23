"""Gate 0H host driver: run remote.sh on a fresh RunPod GPU pod and bring the logs back.

    python gates/0h-runpod/run_pod.py --key <private ssh key> [--gpu "NVIDIA RTX A5000"]

RUNPOD_API_KEY comes from the process environment, else (Windows) the user
environment; it is never printed. The ssh key pair is the caller's (kept outside
the repo); its public half goes to the pod as PUBLIC_KEY. The pod is a plain
ubuntu:22.04 on community cloud with NVIDIA_DRIVER_CAPABILITIES=all, so the
container toolkit grants the graphics (Vulkan) driver libraries, not only compute.
It is terminated in every branch (finally), and the wall clock bounds the run.
Logs land in gates/0h-runpod/<--out>/ (run1/, run2/, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
API = "https://rest.runpod.io/v1"
WALL_S = 75 * 60


def api_key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k and os.name == "nt":
        k = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "[Environment]::GetEnvironmentVariable('RUNPOD_API_KEY','User')"],
                           capture_output=True, text=True).stdout.strip()
    if not k:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return k


def call(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(API + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key()}",
                                          "Content-Type": "application/json",
                                          "User-Agent": "interactor-dress-on-gate-0h/1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:  # the API's reason, not just the status
        raise SystemExit(f"{method} {path}: HTTP {e.code} {e.read()[:500]!r}")
    return json.loads(raw) if raw else None


def pack(dst: Path) -> None:
    """project/ (with the gitignored fit.elf), gates/2-avbd, vendor/xr-grid at HEAD."""
    arch = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD",
                           "project", "gates/2-avbd", "vendor/xr-grid"], capture_output=True, check=True).stdout
    dst.write_bytes(arch)
    with tarfile.open(dst, "a") as t:
        t.add(ROOT / "project" / "fit.elf", arcname="project/fit.elf")


def ssh_opts(key: str) -> list[str]:
    return ["-i", key, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="private key; <key>.pub is sent to the pod")
    ap.add_argument("--gpu", default="NVIDIA RTX A5000,NVIDIA RTX A4000,NVIDIA RTX A4500,NVIDIA GeForce RTX 3090,"
                    "NVIDIA RTX 4000 Ada Generation,NVIDIA GeForce RTX 4090,NVIDIA RTX A6000",
                    help="comma list, first available wins")
    ap.add_argument("--cloud", default="COMMUNITY,SECURE", help="comma list, tried in order")
    ap.add_argument("--out", default="out", help="directory under gates/0h-runpod for this run's logs")
    a = ap.parse_args()
    pub = Path(a.key + ".pub").read_text().strip()
    out = HERE / a.out
    out.mkdir(exist_ok=True)
    tree = out / "tree.tar"
    pack(tree)
    start = ("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server > /dev/null"
             " && mkdir -p /run/sshd /root/.ssh /work && echo \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys"
             " && chmod 600 /root/.ssh/authorized_keys && exec /usr/sbin/sshd -D")
    t0 = time.time()
    pod = None
    for cloud in a.cloud.split(","):
        try:
            pod = call("POST", "/pods", {
                "name": "gate-0h-godot-vulkan", "imageName": "ubuntu:22.04", "cloudType": cloud,
                "gpuTypeIds": a.gpu.split(","), "gpuTypePriority": "availability", "gpuCount": 1,
                "containerDiskInGb": 30, "volumeInGb": 0, "ports": ["22/tcp"], "supportPublicIp": True,
                "env": {"PUBLIC_KEY": pub, "NVIDIA_DRIVER_CAPABILITIES": "all"},
                "dockerStartCmd": ["bash", "-c", start]})
            break
        except SystemExit as e:  # "no instances currently available": try the next cloud
            print(f"{cloud}: {e}", flush=True)
    if pod is None:
        return 1
    pid = pod["id"]
    print(f"pod {pid} {(pod.get('machine') or {}).get('gpuDisplayName') or pod.get('gpu')} {cloud} "
          f"{pod.get('costPerHr')} $/h", flush=True)
    rc = 1
    try:
        ip = port = None
        while time.time() - t0 < 15 * 60:
            p = call("GET", f"/pods/{pid}")
            ip, pm = p.get("publicIp"), (p.get("portMappings") or {})
            port = pm.get("22") if isinstance(pm, dict) else None
            if ip and port:
                probe = subprocess.run(["ssh", *ssh_opts(a.key), "-p", str(port), f"root@{ip}", "true"],
                                       capture_output=True)
                if probe.returncode == 0:
                    break
            time.sleep(10)
        else:
            print("FAIL: pod never answered ssh within 15 min", flush=True)
            return 1
        print(f"ssh up after {time.time() - t0:.0f} s at {ip}:{port}", flush=True)
        o = ssh_opts(a.key)
        subprocess.run(["scp", *o, "-P", str(port), str(tree), str(HERE / "remote.sh"), f"root@{ip}:/work/"], check=True)
        left = WALL_S - (time.time() - t0)
        r = subprocess.run(["ssh", *o, "-p", str(port), f"root@{ip}", "bash /work/remote.sh"],
                           timeout=max(60, left - 120))
        print(f"remote.sh rc={r.returncode} after {time.time() - t0:.0f} s", flush=True)
        subprocess.run(["scp", *o, "-r", "-P", str(port), f"root@{ip}:/work/out/*", str(out) + "/"])
        rc = 0 if r.returncode == 0 else 1
    except subprocess.TimeoutExpired:
        print("FAIL: wall clock", flush=True)
        subprocess.run(["scp", *ssh_opts(a.key), "-r", "-P", str(port), f"root@{ip}:/work/out/*", str(out) + "/"])
    finally:
        call("DELETE", f"/pods/{pid}")
        spent = (time.time() - t0) / 3600 * float(pod.get("costPerHr") or 0)
        print(f"pod {pid} terminated after {time.time() - t0:.0f} s (~${spent:.2f})", flush=True)
        tree.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
