"""Shared by the pixi-launched model services: where the weights live, and which GPU a
launch gets.

models_dir(): the first of the service's own variable, MODELS_DIR, models/ beside the main
checkout's .git (worktrees share one copy), or models/ in this checkout when there is no
git at all (an exported tree, a copied folder).

round_robin_gpu(): no card is pinned. Each launch takes the next GPU in PCI bus order
(nvidia-smi's numbering), from a counter kept next to the weights so every service and
worktree on the machine shares it: the first launch gets GPU 0, the next GPU 1, and so on,
wrapping. Two services launched one after the other land on different cards. A
CUDA_VISIBLE_DEVICES set by the caller wins and the counter is not touched.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # tools/services -> this checkout


def main_checkout() -> Path:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute",
                            "--git-common-dir"], capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).parent
    except (OSError, subprocess.TimeoutExpired):
        pass
    return REPO


def models_dir(*env_names: str) -> Path:
    for n in (*env_names, "MODELS_DIR"):
        if os.environ.get(n):
            return Path(os.environ[n])
    return main_checkout() / "models"


def gpu_indices() -> list[str]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [l.strip() for l in r.stdout.splitlines() if l.strip().isdigit()]


def round_robin_gpu(state_dir: Path, who: str) -> str | None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        print(f"[{who}] GPU {os.environ['CUDA_VISIBLE_DEVICES']} (CUDA_VISIBLE_DEVICES from the caller)", flush=True)
        return os.environ["CUDA_VISIBLE_DEVICES"]
    ids = gpu_indices()
    if not ids:
        print(f"[{who}] no GPU listed by nvidia-smi; torch picks", flush=True)
        return None
    state_dir.mkdir(parents=True, exist_ok=True)
    counter, lock = state_dir / ".gpu-round-robin", state_dir / ".gpu-round-robin.lock"
    deadline = time.monotonic() + 10
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() > deadline:  # a launcher died holding it: take it over
                lock.unlink(missing_ok=True)
                deadline = time.monotonic() + 10
            time.sleep(0.05)
    try:
        try:
            n = int(counter.read_text().strip() or 0)
        except (OSError, ValueError):
            n = 0
        counter.write_text(f"{n + 1}\n")
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)
    pick = ids[n % len(ids)]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = pick
    print(f"[{who}] GPU {pick} of {len(ids)} (round robin, launch {n})", flush=True)
    return pick
