"""Smoke tests for the win-64 Pixal3D service. Every mode runs under a wall clock and
writes its evidence to runs/ (kept in git; the GLB/USD outputs are not).

  pixi run check        imports + one real kernel from each CUDA extension
  pixi run smoke-stub   WEFTSPUN_STUB=1 server: the HTTP/USD contract, no GPU, no weights
  pixi run smoke        real server: POST /predict then /extract on an alpha image,
                        -> USD layer with a UsdGeom.Mesh; wall times and VRAM
"""

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_IMAGE = HERE / "src" / "pixal3d" / "assets" / "images" / "17_img.png"


def log(fh, *a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def imports_only(fh):
    import torch

    log(fh, f"torch {torch.__version__} cuda {torch.version.cuda} "
            f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    if not torch.cuda.is_available():
        log(fh, "FAIL no CUDA device")
        return 1
    import triton

    log(fh, f"triton {triton.__version__}")
    failed = []
    for name in ("cumesh", "flex_gemm", "o_voxel", "utils3d", "utils3d_moge", "nvdiffrast.torch",
                 "nvdiffrec_render", "flex_gemm2.nn", "moge.model.v3", "pxr.UsdGeom"):
        try:
            __import__(name)
            log(fh, f"  ok   import {name}")
        except Exception as exc:
            log(fh, f"  FAIL import {name}: {type(exc).__name__}: {exc}")
            failed.append(name)

    # A file is not a capability: run one kernel from each extension.
    def run(label, fn):
        try:
            log(fh, f"  ok   {label}: {fn()}")
        except Exception as exc:
            log(fh, f"  FAIL {label}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
            failed.append(label)

    def nvdr():
        import nvdiffrast.torch as dr
        ctx = dr.RasterizeCudaContext()
        pos = torch.tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [0, 0.8, 0, 1]]], device="cuda")
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
        rast, _ = dr.rasterize(ctx, pos, tri, resolution=[64, 64])
        return f"rasterize covered {int((rast[..., 3] > 0).sum())} px"

    def fg():
        from flex_gemm.ops.grid_sample import grid_sample_3d
        import inspect
        return f"grid_sample_3d{inspect.signature(grid_sample_3d)} importable"

    def fg_conv():
        # flex_gemm's triton path: a sparse submanifold conv through pixal3d's wrapper
        from pixal3d.modules import sparse as sp
        coords = torch.stack(torch.meshgrid(*[torch.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)
        coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.long), coords], 1).int().cuda()
        x = sp.SparseTensor(feats=torch.randn(len(coords), 16, device="cuda"), coords=coords)
        conv = sp.SparseConv3d(16, 16, 3).cuda()
        y = conv(x)
        torch.cuda.synchronize()
        return f"SparseConv3d {tuple(x.feats.shape)} -> {tuple(y.feats.shape)} ({sp.config.CONV})"

    def cm():
        import cumesh
        v = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device="cuda")
        f = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int32, device="cuda")
        m = cumesh.CuMesh()
        m.init(v, f)
        return f"CuMesh.init tetra -> {m.num_vertices} verts {m.num_faces} faces"

    def naf():
        sys.path.insert(0, str(HERE / "src" / "naf"))
        from src.layers.attentions import na2d_block
        # brute force: every hi-res query attends to its low-res block's window
        B, hk, d, N, D, k = 1, 12, 4, 2, 8, 9
        q = torch.randn(B, hk * d, hk * d, N, D, device="cuda", dtype=torch.float64)
        kl = torch.randn(B, hk, hk, N, D, device="cuda", dtype=torch.float64)
        vl = torch.randn(B, hk, hk, N, D, device="cuda", dtype=torch.float64)
        out = na2d_block(q, kl, vl, (k, k))
        ref = torch.empty_like(out)
        for i in range(hk * d):
            si = min(max(i // d - k // 2, 0), hk - k)
            for j in range(hk * d):
                sj = min(max(j // d - k // 2, 0), hk - k)
                kk = kl[:, si:si + k, sj:sj + k].reshape(B, k * k, N, D)
                vv = vl[:, si:si + k, sj:sj + k].reshape(B, k * k, N, D)
                a = (torch.einsum("bnd,bmnd->bnm", q[:, i, j], kk) * D ** -0.5).softmax(-1)
                ref[:, i, j] = torch.einsum("bnm,bmnd->bnd", a, vv)
        return f"na2d_block vs brute force max|diff| {float((out - ref).abs().max()):.2e}"

    run("nvdiffrast", nvdr)
    run("flex_gemm", fg)
    run("flex_gemm conv (triton)", fg_conv)
    run("cumesh", cm)
    run("naf na2d_block", naf)
    return 1 if failed else 0


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def gpu_poll(samples, stop):
    """nvidia-smi memory.used per GPU, max over the run (MiB)."""
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=10)
            for line in out.strip().splitlines():
                name, used = [x.strip() for x in line.rsplit(",", 1)]
                samples[name] = max(samples.get(name, 0), int(used))
        except Exception:
            pass
        if stop.wait(0.5):
            break


def service(fh, stub, image, wall, resolution):
    import requests

    RUNS.mkdir(exist_ok=True)
    tag = "stub" if stub else "real"
    port = free_port()
    env = dict(os.environ, PORT=str(port), HOST="127.0.0.1")
    if stub:
        env["WEFTSPUN_STUB"] = "1"
    else:
        env.pop("WEFTSPUN_STUB", None)
    t_start = time.perf_counter()
    deadline = t_start + wall
    samples, stop = {}, threading.Event()
    baseline = {}
    gpu_poll(baseline, _once())
    poller = threading.Thread(target=gpu_poll, args=(samples, stop), daemon=True)
    poller.start()
    server_log = open(RUNS / f"{tag}-server.log", "w")
    proc = subprocess.Popen([sys.executable, str(HERE / "serve.py")], env=env,
                            stdout=server_log, stderr=subprocess.STDOUT, cwd=HERE)
    base = f"http://127.0.0.1:{port}"
    result = {"mode": tag, "port": port}
    rc = 1
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}; see runs/{tag}-server.log")
            if time.perf_counter() > deadline:
                raise TimeoutError("wall clock expired waiting for /health")
            try:
                h = requests.get(base + "/health", timeout=2).json()
                if h.get("ready"):
                    break
            except Exception:
                pass
            time.sleep(1)
        result["load_s"] = time.perf_counter() - t_start
        log(fh, f"health {h} after {result['load_s']:.1f}s (model load included)")

        img_b64 = base64.b64encode(Path(image).read_bytes()).decode()
        t0 = time.perf_counter()
        r = requests.post(base + "/predict", json={"image": img_b64, "seed": 42, "nviews": 4,
                                                   "resolution": resolution},
                          timeout=max(1, deadline - time.perf_counter()))
        r.raise_for_status()
        pred = r.json()
        result["predict_wall_s"] = time.perf_counter() - t0
        result["predict_server"] = {k: pred.get(k) for k in ("seconds", "vram_peak_mib", "camera_params")}
        state_bytes = len(base64.b64decode(pred["state"]))
        log(fh, f"/predict {result['predict_wall_s']:.1f}s state {state_bytes} B views {len(pred['views'])} "
                f"server {result['predict_server']}")
        for i, v in enumerate(pred["views"][:4]):
            (RUNS / f"{tag}-view{i}.png").write_bytes(base64.b64decode(v))

        t0 = time.perf_counter()
        r = requests.post(base + "/extract", json={"state": pred["state"]},
                          timeout=max(1, deadline - time.perf_counter()))
        r.raise_for_status()
        ext = r.json()
        result["extract_wall_s"] = time.perf_counter() - t0
        result["extract_server"] = {k: ext.get(k) for k in ("seconds", "vram_peak_mib")}
        glb = RUNS / f"{tag}-output.glb"
        usda = RUNS / f"{tag}-layer.usda"
        glb.write_bytes(base64.b64decode(ext["glb"]))
        usda.write_bytes(base64.b64decode(ext["layer"]))
        log(fh, f"/extract {result['extract_wall_s']:.1f}s glb {glb.stat().st_size} B layer "
                f"{usda.stat().st_size} B server {result['extract_server']}")

        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(usda))
        meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        pts = sum(len(m.GetPointsAttr().Get() or []) for m in meshes)
        tris = sum(len(m.GetFaceVertexCountsAttr().Get() or []) for m in meshes)
        result["usd"] = {"upAxis": UsdGeom.GetStageUpAxis(stage),
                         "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
                         "meshes": len(meshes), "points": pts, "faces": tris,
                         "normals": all(m.GetNormalsAttr().HasValue() for m in meshes)}
        log(fh, f"USD {result['usd']}")
        ok = result["usd"]["upAxis"] == "Y" and result["usd"]["metersPerUnit"] == 1.0
        if stub:
            ok = ok and pred.get("stub") is True and ext.get("stub") is True
        else:
            ok = ok and len(meshes) > 0 and pts > 0 and tris > 0
        rc = 0 if ok else 1
    except Exception as exc:
        log(fh, f"FAIL {type(exc).__name__}: {exc}")
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except Exception:
            proc.kill()
        stop.set()
        poller.join(5)
        server_log.close()
    result["gpu_mem_used_mib_max"] = samples
    result["gpu_mem_used_mib_before"] = baseline
    result["total_wall_s"] = time.perf_counter() - t_start
    result["pass"] = rc == 0
    (RUNS / f"{tag}.json").write_text(json.dumps(result, indent=2))
    log(fh, f"{'PASS' if rc == 0 else 'FAIL'} {tag} in {result['total_wall_s']:.1f}s; "
            f"nvidia-smi max used {samples} (before {baseline})")
    return rc


def _once():
    e = threading.Event()
    e.set()
    return e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--imports-only", action="store_true")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--wall", type=float, default=1800.0, help="seconds, for the whole run")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    name = "check" if a.imports_only else ("stub" if a.stub else "real")
    with open(RUNS / f"{name}.log", "w") as fh:
        if a.imports_only:
            sys.path[:0] = [str(HERE / "src" / d) for d in ("pixal3d", "moge", "flexgemm2-pkg")]
            sys.exit(imports_only(fh))
        sys.exit(service(fh, a.stub, a.image, a.wall, a.resolution))
