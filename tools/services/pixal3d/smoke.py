"""Smoke tests for the win-64 Pixal3D service. Every mode runs under a wall clock and
writes its evidence to runs/ (kept in git; the GLB/USDZ/PNG outputs are not).

  pixi run check        imports + one real kernel from each CUDA extension
  pixi run smoke-stub   WEFTSPUN_STUB=1 server: the HTTP/USDZ contract, no GPU, no weights
  pixi run smoke        real server: POST /predict then /extract on an alpha image,
                        -> one .usdz (check_usdz); wall times and VRAM
  python smoke.py --texture-sizes 2048,1536,1024
                        the same, one extract per texture size on one predict (a sweep:
                        sizes are recorded, only the server default must fit the budget)
  python smoke.py --usdz FILE [--glb FILE]
                        check_usdz on a package on disk (and compare_glb against its GLB)
"""

import argparse
import base64
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_IMAGE = HERE / "src" / "pixal3d" / "assets" / "images" / "17_img.png"
# RunPod carries a job result of at most 20 MB (/runsync); the base64 .usdz must
# leave room for the rest of the JSON.
B64_BUDGET = 18_000_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def check_usdz(path, budget_b64=B64_BUDGET):
    """Checks one /extract result, a .usdz on disk, and returns a report whose
    "problems" list is empty when it passes:
      - package: the first file is the root layer, a .usdc (crate, "PXR-USDC"); every
        file is stored uncompressed at a 64-byte aligned offset (the usdz rules);
      - stage: upAxis Y, metersPerUnit 1, a default prim, exactly one UsdGeom.Mesh
        whose counts agree (3 indices per face, every index < points, one normal and
        one st per point);
      - material: the mesh is bound (MaterialBindingAPI) to a Material whose surface is
        a UsdPreviewSurface with diffuseColor, metallic and roughness read from
        UsdUVTexture nodes; every texture's file resolves inside the package and is a
        PNG PIL opens;
      - UsdValidation: every registered validator, no error;
      - size: the base64 of the package within budget_b64 (None: recorded only)."""
    import io

    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdValidation

    path = Path(path)
    size = path.stat().st_size
    rep = {"bytes": size, "b64_bytes": 4 * ((size + 2) // 3), "budget_b64": budget_b64}
    problems = rep["problems"] = []

    zf = Sdf.ZipFile.Open(str(path))
    names = list(zf.GetFileNames()) if zf else []
    rep["files"] = {}
    for n in names:
        info = zf.GetFileInfo(n)
        rep["files"][n] = info.size
        if info.compressionMethod != 0:
            problems.append(f"{n}: compressed (method {info.compressionMethod})")
        if info.dataOffset % 64:
            problems.append(f"{n}: data at offset {info.dataOffset}, not 64-byte aligned")
    if not names or not names[0].endswith(".usdc"):
        problems.append(f"first file in the package is not a .usdc layer: {names[:1]}")
    elif bytes(zf.GetFile(names[0])[:8]) != b"PXR-USDC":
        problems.append(f"{names[0]} is not a USD crate")
    layer_bytes = sum(v for k, v in rep["files"].items() if k.endswith((".usdc", ".usda", ".usd")))
    tex_bytes = sum(v for k, v in rep["files"].items() if k.endswith(".png"))
    rep["share"] = {"layer": layer_bytes / size, "textures": tex_bytes / size,
                    "zip_overhead": (size - layer_bytes - tex_bytes) / size}

    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as exc:
        problems.append(f"the stage does not open: {str(exc).strip().splitlines()[0][:300]}")
        return rep
    rep["upAxis"] = UsdGeom.GetStageUpAxis(stage)
    rep["metersPerUnit"] = UsdGeom.GetStageMetersPerUnit(stage)
    rep["defaultPrim"] = str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None
    if rep["upAxis"] != "Y" or rep["metersPerUnit"] != 1.0 or not rep["defaultPrim"]:
        problems.append(f"stage metadata: upAxis {rep['upAxis']}, metersPerUnit {rep['metersPerUnit']}, "
                        f"defaultPrim {rep['defaultPrim']}")
    meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    rep["meshes"] = len(meshes)
    if len(meshes) != 1:
        problems.append(f"{len(meshes)} UsdGeom.Mesh prims, expected 1")
    rep["points"] = rep["faces"] = 0
    rep["textures"] = {}
    for m in meshes:
        pts = m.GetPointsAttr().Get() or []
        counts = m.GetFaceVertexCountsAttr().Get() or []
        idx = m.GetFaceVertexIndicesAttr().Get() or []
        normals = m.GetNormalsAttr().Get() or []
        st = UsdGeom.PrimvarsAPI(m).GetPrimvar("st")
        st_vals = st.Get() if st else None
        rep["points"] += len(pts)
        rep["faces"] += len(counts)
        name = m.GetPath().name
        if not len(pts) or not len(counts):
            problems.append(f"{name}: empty ({len(pts)} points, {len(counts)} faces)")
        if set(counts) - {3} or len(idx) != 3 * len(counts) or (len(idx) and max(idx) >= len(pts)):
            problems.append(f"{name}: face counts/indices disagree with {len(pts)} points")
        if len(normals) != len(pts) or m.GetNormalsInterpolation() != UsdGeom.Tokens.vertex:
            problems.append(f"{name}: {len(normals)} normals ({m.GetNormalsInterpolation()}) for {len(pts)} points")
        if st_vals is None or len(st_vals) != len(pts) or st.GetInterpolation() != UsdGeom.Tokens.vertex:
            problems.append(f"{name}: primvars:st missing or not one per point")

        material, _ = UsdShade.MaterialBindingAPI(m.GetPrim()).ComputeBoundMaterial()
        if not material:
            problems.append(f"{name}: no material bound")
            continue
        rep["material"] = str(material.GetPath())
        surface = material.ComputeSurfaceSource()[0]
        if not surface or surface.GetIdAttr().Get() != "UsdPreviewSurface":
            problems.append(f"{rep['material']}: surface is not a UsdPreviewSurface")
            continue
        wired = {}
        for inp in surface.GetInputs():
            src = inp.GetConnectedSources()[0]
            if src:
                tex = UsdShade.Shader(src[0].source.GetPrim())
                if tex.GetIdAttr().Get() == "UsdUVTexture":
                    wired[inp.GetBaseName()] = f"{tex.GetInput('file').Get().path}:{src[0].sourceName}"
        rep["wired"] = wired
        for need in ("diffuseColor", "metallic", "roughness"):
            if need not in wired:
                problems.append(f"{rep['material']}: {need} is not read from a texture")
        for prim in Usd.PrimRange(material.GetPrim()):
            shader = UsdShade.Shader(prim)
            if not shader or shader.GetIdAttr().Get() != "UsdUVTexture":
                continue
            asset = shader.GetInput("file").Get()
            # resolved inside this package: <package>.usdz[<path>]
            inside = asset.resolvedPath.endswith(f"{path.name}[{asset.path}]")
            entry = rep["textures"][asset.path] = {"resolves_in_package": inside}
            if not inside:
                problems.append(f"{asset.path}: does not resolve inside the package ({asset.resolvedPath!r})")
            if asset.path not in names:
                problems.append(f"{asset.path}: not a file in the package")
                continue
            raw = bytes(zf.GetFile(asset.path))
            entry["bytes"] = len(raw)
            if not raw.startswith(PNG_SIGNATURE):
                problems.append(f"{asset.path}: not a PNG")
                continue
            img = Image.open(io.BytesIO(raw))
            img.load()
            entry["size"], entry["mode"] = list(img.size), img.mode
            entry["colorSpace"] = shader.GetInput("sourceColorSpace").Get()

    registry = UsdValidation.ValidationRegistry()
    errors = [e for e in UsdValidation.ValidationContext(registry.GetOrLoadAllValidators()).Validate(stage)
              if not e.HasNoError()]
    rep["validation"] = [f"{e.GetType()} {e.GetName()}: {e.GetMessage()}" for e in errors]
    for e in errors:
        if e.GetType() == UsdValidation.ValidationErrorType.Error:
            problems.append(f"UsdValidation {e.GetName()}: {e.GetMessage()}")
    rep["fits_budget"] = budget_b64 is None or rep["b64_bytes"] <= budget_b64
    if not rep["fits_budget"]:
        problems.append(f"base64 {rep['b64_bytes']} B over the {budget_b64} B budget")
    return rep


def compare_glb(usdz, glb):
    """The package against the GLB it was made from (the server keeps the GLB on its
    disk): same points, faces, UVs; the same RGB texels in every texture. Returns
    (report, problems)."""
    import io

    import numpy as np
    import trimesh
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom

    problems = []
    scene = trimesh.load(str(glb), force="scene")
    tm = []
    for node in scene.graph.nodes_geometry:  # in the scene's frame, as the server writes it
        transform, geom_name = scene.graph[node]
        if isinstance(scene.geometry[geom_name], trimesh.Trimesh):
            tm.append(scene.geometry[geom_name].copy())
            tm[-1].apply_transform(transform)
    stage = Usd.Stage.Open(str(usdz))
    um = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(tm) != 1 or len(um) != 1:
        return {"glb_meshes": len(tm), "usd_meshes": len(um)}, ["compare_glb expects one mesh each"]
    t, u = tm[0], um[0]
    pts = np.asarray(u.GetPointsAttr().Get(), np.float64)
    idx = np.asarray(u.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    st = np.asarray(UsdGeom.PrimvarsAPI(u).GetPrimvar("st").Get(), np.float64)
    rep = {"points_max_abs": float(np.abs(pts - t.vertices).max()) if pts.shape == t.vertices.shape else None,
           "faces_equal": bool(idx.shape == t.faces.shape and (idx == t.faces).all()),
           "st_max_abs": float(np.abs(st - t.visual.uv).max()) if st.shape == t.visual.uv.shape else None}
    if rep["points_max_abs"] is None or rep["points_max_abs"] > 1e-6:
        problems.append(f"points differ from the GLB: {rep['points_max_abs']}")
    if not rep["faces_equal"]:
        problems.append("faces differ from the GLB")
    if rep["st_max_abs"] is None or rep["st_max_abs"] > 1e-6:
        problems.append(f"st differs from the GLB UVs: {rep['st_max_abs']}")
    # the convention, from the GLB's own bytes rather than trimesh's reading of them:
    # glTF puts uv (0,0) at the image's top left, USD puts st (0,0) at its bottom left
    raw = Path(glb).read_bytes()
    n_json = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20 + n_json])
    binary = raw[20 + n_json + 8:]
    prim = gltf["meshes"][0]["primitives"][0]
    if len(gltf["meshes"]) == 1 and len(gltf["meshes"][0]["primitives"]) == 1 and "TEXCOORD_0" in prim["attributes"]:
        acc = gltf["accessors"][prim["attributes"]["TEXCOORD_0"]]
        view = gltf["bufferViews"][acc["bufferView"]]
        uv = np.frombuffer(binary, np.float32, acc["count"] * 2,
                           view.get("byteOffset", 0) + acc.get("byteOffset", 0)).reshape(-1, 2)
        flip = np.abs(st - np.stack([uv[:, 0], 1 - uv[:, 1]], 1)).max() if st.shape == uv.shape else None
        rep["st_vs_gltf_uv_flipped_max_abs"] = None if flip is None else float(flip)
        if flip is None or flip > 1e-6:
            problems.append(f"st is not (u, 1 - v) of the GLB's TEXCOORD_0: {flip}")
    zf = Sdf.ZipFile.Open(str(usdz))
    mat = t.visual.material
    for slot, name in (("baseColorTexture", "base_color"), ("metallicRoughnessTexture", "metallic_roughness"),
                       ("normalTexture", "normal")):
        src = getattr(mat, slot, None)
        if src is None:
            continue
        rel = f"textures/material0_{name}.png"
        if rel not in zf.GetFileNames():
            problems.append(f"{rel} missing for the GLB's {slot}")
            continue
        a = np.asarray(src.convert("RGB"))
        b = np.asarray(Image.open(io.BytesIO(bytes(zf.GetFile(rel)))).convert("RGB"))
        rep[f"{name}_rgb_equal"] = bool(a.shape == b.shape and (a == b).all())
        if not rep[f"{name}_rgb_equal"]:
            problems.append(f"{rel}: RGB texels differ from the GLB's {slot}")
    return rep, problems


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


def service(fh, stub, image, wall, resolution, texture_sizes=(None,), tag=None):
    import requests

    RUNS.mkdir(exist_ok=True)
    tag = tag or ("stub" if stub else "real")
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

        ok = True
        result["extract"] = {}
        for ts in texture_sizes:  # None: the server's default, what a client gets
            label = "default" if ts is None else str(ts)
            body = {"state": pred["state"]} if ts is None else {"state": pred["state"], "texture_size": ts}
            t0 = time.perf_counter()
            r = requests.post(base + "/extract", json=body, timeout=max(1, deadline - time.perf_counter()))
            r.raise_for_status()
            ext = r.json()
            wall_s = time.perf_counter() - t0
            usdz = RUNS / (f"{tag}-asset.usdz" if ts is None else f"{tag}-asset-t{ts}.usdz")
            usdz.write_bytes(base64.b64decode(ext["usd_b64"]))
            # a sweep's named sizes are measurements; the default must fit the budget
            rep = check_usdz(usdz, B64_BUDGET if ts is None else None)
            rep["fits_budget"] = rep["b64_bytes"] <= B64_BUDGET
            if Path(ext.get("glb_path", "")).is_file():  # same machine: the kept GLB
                glb = RUNS / (usdz.stem.replace("-asset", "-output") + ".glb")
                glb.write_bytes(Path(ext["glb_path"]).read_bytes())
                rep["glb_bytes"] = glb.stat().st_size
                rep["compare_glb"], more = compare_glb(usdz, glb)
                rep["problems"] += more
            if ext.get("format") != "usdz" or len(ext["usd_b64"]) != rep["b64_bytes"]:
                rep["problems"].append(f"response format {ext.get('format')!r}, usd_b64 {len(ext['usd_b64'])} B")
            if set(ext) & {"glb", "layer"}:
                rep["problems"].append(f"response still carries {sorted(set(ext) & {'glb', 'layer'})}")
            if stub and ext.get("stub") is not True:
                rep["problems"].append("stub server answered stub=false")
            result["extract"][label] = x = {
                "wall_s": wall_s, "response_bytes": len(r.content), "texture_size": ext.get("texture_size"),
                **{k: ext.get(k) for k in ("seconds", "vram_peak_mib", "usdz_seconds", "usd_bytes", "usd_files")},
                "usdz": rep}
            log(fh, f"/extract texture_size {ext.get('texture_size')} ({label}) {wall_s:.1f}s: usdz {rep['bytes']} B, "
                    f"base64 {rep['b64_bytes']} B (budget {B64_BUDGET}: {'fits' if rep['fits_budget'] else 'OVER'}), "
                    f"response {len(r.content)} B; files {rep['files']}; share {rep['share']}")
            log(fh, f"  USDZ meshes {rep['meshes']} points {rep['points']} faces {rep['faces']} "
                    f"material {rep.get('material')} wired {rep.get('wired')} textures {rep['textures']}")
            log(fh, f"  validation {rep['validation']} compare_glb {rep.get('compare_glb')} server "
                    f"{ {k: x[k] for k in ('seconds', 'vram_peak_mib', 'usdz_seconds')} }")
            for p in rep["problems"]:
                log(fh, f"  PROBLEM {p}")
            ok = ok and not rep["problems"]
        if stub:
            ok = ok and pred.get("stub") is True
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
    ap.add_argument("--texture-sizes", default="",
                    help="comma list: one extract per size on one predict; 'default' = the server's")
    ap.add_argument("--name", default=None, help="evidence name under runs/ (default check|stub|real)")
    ap.add_argument("--usdz", default=None, help="check a .usdz on disk instead of running a server")
    ap.add_argument("--glb", default=None, help="with --usdz: the GLB it was made from (compare_glb)")
    a = ap.parse_args()
    if a.usdz:
        rep = check_usdz(a.usdz)
        if a.glb:
            rep["compare_glb"], more = compare_glb(a.usdz, a.glb)
            rep["problems"] += more
        print(json.dumps(rep, indent=2))
        sys.exit(1 if rep["problems"] else 0)
    sizes = [None if s.strip() == "default" else int(s) for s in a.texture_sizes.split(",") if s.strip()]
    RUNS.mkdir(exist_ok=True)
    name = a.name or ("check" if a.imports_only else ("stub" if a.stub else "real"))
    with open(RUNS / f"{name}.log", "w") as fh:
        if a.imports_only:
            sys.path[:0] = [str(HERE / "src" / d) for d in ("pixal3d", "moge", "flexgemm2-pkg")]
            sys.exit(imports_only(fh))
        sys.exit(service(fh, a.stub, a.image, a.wall, a.resolution, sizes or [None], name))
