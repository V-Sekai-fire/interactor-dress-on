"""One real kernel from every CUDA extension module in the own-built wheels, each checked
against a reference that does not use that module (pure torch, the library's own torch
fallback, the CPU path, or an analytic answer).

  python kernels.py        # exit 0 iff every check passes; one line per check

Modules covered: cumesh._C, cumesh._cubvh, cumesh._xatlas (host code, through
uv_unwrap), flex_gemm.kernels.cuda (+ its Triton GEMMs), o_voxel._C, _nvdiffrast_c,
nvdiffrec_render.renderutils._C. The calls are the ones Pixal3D makes (CuMesh.init /
simplify / uv_unwrap(return_vmaps=True) and cuBVH in o_voxel.postprocess.to_glb,
sparse_submanifold_conv3d and grid_sample_3d, flexible_dual_grid_to_mesh in the shape
decoder, nvdiffrast rasterize / interpolate / texture, renderutils' PBR shading).
"""
import math
import subprocess
import sys
import time

import torch

RESULTS = []


def check(label, fn):
    t0 = time.perf_counter()
    try:
        ok, msg = fn()
        torch.cuda.synchronize()
    except Exception as exc:  # a failed load or launch is a FAIL line, not a crash
        ok, msg = False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
    RESULTS.append((label, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {msg} ({time.perf_counter() - t0:.2f} s)", flush=True)


def icosphere(level):
    """Unit icosphere, outward-wound faces (float32 [V,3], int32 [F,3], on the CPU)."""
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    for _ in range(level):
        mid, nf = {}, []

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                mid[k] = len(v)
                v.append([(v[a][i] + v[b][i]) / 2 for i in range(3)])
            return mid[k]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    V = torch.tensor(v, dtype=torch.float32)
    V = V / V.norm(dim=1, keepdim=True)
    F = torch.tensor(f, dtype=torch.int32)
    n = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1)
    inward = (n * V[F].mean(1)).sum(1) < 0
    F[inward] = F[inward][:, [0, 2, 1]]
    return V, F


def cube():
    V = torch.tensor([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], dtype=torch.float32)
    F = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=torch.int32)
    return V, F


# ---------------------------------------------------------------- cumesh
def cumesh_normals():
    import cumesh
    V, F = icosphere(4)
    m = cumesh.CuMesh()
    m.init(V.cuda(), F.cuda())
    m.compute_vertex_normals()
    n = m.read_vertex_normals().cpu()
    dot = (n * V).sum(1)
    return bool(dot.min() > 0.99), f"CuMesh {m.num_vertices} v / {m.num_faces} f, vertex normal . position min {dot.min():.5f}"


def cumesh_simplify():
    import cumesh
    V, F = icosphere(4)
    m = cumesh.CuMesh()
    m.init(V.cuda(), F.cuda())
    m.simplify(1000)
    v, f = m.read()
    r = v.norm(dim=1)
    ok = 0 < f.shape[0] <= 1000 and float((r - 1).abs().max()) < 0.05
    return ok, f"simplify {F.shape[0]} -> {f.shape[0]} faces (target 1000), max |r-1| {float((r - 1).abs().max()):.4f}"


def cumesh_uv_unwrap():
    import cumesh
    V, F = icosphere(3)
    m = cumesh.CuMesh()
    m.init(V.cuda(), F.cuda())
    v, f, uv, vmap = m.uv_unwrap(return_vmaps=True)
    v, f, uv, vmap = v.cpu(), f.cpu(), uv.cpu(), vmap.cpu().long()
    same = torch.equal(v, V[vmap])
    inside = bool((uv >= 0).all() and (uv <= 1).all())
    ok = same and inside and f.shape[0] == F.shape[0] and bool(f.max() < v.shape[0])
    return ok, (f"uv_unwrap {F.shape[0]} f -> {f.shape[0]} f, {v.shape[0]} v (seams split), "
                f"v == V[vmap] {same}, uv in [0,1] {inside}")


def cumesh_bvh():
    import cumesh
    V, F = cube()
    bvh = cumesh.cuBVH(V.cuda(), F.cuda())
    g = torch.Generator().manual_seed(0)
    p = torch.rand(4096, 3, generator=g) * 2 - 1
    a = p.abs()
    inside = (a <= 0.5).all(1)
    ref = torch.where(inside, (0.5 - a).min(1).values, (a - 0.5).clamp_min(0).norm(dim=1))
    d, _, _ = bvh.unsigned_distance(p.cuda())
    err = float((d.cpu() - ref).abs().max())
    return err < 1e-5, f"cuBVH unsigned_distance, 4096 points vs the exact box distance: max err {err:.2e}"


# ---------------------------------------------------------------- flex_gemm
def sparse_grid(res=24, occupancy=0.3, ch=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    occ = torch.rand(res, res, res, generator=g) < occupancy
    xyz = occ.nonzero().int()
    coords = torch.cat([torch.zeros(len(xyz), 1, dtype=torch.int32), xyz], 1).contiguous()
    feats = torch.randn(len(xyz), ch, generator=g)
    return feats.cuda().contiguous(), coords.cuda(), torch.Size([1, ch, res, res, res])


def flex_gemm_neighbor_map():
    from flex_gemm import kernels
    from flex_gemm.ops import spconv, utils
    from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
    feats, coords, shape = sparse_grid()
    keys, vals = utils.init_hashmap(shape, int(2.0 * coords.shape[0]), coords.device)
    cuda_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
        keys, vals, coords, *shape[2:], 3, 3, 3, 1, 1, 1)
    old = spconv.ALGORITHM
    spconv.set_algorithm(spconv.Algorithm.EXPLICIT_GEMM)
    try:
        torch_map = SubMConv3dFunction._compute_neighbor_cache_torch(coords, shape, (3, 3, 3), (1, 1, 1))['neighbor_map']
    finally:
        spconv.set_algorithm(old)
    same = torch.equal(cuda_map.cpu().long(), torch_map.cpu().long())
    return same, f"hashmap neighbour map ({coords.shape[0]} voxels x 27) == the torch fallback: {same}"


def flex_gemm_conv():
    from flex_gemm.ops import spconv
    feats, coords, shape = sparse_grid()
    Ci = Co = feats.shape[1]
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(Co, 3, 3, 3, Ci, generator=g) / math.sqrt(27 * Ci)).cuda()
    b = torch.randn(Co, generator=g).cuda()
    out, _ = spconv.sparse_submanifold_conv3d(feats, coords, shape, w, b)
    dense = torch.zeros(1, Ci, *shape[2:], device="cuda")
    x, y, z = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
    dense[0, :, x, y, z] = feats.t()
    torch.backends.cudnn.allow_tf32 = False
    ref = torch.nn.functional.conv3d(dense, w.permute(0, 4, 1, 2, 3), b, padding=1)[0, :, x, y, z].t()
    rel = float((out - ref).abs().max() / ref.abs().max())
    return rel < 1e-2, (f"sparse_submanifold_conv3d ({spconv.ALGORITHM}, Triton) {tuple(feats.shape)} -> "
                        f"{tuple(out.shape)} vs dense conv3d: max err / max |ref| {rel:.2e} (TF32 bound 1e-2)")


def flex_gemm_grid_sample():
    from flex_gemm.ops.grid_sample import grid_sample_3d, grid_sample_3d_torch
    feats, coords, shape = sparse_grid(ch=8)
    g = torch.Generator().manual_seed(2)
    q = (torch.rand(1, 5000, 3, generator=g) * (shape[2] - 1)).cuda()
    out = grid_sample_3d(feats, coords, shape, q, mode="trilinear")
    ref = grid_sample_3d_torch(feats, coords, shape, q, "trilinear")
    err = float((out - ref).abs().max())
    return err < 1e-4, f"grid_sample_3d trilinear, 5000 points vs grid_sample_3d_torch: max err {err:.2e}"


# ---------------------------------------------------------------- o_voxel
def o_voxel_serialize():
    import o_voxel
    from o_voxel import _C
    g = torch.Generator().manual_seed(3)
    c = torch.randint(0, 1024, (100000, 3), generator=g, dtype=torch.int32)
    x, y, z = [c[:, i].contiguous() for i in range(3)]
    zc, hc = _C.z_order_encode_cuda(x.cuda(), y.cuda(), z.cuda()), _C.hilbert_encode_cuda(x.cuda(), y.cuda(), z.cuda())
    same = torch.equal(zc.cpu(), _C.z_order_encode_cpu(x, y, z)) and torch.equal(hc.cpu(), _C.hilbert_encode_cpu(x, y, z))
    back = torch.equal(o_voxel.serialize.decode_seq(hc, mode="hilbert").cpu(), c) and \
        torch.equal(o_voxel.serialize.decode_seq(zc, mode="z_order").cpu(), c)
    return same and back, f"z-order + hilbert encode, 1e5 coords: CUDA == CPU {same}, CUDA decode round trip {back}"


def o_voxel_dual_grid():
    import o_voxel
    V, F = icosphere(4)
    V = V * 0.4
    res = 64
    idx, dual, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        V, F.long(), grid_size=res, aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2)
    local = dual * res - idx
    v, f = o_voxel.convert.flexible_dual_grid_to_mesh(
        idx.cuda().int(), local.cuda().float(), inter.cuda().bool(), None,
        aabb=[[-.5, -.5, -.5], [.5, .5, .5]], grid_size=res)
    err = float((v.norm(dim=1) - 0.4).abs().max())
    ok = f.shape[0] > 0 and err < 1.0 / res
    return ok, (f"sphere r=0.4 -> flexible dual grid ({idx.shape[0]} voxels at {res}^3, CPU) -> mesh "
                f"(CUDA hashmap) {v.shape[0]} v / {f.shape[0]} f, max |r-0.4| {err:.4f} (voxel {1 / res:.4f})")


# ---------------------------------------------------------------- nvdiffrast
def nvdiffrast_raster():
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    P = torch.tensor([[-0.71, -0.63, 0.1, 1.0], [0.83, -0.52, 0.2, 1.0], [0.07, 0.91, 0.3, 1.0]])
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    H = W = 64
    rast, _ = dr.rasterize(ctx, P[None].cuda(), tri, resolution=[H, W])
    cov = (rast[0, ..., 3] > 0).cpu()
    ys, xs = torch.meshgrid((torch.arange(H) + 0.5) / H * 2 - 1, (torch.arange(W) + 0.5) / W * 2 - 1, indexing="ij")

    def edge(a, b):
        return (P[b, 0] - P[a, 0]) * (ys - P[a, 1]) - (P[b, 1] - P[a, 1]) * (xs - P[a, 0])
    ref = (edge(0, 1) > 0) & (edge(1, 2) > 0) & (edge(2, 0) > 0)
    same = torch.equal(cov, ref)
    attr, _ = dr.interpolate(P[None, :, :2].cuda().contiguous(), rast, tri)
    xy = torch.stack([xs, ys], -1)
    ierr = float((attr[0].cpu() - xy)[ref].abs().max())
    tex = torch.arange(32, dtype=torch.float32).repeat(32, 1)[None, ..., None].cuda().contiguous()
    uv = torch.rand(1, 16, 16, 2, generator=torch.Generator().manual_seed(4)) * (30 / 32) + 1 / 32
    t = dr.texture(tex, uv.cuda().contiguous(), filter_mode="linear", boundary_mode="clamp")
    terr = float((t[0, ..., 0].cpu() - (uv[0, ..., 0] * 32 - 0.5)).abs().max())
    ok = same and ierr < 1e-4 and terr < 1e-3
    return ok, (f"rasterize {int(cov.sum())} px == edge-function reference {same}; interpolate max err {ierr:.1e}; "
                f"texture (linear ramp) max err {terr:.1e}")


# ---------------------------------------------------------------- nvdiffrec_render
def nvdiffrec_pbr():
    import nvdiffrec_render.renderutils as ru
    g = torch.Generator().manual_seed(5)
    S = (1, 32, 32, 3)

    def r(*s, lo=0.0, hi=1.0):
        return (torch.rand(*s, generator=g) * (hi - lo) + lo).cuda()
    kd, arm = r(*S), r(*S)
    arm[..., 1] = arm[..., 1] * 0.9 + 0.1
    pos = r(*S, lo=-1, hi=1)
    nrm = torch.nn.functional.normalize(r(*S, lo=-1, hi=1), dim=-1)
    view, light = r(1, 1, 1, 3, lo=2, hi=3), r(1, 1, 1, 3, lo=2, hi=3)
    ins_c = [t.clone().requires_grad_(True) for t in (kd, arm, pos, nrm)]
    ins_p = [t.clone().requires_grad_(True) for t in (kd, arm, pos, nrm)]
    out_c = ru.pbr_bsdf(*ins_c, view, light)
    out_p = ru.pbr_bsdf(*ins_p, view, light, use_python=True)
    tgt = r(*S)
    ((out_c - tgt) ** 2).mean().backward()
    ((out_p - tgt) ** 2).mean().backward()
    fwd = float((out_c - out_p).abs().max() / out_p.abs().max())
    bwd = max(float((a.grad - b.grad).abs().max() / b.grad.abs().max()) for a, b in zip(ins_c, ins_p))
    ok = fwd < 1e-4 and bwd < 1e-3
    return ok, f"pbr_bsdf fwd+bwd vs its PyTorch reference (use_python=True): fwd {fwd:.1e}, grads {bwd:.1e}"


def main():
    cap = torch.cuda.get_device_capability(0)
    try:
        drv = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
                                      text=True).strip()
    except Exception:
        drv = "?"
    print(f"device {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]}), driver {drv}; "
          f"torch {torch.__version__} (cuda {torch.version.cuda}); python {sys.version.split()[0]}")
    import triton
    print(f"triton {triton.__version__}")
    import cumesh, flex_gemm, o_voxel, nvdiffrast, nvdiffrec_render  # noqa: E401
    import importlib
    for mod in ("cumesh._C", "cumesh._cubvh", "cumesh._xatlas", "flex_gemm.kernels.cuda", "o_voxel._C",
                "_nvdiffrast_c", "nvdiffrec_render.renderutils._C"):
        try:
            print(f"  load {mod}: {importlib.import_module(mod).__file__}")
        except Exception as exc:
            print(f"  FAIL load {mod}: {type(exc).__name__}: {exc}")
            RESULTS.append((f"load {mod}", False))
    for label, fn in [
        ("cumesh._C compute_vertex_normals", cumesh_normals),
        ("cumesh._C simplify", cumesh_simplify),
        ("cumesh._C+_xatlas uv_unwrap", cumesh_uv_unwrap),
        ("cumesh._cubvh unsigned_distance", cumesh_bvh),
        ("flex_gemm.kernels.cuda neighbour map", flex_gemm_neighbor_map),
        ("flex_gemm spconv (Triton)", flex_gemm_conv),
        ("flex_gemm.kernels.cuda grid_sample_3d", flex_gemm_grid_sample),
        ("o_voxel._C serialize", o_voxel_serialize),
        ("o_voxel._C dual grid -> mesh", o_voxel_dual_grid),
        ("_nvdiffrast_c rasterize/interpolate/texture", nvdiffrast_raster),
        ("nvdiffrec_render.renderutils._C pbr_bsdf", nvdiffrec_pbr),
    ]:
        check(label, fn)
    failed = [l for l, ok in RESULTS if not ok]
    print(f"{'PASS' if not failed else 'FAIL'} {len(RESULTS) - len(failed)}/{len(RESULTS)} checks"
          + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
