# Gate 6a — the brick-grid SDF against OpenVDB

**Result: PASS (9/9 checks on each of two point populations), control fails as
it must. The Lean spline kernel is still pending, so check (a) covers the
reference sampler only.**

`fit.elf` cannot carry OpenVDB. cloth-fit's `FitForm` used it for two things:
an avatar SDF (`meshToSignedDistanceField(xform, Vec3s points, tris, {}, 150, 1)`,
a 414 MB narrow band) and the openvdb fork's tricubic B-spline
`SplineSampler::sampleHessian`. Both are replaced in `vendor/cloth-fit`:

- `SdfGrid`: a lazily filled 8³-brick grid. Voxel (i,j,k) is at world
  (i,j,k)·h. Its value is clamp(sd, −h, 150h), where sd is the exact `igl::AABB`
  distance to the float-rounded avatar, signed by `igl::fast_winding_number`
  (inside when w > 0.5). Grids are cached per (avatar hash, voxel size).
- `SdfSpline`: the call site of the Lean kernel
  `lean/Fit/SdfSplineHessian.lean → kernels/fit/cpp/sdf_spline_hessian_emit.cpp`.
  It sits behind `FIT_KERNELS_PENDING` until the Lean source is on main.

`vendor/cloth-fit/CITATION.cff` lists the changes. This gate asks two
questions. Does the replacement reproduce what OpenVDB gave FitForm? And would
it catch a grid that is wrong by half a voxel?

## Run

```sh
gates/6-fit/sdf/run.sh      # native, CPU only; ~2 min; writes C:/b/g6a
```

`run.sh` does four things:

1. It builds `gate6a.exe` (llvm-mingw 20260826, `-O2`, x86-64 baseline, no FMA)
   from `gate6a.cpp`, `spline_ref.h` and the vendored `SdfGrid.cpp` and
   `SdfSpline.cpp`. libigl and json come from the `.forks/` org forks; Eigen
   comes from the CPM cache.
2. `make_points.py` makes the points: FitForm<4>'s 15 samples per face (the
   barycentrics of `upsample_standard<4>`) on the 1-thread oracle's final
   garment `C:/b/cf-up-out1/step_garment_252.obj`.
3. `C:/b/cf-up/openvdb_dump.exe` dumps OpenVDB at those points. It rebuilds
   FitForm's grid from the oracle's normalised `target_avatar.obj`, with faces
   from the input `FoxGirl/avatar.obj`.
4. `gate6a.exe` compares the two.

## Populations

| population | faces | samples |
|---|---|---|
| `fit`: the samples FitForm actually evaluates (faces not in `not_fit_fids`) | 1,401 | 21,015 |
| `all`: every garment face, so there are more samples near the avatar | 5,220 | 78,300 |

The plan's estimate of "48k fit samples" is wrong. `no-fit.txt` marks 3,819
of the skirt's 5,220 faces, which leaves 1,401 × 15 = 21,015 samples.

## Checks and results (`results.txt`, `run.log`)

"Near" means |v_vdb| < 5 voxels: 20,942 samples for `fit` and 31,738 for `all`.

| check | criterion | fit | all |
|---|---|---|---|
| a.index | p·(1/h), floor, uvw equal OpenVDB's `worldToIndex`, bit for bit | 21,015/21,015 | 78,300/78,300 |
| a.sampler | reference on OpenVDB's own 4³ stencils vs `sampleHessian`, ≤ 1 ULP | **0 ULP**, all bitwise | **0 ULP**, all bitwise |
| a.kernel | Lean emit vs reference, ≤ 1 ULP | pending | pending |
| b.value | \|dv\| ≤ 0.25 voxel for ≥ 99% of near samples | 100% (p99 0.027, max 0.058 voxel) | 100% (p99 0.028, max 0.084) |
| b.sign | sign agreement ≥ 99.5% (near) | 100% | 100% |
| b.gradient | angle ≤ 10° for ≥ 95% (near) | 100% (p95 0.81°, max 3.0°) | 100% (p95 0.77°, max 5.1°) |
| b.pure | brick value == voxel recomputed alone, bit for bit | 19,794/19,794 | 119,499/119,499 |
| b.order | fresh grid filled in reverse point order gives the same samples, bit for bit | 21,015/21,015 | 78,300/78,300 |
| cache | same (V,F,h) → one grid; a different h → another grid | pass | pass |
| **c.control** | grid shifted +0.5 voxel in x must fail (b) | **fails**: value 58.6%, sign 99.41% | **fails**: value 56.0%, sign 99.35% |

Details:

- **Voxel values.** Compared voxel by voxel (INFO), 99.40% (fit) and 99.70%
  (all) of the near stencil voxels agree within 0.25 voxel. Median |dv| is 1e-5
  voxel and the worst is 0.9987 voxel. Signs agree on 99.95% and 99.99%. The
  sampled field agrees better than the voxels do, because the B-spline smooths
  the few outlier voxels.
- **Control.** The shifted grid stays within 10° on gradients (99.8%, 99.3%),
  because a half-voxel shift barely turns the normals. The value and sign
  criteria are what reject it.
- **What (a) shows.** Equal stencils give equal outputs, bit for bit. The
  summation order of the spline sum (per-axis tables, then i, j, k ascending,
  ((v·a)·b)·c) is the one the Lean kernel must reproduce.

## Cost (INFO lines, one thread)

| population | bricks | voxels | memory | distance + winding queries | fill | per sample, including fill |
|---|---|---|---|---|---|---|
| fit | 136 | 69,632 | 0.56 MB | 69,632 each | 0.44–0.62 s | 23–30 µs |
| all | 671 | 343,552 | 2.77 MB | 343,552 each | 1.48–1.57 s | 20–22 µs |

The AABB and winding-number BVH take 0.02–0.06 s to set up.

For comparison, OpenVDB builds 46,523,937 active voxels (413,824,624 bytes) in
3.4–9.8 s with TBB threads on this desk.

These sizes match the plan's 2–12 MB estimate. The garment moves during a fit,
so the brick count will grow over a run. Gate 6.P measures that.

## Stand-in kernel: an ABI check, and a hazard for the Lean kernel (`standin.log`)

The Lean kernel does not exist yet, so the non-pending path of `SdfSpline.cpp`
was exercised with a stand-in:

- A scratch-only Slang file, written by hand, compiled with `slangc -target cpp`
  into `FIT_KERNELS_DIR`.
- It is **not committed**. AGENTS.md rule 2 forbids a hand-written kernel in
  the tree.

What it showed:

1. **The call-site contract works.** `GlobalParams_0` carries `stencil_0`,
   `uvw_0` and `result_0`, and `main_0_Thread` dispatches one lane per sample.
   Output is **0 ULP** from the reference on both populations. The buffer is
   named `result` because `out` is a Slang keyword.
2. **Float literals break the kernel.** An unsuffixed Slang literal is
   `float`: `2.0 / 3.0` folds to 0.66666668653488159. With such literals the
   kernel check **fails**. The largest error, about 8.7e18 ULP, is the distance
   between two values of opposite sign.

   The Lean emit must print double literals (`2.0l`). With them, 2/3 folds to
   0.66666666666666663 and the check passes. Check (a.kernel) catches this
   mistake.

## Evidence kept

`run.log` (full run), `results.txt` (verdict lines), `standin.log`.

The points and dumps are not committed. They are regenerated by `run.sh` under
`C:/b/g6a`:

| file | sha256 prefix |
|---|---|
| `points_fit.txt` | 8e49d9bc… |
| `points_all.txt` | bb6c58bb… |

The dump `.jsonl` files change hash from run to run, because their header
records the build time.

## Not covered here

- **The Lean kernel itself (a.kernel).** Once
  `kernels/fit/cpp/sdf_spline_hessian_emit.cpp` exists, `run.sh` compiles it in
  and checks it automatically.
- **The full fit.** FitForm is now OpenVDB-free and compiles in the native
  polyfem build (`POLYFEM_THREADING=NONE`). `SdfGrid.cpp` and `SdfSpline.cpp`
  also cross-compile for rv64gc. But the solve needs the kernel: while
  `FIT_KERNELS_PENDING` is set, `solution_changed` throws.
