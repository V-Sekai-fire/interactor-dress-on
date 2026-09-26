# Gate ANNY-K: the ANNY forward and backward kernels against finite differences

**Result: PASS.** Every forward output and every gradient of the
Lean-emitted ANNY kernels (`kernels/anny`, from `lean/Anny`) agrees with a
double-precision restatement of the same model and its central finite
differences: to 3e-7 relative on a small model, and to 1.7e-6 on one the
size of ANNY's mesh and rig. The flat control, the same backward fed a
residual scaled by 1.25, fails every gradient group in both cases.

This is the gradient half of an in-guest ANNY fit. With it, the guest's
L-BFGS-B (`guest/drape/lbfgsb.h`, Gate 5) can fit blendshape coefficients,
joint rotations and the root translation to a target mesh. `AnnyInverter`
does that in PyTorch today. This gate runs on the host only, over the
kernels' `slangc -target cpp` emits built with `-ffp-contract=off`, the
same way `tests/drape_kernels` tests the drape kernels. It has no GPU arm
and no guest arm yet.

## The chain

| kernel | computes |
|---|---|
| `anny_blend` | `vanny = templ + Σ_c coeffs[c] · blend[c]` |
| `anny_csr_gemv3` | `jpos = M · vanny` (the sparse joint regressor) |
| `anny_fk` | `world`, `bone` from `rot6`, `trans`, `jpos`, bind rotations `rb` |
| `anny_lbs` | `verts = Σ_j w · bone_j · [vanny, 1]` |
| `anny_vert_residual` | `dverts = verts − target` (L = ½ dverts·dverts) |
| `anny_lbs_backward_bone` | `dbone` (df32, vertex order) |
| `anny_lbs_backward_bind` | `dvanny = Σ_j w · R_jᵀ dverts` |
| `anny_fk_backward` | `drot6`, `dtrans`, `djpos` (leaves-first walk) |
| `anny_csr_gemv3` on `Mᵀ` | `dvanny += Mᵀ · djpos` |
| `anny_blend_backward` | `dcoeffs = blendᵀ · dvanny` (df32) |

Rotations are 3×3 matrices, parameterised by their 6D truncation: the first
two rows, re-orthonormalised (AGENTS.md rule 11).

The following are not differentiated:
- the bind rotations `rb`: Sinew's `skeleton_fit` (Newton–Schulz / Kabsch)
  is a stop-gradient here, and the reference holds them fixed too;
- the phenotype → coefficient interpolation, which is host scalars.

## Run

```
kernels/anny/gen.sh              # Lean -> slang/ -> cpp/ + SPIR-V + AnnyKernelTable.inc
CXX=clang++ tests/anny_kernels/build.sh   # kernels/test.log, kernels/control.log
```

The random models draw from the engine's raw output only. Every input is
rounded to float, and both sides read the rounded values. The loss is
½|verts − target|², over x = [coeffs | rot6 | trans]. The central
difference uses h = 1e-5, in double.

## Results ([`kernels/test.log`](kernels/test.log))

| case | P | NB | J | coords differenced | forward | dcoeffs | drot6 | dtrans |
|---|---|---|---|---|---|---|---|---|
| small | 40 | 5 | 6 | 44 of 44 | 2.75e-7 | 3.15e-7 | 2.88e-7 | 2.30e-7 |
| anny-size | 13,718 | 20 | 104 | 62 of 647 | 5.75e-7 | 1.22e-6 | 1.68e-6 | 6.90e-7 |

The forward column is max|verts − ref| / max|ref|. Each gradient column
is |g − fd| / |fd| over the differenced coordinates, and passes below
1e-4. The anny-size model has ANNY's vertex count (13,718) and rig size
(104 joints), with random weights, tree and regressor, not ANNY's data.

## Control ([`kernels/control.log`](kernels/control.log))

The same kernels with the backward fed `dverts × 1.25`. The forward still
passes, and every gradient group is off by exactly 2.50e-1 in both cases.
That shows the comparison sees the backward's output. A check that
compared nothing would pass here.

## Not done yet

- A GPU arm (`AvbdRd`-style dispatch through `rdc::Device`, the SPIR-V
  from `gen.sh`) and a guest arm (an `anny.elf` whose objective feeds
  `lbfgsb.cpp`).
- ANNY's real data (its template, blendshapes, regressor, weights and bind
  rotations, from `entities-anny-creator`'s bake), and the comparison with
  AnnyInverter's 2.458 / 2.281 mm.
- The gradient through the bind rotations, and a 2D-keypoint loss
  (`loop1_fit`).
