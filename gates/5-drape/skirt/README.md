# Gate 5 G10: the fitted skirt at drape scale 10

**Result: fixed, with a regression gate.** Gate 8's loop drapes fit.elf's
fitted LCL skirt (2682 vertices, 44 waist pins, 14 skeleton capsules) at
drape scale 10. On rd every position was NaN after step 1, while cpu stayed
finite. The cause is one bending hinge whose rest curvature is at float
noise. On the GPU its sum rounds to exactly 0, and the bending kernels divide
by it. The three bending kernels now guard |s| = 0 in Lean. G10 (in
`project/gate_drape.gd`) runs this scene on cpu and rd and passes. The same
cut adds a triangle-mesh body collider, which keeps every skirt vertex out of
the body where the capsules let 186 in.

Evidence is in this directory. The scene files are described in
[`CITATION.cff`](CITATION.cff); the probe is `project/probe_drape_skirt.gd`,
and the jobs are `mesh_parity` and `mesh_bisect` in
`guest/drape/drape_jobs.cpp`. Everything ran on Godot 4.7.2 and an RTX 4090.

## Reproduction ([`probe.txt`](probe.txt), before the fix)

One rd step from the fitted skirt at scale 10, with no colliders:

| variant | result |
|---|---|
| scale 10 | 2682 of 2682 vertices non-finite |
| scale 5 | finite |
| scale 10, `selfCollision=0` | 2682 non-finite: not the self-collision scan |
| scale 10, `membrane=0` | 2682 non-finite: not the membrane |
| scale 10, `bending=0` | **finite**: the bending |
| scale 10, `iters=1` | 72 non-finite after one solver iteration |
| scale 10, `colors=0` | 1895 non-finite |

## Bisection ([`bisect_prefix.txt`](bisect_prefix.txt), before; [`bisect_fixed.txt`](bisect_fixed.txt), after)

`mesh_bisect` computes the step's predictor once. It then runs rd's
`run(k)` from that predictor for k = 1..16, reading back on the next tick,
against k cpu iterations. In the first iteration that leaves cpu, it runs
every (colour, stage) on both backends and compares every buffer: stage 0 is
init plus the four force kernels, stages 1-4 are the gathers, and stage 5 is
the solve. The stop is a test hook on both backends
(`setDebugStopForTest`, `readDebugForTest`). AvbdCpu now records all force
kernels before the gathers, as AvbdRd does. The forces read only positions,
so the reordering does not change the arithmetic.

- **Iteration 1:** rd has 216 non-finite floats (72 vertices). Where both
  are finite, cpu and rd are 4.8e-7 apart.
- **Colour 0, after init + forces:** `bendGrad` holds 12 non-finite floats
  on rd and 0 on cpu. That is one bending constraint's four rows, and it is
  the first non-finite buffer. `triGrad` is finite (1.2e-5 apart).
- **The constraint is bending 2443** (vertices 274, 2575, 1673, 1766). Its
  nTarget is 1.659e-6 and its k 4.8e-4. The four positions are the rest
  positions, bit-identical on both backends.

## Root cause

The kernel computes s = Σ wᵢ pᵢ, len = |s|, then e = s (1 − n_c / len).
Only the rest curvature n_c is guarded (n_c > 1e-6); len is not.

- **len is at float noise.** For hinge 2443, the double-precision |s| at rest
  is 1.66e-6. The terms wᵢpᵢ are up to 37, whose float ulp is 3.8e-6, so the
  float |s| is rounding noise.
- **On the GPU, len is exactly 0.** [`fma_check.txt`](fma_check.txt) shows the
  float sums. Summed as written, s = (0, −3.8e-6, 0). With one of the FMA
  contractions a compiler may form, fma(w1,x1, w0·x0) + fma(w3,x3, w2·x2),
  all three components are exactly 0. The GPU driver's build of the SPIR-V
  landed on 0 (rd's grad rows are non-finite). The guest CPU's clang build of
  slangc's cpp did not (cpu stays finite). Which contraction each one formed
  was not inspected.
- **So the result is NaN.** n_c/0 = inf, then 1 − inf = −inf, then 0·(−inf)
  = NaN. The NaN goes into the hinge's four gradient rows. The Gauss-Seidel
  colours spread it to 72 vertices in one iteration, and to all 2682 within
  the step's 16 iterations.
- **Why scale 10 and not 5.** n_c scales with the drape scale. At scale 5 this
  hinge's n_c is 8.3e-7, under the 1e-6 threshold, so the hinge is skipped.
  At scale 10 it is 1.66e-6 and active. Of 7758 hinges, 7757 are active at
  scale 5 and all 7758 at scale 10 ([`bend_check.txt`](bend_check.txt)).
  Gate 8's similarity-fit fixture was finite at scale 10. It was not checked
  hinge by hinge.

This rules out the other suspects:

- **Padding and capacity:** the topology is the same as the finite
  similarity-fit fixture.
- **The Gate 0F read-only-first hazard:** the NaN appears in the first
  span's force output, from rest positions, with nothing written before.
- **Near-zero rest edges or areas:** the smallest edge is 3.8e-2 and the
  smallest height 2.5e-2 at scale 10; the largest |invUV| is 38.

## The fix (Lean, rule 2)

In `lean/Cloth/SlangCodegen/TriangleBending{ForceAl,DualUpdate,ForceAlBackward}.lean`:

- **The forward kernels:** `scale = (len > 0) ? n_c / len : 1`. At |s| = 0
  the residual's direction is undefined, so e is taken as 0, the same zero
  contribution the n_c ≤ 1e-6 early-out gives.
- **The backward kernel:** `factor` and `vn` are 0 in that branch, the
  derivative of the forward's constant e = 0.
- **Re-emitted:** the `native_decide` pins were updated, `lake build` passes
  (95 jobs), and `AVBD_EMIT=1` re-emitted the kernels. Only these three
  kernels' Slang and cpp changed. Where len > 0 the arithmetic is unchanged.

`lean/` is a subtree of cloth-dynamics/lean, so this change belongs upstream
as well.

## G10: the regression (in `gate_drape.gd`)

The gate hands the scene to the guest through `drape_job_data`
(`mesh_obj`, `mesh_pins`, `mesh_capsules`, `body_obj`), then runs:

| job | after the fix | before the fix (control) |
|---|---|---|
| `mesh_bisect scale=10` | PASS: finite, within 9.5e-7 of cpu at all 16 iterations of step 1 | FAIL: bendGrad non-finite at iteration 1, colour 0, init + forces |
| `mesh_parity scale=10 caps=1 steps=5 tol=1e-4` | PASS: max\|x_cpu − x_rd\| = 2.98e-6 over 5 steps | FAIL: rd 8046 of 8046 floats non-finite at every step |
| `mesh_parity scale=10 body=1 steps=5 tol=1e-4` | PASS: 2.03e-6 | FAIL: rd non-finite |

The "before" column is [`control_prefix.txt`](control_prefix.txt): the same
gate built with the three kernels' pre-fix Slang and cpp. The same build gave
[`bisect_prefix.txt`](bisect_prefix.txt) and
[`parity_prefix.txt`](parity_prefix.txt). The final ELF is byte-identical to
the one the full Gate 5 run used. Why the tolerance
is 5 steps at 1e-4:

- **Measured over 5 steps:** 3.0e-6 with capsules and 2.0e-6 with the body.
- **Over 30 steps with contact:** 2.8e-3. At the 1-ulp level the two
  backends take different contact and self-collision decisions (759 vs 771
  pushes), and those decisions amplify.
- **Over 30 steps with no collider:** 2.2e-5.

After the fix, 100 rd steps at scale 10 with the 14 capsules (the loop's
failing configuration, [`probe_fixed_100caps.txt`](probe_fixed_100caps.txt))
are finite at 37.7 ms/step. Their ymin is −5.5295, and the counts are
friction 76117, projections 21292 and self-pushes 4362. cpu in the loop's own
run (`drape-fs10cpu.txt`) gave ymin −5.5296, friction 76118, projections 21300
and self-pushes 4383.

## The triangle-mesh body collider

`guest/drape/body_mesh.{h,cpp}` is driver code on the CPU, in double, like the
analytic primitives.

- **The collider:** `PrimKind::Mesh` in `primitives.{h,cpp}` holds a static
  triangle mesh with a BVH (median split, four triangles a leaf; FoxGirl's
  10171 triangles give 8053 nodes).
- **The query:** the closest point uses Ericson's region test. The side comes
  from the region's angle-weighted pseudo-normal (Bærentzen & Aanæs).
- **Contact and projection:** the contact is the capsule's rule, with a skin
  of 0.1 and a band of 0.1. The projection Jacobian is the face's
  I − NNᵀ, or the edge or corner form, for the recompute backward.
- **The API:** `drape_primitive_mesh(positions, triangles, [skin, mu, band,
  depth])` adds it, and `project/main.gd` has its no-argument wrapper
  (rule 8).

Over 30 steps at scale 10 ([`collider_*.txt`](.), cpu and rd agree):

| collider | skirt vertices inside the body after 30 steps | deepest | rd ms/step |
|---|---|---|---|
| body mesh | **0** | 0 | 46.6 |
| 14 capsules | 186 | 0.186 (1.9 cm) | 40.1 |
| none (control) | 387 | 0.334 (3.3 cm) | 20.8 |

"Inside" means the signed distance to the body surface is below 0. All three
start with 0 inside. Over 100 rd steps ([`probe_body_100.txt`](probe_body_100.txt))
the body collider costs 43.6 ms/step against the capsules' 37.7, and the
steps are finite. It was cheap, so the loop can use the body in place of the
capsules. A point more than `depth` (default 1.0 drape unit) inside the body
is not seen; the fit starts intersection-free, so that bound does not bind
here.

## Observed, not fixed here

- **The skirt stretches at scale 10.** After 100 steps, ymin is −3.28 with
  the body collider and −5.53 with the capsules, 33 and 55 cm below the feet
  in body metres. The hem starts at 2.85.
  - Mass and the membrane stiffness (area · kTri) both grow as s², and
    gravity as s, so the cloth sags about s times more against its own
    stiffness at scale s.
  - At scale 1, ymin stays 0.29 after 100 steps (Gate 8's `drape-fs1.txt`).
  - The drape scale is a material choice for the loop, not only a unit
    change.
- **The membrane kernels divide by |f0| and by the in-plane height d**
  (`triangle_membrane_force_al`, `_dual_update`). They would give NaN on a
  collapsed current triangle. No such triangle arose here.

- **G9's drape crossover moved on this boot** ([`g9_this_elf.txt`](g9_this_elf.txt),
  [`g9_main_elf.txt`](g9_main_elf.txt)). At 90 fps, cpu at 196 vertices
  measured 18.8-19.5 ms against Gate 5's 24.8, so cpu is cheaper up to 196,
  rd is cheaper from 256, and G9 fails its "auto at the crossover" check.
  main's own drape.elf, run the same hour, gives the same table (18.82 ms at
  196 vertices, the same FAIL), so the move is the machine, not this cut. The
  `auto` threshold (160) is left alone.

## Files

- **The scene:** `fitted_skirt.obj`, `fitted_skirt_pins.txt`,
  `fitted_skirt_capsules.txt` and `body.obj`. `skirt_prep.py` regenerates the
  first three from Gate 8's integration evidence.
- **Analysis scripts and their output:** `bend_check.py` →
  `bend_check.txt`, and `fma_check.py` → `fma_check.txt`.
- **The probe:**

  ```
  godot --path project --script probe_drape_skirt.gd --rendering-driver vulkan --xr-mode off -- "variants=rd:10;rd:10:bending=0" steps=1 [caps=1] [body=1] out=probe.txt
  godot --path project --script probe_drape_skirt.gd --rendering-driver vulkan --xr-mode off -- job=mesh_bisect jobargs=scale=10 out=bisect_fixed.txt
  ```
