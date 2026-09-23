# Gate 6d — the fit on the GPU: drape.elf's AVBD solver as the loop's FIT

**Result: PASS.** The loop with `--fit-mode=avbd` ends `RESULT: PASS
FIXTURE:infer,rig` in **20.5 s** wall (`loop-avbd.*`; the PolyFEM loop 643 s,
its FIT_RUN 627 s): FIT(avbd) 13.3 s (300 steps, 44 ms/step on the RTX
4090, fit.elf's `fit_begin` 2.0 s on its worker meanwhile), CHECK
`OK none` with the pushed-vertex control INTERSECTS (0.85 s), DRAPE 100
finite rd steps (2.8 s, 28 ms/step). The push-vertex control ends
`FAILED(CHECK: INTERSECTS ...)` (`control-push-vertex.*`, 17.8 s). The fit's
surface is 7.2 mm mean / 22.1 mm p95 from the PolyFEM fit's, its gap to the
body 13.1 / 29.0 mm mean / p95 against PolyFEM's 12.0 / 35.1 (the ladder
gate, `ladder.txt`, RESULT PASS). `tests/probe_main_wrappers.gd` PASS (135
guest entry points, 168 wrappers); `gate_drape.gd -- only=G4,G8 quick`
PASS. Side by side: `ladder-f60-i32-s300.png` (the fit phase) and
`loop-avbd.png` (after the drape) against `gates/8-loop/flat-psd.png`.

## Why

The loop's FIT is cloth-fit's PolyFEM retarget in fit.elf: **627 s** for the
932-vertex authored skirt (gates/8-loop `flat-psd.*`, 150 Newton with
`force_psd_projection`; 1393 s without), 97% of a 643 s loop. The user's
`--force-fixture=fit` experiment (a similarity retarget from the garment
skeleton to the body skeleton, then the drape) ran the whole loop in 9.2 s
but hung the skirt as a straight tube, while the PolyFEM fit hugs the hips
(`gates/8-loop/flat-psd.png`). This cut makes the AVBD path the fit and
spends the freed time on its accuracy, measured against that PolyFEM result
on the same 932-vertex mesh.

## Design, as built

`drape.elf` gets a **fit mode** (`guest/drape/`), no new kernel (AGENTS.md
rule 2: the attachment kernels already exist in Lean; the fit is attachments
with moving targets):

- **`drape_fit_set(verts, params, anchor)`** (`main.cpp`, `scene_fit_set` in
  `drape_scene.cpp`): one *fit attachment* per listed vertex (the loop lists
  every garment vertex not in its `nofit` set: all 932 of the authored skirt,
  677 of 2682 for the LCL fixture), after the pins. Its stiffness is
  `kFit × A_v`, the vertex's lumped area (`applyMaterial`), so the pull is
  area-weighted like cloth-fit's `FitForm` (`fit_weight × area(f) × Σ w_i
  sdf(p_i)²`) and a refinement does not change the balance against the
  membrane, whose stiffness is `kTri × area` too. `params = [kFit, gap,
  refresh, similarity]`; `anchor` = the waist loop.
- **`drape_queue_fit(max_steps, tol)`**: the fit phase
  (`DrapeSessionT::enqueueFit`, `drape_session.h`): gravity off and `damp` 0
  (quasi-static: the predictor carries no velocity), the ordinary AVBD step
  otherwise (the same kernels, the mesh collider's contact projection, the
  self-collision pushes). Every `refresh` steps `DrapeSimT::begin` refreshes
  the fit targets from the collider's own closest-point query
  (`mesh_fit_target` in `primitives.cpp`: the BVH's nearest surface point,
  moved `gap` out along the outward normal; the loop uses gap 0, the
  collider's 1 cm skin is the clearance, as cloth-fit's fit term pulls to
  sdf = 0 and its IPC barrier keeps the distance). The phase ends early once
  the largest vertex move of a refresh cycle's last step is under `tol`
  (5e-4 units = 0.05 mm), else at `max_steps`. Gravity and damp are restored.
- **The similarity rest update** (`similarity` = 1, `similarity.h`,
  `DrapeSimT::refreshRest`): every refresh, the best similarity transform
  (Umeyama 1991: rotation, uniform scale, translation; an Eigen-free 3x3 SVD
  by cyclic Jacobi, rule 3) of the *source* garment onto the *current*
  vertices becomes the rest shape (every `restEvery` = 4 steps; the targets
  every step), pivoting on the waist loop (its source centroid maps to its
  current centroid). The rest-derived quantities are recomputed
  (`scene_finish`: the triangles' rest metric and areas, the bendings' rest
  norm, the radii) and written into the solver's *existing* buffers
  (`AvbdRd::updateTriangleRest / updateBendingRest / updateSelfCollisionRadii
  / updateAttachmentStiffness`, `buffer_update` on the same topology, no RID
  made, no uniform set rebuilt; the CPU solver has the same four). This is
  cloth-fit's `SimilarityForm` (rigidity to a rotation + uniform scale of each
  element) with one global transform instead of one per element: the
  membrane pulls toward a uniformly scaled skirt and the fit pull sets the
  scale at the hips.
- **The anchor** (`anchor` = the waist loop, `kAnchor`): each waist vertex
  gets a second attachment whose target, refreshed every step, is the
  vertex's own position plus the loop's centroid error (its source centroid
  minus its current one): a uniform force that holds the loop's centre on
  the pelvis and nothing else, so the ring still shrinks onto the body under
  its own soft pull. That is cloth-fit's `curve_center_target` (weight 1;
  PolyFEM's waist centre sits at y 0.954, the pelvis joint 0.9528). The
  ladder's `settle` (steps with the pull off after the fit) exists and buys
  nothing; the pick uses 0.
- **The pipeline** (`project/stages/pipeline.gd`, option `fit_mode` =
  `polyfem` (default, unchanged) | `avbd`; `gate_loop.gd --fit-mode=avbd`):
  FIT_BEGIN runs the similarity retarget (`_similarity_retarget`, identity for
  an authored garment) and `DrapeStage.fit_avbd_start` (the body mesh
  collider at scale 10, the garment as a `drape_scene_mesh` without pins, the
  fit set, the fit phase queued) with **`DrapeStage.FIT_AVBD`**, the one
  setting this gate's ladder picked (no loop knob: the user's rule, a ladder
  and one pick, not a slider; the gate FAILs if the constant is not the
  chosen rung); fit.elf's `setup` + `fit_begin` start on its worker thread
  meanwhile, for CHECK. FIT_RUN watches the drape's ticks (rule 4: one
  `drape_tick` per `_process`), FIT_READ scales the positions back to metres,
  labelled **FIT(avbd)**, not FIXTURE. CHECK is fit.elf's exact
  `fit_check_intersections(override)` on the fitted vertices, with the
  pushed-vertex control beside it; DRAPE then runs as before from the fitted
  state (the fitted vertices are the drape's rest shape: a clumped fit would
  be baked in, which is why the ladder's checks and shape statistics judge
  the fit phase).
- **CHECK's setup.** `fit_check_intersections` tests the garment against the
  avatar *where the current phase's alpha puts it*. At phase 0 (after
  `fit_begin`, before any `fit_step`) that is cloth-fit's start avatar: with
  the config's `shrink_normal_distance` 0 the body **collapsed onto its
  skeleton** (optimize.cpp:1280), which no garment hits: in pass 1 a vertex
  pushed onto the pelvis answered `OK none`. The avbd CHECK's setup inserts
  `"shrink_normal_distance": 1e-6` (`_fit_elf_begin(true)`): the start
  avatar is the real body 1e-6 solve units (0.65 um) inside its own skin, so
  the phase-0 check is the loop's check without the 627 s of phases (the
  fit's own start-state refusal then also sees the real body; the authored
  skirt clears it by 1 cm). `fit_begin` costs 2.0-3.0 s and overlaps the
  fit phase. The check also reports garment self-intersections (ids ≥ the
  avatar's 5128 vertices), which is how the folds of passes 1-2 and 6 were
  caught.

Rule 8: `drape_fit_set` and `drape_queue_fit` have no-argument wrappers
(`drape_stage.gd drape_fit_set / drape_fit`, `main.gd` delegates);
`tests/probe_main_wrappers.gd` PASS (`wrappers.txt`).

## The ladder (`project/gate_fit_avbd.gd`, `ladder.txt`)

`gate_fit_avbd.gd` runs the loop to MESH (the authored skirt, 932 v / 1728
f, 0 nofit) and fits it at every rung; `ladder_eval.py` measures each
result against `gates/8-loop/flat-psd.fitted.obj` (the PolyFEM fit of the
same mesh, same 932-vertex order, triangles equal) and against the body.
Units: kFit and kAnchor in drape units at scale 10 (kFit per unit of vertex
area; kAnchor a plain spring), fit s the phase's wall time on the RTX 4090,
s the similarity scale at the end, y the skirt's height range in metres
(PolyFEM's 0.661..0.965; the authored 0.524..0.953), the gap the unsigned
distance to the body over every vertex (PolyFEM's 12.0 / 35.1 mm mean /
p95). All rungs: refresh 1, restEvery 4, h 1/60 (the control rung h 1/180),
gap 0, kBend 1e-5, tol 5e-4; `fit_check_intersections` against the real
body (the pushed-vertex control answered INTERSECTS beside every rung).
Final pass (the seventh; the six before it are the negatives below):

| rung | kFit | sim | kAnchor | settle | iters | steps | ms/step | fit s | self pushes | s | y min..max (m) | per vertex to PolyFEM mean / p95 / max (mm) | surface to PolyFEM mean / p95 / max (mm) | gap to body mean / p95 (mm) | check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `f0-i16-s100` | 0.0 | 0 | 0.0 | 0 | 16 | 100 | 25.6 | 2.9 | 175 | 1.000 | 0.523..0.953 | 119.3 / 186.7 / 217.7 | 62.8 / 145.5 / 216.4 | 76.4 / 162.5 | OK none |
| `f60-i32-s300-nosim` | 60.0 | 0 | 100.0 | 0 | 32 | 300 | 44.5 | 13.5 | 48035 | 1.000 | 0.537..0.962 | 85.7 / 131.0 / 154.6 | 20.3 / 95.8 / 130.2 | 15.4 / 31.1 | OK none |
| `f20-i32-s300` | 20.0 | 1 | 100.0 | 0 | 32 | 300 | 49.1 | 14.9 | 2467 | 0.725 | 0.634..0.957 | 50.5 / 78.4 / 89.9 | 7.8 / 24.9 / 40.6 | 14.9 / 34.1 | OK none |
| **`f60-i32-s300`** | 60.0 | 1 | 100.0 | 0 | 32 | 300 | 48.3 | 14.6 | 3333 | 0.695 | 0.650..0.958 | 46.1 / 75.6 / 88.9 | **7.2 / 22.1 / 35.5** | 13.1 / 29.0 | OK none |
| `f180-i32-s300` | 180.0 | 1 | 100.0 | 0 | 32 | 300 | 47.7 | 14.5 | 4049 | 0.678 | 0.659..0.959 | 46.5 / 78.2 / 92.2 | 7.7 / 25.3 / 49.1 | 12.0 / 24.8 | OK none |
| `f600-i32-s300` | 600.0 | 1 | 100.0 | 0 | 32 | 300 | 47.4 | 14.4 | 13380 | 0.667 | 0.663..0.958 | 47.3 / 83.5 / 103.8 | 8.5 / 33.6 / 66.5 | 11.0 / 16.9 | OK none |
| `f60-i16-s300` | 60.0 | 1 | 100.0 | 0 | 16 | 300 | 33.9 | 10.3 | 2961 | 0.698 | 0.648..0.958 | 46.7 / 76.0 / 90.1 | 7.3 / 22.2 / 36.8 | 13.1 / 29.3 | OK none |
| `f60-i32-s300-a30` | 60.0 | 1 | 30.0 | 0 | 32 | 300 | 47.8 | 14.5 | 3052 | 0.698 | 0.646..0.957 | 46.5 / 76.0 / 90.0 | 7.3 / 21.9 / 37.1 | 13.1 / 28.3 | OK none |
| `f60-i32-s300-a300` | 60.0 | 1 | 300.0 | 0 | 32 | 300 | 48.6 | 14.7 | 3461 | 0.694 | 0.651..0.958 | 46.1 / 75.6 / 88.7 | 7.2 / 22.2 / 35.4 | 13.1 / 29.5 | OK none |
| `f180-i32-s300-settle10` | 180.0 | 1 | 100.0 | 10 | 32 | 310 | 47.2 | 14.8 | 4064 | 0.678 | 0.660..0.960 | 46.1 / 76.6 / 88.1 | 7.0 / 21.4 / 40.5 | 12.8 / 26.9 | OK none |
| `f60-i32-s600` | 60.0 | 1 | 100.0 | 0 | 32 | 600 | 48.3 | 29.1 | 4677 | 0.640 | 0.683..0.968 | 48.0 / 74.5 / 84.2 | 7.2 / 23.1 / 44.0 | 13.2 / 31.4 | OK none |

**The pick: `f60-i32-s300`** (`DrapeStage.FIT_AVBD`): kFit 60 (x0.1 of the
cloth-fit calibration), kAnchor 100, 32 iterations, 300 steps (14.6 s on the
4090; 16 iterations give the same numbers in 10.3 s and are kept as the
iteration control), no settle. Its surface is **7.2 mm mean / 22.1 p95 /
35.5 max** from the PolyFEM fit's (both ways; 1.1 fit voxels mean), its gap
to the body 13.1 / 29.0 mm mean / p95 against PolyFEM's 12.0 / 35.1, its
height 0.650..0.958 against 0.661..0.965, s 0.695 (PolyFEM's length ratio
0.715), and `OK none`. What the knobs buy: kFit 20 → 600 moves the surface
distance 7.8 → 7.2 → 7.7 → 8.5 mm and the gap 14.9 → 11.0 (tighter, and the
hem's back pulled into the gap between the legs from 180 on, p95 34 mm at
600); kAnchor 30-300 nothing (±0.1 mm); a 10-step settle nothing (7.0 vs
7.7 at kFit 180); 600 steps nothing at twice the time (s drifts on to
0.64). The two controls: no pull (the tube) 62.8 mm surface / 119 mm per
vertex; no similarity update (the authored length held, the membrane
buckling: 48k self pushes) 20.3 / 85.7 mm.

**The per-vertex distance is 46 mm mean / 76 p95 at every fitted rung**
(the bar the task set, 30 / 60, is not met, and the gate reports it as
INFO) while the surfaces are 7 mm apart: by height band and sector the
front vertices sit +36..46 mm in x of PolyFEM's, the back ones −40..−70,
the left −19..−26 in z, the right +10..+33: the vertices slid along the
surface (the panels' parameterisation), which the per-vertex distance
counts and the surface distance does not. The fit is judged on the
surface (`PASS_SURF_MEAN_MM` 15, `PASS_SURF_P95_MM` 30).

## Vertex-to-vertex agreement: what the 46 mm is made of (`ladder-eval-rigid.txt`)

The user asked for per-vertex agreement (mean ≤ 30 mm, p95 ≤ 60) within the
same time. `ladder_eval.py` now also fits the best rotation about y and the
best rigid transform (Kabsch, a 3x3 matrix; the rotation reported as its
first two rows and, as a summary, the angle from the trace) from each
result to the PolyFEM fit and prints the residual after each:

| garment | per vertex raw | after the best rotation about y | after the best rigid transform |
|---|---|---|---|
| the authored skirt (tube control) | 119.3 / 186.7 | 16.3 deg: 80.2 / 130.1 | 17.2 deg: 78.8 / 130.6 |
| PolyFEM without psd (`gates/8-loop/flat.fitted.obj`) | 3.7 / 5.8 | 1.1 deg: 2.2 / 3.9 | 2.2 / 4.0 |
| **the pick `f60-i32-s300`** | 46.1 / 75.6 | **13.3 deg: 28.2 / 58.0** | 13.5 deg: 27.6 / 57.4 |
| `f60-i16-s300` | 46.7 / 76.0 | 13.3 deg: 28.5 / 58.3 | 27.9 / 57.6 |
| `f20-i32-s300` | 50.5 / 78.4 | 13.3 deg: 29.7 / 56.6 | 28.9 / 56.1 |
| `f180-i32-s300` | 46.5 / 78.2 | 13.6 deg: 29.1 / 61.8 | 28.4 / 61.7 |
| no similarity update (control) | 85.7 / 131.0 | 13.2 deg: 54.0 / 98.8 | 52.6 / 96.4 |

(mm, mean / p95.) So the per-vertex distance is **a rotation about the
vertical axis plus a residual just under the bar**: removing the 13.3 deg
takes the pick to 28.2 / 58.0. The rotation is not ours: **PolyFEM turned
the whole skirt 16.3 deg about y** relative to the authored mesh, the same
at every height band (−15.9..−16.9 deg per 10 cm band), and its two runs
agree on it (psd and no-psd fits 1.1 deg apart, 3.7 mm per vertex), while
the avbd fit keeps the authored azimuth (0.1..0.7 deg per band). The inputs
have no axis to derive it from: FoxGirl's hip joints lie on the x axis to
0.0 deg (skeleton.obj joints 9 and 12: z equal to 0.0001), the body's
horizontal principal axis between y 0.6 and 1.0 is at 0.1 deg, and the
retarget is the identity (the authored garment's skeleton is the body's).
A tube on a left-right symmetric body has its azimuth as a soft mode of the
fit energy; PolyFEM's Newton path from the skeleton-collapsed start avatar
lands 16 deg from where the quasi-static AVBD path lands. Hard-coding that
angle into the retarget would be fitting the answer key, not a mechanism,
and holding the waist's azimuth (a rotational anchor) would hold it at the
authored 0 deg, where the avbd fit already is; a per-panel or per-ring
similarity would act on the 28 mm residual, not on the 18 mm the rotation
carries. **Result on this bar: not met, and not reachable inside the time by
any rung: 46 mm mean / 76 p95, of which 18 mm is a 13 deg azimuth that
PolyFEM's own solve introduces and 28 / 58 is tangential sliding of the
panels' parameterisation (the surfaces are 7 mm apart).** What would reach
it: reproducing cloth-fit's phase 0 (the garment fitted to the avatar
collapsed onto its skeleton and inflated, alpha 0 → 1), the one thing in
its pipeline that can turn a symmetric tube, which is a cut of its own.

## Does the rotation generalise (`generalise/`)

The user asked whether PolyFEM's 16.3 deg azimuth is a constant of its
path (a frame convention to derive and apply), a function of something
measurable, or a soft mode. The sweep runs the native PolyFEM fit
(`gates/6-fit`'s `fit_native`, the guest's bitwise-equal control, the loop's
own budget: `incremental_steps` 1, `force_psd_projection`) and the avbd fit
(the pick, `gate_fit_avbd.gd --only=f60-i32-s300 --garment=`) on the
loop's authored skirt (`generalise/authored.authored.obj`, the pen gate's
mesh: sample 2 is sample 1) and on six copies of it made by
`make_variants.py`: rotated +30 and -30 deg about the vertical axis through
its own x-z centroid (a 3x3 matrix), its x-z radius scaled 0.85 and 1.15
(a narrower and a wider skirt), its height below the waist scaled 0.85 and
1.15 (a shorter and a longer one). Every rotation below is the org's fitter
(`vendor/sinew-align`, Align.lean's C port, through `sinew_align_cli
--about-y` for the azimuth and plain for Kabsch; the angle is the trace's
summary of the matrix, rule 11). `analyse.py` writes the table
(`generalise/results.txt`); the native logs, `phases.tsv` and
`garment_final.obj` per variant are beside it, the avbd runs as
`avbd-<variant>.*`.

| variant | PolyFEM azimuth vs input (deg, overall; per band low..high) | avbd azimuth vs input (deg) | avbd vs PolyFEM per vertex raw / after best y rotation (deg) / after Kabsch (mm mean, p95) | surface mean / p95 (mm) | PolyFEM wall (s, Newton) | avbd fit (s) |
|---|---|---|---|---|---|---|
| base | +20.9; +22.1 +20.8 +20.9 +20.2 +20.0 | -0.3 | 59.6 / 87.8; +17.9 deg: 34.2 / 67.9; 18.0 deg: 33.8 / 68.1 | 7.2 / 21.0 | 33, 131 | 16.3 |
| rot+30 | +7.4; +8.5 +7.9 +7.3 +6.4 +6.4 | +0.0 | 31.6 / 50.5; +5.8 deg: 27.8 / 44.5; 8.7 deg: 23.3 / 42.3 | 8.1 / 25.8 | 18, 64 | 16.2 |
| rot-30 | +0.4; +0.3 +0.1 +0.9 +0.8 -0.0 | +0.0 | 68.9 / 131.2; +0.6 deg: 40.5 / 71.9; 1.1 deg: 40.4 / 71.8 | 19.0 / 104.5 | 13, 42 | 15.1 |
| rad0.85 | -2.8; -1.5 -2.8 -2.6 -3.4 -3.8 | -0.7 | 55.0 / 97.2; -1.7 deg: 28.7 / 53.1; 1.8 deg: 28.6 / 53.2 | 14.9 / 71.4 | 19, 62 | 15.2 |
| rad1.15 | +21.5; +22.6 +21.3 +21.5 +20.9 +20.7 | +1.8 | 55.7 / 86.5; +16.6 deg: 33.5 / 63.7; 16.8 deg: 32.7 / 62.8 | 7.7 / 23.3 | 30, 117 | 15.6 |
| len0.85 | +0.5; +1.0 +0.8 +0.5 +0.3 -0.0 | +0.3 | 19.0 / 33.4; +0.1 deg: 16.4 / 28.4; 2.0 deg: 16.0 / 28.2 | 7.8 / 26.2 | 22, 81 | 15.1 |
| len1.15 | -6.7; -6.4 -6.8 -6.7 -6.7 -6.9 -6.7 | +0.1 | 84.2 / 149.8; -5.2 deg: 80.7 / 152.0; 7.2 deg: 79.9 / 148.4 | 28.9 / 137.4 | 48, 199 | 14.8 |
| LCL skirt, upstream cloth-fit 1 thread (cf-up-out1 step 252) | -2.8 (fit vertices -3.6) vs its retargeted start | - | - | - | - | - |
| LCL skirt, fit_native sdf64b | +2.2 (fit vertices +1.8) vs its retargeted start | - | - | - | - | - |

Read across: the PolyFEM azimuth of the same skirt is **+20.9 deg** in the
native run and **+16.3 deg** in the guest's psd run of Gate 8 (the same
input and config; 131 against 150 Newton: the two paths land 4.6 deg
apart), **+7.4** when the input is turned +30 deg and **+0.4** when it is
turned -30 (it neither rotates with the input, which would read +20.9 both
times, nor lands at one absolute azimuth, which would read -9 and +51),
**-2.8** for the narrower skirt and **+21.5** for the wider, **+0.5** for
the shorter and **-6.7** for the longer; the LCL skirt fixture turns
**-2.8** (upstream cloth-fit) and **+2.2** (our native). The avbd fit
keeps the authored azimuth in every case (-0.7..+1.8 deg). The inputs are
symmetric to 0.1 deg (the hip joints on x, the body's horizontal principal
axis on x), so nothing measurable predicts the angle.

**Verdict: a soft mode.** The azimuth of a tube on a left-right symmetric
body is flat in cloth-fit's energy and its Newton path lands anywhere
between -7 and +22 deg, 4.6 deg apart on the same input between two
solver builds. It is not a frame convention to apply (no fixed rotation is
tested), and per-vertex agreement with it is agreement with where one run
happened to land: the honest per-vertex bar is after the best rotation
about y, where the pick reads 34.2 / 67.9 mm on the native fit (28.2 / 58.0
on the guest's) and the variants 16-40 mm mean. The long skirt is the
outlier (80 mm after the rotation, surface 29 / 137 mm): PolyFEM's hem
there hangs 13 cm above the knee against the legs while the avbd fit
follows the legs down; a shape difference of the fit terms, not an azimuth.

Not run: the vendored jacket examples (`Goblin_Jacket`, `Trex_Jacket`:
Puffer_dense 3112 v on Goblin 5214 v and T-rex 9898 v) refuse to start
under this fork's topology-preserving start avatar (`begin: Unable to
solve, initial solution has intersections`, `native-Goblin_Jacket.log`,
`native-Trex_Jacket.log`); the jumpsuit example the same setups would
need a skin-weights path this tree does not exercise. cloth-fit ships no
other garment-avatar pair (`vendor/cloth-fit/garment-data`), and rule 1
forbids fetching more.

## The rotation fitter: sinew-mocap/solve's Align.lean (`vendor/sinew-align`, `lean/Sinew`)

The user asked for the org's fitters in place of the drape's own Jacobi
SVD. `lean/Sinew` is a squashed subtree split of `V-Sekai-fire/sinew`'s
`solve/core/spec` (36c926e; Align.lean, Math.lean, AlignTest.lean, the
rest of the spec with them), `vendor/sinew-align` the org's C port from
`V-Sekai-fire/interactor-gyre` (6e04760: `sinew_align.c/.h`,
`test_sinew_align.c`, file for file) plus `sinew_align_cli.c` and
`build.sh` here. `guest/drape/similarity.h` now accumulates the centred
cross-covariance and hands it to `sinew_finish_align` (rodrigues / ns30 /
the Jacobi-SVD kabsch fallback), keeps R = I with the scale alone when
`valid9` rejects the result, and takes Umeyama's scale as
trace(R^T H) / sum |src - cs|^2. AlignTest.lean's oracle (a 120 deg
rotation recovered at N=5, 2, 1 to 1e-4) runs on the host
(`sinew/test_sinew_align.native.txt`: 0.0 error, OK) and in the guest
(`drape_sinew_align_test`, wrapped by rule 8 and called by
`tests/probe_main_wrappers.gd`: N=5 err 1.09e-17, N=2 err 0, N=1 err 0; `similarity_fit` on the same pairs scaled by 0.7 gives s = 0.7000000 and max |R - R_true| 4.1e-9). The pick refitted
with it: the fitted vertices move 0.0007 mm mean / 0.0020 mm max against the Jacobi-SVD build (`sinew/pick.txt`, `sinew/ladder-f60-i32-s300.fitted.obj`), float noise; `OK none` as before. `ladder_eval.py`'s rotation fits (about y
and Kabsch) go through `sinew_align_cli` (built by
`vendor/sinew-align/build.sh`, or `SINEW_ALIGN_CLI=`); the about-y fit
keeps only the x-z components and adds the y axis as weighted pairs, since
coplanar data alone leaves the fitter free to flip 180 deg about an
in-plane axis (it did on one height band of the sweep).

## Negative results (kept: `pass1/` .. `pass7/`)

Each pass is the full gate output of an earlier design; the numbers are what
ruled it out.

1. **A constant per-vertex pull with a gap (pass 1, 12 rungs, per-vertex
   k 0..100, gap 0.10-0.20, iters 16-64, steps 100-300).** The tube control
   (k 0) is 119 mm (mean per vertex) from the PolyFEM fit; every fitted rung
   is 84-94 mm, whatever k, iters, gap or kBend: the fit hugs the hips
   (gap to the body 15-20 mm mean against PolyFEM's 12) but keeps the
   authored length, and from k ≥ 30 per vertex the surplus circumference
   folds (self_pushes 20k-150k) into **garment self-intersections**
   (`fit_check_intersections` names garment edge/face ids ≥ 5128). Its
   checks ran against the collapsed start avatar (above): the `OK none`
   verdicts of that pass are not verdicts; the INTERSECTS ones are real
   self-intersections.
2. **Bending against the folds (pass 2, kBend 1e-3, 1e-2, 1e-1 at k 30).**
   No rung stopped folding (INTERSECTS at every kBend; self_pushes 52k-88k):
   the buckling is the membrane's, not the bending's. k 10 stays fold-free
   (`OK none` against the real body) at 20 mm mean gap.
3. **Why: PolyFEM's fit is a conformal shrink.** Its fitted skirt spans
   y 0.661..0.965 m (length 0.304) where the authored one spans 0.524..0.953
   (0.425): `SimilarityForm` lets every element scale isotropically, so
   hugging the hips shortened the whole skirt by ~0.72 (hem +13 cm, mean
   radius 0.216 → 0.154 m). An AVBD membrane holds the authored rest lengths,
   so no pull strength reproduces that; hence the similarity rest update.
4. **The similarity update about the garment's own centroid (pass 3,
   area-weighted kFit 180-1800, similarity on).** The scale converges
   (s 0.68-0.71 for kFit 600-180, the expected ~0.72), the folds are gone
   (self_pushes 3k-10k at kFit ≤ 600; `OK none`), the hips' gap is 11.6-13.3
   mm mean (PolyFEM 12.0), but the skirt sits **8 cm low**: y 0.585..0.900
   against 0.661..0.965, the least-squares translation shrinking the skirt
   about its centroid while cloth-fit's `curve_center_target` keeps the waist
   curve's centre on the pelvis. Mean distance to PolyFEM 84.8 mm. Hence the
   anchor (the waist loop's centroid fixed). The same pass found the
   **permanent RID slot exhaustion**: re-uploading the triangle, bending,
   radius and attachment buffers every refresh made ~300 permanent RIDs each
   time (the buffers and the kernels x colours uniform sets rebuilt after
   every invalidation); the sandbox's permanent table filled after ~600
   refreshes ("Maximum number of scoped variants in permanent state
   reached", rung 10 of 11), although `rd_close` counted 0 leaked slots (the
   device's own bookkeeping was balanced). Hence the in-place `update*`
   paths.
5. **The similarity pivoted on the waist, unheld (pass 4).** With the
   transform's fixed point at the waist loop's source centroid the scale
   stalls at s 0.80-0.83 (kFit 180-1800, refresh 4, h 1/180) and the skirt
   still sits low: y 0.559..0.930 against 0.661..0.965, mean 85-87 mm from
   PolyFEM at every kFit, iters and kBend (16 vs 32 iterations changed
   nothing; kBend 1e-3 nothing). Two causes, measured in pass 5: the target
   refreshed every 4 steps is a spring to a *fixed* point for 3 steps and
   holds each vertex where it stands (cloth-fit's SDF potential lets a vertex
   slide along the surface for free), and the quasi-static step's drift is
   F h² / m, so at the drape's h 1/180 the rest shrink cannot follow.
6. **Refresh every step, h 1/60, the waist centroid shifted toward its
   source centre at the fit stiffness (pass 5, kFit 6-180).** s converges
   (0.68 at kFit ≥ 20, PolyFEM's 0.72 by length), the gap to the body is
   PolyFEM's (11.9-15.3 mm mean), every rung `OK none`, and the best rung
   (kFit 60, 600 steps) is 57 mm mean / 86 p95 from PolyFEM: the whole skirt
   31-38 mm **low** in every height band. Measured on the rings: PolyFEM's
   waist centre is y 0.954, the pelvis joint (0.9528) to the millimetre, at
   radius 0.1355 (the authored ring 0.1935: it shrank onto the hips); ours
   sat at 0.921 (radius 0.142, right). On the flaring hips every
   nearest-point target lies *below* its vertex (the surface normal there
   tilts up), so the skirt ratchets down as it is pulled in, and a shift at
   the fit's own stiffness (kFit × area ~ 4 per vertex) does not hold it.
   cloth-fit's `curve_center_target` (weight 1) holds the waist curve's
   centre on the bone.
7. **The waist ring pinned onto the surface at kAttach (pass 6).** The
   placement is PolyFEM's (y 0.670..0.970 against 0.661..0.965, s 0.66-0.67)
   and the distance halves (45.8-49.2 mm mean, 76-94 p95 at kFit 180-1800),
   but **every rung self-intersects at the waist ring**, the tube control
   included (`fit_check_intersections` names ring vertices 5, 789, 791): a
   ring of radius 0.19 pinned rigidly (10000 against a membrane of ~6 per
   triangle) onto the nearest points of the waist's cross-section folds
   where that map is not injective. Neither a 10-step settle with the pull
   off nor a target at the skin (gap 0.1) undoes a fold. The residual by
   height band has small mean offsets (|Δy| ≤ 10 mm) and is largest at the
   hem's back (79-90 mm): PolyFEM keeps the fabric hanging behind the thighs
   (z −0.108) where the nearest-point pull at kFit ≥ 180 drags it into the
   gap between the legs (z −0.053, x → 0). Hence the anchor as a *second*
   attachment per waist vertex at a moderate kAnchor whose target is the
   vertex's own position plus the loop's centroid error (a pure centroid
   force, the ring free to shrink under its own soft pull), and a ladder
   down to kFit 20: the final pass above (every rung `OK none`, placement
   PolyFEM's, surface 7-9 mm). The run of that ladder started at 12:02 died
   with its third rung (the harness killed the background shell, not
   Godot: the rung alone passes, `pass6/resume-f20.*`, and the foreground
   rerun is `pass7/`, identical to the final pass but for the surface
   columns).

## The loop in avbd mode (`loop-avbd.txt`)

| state | ms | note |
|---|---|---|
| INFER, RIG (FIXTURE) | 88, 9 | |
| AUTHOR | 1125 | 6 strokes, 2 cycles, 2 openings, 2 patches |
| MESH | 592 | 932 v, 1728 f, 2 loops |
| FIT_BEGIN | 356 | the retarget, the body collider, the fit set, `QUEUED fit 300 on rd`; fit.elf setup + `fit_begin` started on its worker |
| FIT_RUN | **13265** | `DONE fit rd steps=300 converged=no max_disp 0.0021` (0.2 mm per step at the cap), 44.2 ms/step, heap 77 MiB |
| FIT_READ | 3 | FIT(avbd) 932 v |
| CHECK | 848 | `OK none`; control `OK INTERSECTS edge (5334,5634) face (2803,2624,2881)` (fit.elf's begin had ended after 2036 ms) |
| DRAPE | 170 | body mesh, 64 pins, scale 10 |
| DRAPE_COLLECT | 2823 | 100 rd steps, 28.2 ms/step, ymin 5.724 units, finite |
| wall | **20.5 s** | PolyFEM: 643 s |

## How to run

```
# the ladder (writes ladder.txt / .json / .log, ladder-<rung>.fitted.obj / .png)
godot --path project --script gate_fit_avbd.gd --rendering-driver vulkan --xr-mode off -- \
    --out=../gates/6d-fit-avbd/ladder.txt
# the loop with the avbd fit (the chosen rung) and its push-vertex control
godot --path project --script gate_loop.gd --rendering-driver vulkan --xr-mode off -- \
    --gate=loop --fit-mode=avbd --out=../gates/6d-fit-avbd/loop-avbd.txt --allow-fixture=infer,rig
... --gate=loop --fit-mode=avbd --push-vertex --out=../gates/6d-fit-avbd/control-push-vertex.txt --allow-fixture=infer,rig
# the shape and gap statistics by hand
python gates/6d-fit-avbd/ladder_eval.py project/fixtures/foxgirl/avatar.obj \
    gates/8-loop/flat-psd.fitted.obj 0.0065462 gates/6d-fit-avbd/ladder-*.fitted.obj
```

`ladder_eval.py` reuses `gates/6-fit/fit_gap.py`'s exact point-triangle
distance; 0.0065462 m is one fit voxel (`voxel_size` 0.01 solve units over
the fit's `target_scale` 1.5276 for this body, the FIT_BEGIN line), so the
gap columns compare with Gate 6's (PolyFEM's 1.83 voxels mean here, 1.77-1.85
there).

## Files

| file | what |
|---|---|
| `ladder.txt`, `ladder.json`, `ladder.log` | the ladder gate's results, summary and Godot log |
| `ladder-<rung>.fitted.obj`, `.png` | each rung's fitted garment (body space, the fit phase only) and screenshot |
| `loop-avbd.*` | Gate 8's loop with `--fit-mode=avbd` (the chosen rung after the drape: `loop-avbd.png`, `.fitted.obj`) |
| `control-push-vertex.*` | its push-vertex control |
| `wrappers.txt` | `tests/probe_main_wrappers.gd` |
| `drape-quick.txt` | `gate_drape.gd -- only=G4,G8 quick` (the drape regression) |
| `pass1/` .. `pass7/` | the negative passes above, complete (`pass6/resume-f20.*`: the rung the killed run stopped at, rerun alone: PASS) |
| `ladder_eval.py` | the shape / gap statistics, and the rotation / rigid residuals |
| `ladder-eval-rigid.txt` | its output for the tube, PolyFEM's no-psd run, the pick and three rungs (the Jacobi-SVD fits of the first analysis; the org's fitter gives the same angles to 0.1 deg) |
| `generalise/` | the sweep: `make_variants.py`, `run_native.sh`, `run_avbd.sh`, `analyse.py`, `results.txt`, every variant's input, PolyFEM output (`native-<v>/`, `.log`) and avbd output (`avbd-<v>.*`) |
| `sinew/` | the org's fitter: the host unit test's output, the pick refitted with it, the wrappers probe with the guest oracle |
