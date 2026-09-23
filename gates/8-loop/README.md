# Gate 8 — join the loop

**Result: PASS FIXTURE:infer,rig, flat and VR, and over MCP.** Cut 8 on
main 8d59e6a with cut-4-tube (curvenet: the skirt's panels, not its caps) and
cut-5-fix (drape: the bending kernels' |s| = 0 guard, the body mesh collider)
merged, every ELF rebuilt. With FoxGirl's body and skeleton as fixtures, the
scripted pen draws the skirt, curvenet gives **2 cycles, 2 openings (waist,
hem), 2 panel patches** and one welded tube (932 vertices, 1728 triangles, 2
boundary loops, Euler 0); fit.elf fits it in **1393 s** (2 phases, 253
Newton), `fit_check_intersections` answers **`OK none`** (its pushed-vertex
control `INTERSECTS`); drape.elf drapes it **100 finite steps on rd**
(`auto`) against the body's own triangles, 21.5 ms/step (`flat.*`). The VR
run on OXRSys (per-process `XR_RUNTIME_JSON`) has the same state sequence,
every integer equal and the fitted vertices equal to 0 (`vr.*`,
`compare-flat-vr.txt`); the MCP drive ends `DONE FIXTURE:infer,rig`
(`mcp/`). Controls: `--drop-seam` ends `FAILED(MESH)` with 0 cycles,
`--push-vertex` ends `FAILED(CHECK: INTERSECTS ...)`. What still stands in:
infer and rig (Cut 7 and Cut 4b).

## Design

`project/main.gd` is now a thin root (node `Main`, so MCP reaches it at
`/root/Main` in `main.tscn` and in `xr_main.tscn`). It adds one stage node
per ELF and the pipeline, and keeps a one-line delegate for every guest
wrapper (AGENTS.md rule 8):

| node | script | ELF | Sandbox (Gate 0F: memory_max before `program=`) |
|---|---|---|---|
| DressOn | `stages/dress_on_stage.gd` | dress_on.elf, probes.elf | refs 65536 / 4096 |
| Drape | `stages/drape_stage.gd` | drape.elf | 1024 MiB, refs 65536 |
| Curvenet | `stages/curvenet_stage.gd` | curvenet.elf | 1024 MiB, refs 4096, timeout 2^24 units |
| Fit | `stages/fit_stage.gd` | fit.elf | 2048 MiB, refs 4096, timeout 2,500,000 units (Gate 6.P), allocations_max 4e6; `fit_configure_with` remakes it |
| Infer | `stages/infer_stage.gd` | infer.elf (none yet) | fixtures only |
| Pipeline | `stages/pipeline.gd` | — | — |

`stages/sandbox_util.gd` makes every Sandbox: `memory_max` (≤ 5112 MiB),
`references_max`, `execution_timeout` and extras are set before
`program=`; a missing ELF, or one without the stage's entry points
(`has_function`), gives `{sandbox: null, reason}` instead of an error.
`stages/stage_base.gd` gives each stage one worker `Thread` and the rule of
one vmcall in flight per sandbox (`busy()`, `start()`, `poll()`).

The pipeline is a state machine advanced from `_process` (rule 4):

```
IDLE → INFER → RIG → AUTHOR → MESH → FIT_BEGIN → FIT_RUN → FIT_READ → CHECK
     → DRAPE → DRAPE_COLLECT → DONE | FAILED(<STATE>: reason)
```

- **INFER / RIG**: FoxGirl's body and skeleton (`project/fixtures/foxgirl/`,
  converted from cloth-fit's garment-data by
  `tools/fixtures/foxgirl_convert.py`: Blender (x, y, z) → Godot (x, z, −y),
  scaled to 1.7 m, feet on y = 0). The skeleton goes through
  `util/skeleton15.gd`, which maps any rig onto cloth-fit's 15-joint layout
  (names first, else the bone graph; identity for FoxGirl). Labelled FIXTURE.
- **AUTHOR**: pen events (`xr/pen_bridge.gd`, from SketchTool or replayed from
  `xr/pen_source_scripted.gd`) → curvenet `pen_begin / pen_point / pen_end`,
  up to 64 events and one stroke end per frame; then `curvenet_build`. A
  begin event carries the stroke's **boundary** mark, set as curvenet's
  `boundary` pen mode just before `pen_begin` (cut-4-tube, Cassie adaptation
  7): a cycle made only of boundary strokes is an opening and gets no patch.
  The scripted rings are boundary strokes, the seams are not; in the headset
  the thumbstick click (or B, or `dress_on_pen_boundary`) toggles the mode.
- **MESH**: `mesh_build(0.03, 1e-5)` on curvenet's worker thread; the mesh must
  be one tube: 1 component, 2 boundary loops. 0.03 m gives 932 vertices, under
  the plan's ≤ ~1000 for the Stage 8 garment.
- **FIT_BEGIN / FIT_RUN / FIT_READ**: body, skeletons (the authored garment's
  source skeleton is the body's own), garment, `fixtures/foxgirl/fit_config.json`
  (cut-6's foxgirl oracle setup without paths) with `incremental_steps` 2 → 1
  (the fit budget, below; `fit_config` in the summary names every edit);
  `fit_begin` and then one `fit_step` per job on the worker thread,
  `fit_status` read between jobs, until it says `done`; then
  `fit_result_vertices`.
- **CHECK**: `fit_check_intersections` must answer `OK none`. Its control
  runs every time beside it: the garment vertex nearest the pelvis moved onto
  the pelvis joint must answer `INTERSECTS`.
- **DRAPE / DRAPE_COLLECT**: `drape_scene_mesh` with the fitted garment, pinned
  on its waist loop (the boundary loop with the highest mean y; 64 pins). The
  body is **drape.elf's triangle-mesh collider** (`drape_primitive_mesh`,
  cut-5-fix: FoxGirl's 10171 triangles, skin 0.1, band 0.1, depth 1.0 drape
  units): over 30 steps on the LCL skirt it left 0 vertices inside the body
  against the 14 skeleton capsules' 186 (`gates/5-drape/skirt/README.md`).
  `--drape-body=capsules` keeps the capsules (radius = the median distance of
  the body vertices nearest each bone, less DiffCloth's 0.1-unit contact
  offset), `none` drapes with no body. The drape runs in a frame scaled by
  **10** (1 m = 10 units, gravity −98; the offset is 1 cm) and the result is
  scaled back. Cut 8 ran at 5 while rd went NaN at 10 on the fitted skirt;
  cut-5-fix found one near-flat bending hinge whose |s| rounds to exactly 0
  under the GPU's FMA and guarded it in the Lean kernels, so the loop is back
  at 10. 100 steps are queued; `drape_tick` runs once per frame; the
  positions must all be finite.
- **Controls.** `--drop-seam` leaves the back seam out (5 strokes). It must
  end `FAILED(MESH: ...)` **and** the curvenet must report fewer cycles than
  the full skirt's 2 (`counts.cycles`, measured, 0 <= cycles < 2): the seam is
  what closes the front and back panels, so that is the failure the control
  is for; a MESH failure for any other reason (the ring caps of `--no-boundary`, say) does
  not pass it. `--push-vertex` must end `FAILED(CHECK: INTERSECTS ...)`.

Every state records its host-timed duration, its stage's vmcall time and
the stage sandbox's heap (`get_heap_usage`) when it ends (`STATE` lines).

**Missing stages.** A stage whose ELF is missing or too old FAILs the run as
`FAILED(<STATE>: <stage> missing (<reason>))` unless `--allow-fixture` names
it; then its fixture stands in and the run is labelled FIXTURE for it:
curvenet → the LCL skirt with its own skeleton and no-fit list; fit → the
similarity map from the garment skeleton to the body skeleton (cloth-fit's
normalisation; identity for an authored garment), CHECK not run; drape → not
run. `--force-fixture` does the same for a stage that is present (to reach
what follows a failing stage).

**Scoring with fixtures** (`gate_loop.gd`, decided in this cut):

| fixture | verdict when every measured criterion passes | why |
|---|---|---|
| none | `PASS` | |
| infer, rig | `PASS FIXTURE:infer,rig` | the critical-path plan runs the loop on FoxGirl's body and skeleton until Cut 7 / Cut 4b land; they are inputs, and no Gate 8 criterion measures them. The label says what stood in. |
| curvenet, fit or drape (any) | `INCOMPLETE` | a criterion is then not measured (cycles, patches, mesh; fit done and intersections; drape steps), so the run is never PASS, whatever else passed |

The same holds for the controls (`PASS (control drop-seam) FIXTURE:infer,rig`,
or `INCOMPLETE` with a curvenet, fit or drape fixture). Any failed criterion is
`FAIL`.

**The pen.** `tools/sync_xr_pen.sh` copies `vendor/xr-grid/addons/procedural_3d_grid`
into `project/addons/` with a CITATION.cff; `gate_loop.gd` checks the copy
file by file (`PEN_SYNC`, the diff −r) and `tools/sync_xr_pen.sh --check` does
the same with diff. `xr_main.tscn` has `XROrigin3D`, two `XRController3D` with
xr-grid's `hand.gd` and `SketchTool` (canvas = `Body`), the Body, and
`PenBridge`, which watches each SketchTool's `active` edge and sends
begin / point / end in Body space (strokes.gd is not used: its
`just_pressed` is always false). `--pen=scripted` (the default) replays
`pen_source_scripted` through the same calls and draws the strokes with
SimpleSketch.

**The scripted skirt** (`xr/pen_source_scripted.gd`): a waist ring at the
pelvis joint's height and a hem ring at the knees', each as two half-rings
meeting front (+z) and back (−z), plus a front and a back seam, 6 open
strokes in body-local coordinates. Ring radius = the largest horizontal
distance of body vertices within 2 cm of the ring's height (hands left out)
+ 1 cm: 0.170 m and 0.196 m on FoxGirl. The straight cone between those
rings passes 2.5 cm *inside* the hips, and `fit_begin` refuses a garment that
starts intersecting the body, so both rings are grown by the deficit
(3.5 cm) to 0.205 m and 0.231 m; the cone then clears the body by 1 cm.
The rings are drawn as boundary strokes. The graph should have 4 knots of
degree 3, 6 edges, 2 cycles (front and back panel), 2 openings (the rings),
2 patches. The Cut 8 task text says "8 curves": 4 knots of degree 3
have 6 edges (handshake lemma), so the gate records the curvenet's count as
INFO and checks cycles, patches and the mesh.

## How to run

Flat (the gate; results stream to `--out`, `<out>.json` is the summary,
`<out>.png` the screenshot):

```
godot --path project --script gate_loop.gd --rendering-driver vulkan --xr-mode off -- \
    --gate=loop --out=../gates/8-loop/results.txt --wallclock=3600 --allow-fixture=infer,rig
# controls
... -- --gate=loop --drop-seam   --out=../gates/8-loop/control-drop-seam.txt  --allow-fixture=infer,rig
... -- --gate=loop --push-vertex --out=../gates/8-loop/control-push-vertex.txt --allow-fixture=infer,rig
# the authoring half only (stops after MESH)
... -- --gate=pen --out=../gates/8-loop/pen.txt --allow-fixture=infer,rig
```

VR (per process, never the system default: AGENTS.md rule 9). OXRSys, the
Windows port, with its Qt simulator showing the stream:

```
XR_RUNTIME_JSON="$PWD/tools/oxrsys/build/windows/runtime/oxrsys-runtime.json" \
  godot --path project --script gate_loop.gd --rendering-driver vulkan --xr-mode on -- \
    --gate=loop --out=../gates/8-loop/results-vr.txt --wallclock=3600 --allow-fixture=infer,rig
```

The VR run must equal the flat run: same STATE sequence, same counts
(cycles, openings, patches, mesh vertices / triangles / loops), same fit and
CHECK answers, drape finite, and the fitted vertices equal to 1e-6:

```
python gates/8-loop/compare_runs.py flat vr      # compares flat.json/.fitted.obj with vr's
```

The gate is a SceneTree script that instantiates `xr_main.tscn`, so it runs
with `--script gate_loop.gd`; `godot ... res://xr_main.tscn -- --gate=loop`
alone plays the scene without the gate (the scene does not read `--gate`). `--quit-after` never
fires under `--xr-mode on`; the gate quits on its own wall clock, and a
watchdog thread kills the process 60 s after it if the frame loop has stalled.

MCP (Gate 0E's shape; plays `xr_main.tscn` flat on port 8795, drives
`dress_on_stages`, `dress_on_run`, polls `dress_on_status`, reads
`dress_on_result`; responses in `mcp/`):

```
gates/8-loop/mcp_run.sh                       # ALLOW=infer,rig by default
```

No-ELF tests: `tests/test_loop_units.gd` (skeleton15, the scripted pen,
mesh_topo; 34 checks), `tests/probe_main_wrappers.gd` (every old and new
wrapper on `/root/Main`, and a source audit that every `ADD_API_FUNCTION` /
`add_sandbox_api_function` in `guest/**/main.cpp` has one, through
`tests/wrapper_audit.gd`), `tests/parse_check.gd`. `run_regress.sh` re-runs
Gates 1, 2, 4 (and Gate 4's wrapper smoke) and 0F on this tree.

## What runs today (cut-8 = main 8d59e6a + Cut 8 + cut-4-tube + cut-5-fix)

Every ELF rebuilt here (`build.sh`, `GGML_SRC` = the org's ggml): dress_on,
probes, drape and fit are byte-identical to the merged branches' builds;
curvenet.elf differs from cut-4-tube's only in the build path strings
(55 `__FILE__` paths one character shorter). Runs on an RTX 4090 / 16
threads; the three long runs (flat, VR, MCP) ran at the same time, each
fit on its own worker thread.

| run | result | file |
|---|---|---|
| parse check: 29 scripts and scenes | **PASS** | `parse_check.txt` |
| units (skeleton15; the scripted pen, now with its boundary marks and the `no_boundary` control; mesh_topo) | **PASS** 34/34 | `units.txt` |
| rule 8, source audit: **114** guest entry points (dress_on 8, probes 36, curvenet 23, drape **30** with `drape_primitive_mesh`, fit 17) reached from 118 `/root/Main` wrappers, every argument defaulted; controls as before | **PASS** | `wrappers.txt` |
| rule 8, runtime: 49 + 73 + 8 wrappers callable with no argument | **PASS** | `wrappers.txt` |
| Gate 1 / Gate 2 / Gate 4 (10/10 checks, guest = native, `skirt_tube` included) / Gate 4 wrapper smoke (17 calls) / Gate 0F (PASS=33 FAIL=5 INFO=27, the 5 FAIL the no-filesystem negative) | **PASS** | `regress/` |
| `--gate=pen` (the authoring half) | **PASS FIXTURE:infer,rig**: cycles 2, openings 2, patches 2, 4 knots of degree 3, 6 curves; 932 v 1728 f, 2 loops, 1 component | `pen.*` |
| **Gate 8 flat** | **PASS FIXTURE:infer,rig** (every criterion below) | `flat.*` |
| **Gate 8 VR** (OXRSys Runtime 1.2.0, `XR_RUNTIME_JSON` on the process) | **PASS FIXTURE:infer,rig**; equal to flat: 26/26 items, fitted vertices max \|Δ\| = 0; the process segfaults in Godot's OpenXR teardown **after** writing RESULT (rc 139, as before the merge) | `vr.*`, `compare-flat-vr.txt` |
| **MCP drive** (`mcp_run.sh`: `dress_on_stages`, `dress_on_run ["infer,rig"]`, `dress_on_status` polled, `dress_on_result`) | **MCP RESULT: PASS**, pipeline `DONE FIXTURE:infer,rig t=1411.4s`; the result has cycles 2, openings 2, `OK none`, drape finite, body mesh | `mcp/` |
| control `--drop-seam` | **PASS (control drop-seam) FIXTURE:infer,rig**: `FAILED(MESH: ... 0 triangles ...)`, cycles 0, openings 2, patches 0 | `control-drop-seam.*` |
| control `--push-vertex` | **PASS (control push-vertex) FIXTURE:infer,rig**: the same fit, then `FAILED(CHECK: INTERSECTS edge (5357,5936) face (3074,3548,2588))` | `control-push-vertex.*` |
| probe `--no-boundary` (rings as ordinary strokes) | FAIL, as it should: cycles 4, openings 0, patches 4 (two are caps), the mesh 3 loops | `probe-no-boundary.*` |
| pen copy == vendor/xr-grid | **PASS** 32 files | every run |

The flat run's criteria, from `flat.txt`:

| criterion | measured |
|---|---|
| cycles / openings / patches | 2 / 2 / 2 (the front and back panels) |
| curvenet | 6 curves, 4 knots of degrees [3, 3, 3, 3], 6 edges, 4 nodes |
| mesh | 932 v, 1728 f, 2 loops, 1 component, Euler 0 |
| fit | done: 2 phases, 253 Newton, energy 0.011207282055103298, io_attempts 0 |
| intersections | `OK none`; control (the vertex nearest the pelvis moved onto it) `OK INTERSECTS` |
| drape | 100 of 100 rd steps, 932 finite vertices, ymin 5.892 units (0.589 m), 64,586 friction events, 31,494 projections, 455 self pushes |
| screenshot | `flat.png` 1152×648 |

Times and heap per state (host-timed; heap = the stage sandbox's
`get_heap_usage` when the state ends):

| state | flat ms | VR ms | vmcall ms (flat) | heap |
|---|---|---|---|---|
| INFER (FIXTURE) | 100 | 122 | 0 | — |
| RIG (FIXTURE) | 10 | 51 | 0 | — |
| AUTHOR (6 strokes) | 1151 | 1266 | 1098 | curvenet 2.9 MiB |
| MESH | 626 | 663 | 618 | curvenet 3.1 MiB |
| FIT_BEGIN | 2441 | 2646 | 2438 | fit 8.3 MiB |
| FIT_RUN | **1,393,079** | 1,394,224 | 1,393,073 | fit 51.6 MiB |
| FIT_READ | 3 | 16 | 0.1 | fit 51.6 MiB |
| CHECK (+ control) | 964 | 882 | 963 | fit 51.6 MiB |
| DRAPE (setup) | 466 | 449 | 447 | drape 2.7 MiB |
| DRAPE_COLLECT (100 steps) | 2153 (21.5 ms/step) | 4510 (45.1 ms/step) | 1976 | drape 29.5 MiB |
| wall | 1401 s | 1405 s | | |

The main thread kept rendering during the fit: 3,260,154 frames flat
(~2340 frames/s, vsync off) and 92,168 in VR (~66 frames/s, the XR frame
loop).

## The fit budget, measured

The plan extrapolated the Stage 8 garment at "≤ ~1,000 vertices, ~60
Newton (incremental_steps 1, both solves capped at 30) ≈ 4–5 min". Measured
on the 932-vertex authored skirt:

| config | phase 0 (AL) | phase 1 (reduced) | fit | verdict |
|---|---|---|---|---|
| incremental_steps 1, both caps 30 (`flat-cap30.*`) | 983 s, 150 Newton in **5** minimizes, 8.59e11 instructions | 117 s, then **throws** at its 30-iteration limit (`[SparseNewton] Reached iteration limit`) | FAIL at FIT_RUN | the caps cost more than they save |
| incremental_steps 1, the config's caps (AL 50, Newton 5000) (`flat.*`, the default) | 1013 s, 144 Newton in 3 minimizes, 8.19e11 instructions (781k units of 2^20) | 380 s, 109 Newton, 4.07e11 instructions | **1393 s, 253 Newton**, done | PASS |

Why the caps backfire: at 30 the augmented Lagrangian's first minimize stops
short of its constraint, so the AL raises its weight and minimizes again (5
times, 150 Newton, against 3 and 144 at 50); and polysolve's reduced solve
treats its iteration limit as an error unless `allow_out_of_iterations` is
set, which cut-6's config does not. So the loop keeps the config's caps and
takes only `incremental_steps` 1 (2 phases instead of 4). The fit is 23 min,
not 4–5: the ~60 Newton figure assumed each solve converges inside its cap,
and it does not (253 Newton at ~5.5 s each on 932 vertices, against ~11.8 s
each for the 2682-vertex LCL skirt). The largest phase is 781k units, 3.2×
under the fit Sandbox's `execution_timeout` of 2,500,000; the heap peaks at
51.6 MiB of the 2048 MiB Sandbox. Levers not tried here: a coarser mesh
(`--mesh-edge=0.04`), `allow_out_of_iterations` with a cap on the reduced
solve, and a looser AL tolerance.

The capsule body gives the same PASS (`flat-capsules.*`, `vr-capsules.*`,
equal to each other: `compare-flat-vr-capsules.txt`), but the draped skirt
sinks into the thighs (`flat-capsules.png`); the body mesh keeps it outside
(`flat.png`; `probe-drape-mesh.*` and `probe-drape-capsules.*` drape the
saved fit both ways in 1.9 s: ymin 5.892 against 5.698). Observed, not
measured by the gate: a small notch at each end of the front seam in the
screenshots.

## History: the pre-merge integration (before the rebase)

Kept for the record; the two blockers found here (curvenet's caps and
knots, the rd NaN at scale 10) are fixed by cut-4-tube and cut-5-fix. The
`rebased-*` and `today-*` files are the runs before those merges.

A scratch copy of this project with the ELFs built in the cut-4, cut-5 and
cut-6 worktrees at 22:22 (curvenet.elf md5 4a87bda0, drape.elf 7fdde27a,
fit.elf a4a13ca1; cut-4 at be47a124 with its uncommitted work, now
0ee885d9; cut-5 f4b5d5d2; cut-6 3e6de220, now 9f1f6679). Evidence in
`integration/`.

- **cut-4 (curvenet) — BLOCKER for MESH.** `--gate=pen`: the pen reaches
  curvenet, and the counts read **2 cycles, 2 patches, 6 curves** — but
  **the two patches are the waist and hem caps, not the front and back
  panels** (`integration/pen.png`: two blue disks), 5 knots of degrees
  [3, 3, 2, 3, 1], and the mesh is 2 components with 2 loops, so MESH fails
  (`not a skirt tube`). Two causes, both reproduced by
  `tests/probe_skirt_orders.gd` (`integration/orders.txt`):
  1. each ring's two half-rings form a 2-edge cycle as soon as the second
     half lands, and `find_cycles`' face walk reports it; it is patched as a
     disk before the seams exist, and stays;
  2. a stroke whose end lies exactly on an existing knot shared with an
     earlier stroke does not always merge into it (`nodes=5` after the
     sixth stroke, in every order tried: rings first, seams first, panel by
     panel); with seams first the two panels are patched but one seam is
     left open, so the mesh is a disk with one boundary loop (euler 1).
     `merge_eps` 0.05 and 0.1 change which knot splits, not whether one does.
  The gate's MESH check is what caught it: the counts alone pass.
- **cut-6 (fit): works in the loop.** `--force-fixture=curvenet` (the LCL
  skirt, 2682 v, on its own skeleton) with the real fit.elf, **two runs**
  (`integration/fitdrape.txt`, `integration/fitdrape3.*`): FIT_BEGIN OK in
  3.7 s and 4.3 s (target_scale 1.5276, io_attempts 0); FIT_RUN, 4 phases on
  the worker thread, **2446 s and 2714 s**. The second run logs each phase:
  phase 0 AL 1150 s (newton 49), 1 reduced 395 s (44), 2 AL 568 s (49),
  3 reduced **602 s** (66). Both end at 208 Newton iterations and final
  energy 0.0013345137800919919, heap 99 MiB, with the main thread at ~2170
  frames/s meanwhile (5,293,296 and 5,886,809 frames); CHECK `OK none`,
  control `OK INTERSECTS edge (5332,7022) face (2056,1586,2613)` in both. The
  fitted garment is saved as `integration/fitdrape3.fitted.obj` (reuse with
  `--force-fixture=curvenet,fit --fit-from=...`). Phases 0 and 2 (AL) stop at
  their 50-iteration cap (newton 49); that is cut-6's setting, noted here.
- **cut-5 (drape): works at scale 5, NaN at 10 on rd.** On the fitted skirt
  (`integration/drape/`): at scale 10, rd gives non-finite positions on the
  **first** step with or without capsules (friction 0, projections 0), while
  cpu stays finite for 100 steps (1.47 s/step) — an rd-only fault for cut-5.
  Scales 1, 2 and 5 are finite on rd; the similarity-placed skirt is finite
  at 10. With scale 5 and 14 capsules: **100 finite steps, 42 ms/step**,
  119924 friction events, 12797 projections (`drape-fs5c.*`); the skirt
  sags (the sphere demo's kTri 150) and the thighs poke through the capsule
  approximation. The default is now 5.

## What is still waiting

| stage | waiting on | then |
|---|---|---|
| INFER, RIG | Cut 7 (Pixal3D) and Cut 4b (skin-tokens) | the FIXTURE label goes |
| FIT_* | a faster fit (23 min for 932 vertices; levers above) | the loop inside an authoring session |
| VR exit | Godot's OpenXR teardown segfaults after RESULT (rc 139) | a clean exit code |
| a real headset | the manual checklist below | hand-drawn strokes in the loop |

## Manual headset checklist

1. `XR_RUNTIME_JSON` set in the shell that starts Godot (OXRSys, or the
   headset runtime's manifest); nothing registered system-wide.
2. `godot --path project --rendering-driver vulkan --xr-mode on res://xr_main.tscn`;
   the log says `xr_main: XR on (...)`.
3. You stand 1.2 m in front of the FoxGirl body (XROrigin at z = 1.2, facing −z).
4. Call `dress_on_run` with pen `xr` (MCP: `dress_on_run ["infer,rig", "xr"]`);
   the status reads `AUTHOR`.
5. Hold a trigger to draw: a yellow (left) or blue (right) ribbon follows the
   controller; each press is one stroke.
6. Click a thumbstick (or press B, or MCP `dress_on_pen_boundary [true]`);
   the log says `pen: boundary mode on`. Draw the waist ring as two halves
   (front to back on each side) and the hem ring the same: these are the
   openings. Click again (`boundary mode off`) and draw a front and a back
   seam, each ending on the ring points. A ring drawn with the mode off gets
   a cap patch (the `--no-boundary` probe: 4 patches, 3 loops).
7. Press a menu button (or `dress_on_author_done`) to end the authoring.
8. The status moves through MESH, FIT_* (~23 min for a ~900-vertex skirt;
   the headset keeps rendering), CHECK, DRAPE; the garment turns blue
   (mesh), green (fit), magenta (drape).
9. `dress_on_result` has the per-state record. A/B buttons are xr-grid's own
   debug save/load of the stroke mesh (they write `res://test_save.mesh`);
   avoid them.
