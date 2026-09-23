# Gate 8 — join the loop

**Result: NOT YET RUN END TO END. The harness, the composition, the
fixtures and the controls are in and tested; on this branch the loop stops
at AUTHOR with `FAILED(AUTHOR: curvenet missing ...)`, as it must, because
curvenet.elf (cut-4), fit.elf (cut-6) and the drape API (cut-5) are not
merged yet. A pre-merge run against those branches' built ELFs got past
AUTHOR only with the curvenet fixture: then fit.elf fitted the LCL skirt to
FoxGirl in 42 min (208 Newton iterations, reproducible to the last digit),
the intersection check said `OK none` (its control `INTERSECTS`), and the
drape ran 100 finite steps on rd. The blocker is in cut-4 (below).**

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
| Fit | `stages/fit_stage.gd` | fit.elf | 2048 MiB, refs 4096, timeout 2^30, allocations_max 4e6 (cut-6's) |
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
  up to 64 events and one stroke end per frame; then `curvenet_build`.
- **MESH**: `mesh_build(0.03, 1e-5)` on curvenet's worker thread; the mesh must
  be one tube: 1 component, 2 boundary loops.
- **FIT_BEGIN / FIT_RUN / FIT_READ**: body, skeletons (the authored garment's
  source skeleton is the body's own), garment, `fixtures/foxgirl/fit_config.json`
  (cut-6's foxgirl oracle setup without paths); `fit_begin` and then one
  `fit_step` per job on the worker thread, `fit_status` read between jobs,
  until it says `done`; then `fit_result_vertices`.
- **CHECK**: `fit_check_intersections` must answer `OK none`. Its control
  runs every time beside it: the garment vertex nearest the pelvis moved onto
  the pelvis joint must answer `INTERSECTS`.
- **DRAPE / DRAPE_COLLECT**: `drape_scene_mesh` with the fitted garment, pinned
  on its waist loop (the boundary loop with the highest mean y). Cut-5's drape
  has primitives only (sphere, plane, capsule), **so the body is approximated
  by 14 capsules along the skeleton's bones** (radius = the median distance of
  the body vertices nearest each bone, less the capsule's own contact offset).
  DiffCloth's capsule has a fixed 0.1-unit contact offset, so the drape runs
  in a frame scaled by 5 (1 m = 5 units, gravity −49; the offset is 2 cm)
  and the result is scaled back. At 10 the rd backend gives NaN on the first
  step for the fitted skirt (below). 100 steps are queued; `drape_tick` runs once per frame; the
  positions must all be finite.

Every state records its host-timed duration, its stage's vmcall time and
the stage sandbox's heap (`get_heap_usage`) when it ends (`STATE` lines).

**Missing stages.** A stage whose ELF is missing or too old FAILs the run as
`FAILED(<STATE>: <stage> missing (<reason>))` unless `--allow-fixture` names
it; then its fixture stands in and the run is labelled FIXTURE for it:
curvenet → the LCL skirt with its own skeleton and no-fit list; fit → the
similarity map from the garment skeleton to the body skeleton (cloth-fit's
normalisation; identity for an authored garment), CHECK not run; drape → not
run. `--force-fixture` does the same for a stage that is present (to reach
what follows a failing stage). A criterion whose stage was a fixture is not
measured, and such a run is `RESULT: INCOMPLETE`, never PASS.

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
The graph should have 4 knots of degree 3, 6 edges, 2 cycles (front and back
panel), 2 patches. The Cut 8 task text says "8 curves": 4 knots of degree 3
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
(cycles, patches, mesh vertices / triangles / loops), same CHECK answer, drape
finite (compare `results.json` with `results-vr.json`). `--quit-after` never
fires under `--xr-mode on`; the gate quits on its own wall clock, and a
watchdog thread kills the process 60 s after it if the frame loop has stalled.

MCP (Gate 0E's shape; plays `xr_main.tscn` flat on port 8795, drives
`dress_on_stages`, `dress_on_run`, polls `dress_on_status`, reads
`dress_on_result`; responses in `mcp/`):

```
gates/8-loop/mcp_run.sh                       # ALLOW=infer,rig by default
```

No-ELF tests: `tests/test_loop_units.gd` (skeleton15, the scripted pen,
mesh_topo; 32 checks), `tests/probe_main_wrappers.gd` (every old and new
wrapper on `/root/Main`, a sample called), `tests/parse_check.gd`.
`run_regress.sh` re-runs Gates 1, 2 and 0F on this tree.

## What runs today (this branch, 279b31b + Cut 8)

| run | result | file |
|---|---|---|
| units (skeleton15 identity + 3 permuted rigs + a permuted named 20-joint rig + 2 refusals; pen deterministic, 4 knots of degree 3, clearance; tube loops) | **PASS** 32/32 | `units.txt` |
| every old wrapper (49) and new (51) on `/root/Main`, 16 called | **PASS** | `wrappers.txt` |
| Gate 1 `gate_rd_compute.gd` on this tree | **PASS** | `regress/` |
| Gate 2 `gate_avbd.gd` on this tree | **PASS** | `regress/gate_avbd.results.txt` |
| Gate 0F `gate_runtime.gd` on this tree | **PASS=31 FAIL=5 INFO=27** (as committed; the 5 FAIL are the no-filesystem negative) | `regress/gate_runtime.results.txt` |
| Gate 8 flat, `--allow-fixture=infer,rig` | **FAILED(AUTHOR: curvenet missing (curvenet.elf not found))**, as designed | `today-infer-rig.*` |
| Gate 8 flat, every stage a fixture | **INCOMPLETE** (DONE; cycles, fit, intersections, drape not measured) | `today-all-fixtures.*` |
| Gate 8 VR on OXRSys (`XR_RUNTIME_JSON` per process), `--allow-fixture=infer,rig` | the same STATE sequence and FAILED(AUTHOR) as flat; OpenXR reports `OXRSys Runtime 1.2.0`; the screenshot comes from a spectator SubViewport (the root viewport reads back black in XR). **The process segfaults on exit** (rc 139) in Godot's OpenXR teardown, after OXRSys has destroyed its instance and after RESULT is written | `today-vr-infer-rig.*` |
| MCP drive | **PASS** (the chain; pipeline FAILED(AUTHOR: curvenet missing)) | `mcp/` |
| pen copy == vendor/xr-grid | **PASS** 32 files | every `today-*.txt` |

## Pre-merge integration (not on this branch)

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
  skirt, 2682 v, on its own skeleton) with the real fit.elf
  (`integration/fitdrape3.*`): FIT_BEGIN OK in 3.7–5.0 s (target_scale
  1.5276, io_attempts 0); FIT_RUN 4 phases on the worker thread, **2446–2518 s**
  (phase 0 AL 1082–1156 s, newton 49; 1 reduced 338–395 s; 2 AL 499–568 s;
  3 reduced 526 s), 208 Newton iterations, final energy
  0.0013345137800919919 in all three runs, heap 99 MiB, the main thread at
  ~2100 frames/s meanwhile; CHECK `OK none`, control `OK INTERSECTS edge
  (5332,7022) face (2056,1586,2613)`. The fitted garment is saved as
  `integration/fitdrape3.fitted.obj` (reuse with `--force-fixture=curvenet,fit
  --fit-from=...`). Phase 0 AL stops at its 50-iteration cap (newton 49)
  both times; that is cut-6's setting, noted here.
- **cut-5 (drape): works at scale 5, NaN at 10 on rd.** On the fitted skirt
  (`integration/drape/`): at scale 10, rd gives non-finite positions on the
  **first** step with or without capsules (friction 0, projections 0), while
  cpu stays finite for 100 steps (1.47 s/step) — an rd-only fault for cut-5.
  Scales 1, 2 and 5 are finite on rd; the similarity-placed skirt is finite
  at 10. With scale 5 and 14 capsules: **100 finite steps, 42 ms/step**,
  119924 friction events, 12797 projections (`drape-fs5c.*`); the skirt
  sags (the sphere demo's kTri 150) and the thighs poke through the capsule
  approximation. The default is now 5.

## What is waiting on which branch

| stage | waiting on | then |
|---|---|---|
| AUTHOR, MESH | **cut-4** merge, and the two findings above fixed (no caps from ring cycles; stroke ends merge into shared knots) | 2 panels → one tube, 2 loops |
| FIT_*, CHECK | **cut-6** merge (fit.elf is built by build.sh, not committed); a ~500-vertex authored skirt should fit well inside the 42 min the 2682-vertex LCL skirt takes | fit done, `OK none`, control INTERSECTS (all seen pre-merge) |
| DRAPE | **cut-5** merge; the rd NaN at scale 10 is theirs to look at; a mesh collider would replace the capsules | 100 finite steps (seen pre-merge at scale 5) |
| INFER, RIG | Cut 7 (Pixal3D) and Cut 4b (skin-tokens) | the FIXTURE label goes |

Merging: cut-4, cut-5 and cut-6 each add their wrappers to the old
`main.gd`; take this branch's `main.gd` and move any new wrapper into its
stage file (their names are already delegated here). `util/mesh_wire.gd`
(cut-4) and `util/obj_io.gd` (cut-6, uncommitted there) are copied
verbatim.

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
6. Draw the waist ring as two halves (front to back on each side), the hem
   ring the same, then a front and a back seam, each ending on the ring
   points.
7. Press a menu button (or `dress_on_author_done`) to end the authoring.
8. The status moves through MESH, FIT_* (minutes; the main thread keeps
   rendering), CHECK, DRAPE; the garment turns blue (mesh), green (fit),
   magenta (drape).
9. `dress_on_result` has the per-state record. A/B buttons are xr-grid's own
   debug save/load of the stroke mesh (they write `res://test_save.mesh`);
   avoid them.
