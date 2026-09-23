# Gate 8 — join the loop

**Result: FAIL at MESH on main's curvenet.elf (not yet end to end).** This
branch is now Cut 8 on main at 8d59e6a (Stages 1, 2, 4, 5, 6 and Gate 0F
merged). With `--allow-fixture=infer,rig` the loop gets through INFER, RIG
and AUTHOR (the scripted pen in curvenet.elf: 2 cycles, 2 patches) and
**fails at MESH**: the two patches are the waist and hem caps, not the front
and back panels, so the mesh is 2 components (`rebased-infer-rig.*`). That is
the cut-4 blocker found before the merge (below), unchanged in main's
curvenet. Past it, the rest of the loop works on main's ELFs: fit.elf fitted
the LCL skirt to FoxGirl in the loop in two runs, 2446 s and 2714 s (208
Newton iterations and the same final energy to the last digit; CHECK `OK
none`, control `INTERSECTS`), and main's drape.elf drapes that fit 100 finite
steps on rd at 38.8 ms/step (`rebased-drape-from-fit.*`, INCOMPLETE because
curvenet and fit are fixtures there).

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
- **Controls.** `--drop-seam` leaves the back seam out (5 strokes). It must
  end `FAILED(MESH: ...)` **and** the curvenet must report fewer cycles than
  the full skirt's 2 (`counts.cycles`, measured, 0 <= cycles < 2): the seam is
  what closes the front and back panels, so that is the failure the control
  is for; a MESH failure for any other reason (the ring caps below, say) does
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
wrapper on `/root/Main`, and a source audit that every `ADD_API_FUNCTION` /
`add_sandbox_api_function` in `guest/**/main.cpp` has one, through
`tests/wrapper_audit.gd`), `tests/parse_check.gd`. `run_regress.sh` re-runs
Gates 1, 2, 4 (and Gate 4's wrapper smoke) and 0F on this tree.

## What runs today (this branch: main 8d59e6a + Cut 8)

| run | result | file |
|---|---|---|
| parse check: 29 scripts and scenes (the stages, main's gate scripts, the tests) | **PASS** | `parse_check.txt` |
| units (skeleton15 identity + 3 permuted rigs + a permuted named 20-joint rig + 2 refusals; pen deterministic, 4 knots of degree 3, clearance; tube loops) | **PASS** 32/32 | `units.txt` |
| rule 8, source audit: 113 guest entry points (dress_on 8, probes 36, curvenet 23, drape 29, fit 17) each reached from a `/root/Main` wrapper with every argument defaulted (117 wrappers); controls: `p_memalign`'s delegate deleted → exactly `dress_on:p_memalign` missing; `rd_calls`' defaults stripped → exactly its 2 parameters | **PASS** | `wrappers.txt` |
| rule 8, runtime: 49 old (279b31b) + 73 main (8d59e6a's, plus `rd_bench_quiet`, `rd_calls`, `rd_set_probe`, `drape_tick`, `drape_job_tick`, which had none) + 8 Cut 8 wrappers callable with no argument; Gate 6's 4 properties forwarded; 23 called | **PASS** | `wrappers.txt` |
| Gate 1 `gate_rd_compute.gd` | **PASS** | `regress/gate_rd_compute.log` |
| Gate 2 `gate_avbd.gd` | **PASS** | `regress/gate_avbd.results.txt` |
| Gate 4 `gate_curvenet.gd` (check 7 now audits the delegates into `stages/curvenet_stage.gd`) and `probe_curvenet_wrappers.gd` (17 calls) | **PASS** | `regress/gate_curvenet.results.txt`, `regress/probe_curvenet_wrappers.log` |
| Gate 0F `gate_runtime.gd` | **PASS=33 FAIL=5 INFO=27** (main's 31 + probe 17's two memalign lines; the 5 FAIL are the no-filesystem negative) | `regress/gate_runtime.results.txt` |
| Gate 8 flat, `--allow-fixture=infer,rig` | **FAIL**: `FAILED(MESH: not a skirt tube: 843 triangles, 2 components, 2 boundary loops)`; cycles 2, patches 2 (the caps), knots 5 of degrees [3, 3, 2, 3, 1]; AUTHOR 1.27 s, MESH 0.11 s | `rebased-infer-rig.*`, `rebased-pen.*` (`--gate=pen`, the same) |
| Gate 8 control `--drop-seam` | **FAIL**, correctly: it ends FAILED(MESH) but with **cycles 2** (the caps again), not fewer than the full skirt's; the old check (any FAILED(MESH)) would have passed it | `rebased-control-drop-seam.*` |
| Gate 8 `--force-fixture=curvenet,fit --fit-from=integration/fitdrape3.fitted.obj` | **INCOMPLETE** (DONE; drape 100 finite steps on rd, 38.8 ms/step, 119,924 friction events, 12,797 projections, as before the merge; cycles, fit and intersections not measured) | `rebased-drape-from-fit.*` |
| pen copy == vendor/xr-grid | **PASS** 32 files | every `rebased-*.txt` |

Not re-run on the rebased tree: the full fit in the loop (~45 min; the runs
below are on cut-6's fit.elf, whose sources main merged), the VR run and the
MCP drive. Before the rebase (279b31b + Cut 8, no curvenet/fit/drape API on
the branch) they gave: VR on OXRSys the same STATE sequence as flat
(`FAILED(AUTHOR: curvenet missing)`), OpenXR `OXRSys Runtime 1.2.0`, and a
segfault on exit (rc 139) in Godot's OpenXR teardown after RESULT is written
(`today-vr-infer-rig.*`); the MCP chain PASS (`mcp/`); every stage a fixture
INCOMPLETE (`today-all-fixtures.*`).

## Pre-merge integration (before the rebase)

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
| AUTHOR, MESH | **curvenet (Cut 4 follow-up)**: the two findings above (no caps from ring cycles; stroke ends merge into shared knots) | 2 panels → one tube, 2 loops; the drop-seam control then has its fewer cycles |
| FIT_*, CHECK | nothing on main (fit.elf is built by build.sh, not committed); a ~500-vertex authored skirt should fit well inside the 41–45 min of the 2682-vertex LCL skirt | fit done, `OK none`, control INTERSECTS (seen in the loop) |
| DRAPE | the rd NaN at scale 10 (Cut 5's to look at); a mesh collider would replace the capsules | 100 finite steps (seen on main's drape.elf at scale 5) |
| INFER, RIG | Cut 7 (Pixal3D) and Cut 4b (skin-tokens) | the FIXTURE label goes |

The merge: main's `main.gd` (cut-4, cut-5 and cut-6 wrappers on the old
single-file root) gave way to this branch's thin root, and every wrapper
there moved into its stage file with its body unchanged, main.gd keeping a
same-named delegate with the same defaults (curvenet's pipeline calls became
`pen_begin_at` / `pen_point_at` so the scripted-pen wrappers keep their MCP
shape; the fit stage took over `fit_configure(_with)`, `foxgirl_arrays`, the
config overrides and the probes; the drape stage `drape_optimize`,
`drape_job_data`, `lbfgsb_load_oracle` and the rest). `util/mesh_wire.gd` and
`util/obj_io.gd` were byte-identical on both sides.

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
