# Gate 8: the whole loop on xr_main.tscn -- body -> rig -> pen -> curvenet ->
# mesh -> fit -> intersection check -> drape -- flat or in VR.
#
#   godot --path project --script gate_loop.gd --rendering-driver vulkan --xr-mode off -- \
#       --gate=loop --out=../gates/8-loop/results.txt --wallclock=3600 --allow-fixture=infer,rig
#   XR_RUNTIME_JSON=<runtime json> godot ... --xr-mode on -- --gate=loop --out=...vr.txt
#
# Args (after --):
#   --gate=loop|pen        pen stops after MESH (the authoring half)
#   --out=<file>           results, one line per event, flushed (stdout is
#                          buffered when redirected); <file>.json the summary,
#                          <file>.png the screenshot
#   --wallclock=<s>        quit after this many seconds in any state (default 3600);
#                          a watchdog thread kills the process 60 s later if
#                          the frame loop itself has stalled (XR handover)
#   --allow-fixture=a,b    stages whose fixture may stand in (infer, rig,
#                          curvenet, fit, drape); each is labelled FIXTURE
#   --force-fixture=a,b    run these stages as their fixture even when their ELF
#                          is present (to reach the stages after a failing one)
#   --pen=scripted|xr      scripted replays xr/pen_source_scripted.gd through
#                          the pen bridge; xr waits for SketchTool strokes
#   --drop-seam            control: the back seam is not drawn; must end
#                          FAILED(MESH: ...) with fewer curvenet cycles than
#                          the full skirt's 2 (the seam is what closes them)
#   --push-vertex          control: must end FAILED(CHECK: INTERSECTS ...)
#   --drape-steps=N --mesh-edge=m --drape-backend=cpu|rd|auto --drape-scale=s
#   --drape-body=mesh|capsules|none  the drape's body collider (default mesh)
#   --no-capsules          drape with no body collider (a control for the drape)
#   --no-boundary          the rings are ordinary strokes (not curvenet boundary
#                          strokes): their caps are patched too (a probe, not a
#                          gate control)
#   --fit-incremental-steps=N --fit-max-iterations=N
#                          the fit budget (defaults 1 and -1); -1 keeps
#                          fit_config.json's (2; AL 50 / Newton 5000). A cap
#                          below the config's makes the reduced solve throw
#   --fit-from=<obj>       with fit as a fixture: these vertices are the fit
#                          (every run that fits writes <out>.fitted.obj)
#
# PASS (loop): pen copy == vendor/xr-grid; 2 cycles, 2 openings (the waist and
# hem rings, drawn as boundary strokes) and 2 patches; the mesh
# is one tube (2 boundary loops); fit ran to done; fit_check_intersections
# "OK none" (and its pushed-vertex control INTERSECTS); drape ran N steps
# and every position is finite; the screenshot was written.
#
# Fixtures and the verdict:
#   infer, rig        allowed. The critical-path plan runs the loop on the
#                     FoxGirl body and skeleton until Cut 7 / Cut 4b land; no
#                     Gate 8 criterion measures them (they are inputs), so a
#                     run that meets every criterion is "PASS FIXTURE:infer,rig",
#                     the label saying what stood in.
#   curvenet, fit,    a criterion of the gate is then not measured (cycles and
#   drape             mesh; fit and intersections; drape steps): the run is
#                     "INCOMPLETE", never PASS, whatever else passed.
# A control that passes is "PASS (control <name>)", with the same fixture
# label. Any curvenet, fit or drape fixture makes a control INCOMPLETE too.
extends SceneTree

const SCENE := "res://xr_main.tscn"
# Fixtures the critical-path plan allows under a PASS (labelled); any other
# fixture makes the run INCOMPLETE.
const FIXTURE_OK := ["infer", "rig"]
# xr/pen_source_scripted.gd's expected cycle count for the full skirt.
const FULL_SKIRT_CYCLES := 2
const PEN_SRC := "res://../vendor/xr-grid/addons/procedural_3d_grid"
const PEN_DST := "res://addons/procedural_3d_grid"

var _args := {}
var _out: FileAccess
var _out_path := ""
var _t0 := 0
var _wall_s := 3600.0
var _main: Node = null
var _phase := "boot"
var _frames := 0
var _done_frames := 0
var _rc := 1
var _watch: Thread = null
var _watch_stop := false
var _pen_sync := ""
var _spectator: SubViewport = null

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _arg(k: String, d: String = "") -> String:
	return str(_args.get(k, d))

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv := a.substr(2).split("=", true, 1)
			_args[kv[0]] = kv[1] if kv.size() > 1 else "1"
	_wall_s = float(_arg("wallclock", "3600"))
	_out_path = _arg("out", "../gates/8-loop/results.txt")
	if not _out_path.is_absolute_path():
		_out_path = ProjectSettings.globalize_path("res://").path_join(_out_path).simplify_path()
	DirAccess.make_dir_recursive_absolute(_out_path.get_base_dir())
	_out = FileAccess.open(_out_path, FileAccess.WRITE)
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	var rt := OS.get_environment("XR_RUNTIME_JSON")
	_say("Gate 8 -- the loop. Godot %s, %s, %s" % [Engine.get_version_info().string, OS.get_processor_name(),
			Time.get_datetime_string_from_system()])
	_say("args: %s" % " ".join(OS.get_cmdline_user_args()))
	_say("XR_RUNTIME_JSON: %s" % (rt.get_file() if rt != "" else "(unset)"))
	_watch = Thread.new()
	_watch.start(_watchdog)
	_pen_sync = _check_pen_copy()
	_say("PEN_SYNC: " + _pen_sync)
	var ps: PackedScene = load(SCENE)
	if ps == null:
		_say("FAIL: cannot load %s" % SCENE)
		_finish("FAIL")
		return
	_main = ps.instantiate()
	root.add_child(_main)
	_phase = "wait"

func _watchdog() -> void:
	var limit_ms := int((_wall_s + 60.0) * 1000.0)
	while not _watch_stop:
		OS.delay_msec(200)
		if Time.get_ticks_msec() - _t0 > limit_ms:
			if _out != null:
				_out.store_line("WATCHDOG: the frame loop did not quit within wallclock + 60 s; killing the process")
				_out.store_line("RESULT: FAIL (watchdog)")
				_out.flush()
			OS.kill(OS.get_process_id())
			return

# diff -r of the pen copy against vendor/xr-grid, in GDScript (CITATION.cff
# is the copy's own and is skipped).
func _check_pen_copy() -> String:
	var a := _list(ProjectSettings.globalize_path(PEN_SRC))
	var b := _list(ProjectSettings.globalize_path(PEN_DST))
	b.erase("CITATION.cff")
	if a.is_empty():
		return "FAIL: %s is empty or missing" % PEN_SRC
	var diffs := []
	for rel in a:
		if not b.has(rel):
			diffs.append("only in vendor: " + rel)
		elif FileAccess.get_file_as_bytes(ProjectSettings.globalize_path(PEN_SRC).path_join(rel)) != \
				FileAccess.get_file_as_bytes(ProjectSettings.globalize_path(PEN_DST).path_join(rel)):
			diffs.append("differs: " + rel)
	for rel in b:
		if not a.has(rel):
			diffs.append("only in project: " + rel)
	if diffs.is_empty():
		return "PASS %d files identical to vendor/xr-grid" % a.size()
	return "FAIL " + "; ".join(diffs)

func _list(dir: String, rel: String = "") -> Array:
	var out := []
	var d := DirAccess.open(dir.path_join(rel))
	if d == null:
		return out
	for f in d.get_files():
		out.append(rel.path_join(f) if rel != "" else f)
	for sub in d.get_directories():
		out.append_array(_list(dir, rel.path_join(sub) if rel != "" else sub))
	return out

func _opts() -> Dictionary:
	var o := {
		"allow_fixture": _arg("allow-fixture", ""),
		"force_fixture": _arg("force-fixture", ""),
		"pen": _arg("pen", "scripted"),
		"drop_seam": _args.has("drop-seam"),
		"push_vertex": _args.has("push-vertex"),
	}
	if _args.has("drape-steps"):
		o.drape_steps = int(_arg("drape-steps"))
	if _args.has("mesh-edge"):
		o.mesh_edge = float(_arg("mesh-edge"))
	if _args.has("drape-backend"):
		o.drape_backend = _arg("drape-backend")
	if _args.has("drape-scale"):
		o.drape_scale = float(_arg("drape-scale"))
	if _args.has("no-capsules"):
		o.drape_capsules = false
	if _args.has("drape-body"):
		o.drape_body = _arg("drape-body")
	if _args.has("fit-from"):
		o.fit_from = _arg("fit-from")
	if _args.has("no-boundary"):
		o.no_boundary = true
	if _args.has("fit-incremental-steps"):
		o.fit_incremental_steps = int(_arg("fit-incremental-steps"))
	if _args.has("fit-max-iterations"):
		o.fit_max_iterations = int(_arg("fit-max-iterations"))
	if _arg("gate", "loop") == "pen":
		o.stop_after = "MESH"
	return o

func _process(_dt: float) -> bool:
	_frames += 1
	if _phase == "done":
		return false
	if (Time.get_ticks_msec() - _t0) / 1000.0 > _wall_s:
		_say("TIMEOUT: wall clock %.0f s in %s: %s" % [_wall_s, _phase,
				_main.dress_on_status() if _main != null and _main.get("pipeline") != null else "-"])
		_finish("FAIL (timeout)")
		return false
	match _phase:
		"wait":
			if _frames < 5:
				return false
			if _main.get("pipeline") == null:
				_say("FAIL: Main has no pipeline (a script did not parse; see the log)")
				_finish("FAIL")
				return false
			_main.pipeline.state_changed.connect(_on_state)
			_main.pipeline.progress.connect(func(t: String): _say("      " + t))
			_say("stages: " + _main.dress_on_stages())
			var w = _main.get_node_or_null("World")
			_say("view: %s" % ("XR " + str(w.xr_runtime) if w != null and w.xr_on else "flat"))
			var r: String = _main.dress_on_run_opts(_opts())
			_say("run: " + r)
			if not r.begins_with("STARTED"):
				_finish("FAIL")
				return false
			_phase = "run"
		"run":
			var st: String = _main.pipeline.state
			if st == "DONE" or st == "FAILED":
				_phase = "settle"
				_done_frames = _frames
				_make_spectator()
		"settle":
			# A few frames so the last garment is drawn before the screenshot.
			if _frames - _done_frames >= 5:
				_evaluate()
	return false

# In XR the root viewport renders to the headset and reads back black here, so
# the screenshot comes from a spectator: a SubViewport on the same World3D
# with a camera where FlatCamera is.
func _make_spectator() -> void:
	var w = _main.get_node_or_null("World")
	if w == null or not w.xr_on:
		return
	_spectator = SubViewport.new()
	_spectator.size = Vector2i(1152, 648)
	_spectator.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	var cam := Camera3D.new()
	var flat: Camera3D = w.get_node("FlatCamera")
	cam.fov = flat.fov
	_spectator.add_child(cam)
	root.add_child(_spectator)
	cam.global_transform = flat.global_transform
	cam.current = true

func _on_state(s: String, rec: Dictionary) -> void:
	if s == "DONE" or s == "FAILED":
		_say("=> %s %s" % [s, rec.get("reason", "")])
		return
	_say("STATE %-13s %8d ms  vm %9.1f ms  heap %s  frames %d  %s%s" % [s, rec.ms, rec.vm_ms,
			_mib(rec.heap), rec.frames, "FIXTURE " if rec.fixture and not str(rec.note).begins_with("FIXTURE") else "",
			rec.note])

func _mib(b: int) -> String:
	return "-" if b < 0 else "%.1f MiB" % (b / 1048576.0)

func _evaluate() -> void:
	var p = _main.pipeline
	var s: Dictionary = p.summary()
	var fx: PackedStringArray = p.fixtures
	var png := _out_path.get_basename() + ".png"
	var shot := "not taken"
	var img: Image = (_spectator if _spectator != null else root).get_texture().get_image()
	if img != null and img.save_png(png) == OK:
		shot = "%s (%dx%d)" % [png.get_file(), img.get_width(), img.get_height()]
	if p.data.has("fitted") and p.data.has("garment") and not p.data.get("fit_fixture", false):
		var fo := FileAccess.open(_out_path.get_basename() + ".fitted.obj", FileAccess.WRITE)
		if fo != null:
			var fv: PackedFloat32Array = p.data.fitted
			var ft: PackedInt32Array = p.data.garment.triangles
			fo.store_line("# fit.elf result (body space) of Gate 8 run %s" % _out_path.get_file())
			for i in range(0, fv.size(), 3):
				fo.store_line("v %.9f %.9f %.9f" % [fv[i], fv[i + 1], fv[i + 2]]) # GDScript has no %g
			for i in range(0, ft.size(), 3):
				fo.store_line("f %d %d %d" % [ft[i] + 1, ft[i + 1] + 1, ft[i + 2] + 1])
			fo.close()
	var js := FileAccess.open(_out_path.get_basename() + ".json", FileAccess.WRITE)
	if js != null:
		js.store_string(JSON.stringify(s, "  "))
		js.close()
	_say("FIXTURE: %s" % (",".join(fx) if not fx.is_empty() else "none"))
	var st := {"ok": true} # a lambda captures locals by value; a Dictionary by reference
	var unmeasured := []
	var lines := []
	var check := func(name: String, pass_: bool, detail: String) -> void:
		lines.append("%s %s: %s" % ["PASS" if pass_ else "FAIL", name, detail])
		if not pass_:
			st.ok = false
	check.call("pen_sync", _pen_sync.begins_with("PASS"), _pen_sync)
	var ctrl := ""
	if _args.has("drop-seam"):
		ctrl = "drop-seam"
		# The specific failure, not any FAILED(MESH): without the back seam the
		# graph closes fewer cycles than the full skirt's (PenSource expects
		# 2), and so the mesh is not a tube.
		var cc: Dictionary = s.get("counts", {})
		if fx.has("curvenet"):
			unmeasured.append("drop-seam cycles (curvenet FIXTURE)")
		else:
			var cyc = cc.get("cycles")
			check.call("control drop-seam", p.state == "FAILED" and p.reason.begins_with("MESH:") and cyc != null
					and int(cyc) >= 0 and int(cyc) < FULL_SKIRT_CYCLES,
					"%s(%s), cycles=%s patches=%s; want FAILED(MESH: ...) with 0 <= cycles < %d" % [p.state, p.reason,
					str(cyc), str(cc.get("patches")), FULL_SKIRT_CYCLES])
	elif _args.has("push-vertex"):
		ctrl = "push-vertex"
		check.call("control push-vertex", p.state == "FAILED" and p.reason.begins_with("CHECK: INTERSECTS"),
				"%s(%s), want FAILED(CHECK: INTERSECTS ...)" % [p.state, p.reason])
	else:
		check.call("pipeline", p.state == "DONE", "%s %s" % [p.state, p.reason])
		var c: Dictionary = s.get("counts", {})
		if fx.has("curvenet"):
			unmeasured.append("cycles/patches (curvenet FIXTURE)")
		elif not c.is_empty():
			check.call("cycles", c.get("cycles") == 2, "cycles=%s (want 2)" % str(c.get("cycles")))
			check.call("openings", c.get("openings") == 2, "openings=%s (want 2: the waist and the hem)" % str(c.get("openings")))
			check.call("patches", c.get("patches") == 2, "patches=%s (want 2: the front and back panels)" % str(c.get("patches")))
			var degs: Array = c.get("knot_degrees", [])
			lines.append("INFO curvenet: curves=%s knots=%s degrees=%s edges=%s (4 knots of degree 3 have 6 edges)" % [
					str(c.get("curves")), str(c.get("knots")), str(degs), str(c.get("edges"))])
		if s.has("mesh") and not fx.has("curvenet"):
			var m: Dictionary = s.mesh
			check.call("mesh", m.get("loops") == 2 and m.get("components") == 1,
					"%s v %s f, %s loops, %s components" % [m.get("vertices"), m.get("triangles"), m.get("loops"), m.get("components")])
		if _arg("gate", "loop") == "loop":
			if fx.has("fit"):
				unmeasured.append("fit done + intersections (fit FIXTURE)")
			elif p.state == "DONE":
				check.call("intersections", str(s.get("check", "")).begins_with("OK none"), str(s.get("check", "")))
				check.call("intersections control", str(s.get("check_control", "")).find("INTERSECTS") >= 0,
						"one vertex pushed inside: %s" % str(s.get("check_control", "")))
			if fx.has("drape"):
				unmeasured.append("drape steps (drape FIXTURE)")
			elif p.state == "DONE":
				var d = s.get("drape", {})
				check.call("drape", typeof(d) == TYPE_DICTIONARY and d.get("finite", false),
						"%s steps queued, %s" % [str(p.opts.drape_steps), str(d)])
		check.call("screenshot", shot != "not taken", shot)
	# A fixture outside infer/rig always leaves a criterion unmeasured, even
	# when the run stopped before reaching it.
	for f in fx:
		if not (f in FIXTURE_OK) and not unmeasured.any(func(u): return str(u).contains(f + " FIXTURE")):
			unmeasured.append("%s (%s FIXTURE)" % [f, f])
	for l in lines:
		_say(l)
	if not unmeasured.is_empty():
		_say("NOT MEASURED: " + "; ".join(unmeasured))
	var ok: bool = st.ok
	var verdict := "PASS" if ok else "FAIL"
	if ok and ctrl != "":
		verdict = "PASS (control %s)" % ctrl
	if ok and not unmeasured.is_empty():
		verdict = "INCOMPLETE"
	if not fx.is_empty() and verdict.begins_with("PASS"):
		verdict += " FIXTURE:" + ",".join(fx)
	_rc = 0 if verdict.begins_with("PASS") else 1
	_finish(verdict)

func _finish(verdict: String) -> void:
	_say("wall %.1f s, frames %d" % [(Time.get_ticks_msec() - _t0) / 1000.0, _frames])
	_say("RESULT: " + verdict)
	_phase = "done"
	_watch_stop = true
	if _watch != null:
		_watch.wait_to_finish()
		_watch = null
	var busy := []
	var ok_main: bool = _main != null and _main.get("pipeline") != null
	if ok_main:
		for stg in [_main.curvenet, _main.fit, _main.drape]:
			if stg != null and stg.busy():
				busy.append(stg.busy_text())
	if ok_main and busy.is_empty() and _main.drape.sandbox != null and _main.drape.drape_api_missing() == "":
		_say("drape rd_close: " + str(_main.drape.call_now("rd_close")))
	if _out != null:
		_out.close()
		_out = null
	if not busy.is_empty():
		# A worker vmcall cannot be interrupted, and quit() would wait for it
		# (the stage joins its Thread on exit). The wall clock is the rule.
		print("exiting with a vmcall in flight: ", busy)
		OS.kill(OS.get_process_id())
		return
	quit(_rc)
