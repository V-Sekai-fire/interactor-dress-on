# Gate 6d: the avbd fit ladder. The authored skirt (the loop to MESH, as
# gate_loop.gd --gate=pen) fitted by drape.elf's fit phase at each rung of
# LADDER, on xr_main.tscn; every rung is timed, saved as an OBJ (body space,
# the PolyFEM fit's vertex order), screenshotted (the fit phase only: gravity
# off, before the loop's drape; gate_loop.gd --fit-mode=avbd screenshots the
# chosen rung after the drape), and judged by fit.elf's
# fit_check_intersections against the real body (with the pushed-vertex
# control beside it). Then gates/6d-fit-avbd/ladder_eval.py measures every
# rung against the PolyFEM fit of the same mesh (gates/8-loop/
# flat-psd.fitted.obj: per-vertex distance mean / p95 / max) and against the
# body (the fit gap mean / p95), and the table goes into ladder.txt.
#
#   godot --path project --script gate_fit_avbd.gd --rendering-driver vulkan --xr-mode off -- \
#       --out=../gates/6d-fit-avbd/ladder.txt [--wallclock=900] [--only=name,name]
#
# The rungs are this file's; the loop has no knob for them (the user's rule:
# a ladder and one pick, not a slider). The pick is drape_stage.gd's
# FIT_AVBD, and this gate FAILs if that constant is not the CHOSEN rung's
# numbers, so the constant and the ladder cannot drift apart.
#
# PASS: every rung ran to DONE with finite positions; the chosen rung's
# surface is within PASS_SURF_MEAN_MM / PASS_SURF_P95_MM of the PolyFEM fit's
# (each vertex of either mesh to the other's nearest triangle: the shape),
# its check is "OK none" and its control INTERSECTS; the two controls stay
# far from the PolyFEM fit: no pull (the tube) and no similarity rest update
# (the authored length held), each at least 1.5x the chosen rung's mean
# per-vertex distance. The per-vertex distance (same 932-vertex order) is
# reported with its own bar, PASS_MEAN_MM / PASS_P95_MM, as INFO: it also
# counts where a vertex sits on the surface (the front and back sectors slid
# 40-70 mm in x against PolyFEM's while the surfaces are 7 mm apart).
extends SceneTree

const SCENE := "res://xr_main.tscn"
const MeshTopo := preload("res://util/mesh_topo.gd")
const REF := "res://../gates/8-loop/flat-psd.fitted.obj"
const AVATAR := "res://fixtures/foxgirl/avatar.obj"
const EVAL := "res://../gates/6d-fit-avbd/ladder_eval.py"
# One fit voxel in metres: fit_config.json's voxel_size 0.01 solve units over
# the fit's target_scale 1.5276 for this body (the FIT_BEGIN line).
const VOXEL_M := 0.0065462
const CHOSEN := "f60-i32-s300"
const PASS_MEAN_MM := 30.0 # per vertex (INFO): the PolyFEM skirt is 304 mm long
const PASS_P95_MM := 60.0
const PASS_SURF_MEAN_MM := 15.0 # surface to surface: 2.3 fit voxels, PolyFEM's own gap to the body is 12 mm
const PASS_SURF_P95_MM := 30.0
# Drape units (1 m = 10 units). Base: the loop's collider (skin 0.1 = 1 cm,
# band 0.1, depth 1.0), the target on the body surface itself (gap 0, as
# cloth-fit's fit term pulls to sdf = 0 and its contact keeps the clearance),
# tol 5e-4 units (0.05 mm per step), the targets refreshed every step (a
# stale target is a spring to a fixed point and holds a vertex where it
# stands; cloth-fit's SDF potential lets it slide along the surface) and the
# rest shape (similarity 1) every 4, the waist loop's centre held at its
# source position by a second attachment per waist vertex at kAnchor (the
# anchor: cloth-fit's curve_center_target), the drape's kBend 1e-5, the fit
# phase's time step h 1/60 (the quasi-static drift per step is F h^2 / m:
# nine times the drape's 1/180, whose s stalled at 0.81 in 300 steps, pass 5).
# kFit multiplies the vertex's lumped area (drape_scene.cpp applyMaterial):
# 600 makes a face's three corner springs equal 2 x the membrane's kTri x
# area (fit_weight 2 over similarity 1 with kTri standing for the similarity
# weight); passes 5-6 showed the strong end pulls the hem's back into the gap
# between the legs, where PolyFEM keeps it hanging, so the ladder is x0.03,
# x0.1, x0.3 and x1 of 600.
const BASE := {"tol": 0.0005, "refresh": 1, "gap": 0.0, "similarity": 1.0, "h": 1.0 / 60.0, "restEvery": 4, "settle": 0,
		"kAnchor": 100.0, "kBend": 0.00001, "skin": 0.1, "band": 0.1, "depth": 1.0}
const LADDER := [
	{"name": "f0-i16-s100", "k": 0.0, "iters": 16, "steps": 100, "similarity": 0.0, "h": 1.0 / 180.0, "kAnchor": 0.0}, # control: no pull, the tube
	{"name": "f60-i32-s300-nosim", "k": 60.0, "iters": 32, "steps": 300, "similarity": 0.0}, # control: the authored length held
	{"name": "f20-i32-s300", "k": 20.0, "iters": 32, "steps": 300},            # x0.03
	{"name": "f60-i32-s300", "k": 60.0, "iters": 32, "steps": 300},            # x0.1
	{"name": "f180-i32-s300", "k": 180.0, "iters": 32, "steps": 300},          # x0.3
	{"name": "f600-i32-s300", "k": 600.0, "iters": 32, "steps": 300},          # x1
	{"name": "f60-i16-s300", "k": 60.0, "iters": 16, "steps": 300},            # the iteration control
	{"name": "f60-i32-s300-a30", "k": 60.0, "iters": 32, "steps": 300, "kAnchor": 30.0},
	{"name": "f60-i32-s300-a300", "k": 60.0, "iters": 32, "steps": 300, "kAnchor": 300.0},
	{"name": "f180-i32-s300-settle10", "k": 180.0, "iters": 32, "steps": 300, "settle": 10},
	{"name": "f60-i32-s600", "k": 60.0, "iters": 32, "steps": 600},            # the step budget
]

var _args := {}
var _out: FileAccess
var _out_path := ""
var _t0 := 0
var _wall_s := 900.0
var _main: Node = null
var _phase := "boot"
var _frames := 0
var _mark := 0
var _rung := -1
var _rung_t0 := 0
var _rows := []
var _only := {}
var _fit_begin := ""
var _fails := []
var _fitted := PackedFloat32Array()

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
	_wall_s = float(_arg("wallclock", "900"))
	for n in _arg("only", "").split(",", false):
		_only[n] = true
	_out_path = _arg("out", "../gates/6d-fit-avbd/ladder.txt")
	if not _out_path.is_absolute_path():
		_out_path = ProjectSettings.globalize_path("res://").path_join(_out_path).simplify_path()
	DirAccess.make_dir_recursive_absolute(_out_path.get_base_dir())
	_out = FileAccess.open(_out_path, FileAccess.WRITE)
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_say("Gate 6d -- the avbd fit ladder. Godot %s, %s, %s, %s" % [Engine.get_version_info().string,
			OS.get_processor_name(), RenderingServer.get_video_adapter_name(), Time.get_datetime_string_from_system()])
	_say("args: %s" % " ".join(OS.get_cmdline_user_args()))
	var ps: PackedScene = load(SCENE)
	if ps == null:
		_say("FAIL: cannot load %s" % SCENE)
		_finish("FAIL")
		return
	_main = ps.instantiate()
	root.add_child(_main)
	_phase = "wait"

func _rung_setting(r: Dictionary) -> Dictionary:
	var p := BASE.duplicate()
	for k in r:
		if k != "name":
			p[k] = r[k]
	return p

func _process(_dt: float) -> bool:
	_frames += 1
	if _phase == "done":
		return false
	if (Time.get_ticks_msec() - _t0) / 1000.0 > _wall_s:
		_say("TIMEOUT: wall clock %.0f s in %s" % [_wall_s, _phase])
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
			_say("stages: " + _main.dress_on_stages())
			var miss: String = _main.drape.fit_api_missing() if _main.drape.available() else _main.drape.reason
			if miss != "":
				_say("FAIL: drape %s" % miss)
				_finish("FAIL")
				return false
			_main.pipeline.state_changed.connect(_on_state)
			var r: String = _main.dress_on_run_opts({"allow_fixture": "infer,rig", "stop_after": "MESH"})
			_say("run to MESH: " + r)
			if not r.begins_with("STARTED"):
				_finish("FAIL")
				return false
			_phase = "mesh"
		"mesh":
			var st: String = _main.pipeline.state
			if st == "FAILED":
				_say("FAIL: the loop did not reach MESH: %s" % _main.pipeline.reason)
				_finish("FAIL")
				return false
			if st != "DONE":
				return false
			var p = _main.pipeline
			var g: Dictionary = p.data.garment
			_say("garment: %d v %d f, %d nofit; body %d v %d f" % [g.vertices.size() / 3, g.triangles.size() / 3,
					g.nofit.size(), p.data.body_v.size() / 3, p.data.body_f.size() / 3])
			# fit.elf's begin on its worker, for the checks (the CHECK state's
			# setup: the real body as the phase-0 avatar, pipeline _fit_elf_begin).
			if _main.fit.available():
				var e: String = p._fit_elf_begin(true)
				_say("fit.elf begin: %s" % ("STARTED on the worker" if e == "" else "FAIL " + e))
				if e != "":
					_fit_begin = "FAIL " + e
			else:
				_fit_begin = "not available (%s)" % _main.fit.reason
				_say("fit.elf: %s; the checks are not measured" % _fit_begin)
			_phase = "rung_start"
		"rung_start":
			_rung += 1
			while _rung < LADDER.size() and not _only.is_empty() and not _only.has(LADDER[_rung].name):
				_rung += 1
			if _rung >= LADDER.size():
				_evaluate()
				return false
			var r: Dictionary = LADDER[_rung]
			var s := _rung_setting(r)
			var p = _main.pipeline
			var g: Dictionary = p.data.garment
			var start: PackedFloat32Array = p._similarity_retarget(g.vertices, g.source_joints, p.data.joints)
			var fit_verts: PackedInt32Array = p._fit_vertices(g.vertices.size() / 3, g.nofit)
			var anchor: PackedInt32Array = p._waist_loop(start, g.triangles)
			_rung_t0 = Time.get_ticks_msec()
			_main.drape.take_vm_us()
			var a: Dictionary = _main.drape.fit_avbd_start(start, g.triangles, fit_verts, anchor, p.data.body_v, p.data.body_f,
					s, p.opts.drape_scale, p.opts.drape_mu, p.opts.drape_backend)
			var bad := ""
			for x in a.steps:
				if str(x).begins_with("FAIL") or str(x).begins_with("BUSY"):
					bad = str(x)
			if bad == "" and not str(a.queued).begins_with("QUEUED"):
				bad = str(a.queued)
			_say("RUNG %s: kFit %s gap %s similarity %s iters %d steps<=%d tol %s refresh %d restEvery %d settle %d kAnchor %s h %s kBend %s | %s | %s" % [
					r.name, str(s.k), str(s.gap), str(s.similarity), s.iters, s.steps, str(s.tol), s.refresh, s.restEvery,
					s.settle, str(s.kAnchor), str(s.h), str(s.kBend), str(a.steps[-1]), a.queued])
			if bad != "":
				_say("  FAIL setup: " + bad)
				_fails.append("%s setup: %s" % [r.name, bad])
				_rows.append({"name": r.name, "setting": s, "result": "FAIL " + bad})
				return false
			_phase = "rung_run"
		"rung_run":
			var st: String = _main.drape.drape_status()
			if st.begins_with("RUNNING"):
				return false
			var ms := Time.get_ticks_msec() - _rung_t0
			var vm_ms: float = _main.drape.take_vm_us() / 1000.0
			var r: Dictionary = LADDER[_rung]
			var s := _rung_setting(r)
			var pos: PackedFloat32Array = _main.drape.drape_positions()
			var p = _main.pipeline
			var scale: float = p.opts.drape_scale
			_fitted = PackedFloat32Array()
			_fitted.resize(pos.size())
			for i in pos.size():
				_fitted[i] = pos[i] / scale
			var finite: bool = MeshTopo.all_finite(_fitted) and pos.size() == p.data.garment.vertices.size()
			var ok: bool = st.find("DONE fit") >= 0 and finite
			_say("  %s | host_ms %d vm_ms %.0f ticks %d finite %s heap %.1f MiB" % [st, ms, vm_ms, _main.drape.ticks,
					str(finite), _main.drape.heap() / 1048576.0])
			var row := {"name": r.name, "setting": s, "result": st, "host_ms": ms, "vm_ms": vm_ms, "ticks": _main.drape.ticks,
					"finite": finite, "ok": ok, "steps": _kv(st, "steps="), "ms_per_step": _kv(st, "ms/step="),
					"self_pushes": _kv(st, "self_pushes="), "sim_s": _kv(st, "sim_s="), "sim_rot_deg": _kv(st, "sim_rot_deg="),
					"anchor_shift": _kv(st, "anchor_shift="), "converged": st.find("converged=yes") >= 0}
			_rows.append(row)
			if not ok:
				_fails.append("%s: %s" % [r.name, st])
				_phase = "rung_start"
				return false
			var obj := _out_path.get_base_dir().path_join("ladder-%s.fitted.obj" % r.name)
			_write_obj(obj, _fitted, p.data.garment.triangles, "avbd fit, rung %s of Gate 6d (body space, fit phase only)" % r.name)
			row.obj = obj
			var w = _main.get_node_or_null("World")
			if w != null:
				w._on_garment(_fitted, p.data.garment.triangles, "fit")
			_mark = _frames
			_phase = "rung_shot"
		"rung_shot":
			if _frames - _mark < 5:
				return false
			var r: Dictionary = LADDER[_rung]
			var png := _out_path.get_base_dir().path_join("ladder-%s.png" % r.name)
			var img: Image = root.get_texture().get_image()
			var shot := "not taken"
			if img != null and img.save_png(png) == OK:
				shot = "%s (%dx%d)" % [png.get_file(), img.get_width(), img.get_height()]
			_rows[-1].screenshot = shot
			_phase = "rung_check"
		"rung_check":
			var row: Dictionary = _rows[-1]
			if not _main.fit.available() or _fit_begin.begins_with("FAIL"):
				row.check = "not measured (fit.elf %s)" % _fit_begin
				_say("  check: " + row.check)
				_phase = "rung_start"
				return false
			if _main.fit.busy():
				return false
			if _fit_begin == "":
				var q: Dictionary = _main.fit.poll()
				_fit_begin = "host_ms=%d %s" % [q.host_ms, str(q.result)]
				_say("fit.elf begin: " + _fit_begin)
				if not str(q.result).begins_with("OK begin"):
					_fit_begin = "FAIL " + str(q.result)
					return false
			var p = _main.pipeline
			var t0 := Time.get_ticks_msec()
			var pushed: PackedFloat32Array = p._push_vertex(_fitted)
			row.check_control = _main.fit.check_intersections(pushed)
			row.check = _main.fit.check_intersections(_fitted)
			row.check_ms = Time.get_ticks_msec() - t0
			_say("  check: %s | control (vertex %d onto the pelvis): %s | %d ms" % [row.check, p.data.pushed_vertex,
					row.check_control, row.check_ms])
			_phase = "rung_start"
	return false

func _on_state(s: String, rec: Dictionary) -> void:
	if s == "DONE" or s == "FAILED":
		_say("=> %s %s" % [s, rec.get("reason", "")])
		return
	_say("STATE %-13s %8d ms  vm %9.1f ms  %s" % [s, rec.ms, rec.vm_ms, rec.note])

static func _kv(text: String, key: String) -> float:
	var at := text.find(key)
	if at < 0:
		return -1.0
	var s := text.substr(at + key.length())
	var end := 0
	while end < s.length() and (s[end] == "-" or s[end] == "." or (s[end] >= "0" and s[end] <= "9")):
		end += 1
	return float(s.substr(0, end)) if end > 0 else -1.0

func _write_obj(path: String, v: PackedFloat32Array, f: PackedInt32Array, comment: String) -> void:
	var fo := FileAccess.open(path, FileAccess.WRITE)
	if fo == null:
		_say("  cannot write " + path)
		return
	fo.store_line("# " + comment)
	for i in range(0, v.size(), 3):
		fo.store_line("v %.9f %.9f %.9f" % [v[i], v[i + 1], v[i + 2]])
	for i in range(0, f.size(), 3):
		fo.store_line("f %d %d %d" % [f[i] + 1, f[i + 1] + 1, f[i + 2] + 1])
	fo.close()

# ladder_eval.py over every rung's OBJ: {rung name: {ref_mean, ref_p95,
# ref_max, gap_mean, gap_p95, gap_max, y_min, y_max, same_tris}} (mm; y in m),
# plus "REF" for the PolyFEM fit's own gap and height. Its lines are echoed.
func _run_eval() -> Dictionary:
	var args := [ProjectSettings.globalize_path(EVAL), ProjectSettings.globalize_path(AVATAR),
			ProjectSettings.globalize_path(REF), str(VOXEL_M)]
	var names := []
	for r in _rows:
		if r.has("obj"):
			args.append(r.obj)
			names.append(r.name)
	if names.is_empty():
		return {}
	var out := []
	var t0 := Time.get_ticks_msec()
	var rc := OS.execute("python", args, out, true)
	_say("ladder_eval.py: rc %d in %.1f s" % [rc, (Time.get_ticks_msec() - t0) / 1000.0])
	var stats := {}
	for chunk in out:
		for line in str(chunk).split("\n", false):
			line = line.strip_edges()
			if line == "":
				continue
			_say("  " + line)
			if line.begins_with("REF gap to body:"):
				stats["REF"] = {"gap_mean": _after(line, "gap to body: mean "), "gap_p95": _after(line, "gap to body: mean", "p95 "),
						"gap_max": _after(line, "gap to body: mean", "max "), "y_min": _after(line, "| y "),
						"y_max": _after(line, "| y ", "..")}
				continue
			if line.begins_with("ladder-") and line.find("to REF:") >= 0:
				var nm := line.substr(7, line.find(".fitted.obj") - 7)
				stats[nm] = {"ref_mean": _after(line, "to REF: mean "), "ref_p95": _after(line, "to REF: mean", "p95 "),
						"ref_max": _after(line, "to REF: mean", "max "), "surf_mean": _after(line, "REF surface: mean "),
						"surf_p95": _after(line, "REF surface: mean", "p95 "), "surf_max": _after(line, "REF surface: mean", "max "),
						"gap_mean": _after(line, "gap to body: mean "),
						"gap_p95": _after(line, "gap to body: mean", "p95 "), "gap_max": _after(line, "gap to body: mean", "max "),
						"y_min": _after(line, "| y "), "y_max": _after(line, "| y ", ".."),
						"same_tris": line.find("triangles == REF") >= 0}
	return stats

# The number after `key` (the first one after `from`, if given).
static func _after(line: String, from: String, key: String = "") -> float:
	var at := line.find(from)
	if at < 0:
		return -1.0
	var s := line.substr(at + from.length())
	if key != "":
		var k := s.find(key)
		if k < 0:
			return -1.0
		s = s.substr(k + key.length())
	var end := 0
	while end < s.length() and (s[end] == "-" or s[end] == "." or (s[end] >= "0" and s[end] <= "9")):
		end += 1
	return float(s.substr(0, end)) if end > 0 else -1.0

func _evaluate() -> void:
	_say("")
	var stats := _run_eval()
	for r in _rows:
		if stats.has(r.name):
			r.eval = stats[r.name]
	var js := FileAccess.open(_out_path.get_basename() + ".json", FileAccess.WRITE)
	if js != null:
		js.store_string(JSON.stringify({"rows": _rows, "fit_begin": _fit_begin, "chosen": CHOSEN,
				"FIT_AVBD": _main.drape.FIT_AVBD, "ref": stats.get("REF", {})}, "  "))
		js.close()
	var ref: Dictionary = stats.get("REF", {})
	_say("")
	_say("The ladder (mm; toREF = per-vertex distance to the PolyFEM fit, surf = surface to surface both ways, gap = to the body; REF: gap %s / %s mm mean / p95, y %s..%s m):" % [
			str(ref.get("gap_mean", "?")), str(ref.get("gap_p95", "?")), str(ref.get("y_min", "?")), str(ref.get("y_max", "?"))])
	_say("rung                        kFit sim  kAnc settle iters steps ms/step  fit_s  pushes   s     y_min  y_max  toREF mean  p95    max   surf mean  p95    max   gap mean  p95    check")
	for r in _rows:
		var s: Dictionary = r.setting
		var e: Dictionary = r.get("eval", {})
		_say("%-26s %6s %3d %5s %6d %5d %5d %7.1f %6.1f %7d  %5.3f  %5.3f  %5.3f  %7.1f %6.1f %6.1f  %6.1f %6.1f %6.1f  %6.1f %6.1f   %s" % [r.name,
				str(s.k), int(s.similarity), str(s.kAnchor), int(s.settle), s.iters, int(r.get("steps", -1)),
				float(r.get("ms_per_step", -1)), float(r.get("host_ms", -1)) / 1000.0, int(r.get("self_pushes", -1)),
				float(r.get("sim_s", -1)),
				float(e.get("y_min", -1)), float(e.get("y_max", -1)), float(e.get("ref_mean", -1)), float(e.get("ref_p95", -1)),
				float(e.get("ref_max", -1)), float(e.get("surf_mean", -1)), float(e.get("surf_p95", -1)),
				float(e.get("surf_max", -1)), float(e.get("gap_mean", -1)), float(e.get("gap_p95", -1)),
				str(r.get("check", "-")).left(48)])
	var ok := _fails.is_empty()
	for f in _fails:
		_say("FAIL rung " + f)
	# The pick: the constant in drape_stage.gd must be the CHOSEN rung's numbers.
	var chosen: Dictionary = {}
	for r in LADDER:
		if r.name == CHOSEN:
			chosen = _rung_setting(r)
	var c: Dictionary = _main.drape.FIT_AVBD
	var drift := []
	for k in chosen:
		if not c.has(k) or not is_equal_approx(float(c[k]), float(chosen[k])):
			drift.append("%s: FIT_AVBD %s, %s %s" % [k, str(c.get(k)), CHOSEN, str(chosen[k])])
	if not drift.is_empty():
		ok = false
		_say("FAIL drape_stage.gd FIT_AVBD is not the chosen rung %s: %s" % [CHOSEN, "; ".join(drift)])
	else:
		_say("PASS drape_stage.gd FIT_AVBD == rung %s" % CHOSEN)
	var crow := {}
	var k0 := {}
	var nosim := {}
	for r in _rows:
		if r.name == CHOSEN:
			crow = r
		if r.name.begins_with("f0-"):
			k0 = r
		if r.name.ends_with("-nosim"):
			nosim = r
	if _only.is_empty() or _only.has(CHOSEN):
		var cc := str(crow.get("check", ""))
		var ctl := str(crow.get("check_control", ""))
		var pass_c := cc.begins_with("OK none") and ctl.find("INTERSECTS") >= 0
		if not pass_c:
			ok = false
		_say("%s chosen rung %s: check %s | control %s" % ["PASS" if pass_c else "FAIL", CHOSEN, cc, ctl])
		var ce: Dictionary = crow.get("eval", {})
		if ce.is_empty():
			ok = false
			_say("FAIL no shape statistics for the chosen rung (ladder_eval.py)")
		else:
			var near: bool = float(ce.surf_mean) <= PASS_SURF_MEAN_MM and float(ce.surf_p95) <= PASS_SURF_P95_MM and bool(ce.same_tris)
			if not near:
				ok = false
			_say("%s chosen rung %s surface vs the PolyFEM fit's: mean %.1f mm (want <= %.0f), p95 %.1f (want <= %.0f), max %.1f; y %.3f..%.3f (PolyFEM %s..%s); similarity s %.4f rot %.3f deg; triangles equal %s" % [
					"PASS" if near else "FAIL", CHOSEN, float(ce.surf_mean), PASS_SURF_MEAN_MM, float(ce.surf_p95),
					PASS_SURF_P95_MM, float(ce.surf_max), float(ce.y_min), float(ce.y_max), str(ref.get("y_min", "?")),
					str(ref.get("y_max", "?")), float(crow.get("sim_s", -1)), float(crow.get("sim_rot_deg", -1)), str(ce.same_tris)])
			var pv: bool = float(ce.ref_mean) <= PASS_MEAN_MM and float(ce.ref_p95) <= PASS_P95_MM
			_say("INFO chosen rung %s per vertex vs the PolyFEM fit: mean %.1f mm (bar %.0f: %s), p95 %.1f (bar %.0f), max %.1f: the vertices' places on the surface, not the surface" % [
					CHOSEN, float(ce.ref_mean), PASS_MEAN_MM, "met" if pv else "NOT met", float(ce.ref_p95), PASS_P95_MM,
					float(ce.ref_max)])
			for ctrl in [["no pull (the tube)", k0], ["no similarity rest update (the authored length held)", nosim]]:
				var row: Dictionary = ctrl[1]
				var ke: Dictionary = row.get("eval", {})
				if ke.is_empty():
					ok = false
					_say("FAIL control %s: no shape statistics" % ctrl[0])
					continue
				var far: bool = float(ke.ref_mean) >= 1.5 * float(ce.ref_mean)
				if not far:
					ok = false
				_say("%s control %s: %.1f mm from the PolyFEM fit, the chosen rung %.1f mm (want >= 1.5x)" % [
						"PASS" if far else "FAIL", ctrl[0], float(ke.ref_mean), float(ce.ref_mean)])
	_finish("PASS" if ok else "FAIL")

func _finish(verdict: String) -> void:
	_say("wall %.1f s, frames %d" % [(Time.get_ticks_msec() - _t0) / 1000.0, _frames])
	_say("RESULT: " + verdict)
	_phase = "done"
	if _main != null and _main.get("pipeline") != null and _main.drape.sandbox != null and not _main.drape.drape_status().begins_with("RUNNING"):
		_say("drape rd_close: " + str(_main.drape.call_now("rd_close")))
	if _out != null:
		_out.close()
		_out = null
	if _main != null and _main.get("pipeline") != null and _main.fit != null and _main.fit.busy():
		print("exiting with a fit vmcall in flight")
		OS.kill(OS.get_process_id())
		return
	quit(0 if verdict == "PASS" else 1)
