# probe_drape_skirt -- Gate 8's fitted skirt on the drape API, one variant per
# run: backend x scale x config knobs, N steps, then the positions' finiteness,
# the first non-finite vertex, and the result line. Frame-driven (one
# drape_tick per frame), wall clock in every branch; lines go to
# gates/5-drape/skirt/probe.txt (or out=<file>).
#
#   godot --path project --script probe_drape_skirt.gd --rendering-driver vulkan --xr-mode off -- \
#       variants="rd:10;cpu:10;rd:10:selfCollision=0" steps=1 caps=0
#   [body=1: the body mesh collider]; or one drape job over the same scene,
#   its args with commas for spaces:
#   ... -- job=mesh_bisect jobargs=scale=10,caps=0 out=bisect.txt
extends SceneTree

const DIR := "res://../gates/5-drape/skirt/"
const WALL_S := 600.0

var _sb = null
var _out: FileAccess
var _t0 := 0
var _variants := []
var _cur = null
var _step := 0
var _ticks := 0
var _steps := 1
var _caps := false
var _body := false
var _body_mesh := {}
var _mesh := {}
var _pins := PackedInt32Array()
var _capsules := []
var _done := false
var _job := ""
var _jobargs := ""
var _job_started := false

func _say(s: String) -> void:
	print(s)
	if _out != null:
		_out.store_line(s)
		_out.flush()

func _initialize() -> void:
	_t0 = Time.get_ticks_usec()
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	var spec := "rd:10;cpu:10"
	var out := "probe.txt"
	for a in OS.get_cmdline_user_args():
		if a.begins_with("variants="):
			spec = a.substr(9)
		elif a.begins_with("steps="):
			_steps = int(a.substr(6))
		elif a.begins_with("caps="):
			_caps = a.substr(5) == "1"
		elif a.begins_with("body="):
			_body = a.substr(5) == "1"
		elif a.begins_with("out="):
			out = a.substr(4)
		elif a.begins_with("job="):
			_job = a.substr(4)
		elif a.begins_with("jobargs="):
			_jobargs = a.substr(8).replace(",", " ")
	_out = FileAccess.open(ProjectSettings.globalize_path(DIR + out), FileAccess.WRITE)
	_say("# probe_drape_skirt %s Godot %s %s steps=%d caps=%s body=%s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, RenderingServer.get_video_adapter_name(), _steps, _caps, _body])
	for v in spec.split(";", false):
		var p := v.split(":", false)
		var knobs := {}
		for i in range(2, p.size()):
			var kv := p[i].split("=")
			knobs[kv[0]] = float(kv[1])
		_variants.append({"backend": p[0], "scale": float(p[1]), "knobs": knobs, "name": v})
	_mesh = load("res://util/obj_io.gd").read(ProjectSettings.globalize_path(DIR + "fitted_skirt.obj"))
	for t in FileAccess.get_file_as_string(ProjectSettings.globalize_path(DIR + "fitted_skirt_pins.txt")).split(" ", false):
		_pins.append(int(t))
	for line in FileAccess.get_file_as_string(ProjectSettings.globalize_path(DIR + "fitted_skirt_capsules.txt")).split("\n", false):
		if line.begins_with("#"):
			continue
		var f := PackedFloat32Array()
		for t in line.split(" ", false):
			f.append(float(t))
		_capsules.append(f)
	_body_mesh = load("res://util/obj_io.gd").read(ProjectSettings.globalize_path(DIR + "body.obj"))
	_say("mesh %d v %d f, %d pins, %d capsules, body %d v %d f" % [_mesh.v.size() / 3, _mesh.f.size() / 3, _pins.size(),
			_capsules.size(), _body_mesh.v.size() / 3, _body_mesh.f.size() / 3])
	_sb = ClassDB.instantiate("Sandbox")
	_sb.references_max = 65536
	_sb.program = load("res://drape.elf")

const DEFAULT_KNOBS := {"iters": 16, "selfCollision": 1, "contact": 1, "frictionPred": 1, "colors": 1,
		"membrane": 1, "bending": 1, "selfPasses": 2, "selfK": 16}

func _process(_d: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_usec() - _t0 > int(WALL_S * 1e6):
		_say("WALL CLOCK ran out")
		_quit()
		return true
	if _job != "":
		_job_tick()
		return false
	if _cur == null:
		if _variants.is_empty():
			_quit()
			return true
		_cur = _variants.pop_front()
		var s: float = _cur.scale
		_sb.vmcall("drape_open", _cur.backend)
		_sb.vmcall("drape_primitive", "clear", PackedFloat32Array())
		if _caps:
			for c in _capsules:
				_sb.vmcall("drape_primitive", "capsule", PackedFloat32Array([c[0] * s, c[1] * s, c[2] * s, c[3], c[4], c[5],
						c[6] * s, c[7] * s, 0.3]))
		if _body:
			var bv := PackedFloat32Array()
			bv.resize(_body_mesh.v.size())
			for i in _body_mesh.v.size():
				bv[i] = _body_mesh.v[i] * s
			_say("    " + str(_sb.vmcall("drape_primitive_mesh", bv, _body_mesh.f, PackedFloat32Array([0.1, 0.3]))).substr(0, 160))
		for k in DEFAULT_KNOBS:
			_sb.vmcall("drape_config", k, float(_cur.knobs.get(k, DEFAULT_KNOBS[k])))
		_sb.vmcall("drape_config", "gravityY", -9.8 * s)
		var pos := PackedFloat32Array()
		pos.resize(_mesh.v.size())
		for i in _mesh.v.size():
			pos[i] = _mesh.v[i] * s
		var r := str(_sb.vmcall("drape_scene_mesh", pos, _mesh.f, _pins, PackedFloat32Array()))
		_say("--- %s: %s" % [_cur.name, r.split("\n")[0]])
		_say("    " + str(_sb.vmcall("drape_queue_forward", _steps)))
		_ticks = 0
		return false
	var t := str(_sb.vmcall("drape_tick", Time.get_ticks_usec()))
	_ticks += 1
	if t.begins_with("RUNNING"):
		return false
	var p: PackedFloat32Array = _sb.vmcall("drape_positions")
	var bad := 0
	var first := -1
	for i in range(0, p.size(), 3):
		if not (is_finite(p[i]) and is_finite(p[i + 1]) and is_finite(p[i + 2])):
			bad += 1
			if first < 0:
				first = i / 3
	_say("    %s | ticks=%d non-finite vertices %d of %d (first %d)" % [t, _ticks, bad, p.size() / 3, first])
	_cur = null
	return false

func _job_tick() -> void:
	if not _job_started:
		_job_started = true
		for k in ["mesh_obj", "mesh_pins", "mesh_capsules", "body_obj"]:
			var f: String = {"mesh_obj": "fitted_skirt.obj", "mesh_pins": "fitted_skirt_pins.txt",
					"mesh_capsules": "fitted_skirt_capsules.txt", "body_obj": "body.obj"}[k]
			_say(str(_sb.vmcall("drape_job_data", k, FileAccess.get_file_as_string(ProjectSettings.globalize_path(DIR + f)))))
		_say(str(_sb.vmcall("drape_job_start", _job, "rd", _jobargs)))
		_ticks = 0
		return
	var r := str(_sb.vmcall("drape_job_tick", Time.get_ticks_usec()))
	_ticks += 1
	if r.begins_with("RUNNING"):
		return
	_say("ticks=%d wall_s=%.1f" % [_ticks, (Time.get_ticks_usec() - _t0) / 1e6])
	_say(r)
	_quit()

func _quit() -> void:
	_done = true
	if _sb != null:
		_say(str(_sb.vmcall("rd_close")))
		_sb.free()
		_sb = null
	_say("END")
	if _out != null:
		_out.close()
	quit(0)
