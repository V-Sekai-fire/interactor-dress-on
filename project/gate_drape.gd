# Gate 5 (drape.elf, Task D): DiffCloth's sphere demo forward and backward in
# the guest, Eigen-free, over AvbdCpu and AvbdRd, frame-driven.
#
#   godot --path project --script gate_drape.gd --rendering-driver vulkan --xr-mode off --gpu-index 1 > ../gates/5-drape/drape/run.log 2>&1
#   (-- quick: short versions of every job, for iterating)
#
# Checks (plan Cut 5, G4-G6, G8's rule-4 half):
#  - the sphere scene's faces equal the native iter0/0.obj faces;
#  - the drape_* API end to end on auto (= rd at 625 vertices): scene, 3 steps,
#    positions finite, frame 1 against the native frame 1;
#  - sphere_forward at mu 0.539770 (the native iter0) on rd, 350 steps, and on
#    cpu, 100 steps: frame 1 <= 1e-5 and frame 10 <= 1e-3 from the native
#    frames; frames 50/100/350 reported; every step finite;
#  - the control: mu 0.3 on rd, whose frames must sit further from the native
#    ones at frames >= 50 than ours at the same mu do;
#  - sphere_backward native at mu 0.539770 and 0.01 against backwardLog.txt
#    (dL/dmu 0.01153 and -50.45588, rel 5e-2; the losses reported);
#  - sim_gradcheck: the unrolled mode against central differences on an 8x8
#    pinned panel and a panel on a plane, 20 steps, mu / kTri / density, single
#    colour, rel 5e-2; the multi-colour table (information);
#  - bench_drape on both backends (information: the crossover behind auto);
#  - rd_rule4 == 0 after everything, the probe raises it; rd_close frees all.
# Results stream to gates/5-drape/drape/results.txt; the last line is RESULT.
extends SceneTree

const OUT_DIR := "res://../gates/5-drape/drape/"
const NATIVE := "res://../gates/5-drape/native/"
const WALL_S := 1500.0
const NATIVE_FRAMES := [0, 1, 10, 50, 100, 350]
const NATIVE_DLDMU := {"0.539770": 0.01153, "0.010000": -50.45588, "0.375146": 0.00781}
const NATIVE_LOSS := {"0.539770": 0.00132918, "0.010000": 1.65198194, "0.375146": 0.00047714}

var _sb = null
var _out: FileAccess
var _t_start := 0
var _rc := 0
var _phase := "api"
var _done := false
var _quick := false
var _queue := []
var _cur = null
var _job_t0 := 0
var _job_ticks := 0
var _results := {}
var _native := {}      # frame -> PackedFloat32Array
var _native_faces := PackedInt32Array()
var _frames := {}      # job key -> {frame: PackedFloat32Array}
var _api_step := 0
var _api_ticks := 0

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _fail(why: String) -> void:
	_rc = 1
	_say("FAIL: " + why)

func _check(ok: bool, line: String) -> void:
	_say(("PASS " if ok else "FAIL ") + line)
	if not ok:
		_rc = 1

func _finish() -> void:
	if _sb != null:
		var c := str(_sb.vmcall("rd_close"))
		var cok := c.begins_with("CLOSED device=") and c.ends_with("permanent_slots=0")
		_check(cok, "rd_close: " + c)
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	if _out != null:
		_out.close()
		_out = null
	if _sb != null:
		_sb.free()
		_sb = null
	_done = true
	quit(_rc)

func _load_obj(path: String) -> Array:
	var f := FileAccess.open(ProjectSettings.globalize_path(path), FileAccess.READ)
	if f == null:
		return []
	var v := PackedFloat32Array()
	var faces := PackedInt32Array()
	while not f.eof_reached():
		var line := f.get_line()
		if line.begins_with("v "):
			var p := line.split(" ", false)
			v.append(p[1].to_float())
			v.append(p[2].to_float())
			v.append(p[3].to_float())
		elif line.begins_with("f "):
			var p := line.split(" ", false)
			for i in range(1, 4):
				faces.append(p[i].split("/")[0].to_int() - 1)
	f.close()
	return [v, faces]

func _initialize() -> void:
	_t_start = Time.get_ticks_usec()
	_quick = OS.get_cmdline_user_args().has("quick")
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	var name := "results_quick.txt" if _quick else "results.txt"
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT_DIR + name), FileAccess.WRITE)
	_say("# Gate 5 drape (Task D), %s, Godot %s, %s%s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, RenderingServer.get_video_adapter_name(), " (quick)" if _quick else ""])
	for fr in NATIVE_FRAMES:
		var o := _load_obj(NATIVE + "iter0/%d.obj" % fr)
		if o.is_empty():
			_fail("could not read native iter0/%d.obj" % fr)
			_finish()
			return
		_native[fr] = o[0]
		if fr == 0:
			_native_faces = o[1]
	_sb = ClassDB.instantiate("Sandbox")
	if _sb == null:
		_fail("Sandbox class not registered")
		_finish()
		return
	_sb.references_max = 65536
	_sb.program = load("res://drape.elf")
	for p in ["memory_max", "instructions_max", "execution_timeout"]:
		if p in _sb:
			_say("sandbox %s = %s" % [p, str(_sb.get(p))])
	var steps_rd := 60 if _quick else 350
	var steps_cpu := 12 if _quick else 100
	_queue.append(["sphere_forward", "rd", "steps=%d" % steps_rd])
	_queue.append(["sphere_forward", "cpu", "steps=%d" % steps_cpu])
	_queue.append(["sphere_forward", "rd", "steps=%d mu=0.3" % steps_rd])
	_queue.append(["sphere_backward", "rd", "steps=%d mus=0.539770,0.01,0.375146" % steps_rd])
	_queue.append(["sphere_backward", "cpu", "steps=%d mus=0.539770,0.01,0.375146" % steps_rd])
	for sc in ["panel", "plane"]:
		_queue.append(["sim_gradcheck", "cpu", "scene=%s colors=0%s" % [sc, " steps=8" if _quick else ""]])
	for sc in ["panel", "plane"]:
		_queue.append(["sim_gradcheck", "cpu", "scene=%s colors=1%s" % [sc, " steps=8" if _quick else ""]])
	_queue.append(["bench_drape", "cpu", "sizes=8,16,24" if _quick else "sizes=8,16,24,32"])
	_queue.append(["bench_drape", "rd", "sizes=8,16,24" if _quick else "sizes=8,16,24,32,48,64"])

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_usec() - _t_start > int(WALL_S * 1e6):
		_fail("the %d s wall clock ran out in phase '%s'%s" % [int(WALL_S), _phase,
				(" at job %s %s %s" % _cur) if _cur != null else ""])
		_finish()
		return true
	if _phase == "api":
		_api()
	elif _phase == "wrappers":
		_wrappers()
	elif _phase == "jobs":
		_jobs()
	elif _phase == "checks":
		_checks()
		_finish()
	return _done

# --- the API end to end ----------------------------------------------------------

func _api() -> void:
	if _api_step == 0:
		_say(str(_sb.vmcall("drape_open", "cpu")))
		var r := str(_sb.vmcall("drape_scene_sphere_demo"))
		_say(r.split("\n")[0])
		var faces: PackedInt32Array = _sb.vmcall("drape_faces")
		_check(faces == _native_faces, "faces: the sphere scene's %d triangles equal native iter0/0.obj's %d, in order" % [
				faces.size() / 3, _native_faces.size() / 3])
		var f0: PackedFloat32Array = _sb.vmcall("drape_frame", 0)
		_check(_maxdiff(f0, _native[0]) <= 1e-6, "frame 0 (rest) against native: max|dx| = %s" % _g(_maxdiff(f0, _native[0])))
		_say(str(_sb.vmcall("drape_open", "auto")))
		r = str(_sb.vmcall("drape_scene_sphere_demo"))
		_say("api auto: " + r.split("\n")[0])
		_say(str(_sb.vmcall("drape_config", "mu", 0.539770)).substr(0, 60))
		_say(str(_sb.vmcall("drape_queue_forward", 3)))
		_api_step = 1
		return
	var t := str(_sb.vmcall("drape_tick", Time.get_ticks_usec()))
	_api_ticks += 1
	if t.begins_with("RUNNING") and _api_ticks < 1000:
		return
	var pos: PackedFloat32Array = _sb.vmcall("drape_positions")
	var fin := pos.size() == 625 * 3
	for x in pos:
		fin = fin and is_finite(x)
	var f1: PackedFloat32Array = _sb.vmcall("drape_frame", 1)
	var d1 := _maxdiff(f1, _native[1])
	_check(t.begins_with("IDLE DONE forward rd") and fin and d1 <= 1e-5,
			"api: drape_open(auto) picked rd for 625 vertices, 3 steps in %d ticks, positions finite=%s, frame 1 max|dx| = %s: %s" % [
				_api_ticks, fin, _g(d1), t])
	_phase = "wrappers"

# --- main.gd's no-argument wrappers (rule 8), ticked by main.gd's own _process --

var _main: Node = null
var _main_frames := 0

func _wrappers() -> void:
	if _main == null:
		_main = load("res://main.gd").new()
		root.add_child(_main)
		_say("wrappers: " + str(_main.drape_sphere_demo("cpu")).split("\n")[0])
		_say("wrappers: " + str(_main.drape_forward(2)))
		return
	_main_frames += 1
	var st: String = _main.drape_status()
	if not st.begins_with("IDLE") and _main_frames < 600:
		return
	var job: String = _main.drape_job_names()
	_check(st.begins_with("IDLE DONE forward cpu steps=2") and job.begins_with("sphere_forward"),
			"main.gd wrappers: drape_sphere_demo(cpu) + drape_forward(2) -> drape_status '%s' after %d frames; drape_job_names '%s'" % [
				st.substr(0, 60), _main_frames, job])
	_main.queue_free()
	_main = null
	_phase = "jobs"

# --- jobs --------------------------------------------------------------------------

func _jobs() -> void:
	if _cur == null:
		if _queue.is_empty():
			_phase = "checks"
			return
		_cur = _queue.pop_front()
		var r := str(_sb.vmcall("drape_job_start", _cur[0], _cur[1], _cur[2]))
		if not r.begins_with("STARTED"):
			_fail("%s %s %s would not start: %s" % [_cur[0], _cur[1], _cur[2], r])
			_cur = null
			return
		_job_t0 = Time.get_ticks_usec()
		_job_ticks = 0
		return
	var r := str(_sb.vmcall("drape_job_tick", Time.get_ticks_usec()))
	_job_ticks += 1
	if r.begins_with("RUNNING"):
		return
	var key := "%s %s %s" % _cur
	_results[key] = r
	var lines := r.split("\n")
	_say("job %-40s ticks=%6d wall_ms=%9.1f  %s" % [key, _job_ticks, (Time.get_ticks_usec() - _job_t0) / 1000.0, lines[0]])
	for i in range(1, lines.size()):
		_say("    " + lines[i])
	if not r.begins_with("PASS"):
		_rc = 1
	if _cur[0] == "sphere_forward":
		var fr := {}
		for k in NATIVE_FRAMES:
			var a: PackedFloat32Array = _sb.vmcall("drape_job_frame", k)
			if a.size() > 0:
				fr[k] = a
		_frames[key] = fr
	_cur = null

# --- checks -------------------------------------------------------------------------

func _maxdiff(a: PackedFloat32Array, b: PackedFloat32Array) -> float:
	if a.size() != b.size() or a.size() == 0:
		return INF
	var m := 0.0
	for i in range(a.size()):
		m = maxf(m, absf(a[i] - b[i]))
	return m

func _g(x: float) -> String:
	if x == 0.0 or is_nan(x) or is_inf(x):
		return str(x)
	var e := int(floor(log(absf(x)) / log(10.0)))
	if e >= -3 and e < 5:
		return String.num(x, maxi(0, 3 - e))
	return "%.3fe%d" % [x / pow(10.0, e), e]

func _frames_of(prefix: String) -> Dictionary:
	for k in _frames.keys():
		if k.begins_with(prefix):
			return _frames[k]
	return {}

func _checks() -> void:
	_say("--- checks")
	var steps_rd := 60 if _quick else 350
	var steps_cpu := 12 if _quick else 100
	var rd := _frames_of("sphere_forward rd steps=%d" % steps_rd)
	var cpu := _frames_of("sphere_forward cpu")
	var ctl := _frames_of("sphere_forward rd steps=%d mu=0.3" % steps_rd)
	# rd and ctl share a prefix; pick ours by the absent mu key.
	for k in _frames.keys():
		if k == "sphere_forward rd steps=%d" % steps_rd:
			rd = _frames[k]
	for which in [["rd", rd], ["cpu", cpu]]:
		var fr: Dictionary = which[1]
		var parts := []
		var ok := fr.has(1) and fr.has(10)
		for k in NATIVE_FRAMES:
			if fr.has(k):
				var d := _maxdiff(fr[k], _native[k])
				parts.append("%d: %s" % [k, _g(d)])
				if k == 1:
					ok = ok and d <= 1e-5
				if k == 10:
					ok = ok and d <= 1e-3
		_check(ok, "G4 %s frames at mu 0.539770 against native iter0 (max|dx|; frame 1 <= 1e-5, frame 10 <= 1e-3): %s" % [
				which[0], ", ".join(PackedStringArray(parts))])
	if rd.size() > 0 and cpu.size() > 0:
		var parts := []
		for k in NATIVE_FRAMES:
			if rd.has(k) and cpu.has(k):
				parts.append("%d: %s" % [k, _g(_maxdiff(rd[k], cpu[k]))])
		_say("info cpu vs rd frames (max|dx|): " + ", ".join(PackedStringArray(parts)))
	var cparts := []
	var cok := true
	var judged := 0
	for k in [50, 100, 350]:
		if ctl.has(k) and rd.has(k):
			var dc := _maxdiff(ctl[k], _native[k])
			var ds := _maxdiff(rd[k], _native[k])
			cparts.append("%d: mu0.3-native %s vs ours-native %s" % [k, _g(dc), _g(ds)])
			cok = cok and dc > ds
			judged += 1
	_check(cok and judged > 0, "G4 control: mu 0.3 sits further from the native mu-0.54 frames than ours at mu 0.54 (frames >= 50): " +
			", ".join(PackedStringArray(cparts)))
	# G5: native-mode dL/dmu against backwardLog.
	var bw := ""
	var bwc := ""
	for k in _results.keys():
		if k.begins_with("sphere_backward rd"):
			bw = _results[k]
		if k.begins_with("sphere_backward cpu"):
			bwc = _results[k]
	var re := RegEx.new()
	re.compile("RESULT mu=(\\S+) loss=(\\S+) dL/dmu=(\\S+)")
	# The flat control for G5: the same port on the cpu backend, whose
	# trajectory differs from rd's only in float ordering. How far its
	# dL/dmu sits from rd's is the spread the forward's own sensitivity puts
	# on this number, independent of any backward error.
	var cpu_g := {}
	for m in re.search_all(bwc):
		cpu_g[m.get_string(1)] = [m.get_string(2).to_float(), m.get_string(3).to_float()]
	var n := 0
	for m in re.search_all(bw):
		var mu0: String = m.get_string(1)
		if cpu_g.has(mu0):
			var gr := m.get_string(3).to_float()
			var gc: float = cpu_g[mu0][1]
			_say("info G5 spread at mu %s: dL/dmu rd %s vs cpu %s (rel %s); loss rd %s vs cpu %s" % [mu0, _g(gr), _g(gc),
					_g(absf(gr - gc) / maxf(absf(gr), 1e-30)), _g(m.get_string(2).to_float()), _g(cpu_g[mu0][0])])
	for m in re.search_all(bw):
		var mu: String = m.get_string(1)
		if not NATIVE_DLDMU.has(mu):
			continue
		n += 1
		var g := m.get_string(3).to_float()
		var nat: float = NATIVE_DLDMU[mu]
		var rel := absf(g - nat) / absf(nat)
		var loss := m.get_string(2).to_float()
		var natl: float = NATIVE_LOSS[mu]
		var lrel := absf(loss - natl) / natl
		_check(rel <= 5e-2, "G5 native dL/dmu at mu %s: ours %s, backwardLog %s, rel %s (limit 5e-2); loss ours %s native %s (rel %s)" % [
				mu, _g(g), _g(nat), _g(rel), _g(loss), _g(natl), _g(lrel)])
	if n != 3:
		_fail("G5: parsed %d of 3 sphere_backward results" % n)
	_check_rule4()

func _count(s: String, key: String) -> int:
	var re := RegEx.new()
	re.compile("\\b" + key + "=(\\d+)")
	var m := re.search(s)
	return int(m.get_string(1)) if m != null else -1

func _check_rule4() -> void:
	var r4 := str(_sb.vmcall("rd_rule4"))
	var same := _count(r4, "same_frame_syncs")
	var syncs := _count(r4, "syncs")
	_check(same == 0 and syncs > 0, "rd_rule4 after every job: %s (want same_frame_syncs=0 with syncs>0)" % r4)
	var probe := str(_sb.vmcall("rd_rule4_probe"))
	var after := _count(str(_sb.vmcall("rd_rule4")), "same_frame_syncs")
	_check(probe.begins_with("PASS") and after == same + 1, "rd_rule4_probe (positive control): %s; counter now %d" % [probe, after])
