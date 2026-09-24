# Gate 5 (drape.elf): DiffCloth's Simulation and L-BFGS-B in the guest,
# Eigen-free, over AvbdCpu / AvbdRd and the Lean-emitted L-BFGS-B kernels,
# frame-driven (vsync off, one drape_job_tick / drape_tick per frame, a wall
# clock that quits in every branch).
#
#   godot --path project --script gate_drape.gd --rendering-driver vulkan --xr-mode off --gpu-index 1 > ../gates/5-drape/run.log 2>&1
#   (-- only=G3,G7 : those groups only, into results_partial.txt; -- quick : short jobs)
#
# G1 lbfgsb_components cpu+rd: the 20 LBFGSpp component fixtures (1e-4, sets
#    identical); control theta x1.25 fails all 20.
# G2 lbfgsb_problems cpu+rd: 20 LBFGSpp traces (f 1e-6 abs+rel, x 1e-3,
#    iterations max(2, 25%)).
# G3 inverse_min cpu+rd: k_tri 0.5 -> 2.0 and (k_bend, density) (0.4, 2.5) ->
#    (1.5, 1.25) within 0.05; iterations within 2 and parameters within 1e-3 of
#    LBFGSpp on the host-compiled AvbdCpu objective; zero-gradient arm > 0.1.
# G4 sphere_forward: faces equal native; frame 1 <= 1e-5, frame 10 <= 1e-3 from
#    native iter0; frames 50/100/350 reported; control mu 0.3 further from
#    native than ours at frames >= 50; 100 steps cpu+rd finite, 350 rd.
# G5 sphere_backward native: dL/dmu within 5e-2 of backwardLog at mu 0.539770
#    and 0.01, loss at 0.01 prints 1.652; 0.375146 and the cpu spread reported.
# G6 sim_gradcheck: unrolled vs central FD within 5e-2, single colour, 8x8
#    pinned panel and panel on a plane, 20 steps, mu/kTri/density; the
#    colours-on / step / native error table picks the default mode.
# G7 lbfgsb_replay: our L-BFGS-B on backwardLog's values evaluates 0.539770 ->
#    0.010000 -> 0.375146 (printed precision); live native-mode optimize on rd
#    (the sequence reported against native); the recompute (unrolled, step)
#    optimize's final mu against the ground truth 0.3.
# G8 rd_rule4 same_frame_syncs == 0 (the probe raises it); llvm-nm -C drape.elf
#    has 0 Eigen (and 0 LBFGSpp) symbols.
# G9 crossovers: L-BFGS-B ms/iteration cpu vs rd for n = 10..1e5; drape ms/step
#    cpu vs rd with contact + self-collision, 8x8..64x64; drape_open(auto)'s
#    threshold must sit at the measured crossover.
# G10 the fitted skirt at drape scale 10 (gates/5-drape/skirt: Gate 8's fit.elf
#    result, its 44 waist pins, the 14 skeleton capsules, the FoxGirl body):
#    mesh_parity cpu vs rd over 5 steps, capsules and the body mesh collider,
#    both finite and max|x_cpu - x_rd| <= 1e-4 drape units; mesh_bisect finds
#    rd finite and within 1e-3 of cpu through all 16 iterations of step 1.
#    Before the bending kernels' |s| = 0 guard rd was NaN on step 1 here.
# Results stream to gates/5-drape/results.txt; the last line is RESULT.
extends SceneTree

const GATE_DIR := "res://../gates/5-drape/"
const NATIVE := "res://../gates/5-drape/native/"
const ORACLE := "res://../gates/5-drape/oracle/"
const SKIRT := "res://../gates/5-drape/skirt/"
const WALL_S := 3600.0
const NATIVE_FRAMES := [0, 1, 10, 50, 100, 350]
const NATIVE_DLDMU := {"0.539770": 0.01153, "0.010000": -50.45588, "0.375146": 0.00781}
const NATIVE_LOSS := {"0.539770": 0.00132918, "0.010000": 1.65198194, "0.375146": 0.00047714}
const G5_GATED := ["0.539770", "0.010000"]
# The native run's initial mu exactly (guest/drape/drape_scene.h kNativeSphereMu0).
const MU0 := "0.5397701956236457"
const LB_SIZES := [10, 100, 1000, 10000, 100000]
const BENCH_SIZES := [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64]
# The drape bench runs twice: with frames uncapped (as the rest of the gate)
# and at 90 fps, the OpenXR display rate this project deploys at (AGENTS.md
# rule 9). rd's cost is frames per step, so where it crosses cpu depends on
# the frame rate; drape_open(auto)'s threshold is taken at 90 fps.
const BENCH_FPS := [0, 90]
const AUTO_FPS := 90

var _sb = null
var _out: FileAccess
var _t_start := 0
var _rc := 0
var _phase := "data"
var _done := false
var _quick := false
var _only := {}
var _queue := []
var _cur = null
var _job_t0 := 0
var _job_ticks := 0
var _results := {}
var _native := {}
var _native_faces := PackedInt32Array()
var _frames := {}
var _api_step := 0
var _api_ticks := 0
var _main: Node = null
var _main_frames := 0
var _cost := {}          # "lb n" -> {backend: [wall_ms, iterations]}
var _fails := {}         # group -> count
var _opt_plan := []
var _opt_cur = null
var _opt_step := 0
var _opt_ticks := 0
var _opt_results := {}
var _auto_line := ""

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
		var g := line.split(" ")[0]
		_fails[g] = int(_fails.get(g, 0)) + 1

func _on(g: String) -> bool:
	return _only.is_empty() or _only.has(g)

func _finish() -> void:
	if _sb != null:
		var c := str(_sb.vmcall("rd_close"))
		_check(c.begins_with("CLOSED device=") and c.ends_with("permanent_slots=0"), "G8 rd_close: " + c)
	var summary := []
	for g in ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]:
		if _on(g):
			summary.append("%s %s" % [g, "PASS" if int(_fails.get(g, 0)) == 0 else "FAIL(%d)" % int(_fails.get(g, 0))])
	_say("SUMMARY " + ", ".join(PackedStringArray(summary)))
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	if _out != null:
		_out.close()
		_out = null
	if _sb != null:
		_sb.free()
		_sb = null
	_done = true
	quit(_rc)

func _text(path: String) -> String:
	var f := FileAccess.open(ProjectSettings.globalize_path(path), FileAccess.READ)
	if f == null:
		return ""
	var t := f.get_as_text()
	f.close()
	return t

func _files(dir: String) -> PackedStringArray:
	var d := DirAccess.open(ProjectSettings.globalize_path(dir))
	var out := PackedStringArray()
	if d == null:
		return out
	for f in d.get_files():
		if f.ends_with(".txt"):
			out.append(f)
	out.sort()
	return out

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
	for a in OS.get_cmdline_user_args():
		if a == "quick":
			_quick = true
		elif a.begins_with("only="):
			for g in a.substr(5).split(","):
				_only[g] = true
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	var name := "results.txt" if _only.is_empty() and not _quick else "results_partial.txt"
	_out = FileAccess.open(ProjectSettings.globalize_path(GATE_DIR + name), FileAccess.WRITE)
	_say("# Gate 5 drape.elf, %s, Godot %s, %s%s%s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, RenderingServer.get_video_adapter_name(), " (quick)" if _quick else "",
			(" only " + ",".join(PackedStringArray(_only.keys()))) if not _only.is_empty() else ""])
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
	if _sb != null: _sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	if _sb == null:
		_fail("Sandbox class not registered")
		_finish()
		return
	_sb.references_max = 65536
	_sb.program = load("res://drape.elf")
	for p in ["memory_max", "execution_timeout"]:
		if p in _sb:
			_say("sandbox %s = %s" % [p, str(_sb.get(p))])
	_plan()

func _job(g: String, name: String, backend: String, args: String, cost := "", fps := 0) -> void:
	if _on(g):
		_queue.append([g, name, backend, args, cost, fps])

func _plan() -> void:
	var steps_rd := 60 if _quick else 350
	var steps_cpu := 12 if _quick else 100
	for b in ["cpu", "rd"]:
		_job("G1", "lbfgsb_components", b, "")
	for b in ["cpu", "rd"]:
		_job("G2", "lbfgsb_problems", b, "")
	for b in ["cpu", "rd"]:
		_job("G3", "inverse_min", b, "")
	_job("G4", "sphere_forward", "rd", "steps=%d" % steps_rd)
	_job("G4", "sphere_forward", "cpu", "steps=%d" % steps_cpu)
	_job("G4", "sphere_forward", "rd", "steps=%d mu=0.3" % steps_rd)
	_job("G5", "sphere_backward", "rd", "steps=%d mus=%s,0.01,0.375146" % [steps_rd, MU0])
	_job("G5", "sphere_backward", "cpu", "steps=%d mus=%s,0.01,0.375146" % [steps_rd, MU0])
	_job("G5", "sphere_backward", "rd", "steps=%d mus=0.5397701859474182,0.539770,0.5397702" % steps_rd)
	for c in [0, 1]:
		for sc in ["panel", "plane"]:
			_job("G6", "sim_gradcheck", "cpu", "scene=%s colors=%d%s" % [sc, c, " steps=8" if _quick else ""])
	for b in ["cpu", "rd"]:
		_job("G7", "lbfgsb_replay", b, "")
	for n in LB_SIZES:
		if _quick and n > 1000:
			continue
		for b in ["cpu", "rd"]:
			_job("G9", "lbfgsb_bench", b, "n=%d iters=10" % n, "lb %d" % n)
	var sizes := PackedStringArray()
	for n in BENCH_SIZES:
		if not (_quick and n > 24):
			sizes.append(str(n))
	for fps in BENCH_FPS:
		for b in ["cpu", "rd"]:
			_job("G9", "bench_drape", b, "sizes=%s steps=%d" % [",".join(sizes), 6 if _quick else 20], "fps %d" % fps, fps)
	# G10: the fitted skirt at drape scale 10 (5 cpu steps are ~6 s each run).
	_job("G10", "mesh_bisect", "rd", "scale=10")
	_job("G10", "mesh_parity", "rd", "scale=10 caps=1 steps=5 tol=1e-4")
	_job("G10", "mesh_parity", "rd", "scale=10 body=1 steps=5 tol=1e-4")
	# G7 live: the sphere demo on the API's rd session.
	if _on("G7"):
		# [name, spec, max iterations, steps]: the native run as upstream
		# (350 steps, DiffCloth's parameters); the recompute modes over the
		# same 350 steps, and over 60 steps, which end before the
		# self-collision onset (step 69) where the loss is smooth in mu.
		var st := steps_rd
		var mi := 8 if not _quick else 2
		_opt_plan.append(["native", "params=mu steps=%d mode=native" % st, 0, st])
		_opt_plan.append(["unrolled", "params=mu steps=%d mode=unrolled delta=1e-10" % st, mi, st])
		_opt_plan.append(["step", "params=mu steps=%d mode=step delta=1e-10" % st, mi, st])
		_opt_plan.append(["unrolled60", "params=mu steps=60 mode=unrolled delta=1e-12 epsilon=1e-12 epsilon_rel=1e-12", mi, 60])
		_opt_plan.append(["step60", "params=mu steps=60 mode=step delta=1e-12 epsilon=1e-12 epsilon_rel=1e-12", mi, 60])

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_usec() - _t_start > int(WALL_S * 1e6):
		_fail("the %d s wall clock ran out in phase '%s'%s" % [int(WALL_S), _phase,
				(" at job %s %s %s" % [_cur[1], _cur[2], _cur[3]]) if _cur != null else ""])
		_finish()
		return true
	if _phase == "data":
		_data()
	elif _phase == "api":
		_api()
	elif _phase == "wrappers":
		_wrappers()
	elif _phase == "jobs":
		_jobs()
	elif _phase == "optimize":
		_optimize()
	elif _phase == "checks":
		_checks()
		_finish()
	return _done

# --- the oracle into the guest ----------------------------------------------------

func _data() -> void:
	var n := 0
	_sb.vmcall("drape_job_data", "clear", "")
	for f in _files(ORACLE + "components"):
		_sb.vmcall("drape_job_data", f.get_basename(), _text(ORACLE + "components/" + f))
		n += 1
	for f in _files(ORACLE + "problems"):
		_sb.vmcall("drape_job_data", "prob_" + f.get_basename(), _text(ORACLE + "problems/" + f))
		n += 1
	for f in _files(ORACLE + "traces"):
		_sb.vmcall("drape_job_data", "trace_" + f.get_basename(), _text(ORACLE + "traces/" + f))
		n += 1
	var r := ""
	for c in ["k_tri", "k_bend_density"]:
		r = str(_sb.vmcall("drape_job_data", "invmin_" + c, _text(ORACLE + "inverse_min/case_%s.txt" % c)))
		n += 1
	_say("oracle handed to the guest: %d files (20 components, 5 problems, 20 traces, 2 inverse_min): %s" % [n, r])
	if n != 47 or not r.ends_with("47 keys"):
		_fail("expected 47 oracle files")
	# G10's scene: the fitted skirt, its pins and capsules, and the body.
	for kf in [["mesh_obj", "fitted_skirt.obj"], ["mesh_pins", "fitted_skirt_pins.txt"],
			["mesh_capsules", "fitted_skirt_capsules.txt"], ["body_obj", "body.obj"]]:
		r = str(_sb.vmcall("drape_job_data", kf[0], _text(SKIRT + kf[1])))
	_say("G10 scene handed to the guest: " + r)
	if not r.ends_with("51 keys"):
		_fail("expected the 4 G10 scene files after the oracle")
	_phase = "api"

# --- G4's API end to end on auto -----------------------------------------------------

func _api() -> void:
	if not _on("G4"):
		_phase = "wrappers"
		return
	if _api_step == 0:
		_say(str(_sb.vmcall("drape_open", "cpu")))
		var r := str(_sb.vmcall("drape_scene_sphere_demo"))
		_say(r.split("\n")[0])
		var faces: PackedInt32Array = _sb.vmcall("drape_faces")
		_check(faces == _native_faces, "G4 faces: the sphere scene's %d triangles equal native iter0/0.obj's %d, in order" % [
				faces.size() / 3, _native_faces.size() / 3])
		var f0: PackedFloat32Array = _sb.vmcall("drape_frame", 0)
		_check(_maxdiff(f0, _native[0]) <= 1e-6, "G4 frame 0 (rest) against native: max|dx| = %s" % _g(_maxdiff(f0, _native[0])))
		_auto_line = str(_sb.vmcall("drape_open", "auto"))
		_say(_auto_line)
		r = str(_sb.vmcall("drape_scene_sphere_demo"))
		_say("api auto: " + r.split("\n")[0])
		_say(str(_sb.vmcall("drape_config", "mu", MU0.to_float())).substr(0, 60))
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
			"G4 api: drape_open(auto) picked rd for 625 vertices, 3 steps in %d ticks, positions finite=%s, frame 1 max|dx| = %s: %s" % [
				_api_ticks, fin, _g(d1), t])
	_phase = "wrappers"

# --- main.gd's no-argument wrappers (rule 8), ticked by main.gd's own _process ---------

func _wrappers() -> void:
	if not (_on("G4") or _on("G3")):
		_phase = "jobs"
		return
	if _main == null:
		_main = load("res://main.gd").new()
		root.add_child(_main)
		_say("wrappers: " + str(_main.lbfgsb_load_oracle()))
		_say("wrappers: " + str(_main.drape_job("inverse_min", "cpu", "cases=0")))
		return
	_main_frames += 1
	var st: String = _main.drape_job_result()
	if st.begins_with("RUNNING") and _main_frames < 20000:
		return
	var names: String = _main.drape_job_names()
	var opt: String = _main.drape_optimize_result()
	_check(st.begins_with("PASS inverse_min cpu") and names.contains("inverse_min") and names.contains("lbfgsb_replay") and opt.length() > 0,
			"G3 main.gd wrappers: lbfgsb_load_oracle + drape_job(inverse_min, cpu, cases=0) -> '%s' after %d frames; drape_job_names lists inverse_min and lbfgsb_replay" % [
				st.split("\n")[0].substr(0, 120), _main_frames])
	_main.queue_free()
	_main = null
	_phase = "jobs"

# --- jobs ----------------------------------------------------------------------------

func _jobs() -> void:
	if _cur == null:
		if _queue.is_empty():
			_phase = "optimize"
			return
		_cur = _queue.pop_front()
		Engine.max_fps = _cur[5]
		var r := str(_sb.vmcall("drape_job_start", _cur[1], _cur[2], _cur[3]))
		if not r.begins_with("STARTED"):
			_check(false, "%s %s %s %s would not start: %s" % [_cur[0], _cur[1], _cur[2], _cur[3], r])
			_cur = null
			return
		_job_t0 = Time.get_ticks_usec()
		_job_ticks = 0
		return
	var r := str(_sb.vmcall("drape_job_tick", Time.get_ticks_usec()))
	_job_ticks += 1
	if r.begins_with("RUNNING"):
		return
	var wall := (Time.get_ticks_usec() - _job_t0) / 1000.0
	Engine.max_fps = 0
	var key := "%s %s %s%s" % [_cur[1], _cur[2], _cur[3], (" " + _cur[4]) if _cur[1] == "bench_drape" else ""]
	_results[key] = r
	var lines := r.split("\n")
	if _cur[4] != "" and _cur[1] != "bench_drape":
		var re := RegEx.new()
		re.compile("iterations=(\\d+)")
		var m := re.search(r)
		var it := int(m.get_string(1)) if m != null else 0
		if not _cost.has(_cur[4]):
			_cost[_cur[4]] = {}
		_cost[_cur[4]][_cur[2]] = [wall, it, _job_ticks]
		_check(r.begins_with("PASS"), "%s %-24s ticks=%5d wall_ms=%9.1f  %s" % [_cur[0], key, _job_ticks, wall, lines[0]])
		_cur = null
		return
	_say("job %-50s ticks=%6d wall_ms=%9.1f  %s" % [key, _job_ticks, wall, lines[0]])
	for i in range(1, lines.size()):
		_say("    " + lines[i])
	# sim_gradcheck multi-colour and sphere_forward/backward verdicts are
	# judged in the checks; the rest are their own verdicts.
	if _cur[1] in ["lbfgsb_components", "lbfgsb_problems", "inverse_min", "lbfgsb_replay", "mesh_parity", "mesh_bisect"]:
		_check(r.begins_with("PASS"), "%s %s %s: %s" % [_cur[0], _cur[1], _cur[2], lines[0]])
	elif not r.begins_with("PASS"):
		_check(false, "%s %s did not finish cleanly: %s" % [_cur[0], key, lines[0]])
	if _cur[1] == "sphere_forward":
		var fr := {}
		for k in NATIVE_FRAMES:
			var a: PackedFloat32Array = _sb.vmcall("drape_job_frame", k)
			if a.size() > 0:
				fr[k] = a
		_frames[key] = fr
	_cur = null

# --- G7: optimize on the sphere demo through the API (rd session) -------------------------

func _optimize() -> void:
	if _opt_cur == null:
		if _opt_plan.is_empty():
			_phase = "checks"
			return
		_opt_cur = _opt_plan.pop_front()
		_opt_step = 0
	if _opt_step == 0:
		_say("--- G7 optimize %s: %s" % [_opt_cur[0], _opt_cur[1]])
		_say(str(_sb.vmcall("drape_open", "rd")))
		_say(str(_sb.vmcall("drape_scene_sphere_demo")).split("\n")[0])
		_sb.vmcall("drape_config", "mu", 0.3)
		_say(str(_sb.vmcall("drape_queue_forward", _opt_cur[3])))
		_opt_step = 1
		_opt_ticks = 0
		return
	var t := str(_sb.vmcall("drape_tick", Time.get_ticks_usec()))
	_opt_ticks += 1
	if t.begins_with("RUNNING") and _opt_ticks < 2000000:
		return
	if _opt_step == 1:
		_say("target forward (mu 0.3): " + t)
		_say(str(_sb.vmcall("drape_set_target", "trajectory", PackedInt32Array(), PackedFloat32Array(), -1)))
		_sb.vmcall("drape_config", "mu", MU0.to_float())
		var q := str(_sb.vmcall("drape_queue_optimize", _opt_cur[1], PackedFloat32Array([MU0.to_float()]),
				PackedFloat32Array([0.01]), PackedFloat32Array([0.95]), _opt_cur[2]))
		_say(q)
		_opt_ticks = 0
		_opt_step = 2
		_opt_t0 = Time.get_ticks_usec()
		if not q.begins_with("QUEUED"):
			_check(false, "G7 drape_queue_optimize refused: " + q)
			_opt_cur = null
		return
	var res := str(_sb.vmcall("drape_optimize_result"))
	for line in res.split("\n"):
		_say("    " + line)
	_say("    host wall_ms=%.1f ticks=%d" % [(Time.get_ticks_usec() - _opt_t0) / 1000.0, _opt_ticks])
	_opt_results[_opt_cur[0]] = res
	_opt_cur = null

var _opt_t0 := 0

# --- checks ---------------------------------------------------------------------------

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

func _count(s: String, key: String) -> int:
	var re := RegEx.new()
	re.compile("\\b" + key + "=(\\d+)")
	var m := re.search(s)
	return int(m.get_string(1)) if m != null else -1

func _result_of(prefix: String) -> String:
	for k in _results.keys():
		if k == prefix or k.begins_with(prefix + " "):
			return _results[k]
	return ""

func _checks() -> void:
	_say("--- checks")
	if _on("G4"):
		_check_g4()
	if _on("G5"):
		_check_g5()
	if _on("G6"):
		_check_g6()
	if _on("G7"):
		_check_g7()
	_check_g8()
	if _on("G9"):
		_check_g9()

func _check_g4() -> void:
	var steps_rd := 60 if _quick else 350
	var rd: Dictionary = _frames.get("sphere_forward rd steps=%d" % steps_rd, {})
	var cpu := {}
	var ctl: Dictionary = _frames.get("sphere_forward rd steps=%d mu=0.3" % steps_rd, {})
	for k in _frames.keys():
		if k.begins_with("sphere_forward cpu"):
			cpu = _frames[k]
	for which in [["rd", rd, "sphere_forward rd steps=%d" % steps_rd], ["cpu", cpu, "sphere_forward cpu"]]:
		var fr: Dictionary = which[1]
		var parts := []
		var ok := fr.has(1) and fr.has(10) and _result_of(which[2]).begins_with("PASS")
		for k in NATIVE_FRAMES:
			if fr.has(k):
				var d := _maxdiff(fr[k], _native[k])
				parts.append("%d: %s" % [k, _g(d)])
				if k == 1:
					ok = ok and d <= 1e-5
				if k == 10:
					ok = ok and d <= 1e-3
		_check(ok, "G4 %s frames at mu 0.539770 against native iter0, every step finite (max|dx|; frame 1 <= 1e-5, frame 10 <= 1e-3): %s" % [
				which[0], ", ".join(PackedStringArray(parts))])
	if rd.size() > 0 and cpu.size() > 0:
		var parts := []
		for k in NATIVE_FRAMES:
			if rd.has(k) and cpu.has(k):
				parts.append("%d: %s" % [k, _g(_maxdiff(rd[k], cpu[k]))])
		_say("info G4 cpu vs rd frames (max|dx|): " + ", ".join(PackedStringArray(parts)))
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

func _check_g5() -> void:
	var bw := ""
	var bwc := ""
	var bwp := ""
	for k in _results.keys():
		if k.begins_with("sphere_backward rd") and k.contains("0.375146"):
			bw = _results[k]
		elif k.begins_with("sphere_backward rd"):
			bwp = _results[k]
		if k.begins_with("sphere_backward cpu"):
			bwc = _results[k]
	var re := RegEx.new()
	re.compile("RESULT mu=(\\S+) mu_exact=(\\S+) loss=(\\S+) dL/dmu=(\\S+)")
	var cpu_g := {}
	for m in re.search_all(bwc):
		cpu_g[m.get_string(1)] = [m.get_string(3).to_float(), m.get_string(4).to_float()]
	var n := 0
	var g0 := NAN
	var l0 := NAN
	for m in re.search_all(bw):
		var mu: String = m.get_string(1)
		if not NATIVE_DLDMU.has(mu):
			continue
		var g := m.get_string(4).to_float()
		var nat: float = NATIVE_DLDMU[mu]
		var rel := absf(g - nat) / absf(nat)
		var loss := m.get_string(3).to_float()
		var natl: float = NATIVE_LOSS[mu]
		if mu == "0.539770":
			g0 = g
			l0 = loss
		var spread := ""
		if cpu_g.has(mu):
			var gc: float = cpu_g[mu][1]
			spread = "; the cpu backend (same port, float order): dL/dmu %s (rel to rd %s), loss %s" % [_g(gc),
					_g(absf(g - gc) / maxf(absf(g), 1e-30)), _g(cpu_g[mu][0])]
		var line := "native dL/dmu at mu %s (%s): ours (rd) %s, backwardLog %s, rel %s (limit 5e-2); loss ours %s native %s (rel %s)%s" % [
				mu, m.get_string(2), String.num(g, 7), String.num(nat, 7), _g(rel), String.num(loss, 9), String.num(natl, 9),
				_g(absf(loss - natl) / natl), spread]
		if mu in G5_GATED:
			n += 1
			_check(rel <= 5e-2, "G5 " + line)
			if mu == "0.010000":
				var printed := "%.3f" % loss
				_check(printed == "1.652", "G5 loss at mu 0.01 at backwardLog's printed precision (%%.3f): ours %s (%s), backwardLog 1.652 (1.65198194)" % [printed, String.num(loss, 9)])
		else:
			_say("info G5 " + line)
	if n != 2:
		_check(false, "G5: parsed %d of 2 gated sphere_backward results" % n)
	# The flat control on the problem itself: mu moved by less than 2e-7
	# (float32 rounding of native's mu, the printed 0.539770, 0.5397702).
	var parts := PackedStringArray()
	var gmin := g0
	var gmax := g0
	var lmin := l0
	var lmax := l0
	for m in re.search_all(bwp):
		var g := m.get_string(4).to_float()
		var loss := m.get_string(3).to_float()
		parts.append("mu %s: dL/dmu %s loss %s" % [m.get_string(2), String.num(g, 7), String.num(loss, 7)])
		gmin = minf(gmin, g)
		gmax = maxf(gmax, g)
		lmin = minf(lmin, loss)
		lmax = maxf(lmax, loss)
	_say("info G5 flat control (the sphere demo's own conditioning; mu within 2e-7 of native's %s): %s; dL/dmu spans %s..%s (%s of its mean), the loss %s..%s (%s): backwardLog's 0.01153 sits %s below the span" % [
			MU0, "; ".join(parts), String.num(gmin, 6), String.num(gmax, 6), _g((gmax - gmin) / (0.5 * (gmax + gmin))),
			String.num(lmin, 6), String.num(lmax, 6), _g((lmax - lmin) / (0.5 * (lmax + lmin))), _g((gmin - 0.01153) / 0.01153)])

func _check_g6() -> void:
	# The single-colour unrolled verdicts are the jobs' own.
	for sc in ["panel", "plane"]:
		var r := _result_of("sim_gradcheck cpu scene=%s colors=0" % sc)
		_check(r.begins_with("PASS"), "G6 %s single colour: %s" % [sc, r.split("\n")[0]])
	# The table: max rel over mu/kTri/density per (scene, colours, mode).
	var re := RegEx.new()
	re.compile("GC scene=(\\S+) colors=(\\d) mode=(\\S+) param=(\\S+) analytic=\\S+ fd=\\S+ rel=(\\S+)")
	var tab := {}
	var worst := {}
	for k in _results.keys():
		if not k.begins_with("sim_gradcheck"):
			continue
		for m in re.search_all(_results[k]):
			var row := "%s colours=%s" % [m.get_string(1), "on" if m.get_string(2) == "1" else "off"]
			var mode := m.get_string(3)
			var rel := m.get_string(5).to_float()
			if not tab.has(row):
				tab[row] = {}
			tab[row][mode] = maxf(float(tab[row].get(mode, 0.0)), rel)
			worst[mode] = maxf(float(worst.get(mode, 0.0)), rel)
	_say("G6 table: max rel error against central FD over mu, kTri, density (20 steps)")
	_say("  %-24s %10s %10s %10s" % ["scene", "unrolled", "step", "native"])
	var rows := tab.keys()
	rows.sort()
	for row in rows:
		var d: Dictionary = tab[row]
		_say("  %-24s %10s %10s %10s" % [row, _g(d.get("unrolled", NAN)), _g(d.get("step", NAN)), _g(d.get("native", NAN))])
	var best := ""
	for mode in worst.keys():
		if best == "" or worst[mode] < worst[best]:
			best = mode
	_check(best == "unrolled", "G6 default mode: the smallest worst-case error over the table is %s (%s; step %s, native %s): the drape default is unrolled" % [
			best, _g(worst.get(best, NAN)), _g(worst.get("step", NAN)), _g(worst.get("native", NAN))])

func _opt_evals(res: String) -> Array:
	var out := []
	var re := RegEx.new()
	re.compile("(?m)^\\s*(x0|try) mu=(\\S+)")
	for m in re.search_all(res):
		out.append(m.get_string(2).to_float())
	return out

func _check_g7() -> void:
	var nat: String = _opt_results.get("native", "")
	var ev := _opt_evals(nat)
	var seq := PackedStringArray()
	for x in ev:
		seq.append("%.6f" % x)
	var head := nat.split("\n")[0]
	var shape: bool = ev.size() == 3 and absf(ev[0] - 0.539770) < 1e-6 and absf(ev[1] - 0.01) < 1e-7 and head.begins_with("DONE") and head.contains("iterations=1")
	var x2: float = ev[2] if ev.size() > 2 else NAN
	var printed: bool = ev.size() > 2 and "%.6f" % x2 == "0.375146"
	_check(shape and absf(x2 - 0.375146) <= 1e-3,
			"G7 live native-mode optimize (rd, 350 steps, our dL/dmu): evaluations %s (native 0.539770 -> 0.010000 -> 0.375146); 1 iteration, delta stop: %s; third evaluation within 1e-3 of native (%s), at printed precision: %s" % [
				" -> ".join(seq), shape, _g(absf(x2 - 0.375146)), "yes" if printed else "no (the G5 spread)"])
	_say("    " + head)
	# The recompute modes (information: the task reports their final mu).
	for mode in ["unrolled", "step", "unrolled60", "step60"]:
		var r: String = _opt_results.get(mode, "")
		var h := r.split("\n")[0]
		var fin := ""
		var re := RegEx.new()
		re.compile("(?m)^(?:iter \\d+ .*?|\\s*(?:x0|try) )mu=(\\S+)")
		for m in re.search_all(r):
			fin = m.get_string(1)
		var re2 := RegEx.new()
		re2.compile(" mu=(\\S+)")
		var m2 := re2.search(h)
		if m2 != null:
			fin = m2.get_string(1)
		var mu := fin.to_float() if fin != "" else NAN
		var e := _opt_evals(r)
		var steps := "60" if mode.ends_with("60") else "350"
		_say("info G7 %s-mode optimize (recompute; rd, %s steps, from mu0): last mu %s against the ground truth 0.3 (|err| %s), %d evaluations: %s" % [
				mode.trim_suffix("60"), steps, "%.6f" % mu, _g(absf(mu - 0.3)), e.size(), h.substr(0, 170)])

func _check_g8() -> void:
	var r4 := str(_sb.vmcall("rd_rule4"))
	var same := _count(r4, "same_frame_syncs")
	var syncs := _count(r4, "syncs")
	_check(same == 0 and syncs > 0, "G8 rd_rule4 after every job: %s (want same_frame_syncs=0 with syncs>0)" % r4)
	var probe := str(_sb.vmcall("rd_rule4_probe"))
	var after := _count(str(_sb.vmcall("rd_rule4")), "same_frame_syncs")
	_check(probe.begins_with("PASS") and after == same + 1, "G8 rd_rule4_probe (positive control): %s; counter now %d" % [probe, after])
	var elf := ProjectSettings.globalize_path("res://drape.elf")
	var out := []
	var rc := OS.execute("llvm-nm", ["-C", elf], out, true)
	var text: String = out[0] if out.size() > 0 else ""
	var lines := text.split("\n", false)
	var eigen := 0
	var lbfgspp := 0
	for l in lines:
		if l.contains("Eigen"):
			eigen += 1
		if l.to_lower().contains("lbfgspp"):
			lbfgspp += 1
	_check(rc == 0 and lines.size() > 1000 and eigen == 0 and lbfgspp == 0,
			"G8 llvm-nm -C project/drape.elf: %d symbols, %d matching Eigen, %d matching LBFGSpp (rc %d)" % [lines.size(), eigen, lbfgspp, rc])

func _check_g9() -> void:
	_say("G9 L-BFGS-B cost (box QP, 10 iterations, host wall of one job / its iterations)")
	_say("  %8s %12s %12s %8s" % ["n", "cpu ms/it", "rd ms/it", "rd/cpu"])
	var lb_cross := -1
	var lb_ok := true
	for n in LB_SIZES:
		var c: Dictionary = _cost.get("lb %d" % n, {})
		if not (c.has("cpu") and c.has("rd")):
			continue
		var cm: float = c["cpu"][0] / maxf(c["cpu"][1], 1)
		var rm: float = c["rd"][0] / maxf(c["rd"][1], 1)
		lb_ok = lb_ok and c["cpu"][1] > 0 and c["rd"][1] > 0
		_say("  %8d %12.3f %12.3f %8.2f" % [n, cm, rm, rm / maxf(cm, 1e-9)])
		if rm < cm and lb_cross < 0:
			lb_cross = n
	_check(lb_ok, "G9 L-BFGS-B bench: every size ran on both backends; rd first cheaper at n = %s (the optimize and lbfgsb jobs' auto stays cpu)" % (
			str(lb_cross) if lb_cross > 0 else "none up to %d" % LB_SIZES[-1]))
	# The drape step, at each frame rate.
	var re := RegEx.new()
	re.compile("bench_drape (cpu|rd)\\s+(\\d+)x\\d+\\s+nv=\\s*(\\d+) .*?ms/step=(\\S+)")
	var re2 := RegEx.new()
	re2.compile("rd from (\\d+) vertices")
	var m2 := re2.search(_auto_line if _auto_line != "" else str(_sb.vmcall("drape_open", "auto")))
	var thr := int(m2.get_string(1)) if m2 != null else -1
	for fps in BENCH_FPS:
		var ms := {}
		for k in _results.keys():
			if not (k.begins_with("bench_drape") and k.ends_with(" fps %d" % fps)):
				continue
			for m in re.search_all(_results[k]):
				var nv := int(m.get_string(3))
				if not ms.has(nv):
					ms[nv] = {}
				ms[nv][m.get_string(1)] = m.get_string(4).to_float()
		_say("G9 drape step cost at %s (sphere-demo scene at n x n, contact + self-collision, host ms per step over 20 frame-driven steps)" % (
				"uncapped frames" if fps == 0 else "%d fps" % fps))
		_say("  %8s %10s %10s %8s" % ["verts", "cpu", "rd", "rd/cpu"])
		var nvs := ms.keys()
		nvs.sort()
		var last_cpu := 0
		var first_rd := -1
		for nv in nvs:
			var d: Dictionary = ms[nv]
			_say("  %8d %10s %10s %8s" % [nv, _g(d.get("cpu", NAN)), _g(d.get("rd", NAN)),
					_g(float(d.get("rd", NAN)) / float(d.get("cpu", NAN)))])
			if d.has("cpu") and d.has("rd"):
				if d["rd"] < d["cpu"]:
					if first_rd < 0:
						first_rd = nv
				elif first_rd < 0:
					last_cpu = nv
		var line := "crossover at %s: cpu cheaper up to %d vertices, rd from %s" % [
				"uncapped frames" if fps == 0 else "%d fps" % fps, last_cpu, str(first_rd) if first_rd > 0 else "none measured"]
		if fps == AUTO_FPS:
			_check(first_rd > 0 and thr > last_cpu and thr <= first_rd,
					"G9 drape_open(auto) threshold %d vertices sits at the measured %s" % [thr, line])
		else:
			_say("info G9 " + line)
