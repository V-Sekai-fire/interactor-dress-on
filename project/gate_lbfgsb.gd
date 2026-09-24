# Gate 5 G1 and G2 (drape.elf, Task B): the in-guest L-BFGS-B driver
# (guest/drape/lbfgsb.cpp) over the Lean-emitted kernels, on the CPU path
# (slangc cpp) and the GPU path (SPIR-V over rdc::Device), against LBFGSpp.
#
#   godot --path project --script gate_lbfgsb.gd --rendering-driver vulkan --xr-mode off > ../gates/5-drape/lbfgsb/run.log 2>&1
#
# Checks:
#  - G1 lbfgsb_components on cpu and rd: the 20 LBFGSpp component fixtures
#    (M, M*v, theta, pg_inf, the Cauchy point, c, the sets, drt0, drt, g.drt,
#    step_max), rel <= 1e-4 (drt, dg, step_max <= 5e-4); the control (theta
#    x1.25) must fail all 20;
#  - G2 lbfgsb_problems on cpu and rd: the 20 oracle traces (Rosenbrock n2,
#    n10, n100, LBFGSpp's rosenbrock-box n25, box QP n1000; m 10 and 5;
#    DiffCloth's parameters and LBFGSpp's defaults): f within 1e-6 (abs+rel),
#    x within 1e-3, the final active sets identical, iterations within
#    max(2, 25%); per-iteration cost on both backends (the vec crossover);
#  - drape_queue_optimize end to end on the sphere demo (rd session): a mu
#    0.3 target over 60 steps, 3 iterations from mu 0.539770 (information:
#    the mu sequence), which must finish DONE with finite values;
#  - main.gd's wrappers: lbfgsb_load_oracle + drape_job(lbfgsb_problems);
#  - rd_rule4 == 0 after everything; rd_close frees every slot.
# Results stream to gates/5-drape/lbfgsb/results.txt; the last line is RESULT.
extends SceneTree

const OUT_DIR := "res://../gates/5-drape/lbfgsb/"
const ORACLE := "res://../gates/5-drape/oracle/"
const WALL_S := 1500.0

var _sb = null
var _out: FileAccess
var _t_start := 0
var _rc := 0
var _phase := "data"
var _done := false
var _queue := []
var _cur = null
var _job_t0 := 0
var _job_ticks := 0
var _results := {}
var _opt_step := 0
var _opt_ticks := 0
var _main: Node = null
var _main_frames := 0
var _cost := {}
var _cost_said := false

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
		_check(c.begins_with("CLOSED device=") and c.ends_with("permanent_slots=0"), "rd_close: " + c)
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

func _initialize() -> void:
	_t_start = Time.get_ticks_usec()
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(OUT_DIR))
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT_DIR + "results.txt"), FileAccess.WRITE)
	_say("# Gate 5 L-BFGS-B (Task B), %s, Godot %s, %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, RenderingServer.get_video_adapter_name()])
	_sb = ClassDB.instantiate("Sandbox")
	if _sb != null: _sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	if _sb != null: _sb.memory_max = 512 # the Windows addon's default, which the recorded runs used; Linux's is 32 MiB
	if _sb == null:
		_fail("Sandbox class not registered")
		_finish()
		return
	_sb.references_max = 65536
	_sb.program = load("res://drape.elf")
	for backend in ["cpu", "rd"]:
		_queue.append(["lbfgsb_components", backend, ""])
	for backend in ["cpu", "rd"]:
		_queue.append(["lbfgsb_problems", backend, ""])
	# The cost table: one job per trace and backend, timed on the host (a cpu
	# job runs whole inside one or two ticks, where the guest sees one clock).
	for f in _files(ORACLE + "traces"):
		for backend in ["cpu", "rd"]:
			_queue.append(["lbfgsb_problems", backend, "only=" + f.get_basename(), "cost"])

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_usec() - _t_start > int(WALL_S * 1e6):
		_fail("the %d s wall clock ran out in phase '%s'%s" % [int(WALL_S), _phase,
				(" at job %s %s %s" % _cur) if _cur != null else ""])
		_finish()
		return true
	if _phase == "data":
		_data()
	elif _phase == "jobs":
		_jobs()
	elif _phase == "optimize":
		_optimize()
	elif _phase == "wrappers":
		_wrappers()
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
	var r := ""
	for f in _files(ORACLE + "traces"):
		r = str(_sb.vmcall("drape_job_data", "trace_" + f.get_basename(), _text(ORACLE + "traces/" + f)))
		n += 1
	_check(n == 45 and r.ends_with("45 keys"), "oracle handed to the guest: %d files (20 components, 5 problems, 20 traces): %s" % [n, r])
	_phase = "jobs"

# --- G1 / G2 -----------------------------------------------------------------------

func _jobs() -> void:
	if _cur == null:
		if _queue.is_empty():
			_phase = "optimize"
			return
		_cur = _queue.pop_front()
		if _cur.size() > 3 and not _cost_said:
			_cost_said = true
			_say("--- cost: one lbfgsb_problems job per trace and backend, host-timed")
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
	var lines := r.split("\n")
	var wall := (Time.get_ticks_usec() - _job_t0) / 1000.0
	if _cur.size() > 3:
		var trace: String = _cur[2].substr(5)
		var re := RegEx.new()
		re.compile("iters (\\d+)/")
		var m := re.search(r)
		var it := int(m.get_string(1)) if m != null else 0
		if not _cost.has(trace):
			_cost[trace] = {}
		_cost[trace][_cur[1]] = [wall, _job_ticks, it]
		_say("cost %-34s %-3s ticks=%5d wall_ms=%8.1f iterations=%d  %s" % [trace, _cur[1], _job_ticks, wall, it, lines[0]])
		_cur = null
		return
	var key := "%s %s" % [_cur[0], _cur[1]]
	_results[key] = r
	_say("job %-26s ticks=%6d wall_ms=%9.1f  %s" % [key, _job_ticks, wall, lines[0]])
	for i in range(1, lines.size()):
		_say("    " + lines[i])
	_check(r.begins_with("PASS"), key + ": " + lines[0])
	_cur = null

# --- drape_queue_optimize on the sphere demo ----------------------------------------

func _optimize() -> void:
	if _opt_step == 0:
		_say("--- drape_queue_optimize (sphere demo, rd session, 60 steps)")
		_say(str(_sb.vmcall("drape_open", "auto")))
		_say(str(_sb.vmcall("drape_scene_sphere_demo")).split("\n")[0])
		_sb.vmcall("drape_config", "mu", 0.3)
		_say(str(_sb.vmcall("drape_queue_forward", 60)))
		_opt_step = 1
		return
	var t := str(_sb.vmcall("drape_tick", Time.get_ticks_usec()))
	_opt_ticks += 1
	if t.begins_with("RUNNING") and _opt_ticks < 200000:
		return
	if _opt_step == 1:
		_say("target forward: " + t)
		_say(str(_sb.vmcall("drape_set_target", "trajectory", PackedInt32Array(), PackedFloat32Array(), -1)))
		_sb.vmcall("drape_config", "mu", 0.539770)
		var q := str(_sb.vmcall("drape_queue_optimize", "params=mu steps=60 mode=native",
				PackedFloat32Array([0.539770]), PackedFloat32Array([0.01]), PackedFloat32Array([1.0]), 3))
		_say(q)
		_opt_ticks = 0
		_opt_step = 2
		if not q.begins_with("QUEUED"):
			_fail("drape_queue_optimize refused: " + q)
			_phase = "wrappers"
		return
	var res := str(_sb.vmcall("drape_optimize_result"))
	for line in res.split("\n"):
		_say("    " + line)
	var head := res.split("\n")[0]
	_check(t.begins_with("IDLE DONE optimize") and head.begins_with("DONE optimize") and not head.contains("nan"),
			"drape_queue_optimize(mu, 3 iterations, native, 60 steps, target mu 0.3) in %d ticks: %s" % [_opt_ticks, head])
	_phase = "wrappers"

# --- main.gd's no-argument wrappers (rule 8) --------------------------------------------

func _wrappers() -> void:
	if _main == null:
		_main = load("res://main.gd").new()
		root.add_child(_main)
		_say("wrappers: " + str(_main.lbfgsb_load_oracle()))
		_say("wrappers: " + str(_main.drape_job("lbfgsb_problems", "cpu", "only=rosen_n2_m5_tight")))
		return
	_main_frames += 1
	var st: String = _main.drape_job_result()
	if st.begins_with("RUNNING") and _main_frames < 20000:
		return
	var names: String = _main.drape_job_names()
	var opt: String = _main.drape_optimize_result()
	_check(st.begins_with("PASS G2 cpu: 1/1") and names.contains("lbfgsb_problems") and opt.length() > 0,
			"main.gd wrappers: lbfgsb_load_oracle + drape_job(lbfgsb_problems, cpu, only=rosen_n2_m5_tight) -> '%s' after %d frames; drape_optimize_result answers" % [
				st.split("\n")[0], _main_frames])
	_main.queue_free()
	_main = null
	_phase = "checks"

# --- checks ---------------------------------------------------------------------------

func _count(s: String, key: String) -> int:
	var re := RegEx.new()
	re.compile("\\b" + key + "=(\\d+)")
	var m := re.search(s)
	return int(m.get_string(1)) if m != null else -1

func _checks() -> void:
	_say("--- per-iteration cost, cpu vs rd (host wall time of a one-trace job / its iterations; the vec crossover)")
	var keys := _cost.keys()
	keys.sort()
	for k in keys:
		var c: Dictionary = _cost[k]
		if c.has("cpu") and c.has("rd") and c["cpu"][2] > 0 and c["rd"][2] > 0:
			_say("  %-34s cpu %8.3f ms/iter (%d ticks)   rd %8.3f ms/iter (%d ticks)   rd/cpu %.1f" % [k,
					c["cpu"][0] / c["cpu"][2], c["cpu"][1], c["rd"][0] / c["rd"][2], c["rd"][1],
					(c["rd"][0] / c["rd"][2]) / maxf(c["cpu"][0] / c["cpu"][2], 1e-6)])
	var r4 := str(_sb.vmcall("rd_rule4"))
	var same := _count(r4, "same_frame_syncs")
	var syncs := _count(r4, "syncs")
	_check(same == 0 and syncs > 0, "rd_rule4 after every job: %s (want same_frame_syncs=0 with syncs>0)" % r4)
