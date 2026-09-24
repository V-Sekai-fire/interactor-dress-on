# Stage 2 gate: the AVBD solver in the guest, forward and backward, CPU and
# GPU backends from the same Lean-emitted kernels, frame-driven.
#
#   godot --path project --script gate_avbd.gd --rendering-driver vulkan --xr-mode off > ../gates/2-avbd/run.log 2>&1
#
# AGENTS.md rule 4: every job is a stage queue in drape.elf (guest/jobs.h)
# whose tick ends at each GPU submit; this script ticks the current job once
# per _process, so every rd readback lands a frame after its submit. The guard
# rd_rule4() counts syncs that landed in their submit's frame: it must be 0
# after all the jobs, and rd_rule4_probe() (a submit + sync in one call, the
# positive control) must raise it.
#
# The jobs run on cpu, then rd; the two cpu-vs-rd jobs last. Then the checks
# that need more than one job: gradcheck's analytic values against the native
# Vulkan run on the same SPIR-V (native_gradcheck.log, rel 1e-4), the backward
# smoke against native_solver.log (rel 5e-3; native prints %.3g), cpu and rd
# self-collision pair sets identical, and the rule-4 counter.
#
# vsync is off and max_fps 0 so a tick costs a frame, not a refresh. The
# script quits on a 600 s wall clock whatever phase it is in (--quit-after is
# not relied on). Results stream to gates/2-avbd/results.txt; the last line is
# RESULT: PASS or RESULT: FAIL.
extends SceneTree

const OUT_DIR := "res://../gates/2-avbd/"
const WALL_S := 600.0
const IDLE_FRAMES := 240
const BACKEND_JOBS := ["fixture", "backward_smoke", "gradcheck", "stategrad", "gradcheck_duals",
		"self_collision", "bench_fwd", "bench_bwd"]
const BOTH_JOBS := ["two_vertex_bwd", "two_vertex_bwd_nopad"]

var _sb = null
var _out: FileAccess
var _t_start := 0
var _rc := 0
var _phase := "idle"
var _done := false
var _idle_n := 0
var _idle_t0 := 0
var _queue := []
var _cur = null
var _job_t0 := 0
var _job_ticks := 0
var _job_vm_us := 0
var _results := {}

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _fail(why: String) -> void:
	_rc = 1
	_say("FAIL: " + why)

func _finish() -> void:
	# Free the guest's device and its permanent RID slots before the Sandbox
	# (otherwise the device leaks into ObjectDB at exit). Every RID the jobs
	# made must be released by now: permanent_slots=0.
	if _sb != null:
		var c := str(_sb.vmcall("rd_close"))
		var cok := c.begins_with("CLOSED device=") and _count(c, "permanent_slots") == 0
		_say("%s rd_close: %s" % ["PASS" if cok else "FAIL", c])
		if not cok:
			_rc = 1
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	if _out != null:
		_out.close()
		_out = null
	if _sb != null:
		_sb.free()
		_sb = null
	_done = true
	quit(_rc)

func _initialize() -> void:
	_t_start = Time.get_ticks_usec()
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT_DIR + "results.txt"), FileAccess.WRITE)
	_say("# Stage 2 gate, %s, Godot %s, %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, RenderingServer.get_video_adapter_name()])
	_sb = ClassDB.instantiate("Sandbox")
	if _sb != null: _sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	if _sb == null:
		_fail("Sandbox class not registered")
		_finish()
		return
	_sb.program = load("res://drape.elf")
	# Every uniform set costs ~10 scoped references for the call.
	_sb.references_max = 65536
	for b in ["cpu", "rd"]:
		for j in BACKEND_JOBS:
			_queue.append([j, b])
	for j in BOTH_JOBS:
		_queue.append([j, "both"])

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_usec() - _t_start > int(WALL_S * 1e6):
		_fail("the %d s wall clock ran out in phase '%s'%s" % [int(WALL_S), _phase,
				(" at job %s %s" % [_cur[0], _cur[1]]) if _cur != null else ""])
		_finish()
		return true
	if _phase == "idle":
		_idle()
	elif _phase == "jobs":
		_jobs()
	elif _phase == "checks":
		_checks()
		_finish()
	return _done

# The bare frame period with nothing to do: the floor under every
# frame-driven ms/substep below.
func _idle() -> void:
	if _idle_n == 0:
		_idle_t0 = Time.get_ticks_usec()
	_idle_n += 1
	if _idle_n > IDLE_FRAMES:
		var us := Time.get_ticks_usec() - _idle_t0
		_say("idle frame period: %.3f ms over %d frames (vsync off, max_fps 0)" % [us / 1000.0 / IDLE_FRAMES, IDLE_FRAMES])
		_phase = "jobs"

func _jobs() -> void:
	if _cur == null:
		if _queue.is_empty():
			_phase = "checks"
			return
		_cur = _queue.pop_front()
		var r := str(_sb.vmcall("avbd_job_start", _cur[0], _cur[1]))
		if not r.begins_with("STARTED"):
			_fail("%s %s would not start: %s" % [_cur[0], _cur[1], r])
			_results["%s %s" % _cur] = r
			_cur = null
			return
		_job_t0 = Time.get_ticks_usec()
		_job_ticks = 0
		_job_vm_us = 0
		return
	var t := Time.get_ticks_usec()
	var r := str(_sb.vmcall("avbd_job_tick", t))
	_job_vm_us += Time.get_ticks_usec() - t
	_job_ticks += 1
	if r.begins_with("RUNNING"):
		return
	var key := "%s %s" % _cur
	_results[key] = r
	var lines := r.split("\n")
	_say("job %-26s ticks=%5d wall_ms=%9.1f vmcall_ms=%9.1f  %s" % [key, _job_ticks,
			(Time.get_ticks_usec() - _job_t0) / 1000.0, _job_vm_us / 1000.0, lines[0]])
	for i in range(1, lines.size()):
		_say("    " + lines[i])
	if not r.begins_with("PASS"):
		_rc = 1
	_cur = null

# --- checks across jobs ---------------------------------------------------------

func _read_text(name: String) -> String:
	var f := FileAccess.open(ProjectSettings.globalize_path(OUT_DIR + name), FileAccess.READ)
	if f == null:
		return ""
	var s := f.get_as_text()
	f.close()
	return s

# %.3g for GDScript, which formats neither %g nor %e.
func _g(x: float) -> String:
	if x == 0.0 or is_nan(x) or is_inf(x):
		return str(x)
	var e := int(floor(log(absf(x)) / log(10.0)))
	if e >= -3 and e < 5:
		return String.num(x, maxi(0, 2 - e))
	return "%.2fe%d" % [x / pow(10.0, e), e]

func _rel(a: float, n: float) -> float:
	return absf(a - n) / maxf(absf(n), 1e-12)

func _checks() -> void:
	_say("--- checks")
	_check_gradcheck_native()
	_check_smoke_native()
	_check_stategrad_native()
	_check_self_collision()
	_check_rule4()

# gradcheck's analytic values, each backend, against the native Vulkan run on
# the same SPIR-V: rel 1e-4.
func _check_gradcheck_native() -> void:
	var native := _read_text("native_gradcheck.log")
	var re := RegEx.new()
	re.compile("dL/d (\\S+)\\s+analytic=\\s*(\\S+)")
	var nv := {}
	for m in re.search_all(native):
		nv[m.get_string(1)] = m.get_string(2).to_float()
	if nv.size() != 5:
		_fail("native_gradcheck.log: found %d analytic values, want 5" % nv.size())
		return
	for b in ["cpu", "rd"]:
		var ours: String = _results.get("gradcheck " + b, "")
		var worst := 0.0
		var n := 0
		var parts := []
		for m in re.search_all(ours):
			var k: String = m.get_string(1)
			if not nv.has(k):
				continue
			var r: float = _rel(m.get_string(2).to_float(), nv[k])
			worst = maxf(worst, r)
			n += 1
			parts.append("%s %s" % [k, _g(r)])
		var ok := n == 5 and worst <= 1e-4
		_say("%s gradcheck %s vs native analytic: %d/5 compared, worst rel %s (limit 1e-4): %s" % [
				"PASS" if ok else "FAIL", b, n, _g(worst), ", ".join(PackedStringArray(parts))])
		if not ok:
			_rc = 1

func _native_smoke(text: String) -> Dictionary:
	var out := {}
	var num := "([-+0-9.eE]+|-?nan|-?inf)"
	var pats := [
		["v0=\\(%s, %s, %s\\)\\s+v1=\\(%s, %s, %s\\)" % [num, num, num, num, num, num],
				["dx_v0x", "dx_v0y", "dx_v0z", "dx_v1x", "dx_v1y", "dx_v1z"]],
		["\\(s0\\) : %s\\s+\\S+ \\(s0\\)\\s+: %s" % [num, num], ["dL_s0", "dk_s0"]],
		["anchor : \\(%s, %s, %s\\)\\s+\\S+k_attach: %s" % [num, num, num, num],
				["danchor_x", "danchor_y", "danchor_z", "dk_attach"]],
		["k_tri\\s+: %s\\s+\\S+_tri\\s+: \\(%s, %s, %s\\)" % [num, num, num, num],
				["dk_tri", "dl0_tri_x", "dl0_tri_y", "dl0_tri_z"]],
		["n_bend : %s\\s+\\S+k_bend\\s+: %s" % [num, num], ["dn_bend", "dk_bend"]],
	]
	for p in pats:
		var re := RegEx.new()
		re.compile(p[0])
		var m := re.search(text)
		if m == null:
			continue
		for i in range(p[1].size()):
			out[p[1][i]] = m.get_string(i + 1).to_float()
	return out

# The backward smoke against native_solver.log, which prints %.3g: rel 5e-3,
# with a 1e-6 floor for the zeros.
func _check_smoke_native() -> void:
	var nv := _native_smoke(_read_text("native_solver.log"))
	if nv.size() != 18:
		_fail("native_solver.log: parsed %d of 18 smoke values" % nv.size())
		return
	var re := RegEx.new()
	re.compile("smoke (\\w+)=(\\S+)")
	for b in ["cpu", "rd"]:
		var ours: String = _results.get("backward_smoke " + b, "")
		var n := 0
		var bad := []
		for m in re.search_all(ours):
			var k: String = m.get_string(1)
			if not nv.has(k):
				continue
			var a: float = m.get_string(2).to_float()
			n += 1
			if absf(a - nv[k]) > 5e-3 * maxf(absf(a), absf(nv[k])) + 1e-6:
				bad.append("%s ours=%.9g native=%s" % [k, a, str(nv[k])])
		var ok := n == 18 and bad.is_empty()
		_say("%s backward_smoke %s vs native (%%.3g, rel 5e-3): %d/18 compared, %d disagree%s" % [
				"PASS" if ok else "FAIL", b, n, bad.size(), (": " + "; ".join(PackedStringArray(bad))) if not bad.is_empty() else ""])
		if not ok:
			_rc = 1

# Informational: stategrad's analytic columns against the native run's
# (%.6g there), and the finite differences likewise.
func _check_stategrad_native() -> void:
	var row := RegEx.new()
	row.compile("(?m)^\\s+(\\d+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(ok|MISMATCH)\\s*$")
	var native := []
	for m in row.search_all(_read_text("native_stategrad.log")):
		native.append([m.get_string(2).to_float(), m.get_string(3).to_float()])
	if native.size() != 24:
		_say("note: native_stategrad.log: parsed %d of 24 rows" % native.size())
		return
	for b in ["cpu", "rd"]:
		var ours := []
		for m in row.search_all(_results.get("stategrad " + b, "")):
			ours.append([m.get_string(2).to_float(), m.get_string(3).to_float()])
		if ours.size() != 24:
			_say("note: stategrad %s: parsed %d of 24 rows" % [b, ours.size()])
			continue
		var wa := 0.0
		var wf := 0.0
		for i in range(24):
			wa = maxf(wa, absf(ours[i][0] - native[i][0]) / maxf(1.0, absf(native[i][0])))
			wf = maxf(wf, absf(ours[i][1] - native[i][1]) / maxf(1.0, absf(native[i][1])))
		_say("info stategrad %s vs native (24 rows, rel vs max(1,|native|)): analytic worst %s, fd worst %s" % [b, _g(wa), _g(wf)])

func _check_self_collision() -> void:
	var re := RegEx.new()
	re.compile("pairs=(\\d+) expected=(\\d+) hash=(\\w+)")
	var hs := {}
	for b in ["cpu", "rd"]:
		var a := []
		for m in re.search_all(_results.get("self_collision " + b, "")):
			a.append(m.get_string(3))
		hs[b] = a
	var ok: bool = hs["cpu"].size() == 4 and hs["cpu"] == hs["rd"]
	_say("%s self_collision cpu and rd pair sets identical (%d cases, hashes %s)" % [
			"PASS" if ok else "FAIL", hs["cpu"].size(), ", ".join(PackedStringArray(hs["rd"]))])
	if not ok:
		_rc = 1

func _count(s: String, key: String) -> int:
	var re := RegEx.new()
	re.compile("\\b" + key + "=(\\d+)")
	var m := re.search(s)
	return int(m.get_string(1)) if m != null else -1

func _check_rule4() -> void:
	var r4 := str(_sb.vmcall("rd_rule4"))
	var same := _count(r4, "same_frame_syncs")
	var syncs := _count(r4, "syncs")
	var ok := same == 0 and syncs > 0
	_say("%s rd_rule4 after every job: %s (want same_frame_syncs=0 with syncs>0)" % ["PASS" if ok else "FAIL", r4])
	if not ok:
		_rc = 1
	var probe := str(_sb.vmcall("rd_rule4_probe"))
	var after := _count(str(_sb.vmcall("rd_rule4")), "same_frame_syncs")
	var pok := probe.begins_with("PASS") and after == same + 1
	_say("%s rd_rule4_probe (positive control, submit+sync in one call): %s; counter now %d" % [
			"PASS" if pok else "FAIL", probe, after])
	if not pok:
		_rc = 1
