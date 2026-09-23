# Gate 3, census shapes: ggml-rd's ops at the shapes the models dispatch
# (guest/ggml_test/census.cpp), which test-backend-ops' cases do not reach.
#
#   godot --path project --script census_ggml_rd.gd --rendering-driver vulkan --xr-mode off -- --out=k1k5
#
# Two jobs on the pump, frame-driven like gate_ggml_rd.gd (WAIT_GPU after
# every submit, the sync a frame later: AGENTS.md rule 4):
#   census all   every census row on ggml-rd and on the in-guest ggml-cpu,
#                test-backend-ops' comparison (NMSE <= 1e-7, infinities
#                matched): every row OK
#   census all   the control, under GGML_RD_FAULT=1 (every dispatch reads a
#   (fault)      source one element off): every row must FAIL, DIAG_MASK_INF
#                included (test-backend-ops cannot fail it on values: its
#                NMSE sums inf - inf = NaN, and NaN > max is false)
#   perf <i>     one job per row: R in-place applications per graph, one
#                warm-up and 5 timed graphs. The host times each graph from
#                the vmcall that submitted it to the end of the vmcall that
#                synced it (InferHost.wait_us; the guest clock is not a
#                clock). Row 0 is one tiny dispatch: the floor (frame gap,
#                submit, sync), subtracted before dividing by R.
# Results: gates/3-ggml-rd/ops/<out>/census.txt, the last line RESULT.
# Quits on a wall clock whatever it is doing.
extends SceneTree

const InferHost := preload("res://infer_host.gd")
const OUT_DIR := "res://../gates/3-ggml-rd/ops/"
const WALL_S := 1500.0
const TOTAL_MB := 24576
const ITERS := 5

var _sb = null
var _rd: RenderingDevice = null
var _host = null
var _out: FileAccess
var _out_dir := OUT_DIR
var _t0 := 0
var _rc := 0
var _done := false
var _runs: Array = []
var _cur = null
var _rows := -1
var _floor_us := -1.0
var _perf_lines: Array = []

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _verdict(ok: bool, what: String) -> void:
	if not ok:
		_rc = 1
	_say("%s %s" % ["PASS" if ok else "FAIL", what])

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--out="):
			_out_dir = OUT_DIR + a.substr(6) + "/"
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_rd = RenderingServer.create_local_rendering_device()
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_out_dir))
	_out = FileAccess.open(ProjectSettings.globalize_path(_out_dir + "census.txt"), FileAccess.WRITE)
	_say("# Gate 3 census shapes, %s, Godot %s, %s, %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, OS.get_processor_name(),
			RenderingServer.get_video_adapter_name() if _rd != null else "no RenderingDevice"])
	if _rd == null:
		_verdict(false, "no RenderingDevice (run with --rendering-driver vulkan, not --headless)")
		_finish()
		return
	_sb = ClassDB.instantiate("Sandbox")
	_sb.memory_max = 2048
	_sb.program = load("res://ggml_test.elf")
	_sb.references_max = 65536
	_sb.execution_timeout = 1000000
	_say("attach: %s" % str(_sb.vmcall("ggml_attach", _rd, TOTAL_MB)))
	_host = InferHost.new(_sb, _rd, "ggml_pump")
	_runs = [["census", "all", "", "census"], ["census", "all", "GGML_RD_FAULT=1", "census_fault"]]

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_msec() - _t0 > int(WALL_S * 1000):
		_verdict(false, "the %d s wall clock ran out" % int(WALL_S))
		_finish()
		return true
	if _cur == null:
		if _runs.is_empty():
			_verdict(_perf_lines.size() == _rows and _rows > 0, "perf: %d of %d rows timed" % [_perf_lines.size(), _rows])
			_finish()
			return true
		_cur = _runs.pop_front()
		# rule 10: the census rows run ggml-cpu, capped at 5 minutes per vmcall.
		_host.reset(InferHost.ggml_runs_cpu("ggml_probe_start", _cur[0]))
		var r := str(_sb.vmcall("ggml_probe_start", _cur[0], _cur[1], _cur[2]))
		if not r.begins_with("STARTED"):
			_verdict(false, "%s %s would not start: %s" % [_cur[0], _cur[1], r])
			_cur = null
		return false
	var st: String = _host.pump_frame()
	if st == "running":
		return false
	_end_run(st)
	_cur = null
	return false

func _median(a: Array) -> float:
	var s := a.duplicate()
	s.sort()
	if s.is_empty():
		return -1.0
	return float(s[s.size() / 2])

func _end_run(st: String) -> void:
	var text := str(_sb.vmcall("ggml_output"))
	var lf := FileAccess.open(ProjectSettings.globalize_path(_out_dir + "run-%s-%s.log" % [_cur[3], _cur[1]]), FileAccess.WRITE)
	lf.store_string(text)
	lf.close()
	if st == "error":
		_verdict(false, "%s %s: error %s" % [_cur[0], _cur[1], _host.text])
		return
	if _cur[3] == "census_fault":
		var failed := 0
		var seen := 0
		for l in text.split("
"):
			if l.begins_with("CENSUS"):
				seen += 1
				if l.ends_with(" FAIL"):
					failed += 1
				_say("   fault: " + l)
		_verdict(seen == _rows and failed == _rows, "control: GGML_RD_FAULT=1 fails every census row: %d of %d FAIL" % [failed, seen])
		return
	if _cur[0] == "census":
		_rows = 0
		for l in text.split("\n"):
			if l.begins_with("CENSUS"):
				_say(l)
			elif l.begins_with("ROW "):
				_rows += 1
		_verdict(text.contains("RESULT: PASS"), "census all: every row within test-backend-ops' NMSE on ggml-rd vs ggml-cpu (%d rows, %s)" % [
				_rows, _host.summary()])
		for i in _rows:
			_runs.append(["perf", str(i), "", "perf"])
		_say("== perf (host-timed; per-dispatch = (median - floor) / R, R - 1 barriers inside)")
		return
	var perf := ""
	for l in text.split("\n"):
		if l.begins_with("PERF"):
			perf = l
	var timed: Array = _host.wait_us.slice(max(_host.wait_us.size() - ITERS, 0))
	var med := _median(timed)
	var reps := 1
	var bytes := 0
	var re := RegEx.new()
	re.compile("reps=(\\d+).* bytes_per_dispatch=(\\d+)")
	var m := re.search(perf)
	if m != null:
		reps = int(m.get_string(1))
		bytes = int(m.get_string(2))
	var ok := text.contains("RESULT: PASS") and timed.size() == ITERS
	if _cur[1] == "0":
		_floor_us = med
		_say("%s | graph us: median %.0f min %.0f (the floor) %s" % [perf, med, timed.min() if not timed.is_empty() else -1, "" if ok else "FAIL"])
	else:
		var per := (med - _floor_us) / float(reps)
		var gbs: float = bytes / maxf(per, 0.001) / 1000.0
		_say("%s | graph us: median %.0f min %.0f | per dispatch %.1f us, %.0f GB/s %s" % [perf, med,
				timed.min() if not timed.is_empty() else -1, per, gbs, "" if ok else "FAIL"])
	if ok:
		_perf_lines.append(perf)
	else:
		_rc = 1

func _finish() -> void:
	if _sb != null:
		_say("rd: %s" % str(_sb.vmcall("ggml_rd_stats")))
		_say("close: %s" % str(_sb.vmcall("ggml_rd_close")))
	_say("wall_s=%.1f" % ((Time.get_ticks_msec() - _t0) / 1000.0))
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	if _out != null:
		_out.close()
		_out = null
	if _sb != null:
		_sb.free()
		_sb = null
	if _rd != null:
		_rd.free()
		_rd = null
	_done = true
	quit(_rc)
