# Gate 3, K8: FLASH_ATTN_EXT GPU time at the census shapes (ggml_test.elf's
# fa_perf probe: D = 128, 12 heads, f32 Q/K/V, no mask).
#
#   godot --path project --script perf_ggml_fa.gd --rendering-driver vulkan --xr-mode off
#
# The guest clock is not a clock (AGENTS.md), so the host times it. The probe
# computes a one-node graph `reps` times; each compute is a submit, a
# WAIT_GPU yield, and the sync at the start of the next frame's pump, which
# then submits the next. The host stamps every WAIT_GPU return; after the
# first (pipeline creation and uploads), the period between two stamps is
# max(the node's GPU time, one frame) plus the packing, so
#   per-node ms <= the interval between two stamps
# reported as the median (and min and mean) interval after the first; the
# tiny shape (Lq = 16, Lk = 5) shows the frame floor under it. Other
# processes on the same GPU (other agents' Godot runs) only lengthen
# intervals, so the median and min are the fairer numbers. Every
# job must also PASS its sampled check (8 query rows x 12 heads against a
# double reference in the guest). The serial sibling (GGML_RD_SERIAL=1) is
# timed at one shape for comparison.
#
# Results: gates/3-ggml-rd/perf/fa.txt; the last line is RESULT: PASS or FAIL.
# Quits on a wall clock whatever it is doing.
extends SceneTree

const InferHost := preload("res://infer_host.gd")
const OUT := "res://../gates/3-ggml-rd/perf/fa.txt"
const WALL_S := 1500.0
const TOTAL_MB := 24576
# [lq, lk, reps, env]
const SHAPES := [
	[16, 5, 50, ""],
	[4096, 4096, 20, ""],
	[4096, 1029, 30, ""],
	[912, 1029, 50, ""],
	[912, 912, 50, ""],
	[4096, 5, 50, ""],
	[912, 1029, 30, "GGML_RD_SERIAL=1"],
	[4096, 4096, 10, "GGML_RD_SERIAL=1"],
]

var _sb = null
var _rd: RenderingDevice = null
var _host = null
var _out: FileAccess
var _t0 := 0
var _rc := 0
var _done := false
var _jobs: Array = []
var _cur = null
var _stamps: Array = []
var _summary: Array = []

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_rd = RenderingServer.create_local_rendering_device()
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(OUT).get_base_dir())
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT), FileAccess.WRITE)
	_say("# Gate 3 K8 FLASH_ATTN_EXT timing, %s, Godot %s, %s, %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, OS.get_processor_name(),
			RenderingServer.get_video_adapter_name() if _rd != null else "no RenderingDevice"])
	if _rd == null:
		_rc = 1
		_say("FAIL no RenderingDevice (run with --rendering-driver vulkan, not --headless)")
		_finish()
		return
	_sb = ClassDB.instantiate("Sandbox")
	_sb.memory_max = 2048
	_sb.program = load("res://ggml_test.elf")
	_sb.references_max = 65536
	_sb.execution_timeout = 1000000
	_say("attach: %s" % str(_sb.vmcall("ggml_attach", _rd, TOTAL_MB)))
	_host = InferHost.new(_sb, _rd, "ggml_pump")
	_jobs = SHAPES.duplicate()

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_msec() - _t0 > int(WALL_S * 1000):
		_rc = 1
		_say("FAIL the %d s wall clock ran out" % int(WALL_S))
		_finish()
		return true
	if _cur == null:
		if _jobs.is_empty():
			_say("== per FA node (upper bound: the period between WAIT_GPUs)")
			for l in _summary:
				_say(l)
			_finish()
			return true
		_cur = _jobs.pop_front()
		_host.reset()
		_stamps = []
		var r := str(_sb.vmcall("ggml_probe_start", "fa_perf", "%d,%d,%d" % [_cur[0], _cur[1], _cur[2]], _cur[3]))
		if not r.begins_with("STARTED"):
			_rc = 1
			_say("FAIL fa_perf would not start: %s" % r)
			_cur = null
		return false
	var st: String = _host.pump_frame()
	if st == "running":
		if _host.last_kind == InferHost.WAIT_GPU:
			_stamps.append(Time.get_ticks_usec())
		return false
	var text := str(_sb.vmcall("ggml_output"))
	var line := ""
	for l in text.split("\n"):
		if l.begins_with("PROBE fa_perf"):
			line = l
	var ok: bool = st == "done" and text.contains("RESULT: PASS") and _stamps.size() == _cur[2]
	if not ok:
		_rc = 1
	var n := _stamps.size()
	var ms := -1.0
	var med := -1.0
	var mn := -1.0
	if n >= 3:
		ms = (_stamps[n - 1] - _stamps[1]) / 1000.0 / float(n - 2)
		var d: Array = []
		for i in range(1, n - 1):
			d.append((_stamps[i + 1] - _stamps[i]) / 1000.0)
		d.sort()
		med = d[d.size() / 2]
		mn = d[0]
	var gflop: float = 4.0 * 128.0 * _cur[0] * _cur[1] * 12.0 * 1e-9
	var kind: String = _cur[3] if _cur[3] != "" else "tiled"
	_say("%s lq=%d lk=%d reps=%d %s: waits=%d period mean=%.3f median=%.3f min=%.3f ms (%s) | %s" % [
			"OK  " if ok else "FAIL", _cur[0], _cur[1], _cur[2], kind, n, ms, med, mn, _host.summary(), line])
	_summary.append("FLASH_ATTN_EXT D=128 H=12 Lq=%d Lk=%d f32 %s: median %.3f ms (min %.3f, mean %.3f) per node, %.3f GFLOP, %.2f TFLOP/s at the median" % [
			_cur[0], _cur[1], kind, med, mn, ms, gflop, gflop / max(med, 1e-6)])
	_cur = null
	return false

func _finish() -> void:
	if _sb != null:
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
