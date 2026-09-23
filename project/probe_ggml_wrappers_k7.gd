# Rule 8 smoke for family K7: main.gd's K7 presets (ggml_probe_conv_perf,
# ggml_ops_conv) called with no arguments, pumped by main.gd's own _process;
# rule 4 must end at 0. Quits on a 400 s wall clock.
#
#   godot --path project --script probe_ggml_wrappers_k7.gd --rendering-driver vulkan --xr-mode off
#
# Writes gates/3-ggml-rd/ops-k7/wrappers.txt; the last line is RESULT: PASS or FAIL.
extends SceneTree

var _main = null
var _steps := ["ggml_probe_conv_perf", "ggml_ops_conv"]
var _cur := ""
var _out: FileAccess
var _t0 := 0
var _rc := 0

func _say(s: String) -> void:
	print(s)
	_out.store_line(s)
	_out.flush()

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	_out = FileAccess.open(ProjectSettings.globalize_path("res://../gates/3-ggml-rd/ops-k7/wrappers.txt"), FileAccess.WRITE)
	_main = load("res://main.gd").new()
	_main.name = "Main"
	root.add_child(_main)
	_say("attach: " + str(_main.ggml_attach()))

func _process(_d: float) -> bool:
	if Time.get_ticks_msec() - _t0 > 400000:
		_say("FAIL wall clock")
		_rc = 1
		return _end()
	if _cur == "":
		if _steps.is_empty():
			return _end()
		_cur = _steps.pop_front()
		_say("%s -> %s" % [_cur, str(_main.call(_cur))])
		return false
	var st: String = _main.ggml_job_status()
	if st.begins_with("RUNNING"):
		return false
	_say("  status: " + st.substr(0, 200))
	var out: String = _main.ggml_output()
	var ok := false
	for l in out.split("\n"):
		if l.begins_with("PROBE") or l.begins_with("RESULT") or l.contains("tests passed") or l.contains("Backend RD0"):
			_say("  " + l.strip_edges().replace("\u001b[1;32m", "").replace("\u001b[0m", ""))
		if l.begins_with("RESULT: PASS") or (l.contains("Backend RD0:") and l.contains("OK")):
			ok = true
	if not ok:
		_rc = 1
	_cur = ""
	return false

func _end() -> bool:
	var stats: String = _main.ggml_rd_stats()
	var re := RegEx.new()
	re.compile("rule4_same_frame_syncs=(\\d+)")
	var m := re.search(stats)
	if m == null or int(m.get_string(1)) != 0:
		_rc = 1
	_say("rule4: " + (m.get_string(0) if m else "?"))
	_say("close: " + str(_main.ggml_rd_close()))
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	_out.close()
	quit(_rc)
	return true
