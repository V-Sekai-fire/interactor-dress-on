# Rule 8 smoke for G3.graph and G3.cost: main.gd's presets (ggml_graph_qwen,
# ggml_cost_decode, ggml_cost_dit, ggml_graph_sconv, ggml_graph_dit) called
# with no arguments, pumped by main.gd's own _process; each must print
# RESULT: PASS, and rule 4 must end at 0. Quits on a 2400 s wall clock.
#
#   godot --path project --script probe_ggml_wrappers_graph.gd --rendering-driver vulkan --xr-mode off
#   ... ++ fast      only ggml_graph_qwen, ggml_cost_decode and ggml_cost_dit
#
# Writes gates/3-ggml-rd/graph/wrappers.txt; the last line is RESULT: PASS or FAIL.
extends SceneTree

var _main = null
var _steps := ["ggml_graph_qwen", "ggml_cost_decode", "ggml_cost_dit", "ggml_graph_sconv", "ggml_graph_dit"]
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
	if OS.get_cmdline_user_args().has("fast"):
		_steps = ["ggml_graph_qwen", "ggml_cost_decode", "ggml_cost_dit"]
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	_out = FileAccess.open(ProjectSettings.globalize_path("res://../gates/3-ggml-rd/graph/wrappers.txt"), FileAccess.WRITE)
	_main = load("res://main.gd").new()
	_main.name = "Main"
	root.add_child(_main)
	_say("attach: " + str(_main.ggml_attach()))

func _process(_d: float) -> bool:
	if Time.get_ticks_msec() - _t0 > 2400000:
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
		if l.contains("SUMMARY") or l.begins_with("RESULT"):
			_say("  " + l.strip_edges())
		if l.begins_with("RESULT: PASS"):
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
	_say("wall_s=%.1f" % ((Time.get_ticks_msec() - _t0) / 1000.0))
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	_out.close()
	quit(_rc)
	return true
