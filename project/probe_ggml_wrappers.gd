# Rule 8 smoke for Cut 3: instantiate project/main.gd as the MCP host does and
# drive ggml_test.elf through its no-argument wrappers only (ggml_attach, the
# probe and ops presets, ggml_pump, ggml_job_status, ggml_output,
# ggml_rd_stats, ggml_rd_close), the job pumped by main.gd's own _process.
# A second ggml_pump in the frame that started a job must not pump again
# (rule 4), and the rule-4 counter must end at 0. Quits on a 240 s wall clock.
#
#   godot --path project --script probe_ggml_wrappers.gd --rendering-driver vulkan --xr-mode off
#
# Writes gates/3-ggml-rd/ops/wrappers.txt; the last line is RESULT: PASS or FAIL.
extends SceneTree

var _main = null
var _steps := ["ggml_probe_chain", "ggml_probe_files", "ggml_probe_alias_rw", "ggml_ops_fault_short"]
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
	_out = FileAccess.open(ProjectSettings.globalize_path("res://../gates/3-ggml-rd/ops/wrappers.txt"), FileAccess.WRITE)
	_main = load("res://main.gd").new()
	_main.name = "Main"
	root.add_child(_main)
	_say("attach: " + str(_main.ggml_attach()))
	_say("attach again: " + str(_main.ggml_attach()))

func _process(_d: float) -> bool:
	if Time.get_ticks_msec() - _t0 > 240000:
		_say("FAIL wall clock")
		_rc = 1
		return _end()
	if _cur == "":
		if _steps.is_empty():
			return _end()
		_cur = _steps.pop_front()
		var r: String
		if _cur == "ggml_ops_fault_short":
			r = str(_main.ggml_ops_start("-o ADD -b RD0 -p ne=\\[10,5,4,3\\]", "GGML_RD_FAULT=1"))
		else:
			r = str(_main.call(_cur))
		_say("%s -> %s" % [_cur, r])
		# a second pump in the same frame must not run (rule 4)
		_say("  same-frame ggml_pump: " + str(_main.ggml_pump()))
		return false
	var st: String = _main.ggml_job_status()
	if st.begins_with("RUNNING"):
		return false
	_say("  status: " + st)
	var out: String = _main.ggml_output()
	for l in out.split("\n"):
		if l.begins_with("PROBE") or l.begins_with("RESULT") or l.contains("tests passed") or l.contains("Backend RD0"):
			_say("  " + l.strip_edges())
	_cur = ""
	return false

func _end() -> bool:
	var stats: String = _main.ggml_rd_stats()
	_say("stats: " + stats.substr(0, 400))
	_say("close: " + str(_main.ggml_rd_close()))
	_say("status after close: " + str(_main.ggml_job_status()))
	var re := RegEx.new()
	re.compile("rule4_same_frame_syncs=(\\d+)")
	var m := re.search(stats)
	if m == null or int(m.get_string(1)) != 0:
		_rc = 1
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	_out.close()
	quit(_rc)
	return true
