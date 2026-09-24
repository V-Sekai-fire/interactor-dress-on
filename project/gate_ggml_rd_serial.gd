# Gate 3, K8: the serial-sibling A/B. test-backend-ops -o FLASH_ATTN_EXT -b RD0
# in the guest (ggml_test.elf) with GGML_RD_SERIAL=1, so ggml-rd dispatches the
# _serial kernels (one thread per query row, no groupshared; the ones the host
# L2 test runs through their cpp emits) instead of the tiled ones on the GPU.
# Both must pass the same cases; gate_ggml_rd.gd runs the tiled ones.
#
#   godot --path project --script gate_ggml_rd_serial.gd --rendering-driver vulkan --xr-mode off
#
# Results: gates/3-ggml-rd/ops/fa-serial/results.txt (last line RESULT) and
# run-ops_serial.log beside it. Quits on a wall clock whatever it is doing.
extends SceneTree

const InferHost := preload("res://infer_host.gd")
# The ops whose packers have _serial siblings (GGML_RD_SERIAL=1 selects them).
const OPS := "FLASH_ATTN_EXT"
const OUT_DIR := "res://../gates/3-ggml-rd/ops/fa-serial/"
const WALL_S := 1800.0
# The device memory ggml-rd reports as total (free = total - allocated):
# Godot has no call for it, so the host states it (24 GiB here).
const TOTAL_MB := 24576

var _sb = null
var _rd: RenderingDevice = null
var _host = null
var _out: FileAccess
var _t0 := 0
var _rc := 0
var _done := false
var _runs: Array = []
var _cur = null
var _run_t0 := 0
var _headless := false
var _results := {}

func _clean(t: String) -> String:
	var root := ProjectSettings.globalize_path("res://").trim_suffix("/")
	root = root.get_base_dir() # the checkout, not project/
	return t.replace(root, "<checkout>").replace(root.replace("/", "\\"), "<checkout>")

func _say(line: String) -> void:
	line = _clean(line)
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _verdict(ok: bool, what: String) -> void:
	if not ok:
		_rc = 1
	_say("%s %s" % ["PASS" if ok else "FAIL", what])

func _strip_ansi(t: String) -> String:
	var re := RegEx.new()
	re.compile("\u001b\\[[0-9;]*m")
	return re.sub(t, "", true)

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_rd = RenderingServer.create_local_rendering_device()
	_headless = _rd == null
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(OUT_DIR))
	var name := "results-headless.txt" if _headless else "results.txt"
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT_DIR + name), FileAccess.WRITE)
	_say("# Gate 3 G3.ops, %s, Godot %s, %s, %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, OS.get_processor_name(),
			"headless: no RenderingDevice" if _headless else RenderingServer.get_video_adapter_name()])
	_sb = ClassDB.instantiate("Sandbox")
	if _sb != null: _sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	if _sb == null:
		_verdict(false, "the Sandbox class is not registered")
		_finish()
		return
	# memory_max first (it survives program=, Gate 0F): the reference
	# backend holds whole test tensors (up to 3 x 64 MiB) in the guest heap.
	_sb.memory_max = 2048
	_sb.program = load("res://ggml_test.elf")
	_sb.references_max = 65536
	# Host calls are charged against the budget (Gate 0F finding 6), and one
	# pump can run a large reference op on the in-guest CPU.
	_sb.execution_timeout = 1000000
	var a := str(_sb.vmcall("ggml_attach", _rd, TOTAL_MB))
	_say("attach: %s" % a)
	_host = InferHost.new(_sb, _rd, "ggml_pump")
	if _headless:
		_verdict(false, "no RenderingDevice (run with --rendering-driver vulkan, not --headless)")
		_finish()
		return
	_runs = [["ops_serial", "ops", "-o %s -b RD0" % OPS, "GGML_RD_SERIAL=1"]]

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_msec() - _t0 > int(WALL_S * 1000):
		_verdict(false, "the %d s wall clock ran out%s" % [int(WALL_S), (" in run %s" % _cur[0]) if _cur != null else ""])
		_finish()
		return true
	if _cur == null:
		if _runs.is_empty():
			_checks()
			_finish()
			return true
		_cur = _runs.pop_front()
		# rule 10: a job that runs ggml-cpu gets the 5-minute cap per vmcall.
		_host.reset(InferHost.ggml_runs_cpu("ggml_ops_start" if _cur[1] == "ops" else "ggml_probe_start",
				"" if _cur[1] == "ops" else _cur[2]))
		_run_t0 = Time.get_ticks_msec()
		var r: String
		if _cur[1] == "ops":
			r = str(_sb.vmcall("ggml_ops_start", _cur[2], _cur[3]))
		else:
			r = str(_sb.vmcall("ggml_probe_start", _cur[2], _cur[3], _cur[4]))
		if not r.begins_with("STARTED"):
			_verdict(false, "%s would not start: %s" % [_cur[0], r])
			_cur = null
		return false
	var st: String = _host.pump_frame()
	if st == "running":
		return false
	_end_run(st)
	_cur = null
	return false

func _end_run(st: String) -> void:
	var name: String = _cur[0]
	var ms := Time.get_ticks_msec() - _run_t0
	var text := _strip_ansi(str(_sb.vmcall("ggml_output")))
	var lf := FileAccess.open(ProjectSettings.globalize_path(OUT_DIR + "run-%s.log" % name), FileAccess.WRITE)
	lf.store_string(_clean(text))
	lf.close()
	var stats := str(_sb.vmcall("ggml_rd_stats"))
	_say("== %s: %s in %.1f s, %s" % [name, st, ms / 1000.0, _host.summary()])
	if st == "error":
		_say("   error: %s" % _host.text)
	var res := {"state": st, "ms": ms, "text": text, "stats": stats}
	if _cur[1] == "ops":
		res.merge(_parse_ops(text))
		_say("   cases: OK=%d FAIL=%d not_supported=%d | %s | %s" % [res.ok, res.fail, res.unsupported,
				res.passed_line, res.backend_line])
		for op in res.per_op:
			var c = res.per_op[op]
			_say("   %s: OK=%d FAIL=%d not_supported=%d" % [op, c.ok, c.fail, c.unsupported])
		for l in res.fail_lines.slice(0, 5):
			_say("   failed: %s" % l)
	else:
		for l in text.split("\n"):
			if l.begins_with("PROBE") or l.begins_with("RESULT"):
				_say("   " + l)
	_say("   rd: %s" % stats)
	_results[name] = res

func _parse_ops(text: String) -> Dictionary:
	var ok := 0
	var fail := 0
	var unsupported := 0
	var per_op := {}
	var fail_lines := []
	var passed_line := ""
	var backend_line := ""
	# A case is "OP(params): OK|FAIL|not supported [..]"; a failing case's
	# comparison message ("[ADD] ERR = ...") comes first on the same line,
	# since the case is printed after it is judged.
	var re := RegEx.new()
	re.compile("([A-Z][A-Z0-9_]*)\\((.*)\\): (OK|FAIL|not supported)")
	for raw in text.split("\n"):
		var s := raw.strip_edges()
		var m := re.search(s)
		if m != null:
			var op := m.get_string(1)
			if not per_op.has(op):
				per_op[op] = {"ok": 0, "fail": 0, "unsupported": 0}
			var st := m.get_string(3)
			if st == "OK":
				ok += 1
				per_op[op].ok += 1
			elif st == "FAIL":
				fail += 1
				per_op[op].fail += 1
				fail_lines.append(s)
			else:
				unsupported += 1
				per_op[op].unsupported += 1
		elif s.ends_with("tests passed"):
			passed_line = s
		elif s.begins_with("Backend RD0:"):
			backend_line = s
	return {"ok": ok, "fail": fail, "unsupported": unsupported, "per_op": per_op, "fail_lines": fail_lines,
			"passed_line": passed_line, "backend_line": backend_line}

func _probe_pass(name: String) -> bool:
	return _results.has(name) and _results[name].state == "done" and _results[name].text.contains("RESULT: PASS")

func _checks() -> void:
	_say("== verdicts")
	var r = _results.get("ops_serial", {})
	_verdict(r.get("state", "") == "done" and r.get("fail", -1) == 0 and r.get("ok", 0) > 0
			and str(r.get("backend_line", "")).ends_with("OK"),
			"ops_serial: test-backend-ops -o %s -b RD0 with GGML_RD_SERIAL=1: OK=%d FAIL=%d not_supported=%d (%s)" % [
			OPS, r.get("ok", 0), r.get("fail", -1), r.get("unsupported", 0), r.get("backend_line", "")])
	var stats := str(_sb.vmcall("ggml_rd_stats"))
	var re := RegEx.new()
	re.compile("rule4_same_frame_syncs=([0-9]+)")
	var mm := re.search(stats)
	_verdict(mm != null and int(mm.get_string(1)) == 0, "rule 4: no sync in its submit's frame (%s)" % (mm.get_string(0) if mm else "?"))

func _finish() -> void:
	if _sb != null and not _headless:
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
