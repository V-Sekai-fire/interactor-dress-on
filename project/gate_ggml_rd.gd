# Gate 3 (G3.ops): ggml's test-backend-ops in the guest (ggml_test.elf),
# ggml-rd on the GPU against the in-guest ggml-cpu, plus the ggml-rd probes.
#
#   godot --path project --script gate_ggml_rd.gd --rendering-driver vulkan --xr-mode off
#   godot --headless --path project --script gate_ggml_rd.gd      (the no-device control)
#
# Frame-driven: the host owns a local RenderingDevice, attaches it to the
# guest, and advances each job with InferHost.pump_frame() once per frame
# (WAIT_GPU after every submit, so every sync lands a frame after its
# submit: AGENTS.md rule 4). Runs, in order:
#   probe chain        256 in-place ADDs on one tensor, one graph: exact, 255 barriers
#   probe chain        the same under GGML_RD_BARRIER_ALL=1
#   probe independent  64 ADD/MULs into separate outputs: exact, 0 barriers
#   probe independent  the same under GGML_RD_BARRIER_ALL=1: 63 barriers
#   probe files        READ + UPLOAD of a host file into an RD buffer, x + x
#   probe alias rw     x += 1 in place x1000 across barriers, read-write sources: exact
#   probe alias ro     the same with the read-only-source control kernel: must lose counts
#   probe perf         GPU time per op on the census's hottest data-movement shapes
#                      (timestamps; numbers, PASS when every case was timed)
#   probe perf         the same under GGML_RD_BARRIER_ALL=1 (serialised latency)
#   probe mm_perf      MUL_MAT timing on the census's hottest shapes and the
#                      4096x1536x1024 benchmark, each checked against a double sum
#   ops main           test-backend-ops -o <OPS> -b RD0: 0 FAIL, every case OK or not supported,
#                      and no case REQUIRED names (the census's type rows) "not supported"
#   ops barrier_all    the same under GGML_RD_BARRIER_ALL=1: 0 FAIL
#   ops fault          -o ADD with GGML_RD_FAULT=1 (a source offset +1): the
#                      control, it must FAIL
#   ops fault move     the same control on the data-movement ops that read the
#                      source it moves (DUP, CONT, GET_ROWS, CONCAT, REPEAT; a
#                      CPY's src1 is its destination): every case must FAIL
# and then the rule-4 counter (syncs in their submit's frame) must be 0.
# Headless (no RenderingDevice): one run, -o ADD -b RD0, which must print
# "no RD device" and test no RD0 case: the flat control that separates
# "the GPU path is not there" from "the kernels are wrong".
#
# Results: gates/3-ggml-rd/ops/results.txt (results-headless.txt headless),
# each run's full output in run-<name>.log beside it; the last line is
# RESULT: PASS or RESULT: FAIL. Quits on a wall clock whatever it is doing.
#
# An op family runs the same gate on its own ops, into its own folder,
# with user arguments after `++` (or `--`), e.g.
#   ... --script gate_ggml_rd.gd --rendering-driver vulkan --xr-mode off ++ \
#       --ops=IM2COL,CONV_3D --fault-ops=IM2COL,CONV_3D --out=ops-k7 --probe=conv_perf:all
#   --ops=NORM,RMS_NORM   the ops of "ops main" and "ops barrier_all" (OPS)
#   --fault=NORM          the ops of the fault control (FAULT_OPS; also
#                         spelled --fault-ops=)
#   --out=<folder>        results in gates/3-ggml-rd/<folder>/ (e.g. ops-k7,
#                         ops/k1k5); a path starting gates/ is from the checkout
#   --probe=rows_perf[:arg]  one more probe after the ops runs (repeatable);
#                            it must print RESULT: PASS
#   runs=ops_main,probe_mm_perf  only those runs (the verdicts of the runs left
#                         out then FAIL, so a partial run never reads RESULT: PASS)
extends SceneTree

const InferHost := preload("res://infer_host.gd")
# The ops under test, as test-backend-ops -o takes them. An op family adds
# its ops here (the lead merges this line); ADD stays the fault control.
# GGML_GATE_OPS in the environment replaces the list for one run.
const OPS := "ADD,MUL,CPY,DUP,CONT,GET_ROWS,CONCAT,REPEAT,MUL_MAT,FLASH_ATTN_EXT"
const FAULT_OPS := "ADD"
# Cases that must be OK, never "not supported": the census's required type
# rows, as regexes over a case's test-backend-ops parameters, by op.
const REQUIRED := {
	"MUL_MAT": ["^type_a=(f32|f16|bf16),type_b=f32,", "^type_a=f16,type_b=f16,"],
}
const OUT_DIR := "res://../gates/3-ggml-rd/ops/"
const WALL_S := 3600.0
# The data-movement ops whose fault control is meaningful (GGML_RD_FAULT moves
# src1's offset when there is one, else src0's; a CPY never reads its src1).
const FAULT_MOVE_OPS := "DUP,CONT,GET_ROWS,CONCAT,REPEAT"
# Not the i32 cases: test_get_rows fills an i32 source the way it fills the
# row indices (r * be1 * be2 values in [0, m), the rest of the tensor 0), so
# a shifted index mostly picks another all-zero row and the fault is
# invisible (2 of 4 i32 GET_ROWS cases passed under it, 2026-09-23).
const FAULT_MOVE_PARAMS := "^(?!type=i32,)"
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
var _ops := OPS
var _fault_ops := FAULT_OPS
var _out_dir := OUT_DIR
var _extra_probes: Array = []

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

func _parse_user_args() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--ops="):
			_ops = a.trim_prefix("--ops=")
		elif a.begins_with("--fault-ops="):
			_fault_ops = a.trim_prefix("--fault-ops=")
		elif a.begins_with("--fault="):
			_fault_ops = a.trim_prefix("--fault=")
		elif a.begins_with("--probe="):
			var pa := a.trim_prefix("--probe=").split(":", true, 1)
			var parg: String = pa[1] if pa.size() > 1 else ""
			_extra_probes.append(["probe_" + pa[0] + ("_" + parg if parg != "" else ""), "probe", pa[0], parg, ""])
		elif a.begins_with("--out="):
			var o := a.trim_prefix("--out=").trim_suffix("/")
			_out_dir = ("res://../" + o + "/") if o.begins_with("gates/") else ("res://../gates/3-ggml-rd/" + o + "/")

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	_parse_user_args()
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_rd = RenderingServer.create_local_rendering_device()
	if OS.get_environment("GGML_GATE_OPS") != "":
		_ops = OS.get_environment("GGML_GATE_OPS")
	_headless = _rd == null
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_out_dir))
	var name := "results-headless.txt" if _headless else "results.txt"
	_out = FileAccess.open(ProjectSettings.globalize_path(_out_dir + name), FileAccess.WRITE)
	_say("# Gate 3 G3.ops, %s, Godot %s, %s, %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, OS.get_processor_name(),
			"headless: no RenderingDevice" if _headless else RenderingServer.get_video_adapter_name()])
	_sb = ClassDB.instantiate("Sandbox")
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
		_runs = [["ops_no_device", "ops", "-o ADD -b RD0", ""]]
	else:
		var probe_file := _write_probe_file()
		_runs = [
			["probe_chain", "probe", "chain", "256", ""],
			["probe_chain_barrier_all", "probe", "chain", "256", "GGML_RD_BARRIER_ALL=1"],
			["probe_independent", "probe", "independent", "64", ""],
			["probe_independent_barrier_all", "probe", "independent", "64", "GGML_RD_BARRIER_ALL=1"],
			["probe_files", "probe", "files", probe_file, ""],
			["probe_alias_rw", "probe", "alias", "rw", ""],
			["probe_alias_ro_control", "probe", "alias", "ro", ""],
			["probe_perf", "probe", "perf", "move", ""],
			["probe_perf_barrier_all", "probe", "perf", "move", "GGML_RD_BARRIER_ALL=1"],
			["probe_mm_perf", "probe", "mm_perf", "all", ""],
			["ops_main", "ops", "-o %s -b RD0" % _ops, ""],
			["ops_barrier_all", "ops", "-o %s -b RD0" % _ops, "GGML_RD_BARRIER_ALL=1"],
			["ops_fault", "ops", "-o %s -b RD0" % _fault_ops, "GGML_RD_FAULT=1"],
			["ops_fault_move", "ops", "-o %s -p %s -b RD0" % [FAULT_MOVE_OPS, FAULT_MOVE_PARAMS], "GGML_RD_FAULT=1"],
		]
		_runs.append_array(_extra_probes)
		for ua in OS.get_cmdline_user_args():
			if ua.begins_with("runs="):
				var keep := ua.trim_prefix("runs=").split(",")
				_runs = _runs.filter(func(r): return keep.has(r[0]))
				_say("runs selected: %s" % str(keep))

# 4096 f32s with a spread of values (and -0, a tiny normal, the largest
# finite: x + x overflows to inf on both sides), for the READ/UPLOAD probe.
func _write_probe_file() -> String:
	var path := ProjectSettings.globalize_path(_out_dir + "upload_probe.f32")
	var f := FileAccess.open(path, FileAccess.WRITE)
	for i in 4096:
		var v := 0.5 * i - 7.25
		if i == 1:
			v = -0.0
		elif i == 2:
			v = 1.0e-30
		elif i == 3:
			v = 3.4028234e38
		f.store_float(v)
	f.close()
	return path

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
		_host.reset()
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
	var lf := FileAccess.open(ProjectSettings.globalize_path(_out_dir + "run-%s.log" % name), FileAccess.WRITE)
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
		_say("   required cases not supported: %d" % res.required_unsupported.size())
		for l in res.required_unsupported.slice(0, 5):
			_say("   required, not supported: %s" % l)
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
	var required_unsupported := []
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
				for pat in REQUIRED.get(op, []):
					var rq := RegEx.new()
					rq.compile(pat)
					if rq.search(m.get_string(2)) != null:
						required_unsupported.append(s)
		elif s.ends_with("tests passed"):
			passed_line = s
		elif s.begins_with("Backend RD0:"):
			backend_line = s
	return {"ok": ok, "fail": fail, "unsupported": unsupported, "per_op": per_op, "fail_lines": fail_lines,
			"required_unsupported": required_unsupported,
			"passed_line": passed_line, "backend_line": backend_line}

func _probe_pass(name: String) -> bool:
	return _results.has(name) and _results[name].state == "done" and _results[name].text.contains("RESULT: PASS")

func _checks() -> void:
	_say("== verdicts")
	if _headless:
		var r = _results.get("ops_no_device", {})
		var t: String = r.get("text", "")
		_verdict(r.get("state", "") == "done" and t.contains("no RD device") and t.contains("Testing 1 devices")
				and not t.contains("Backend RD0:") and r.get("ok", -1) == 0,
				"headless control: 'no RD device', 1 device (CPU), no RD0 case run")
		return
	for n in ["probe_chain", "probe_chain_barrier_all", "probe_independent", "probe_independent_barrier_all", "probe_files"]:
		_verdict(_probe_pass(n), "%s" % n)
	_verdict(_probe_pass("probe_alias_rw"), "probe_alias_rw: x at b1 and b4 of one buffer, read-write sources, 1000 spans: exact")
	for x in _extra_probes:
		_verdict(_probe_pass(x[0]), "%s %s" % [x[0], x[3]])
	_verdict(_probe_pass("probe_alias_ro_control"),
			"probe_alias_ro_control: the same recording with read-only sources loses increments (the Gate 0F hazard, still there)")
	_verdict(_probe_pass("probe_perf") and _probe_pass("probe_perf_barrier_all"),
			"probe_perf: every hot data-movement shape timed on the GPU, with and without a barrier per dispatch")
	_verdict(_probe_pass("probe_mm_perf"), "probe_mm_perf: every shape timed and within nmse 1e-8 of a double sum")
	for n in ["ops_main", "ops_barrier_all"]:
		var r = _results.get(n, {})
		_verdict(r.get("state", "") == "done" and r.get("fail", -1) == 0 and r.get("ok", 0) > 0
				and str(r.get("backend_line", "")).ends_with("OK"),
				"%s: test-backend-ops -o %s -b RD0: OK=%d FAIL=%d not_supported=%d (%s)" % [n, _ops, r.get("ok", 0),
				r.get("fail", -1), r.get("unsupported", 0), r.get("backend_line", "")])
		_verdict(r.get("state", "") == "done" and r.get("required_unsupported", [null]).is_empty(),
				"%s: every required case (%s) is OK, none not supported (%d are)" % [n, str(REQUIRED),
				r.get("required_unsupported", [null]).size()])
	var m = _results.get("ops_main", {})
	var b = _results.get("ops_barrier_all", {})
	_verdict(m.get("ok", -1) == b.get("ok", -2) and m.get("unsupported", -1) == b.get("unsupported", -2),
			"barrier elision and barrier-after-every-dispatch pass the same cases")
	var f = _results.get("ops_fault", {})
	_verdict(f.get("state", "") == "done" and f.get("fail", 0) > 0 and f.get("ok", -1) == 0,
			"control: GGML_RD_FAULT=1 (a source read one element off) fails every %s case: FAIL=%d OK=%d (%s)" % [
			_fault_ops, f.get("fail", 0), f.get("ok", 0), f.get("backend_line", "")])
	var fm = _results.get("ops_fault_move", {})
	_verdict(fm.get("state", "") == "done" and fm.get("fail", 0) > 0 and fm.get("ok", -1) == 0,
			"control: GGML_RD_FAULT=1 fails every %s case but i32: FAIL=%d OK=%d (%s)" % [FAULT_MOVE_OPS,
			fm.get("fail", 0), fm.get("ok", 0), fm.get("backend_line", "")])
	var stats := str(_sb.vmcall("ggml_rd_stats"))
	var re := RegEx.new()
	re.compile("rule4_same_frame_syncs=(\\d+)")
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
