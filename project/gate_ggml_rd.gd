# Gate 3 (G3.ops): ggml's test-backend-ops in the guest (ggml_test.elf),
# ggml-rd on the GPU against the in-guest ggml-cpu, plus the ggml-rd probes.
#
#   godot --path project --script gate_ggml_rd.gd --rendering-driver vulkan --xr-mode off
#   godot --headless --path project --script gate_ggml_rd.gd      (the CPU fallback; -- fallback=off: the no-device control)
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
#   probe census       the census's hot rows (K1/K5's census.cpp) vs the in-guest ggml-cpu
#   ops main           test-backend-ops -o <OPS> -b RD0: 0 FAIL, every case OK or not supported;
#                      every REQUIRED pattern (the census's type rows) matches at least one OK
#                      case and no "not supported" one
#   ops barrier_all    the same under GGML_RD_BARRIER_ALL=1: 0 FAIL
#   ops fault          -o ADD with GGML_RD_FAULT=1 (a source offset +1): the
#                      control, it must FAIL
#   ops fault move     the same control on the data-movement ops that read the
#                      source it moves (DUP, CONT, GET_ROWS, CONCAT, REPEAT; a
#                      CPY's src1 is its destination): every case must FAIL
#   ops cpu cap control  AGENTS.md rule 10's control, always last: -o MUL_MAT
#                      with the per-vmcall ggml-cpu cap lowered to 8 units; a
#                      pump overruns it, and the run must end as an error
#                      naming execution_timeout (a timeout is a FAIL)
# and then the rule-4 counter (syncs in their submit's frame) must be 0.
# Every job that runs ggml-cpu (the ops runs and probe census) has each pump
# vmcall capped at InferHost.GGML_CPU_TIMEOUT_UNITS (~5 min, rule 10); the
# ggml-rd-only probes keep the Sandbox's 1,000,000.
# Headless (no RenderingDevice): ggml-rd's CPU fallback (the kernels' slangc
# cpp emits on guest memory, guest/ggml-rd/rd_cpu.cpp) stands in as RD0, and
# the ops runs and the fault controls run on it: ops main, ops fault, ops
# fault move (no probes: they measure the GPU). With `fallback=off` the one
# run is -o ADD -b RD0 under GGML_RD_CPU_FALLBACK=0, which must print "no RD
# device" and test no RD0 case: the flat control that separates "the GPU path
# is not there" from "the kernels are wrong" (results-headless-off.txt).
# A HEARTBEAT line every 30 s of a run carries the host summary and the
# ggml-rd counters (dispatches), so a long headless run shows it is moving.
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
#   --params=<regex>      test-backend-ops' -p filter on ops main (no spaces)
#   --env=K=V             one more environment switch for ops main (e.g. GGML_RD_PROFILE=1)
#   fallback=off          headless: the no-device control instead of the fallback runs
#   runs=ops_main,probe_mm_perf  only those runs (the verdicts of the runs left
#                         out then FAIL, so a partial run never reads RESULT: PASS)
extends SceneTree

const InferHost := preload("res://infer_host.gd")
# The ops under test, as test-backend-ops -o takes them. An op family adds
# its ops here (the lead merges this line); ADD stays the fault control.
# GGML_GATE_OPS in the environment replaces the list for one run.
# The 22 census ops, and DUP and MEAN, which share their kernels.
const OPS := ("ADD,MUL,CPY,DUP,CONT,GET_ROWS,CONCAT,REPEAT,MUL_MAT,FLASH_ATTN_EXT,IM2COL,CONV_3D,"
		+ "NORM,RMS_NORM,MEAN,SOFT_MAX,SILU,GELU,GELU_ERF,SIGMOID,NEG,SCALE,DIAG_MASK_INF,ROPE")
const FAULT_OPS := "ADD"
# The census's required type rows (census_union.csv, the plan's Cut 3), as
# regexes over a case's test-backend-ops parameters, by op: each must match
# at least one OK case of a run that tests its op, and no "not supported"
# one. Excluded on purpose: ROPE with frequency factors (ff=1), SOFT_MAX
# with a mask or sinks, FLASH_ATTN_EXT with a mask; no census row has them.
const REQUIRED := {
	"ADD": ["^type=f32,"],
	"MUL": ["^type=f32,"],
	"MUL_MAT": ["^type_a=f32,type_b=f32,", "^type_a=f16,type_b=f32,", "^type_a=bf16,type_b=f32,",
			"^type_a=f16,type_b=f16,", "^type_a=f32,type_b=f32,.*per=\\[0,2,1,3\\]",
			"^type_a=f16,type_b=f32,.*per=\\[0,2,1,3\\]"],
	"CPY": ["^type_src=f32,type_dst=f32,"],
	"CONT": ["^type=f32,", "^type=f16,"],
	"GET_ROWS": ["^type=f32,", "^type=f16,", "^type=bf16,"],
	"CONCAT": ["^type=f32,"],
	"REPEAT": ["^type=f32,"],
	"ROPE": ["^type=f32,.*mode=2,.*ff=0,"],
	"FLASH_ATTN_EXT": ["^hsk=128,hsv=128,nh=12,.*mask=0,sinks=0,max_bias=0\\.000000,logit_softcap=0\\.000000,prec=f32,type_K=f32,type_V=f32"],
	"CONV_3D": ["type_kernel=f16"],
	"IM2COL": ["^type_input=f32,type_kernel=f16,dst_type=f16,"],
	"SOFT_MAX": ["^type=f32,.*mask=0,sinks=0,"],
	"DIAG_MASK_INF": ["^type=f32,"],
	"SCALE": ["^type=f32,"],
	"SIGMOID": ["^type=f32,"],
	"GELU": ["^type=f32,"],
	"GELU_ERF": ["^type=f32,"],
	"NEG": ["^type=f32,"],
	"SILU": ["^type=f32,"],
	"NORM": ["^type=f32,"],
	"RMS_NORM": ["^type=f32,"],
}
const OUT_DIR := "res://../gates/3-ggml-rd/ops/"
# The full list takes about 27 min per ops run in the guest (ggml-cpu is
# emulated; 1590 s and 1612 s on 2026-09-23), ops_main and ops_barrier_all
# both run it, and the whole gate took 3466 s: an hour is too tight.
const WALL_S := 10800.0
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
# Rule 10's control: the ggml-cpu cap lowered to 8 units (8.4e6 instructions,
# ~11 ms), which the first pumps of test-backend-ops overrun.
const CPU_CAP_CONTROL_UNITS := 8

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
var _params := ""
var _env := "" # --env=K=V: added to ops main's environment (e.g. GGML_RD_PROFILE=1)
var _fallback_off := false
var _beat_t0 := 0

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
		elif a.begins_with("--params="):
			_params = a.trim_prefix("--params=")
		elif a.begins_with("--env="):
			_env = a.trim_prefix("--env=")
		elif a == "fallback=off":
			_fallback_off = true
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
	var name := ("results-headless-off.txt" if _fallback_off else "results-headless.txt") if _headless else "results.txt"
	_out = FileAccess.open(ProjectSettings.globalize_path(_out_dir + name), FileAccess.WRITE)
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
	_say("rule 10: ggml-cpu jobs capped at execution_timeout=%d units per vmcall (~300 s), ggml-rd-only jobs at %d" % [
			InferHost.GGML_CPU_TIMEOUT_UNITS, int(_sb.execution_timeout)])
	var pfilter := (" -p " + _params) if _params != "" else ""
	if _headless and _fallback_off:
		_runs = [["ops_no_device", "ops", "-o ADD -b RD0", "GGML_RD_CPU_FALLBACK=0"]]
	elif _headless:
		_runs = [
			["ops_main", "ops", "-o %s -b RD0%s" % [_ops, pfilter], _env],
			["ops_fault", "ops", "-o %s -b RD0" % _fault_ops, "GGML_RD_FAULT=1"],
			["ops_fault_move", "ops", "-o %s -p %s -b RD0" % [FAULT_MOVE_OPS, FAULT_MOVE_PARAMS], "GGML_RD_FAULT=1"],
		]
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
			["probe_census", "probe", "census", "all", ""],
			["ops_main", "ops", "-o %s -b RD0%s" % [_ops, pfilter], _env],
			["ops_barrier_all", "ops", "-o %s -b RD0%s" % [_ops, pfilter], "GGML_RD_BARRIER_ALL=1"],
			["ops_fault", "ops", "-o %s -b RD0" % _fault_ops, "GGML_RD_FAULT=1"],
			["ops_fault_move", "ops", "-o %s -p %s -b RD0" % [FAULT_MOVE_OPS, FAULT_MOVE_PARAMS], "GGML_RD_FAULT=1"],
		]
		_runs.append_array(_extra_probes)
		# Last: a killed vmcall abandons the job's fiber, so no job can follow it.
		_runs.append(["ops_cpu_cap_control", "ops", "-o MUL_MAT -b RD0", ""])
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
		_host.cpu_timeout_units = InferHost.GGML_CPU_TIMEOUT_UNITS
		_host.reset(InferHost.ggml_runs_cpu("ggml_ops_start" if _cur[1] == "ops" else "ggml_probe_start",
				"" if _cur[1] == "ops" else _cur[2]))
		_run_t0 = Time.get_ticks_msec()
		var r: String
		if _cur[1] == "ops":
			r = str(_sb.vmcall("ggml_ops_start", _cur[2], _cur[3]))
		else:
			r = str(_sb.vmcall("ggml_probe_start", _cur[2], _cur[3], _cur[4]))
		if _cur[0] == "ops_cpu_cap_control":
			_host.cpu_timeout_units = CPU_CAP_CONTROL_UNITS # applied from the first pump on
		if not r.begins_with("STARTED"):
			_verdict(false, "%s would not start: %s" % [_cur[0], r])
			_cur = null
		_beat_t0 = Time.get_ticks_msec()
		return false
	var st: String = _host.pump_frame()
	if st == "running":
		if Time.get_ticks_msec() - _beat_t0 >= 30000:
			_beat_t0 = Time.get_ticks_msec()
			_say("   HEARTBEAT %s t=%d s %s | %s" % [_cur[0], (Time.get_ticks_msec() - _run_t0) / 1000,
					_host.summary(), str(_sb.vmcall("ggml_rd_stats")).get_slice(" pipelines", 0)])
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
	var res := {"state": st, "ms": ms, "text": text, "stats": stats, "error": _host.text if st == "error" else ""}
	if _cur[1] == "ops":
		res.merge(_parse_ops(text))
		_say("   cases: OK=%d FAIL=%d not_supported=%d | %s | %s" % [res.ok, res.fail, res.unsupported,
				res.passed_line, res.backend_line])
		for op in res.per_op:
			var c = res.per_op[op]
			_say("   %s: OK=%d FAIL=%d not_supported=%d" % [op, c.ok, c.fail, c.unsupported])
		for l in res.fail_lines.slice(0, 5):
			_say("   failed: %s" % l)
		if not _fallback_off: # the no-device control runs no RD0 case: nothing to require
			_say("   required cases not supported: %d" % res.required_unsupported.size())
			var missing := _required_missing(res)
			_say("   required patterns with no OK case: %d%s" % [missing.size(), (" " + str(missing)) if missing.size() > 0 else ""])
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
	var required_ok := {}
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
				for pat in REQUIRED.get(op, []):
					var rq := RegEx.new()
					rq.compile(pat)
					if rq.search(m.get_string(2)) != null:
						required_ok[op + " " + pat] = required_ok.get(op + " " + pat, 0) + 1
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
			"required_unsupported": required_unsupported, "required_ok": required_ok,
			"passed_line": passed_line, "backend_line": backend_line}

# REQUIRED patterns of the ops this run tested that matched no OK case.
func _required_missing(res: Dictionary) -> Array:
	var tested: PackedStringArray = _ops.split(",")
	var out := []
	for op in REQUIRED:
		if not tested.has(op):
			continue
		for pat in REQUIRED[op]:
			if res.get("required_ok", {}).get(op + " " + pat, 0) == 0:
				out.append(op + " " + pat)
	return out

func _probe_pass(name: String) -> bool:
	return _results.has(name) and _results[name].state == "done" and _results[name].text.contains("RESULT: PASS")

func _checks() -> void:
	_say("== verdicts")
	if _headless and _fallback_off:
		var r = _results.get("ops_no_device", {})
		var t: String = r.get("text", "")
		_verdict(r.get("state", "") == "done" and t.contains("no RD device") and t.contains("Testing 1 devices")
				and not t.contains("Backend RD0:") and r.get("ok", -1) == 0,
				"headless control: GGML_RD_CPU_FALLBACK=0: 'no RD device', 1 device (CPU), no RD0 case run")
		return
	if _headless:
		var m = _results.get("ops_main", {})
		_verdict(m.get("state", "") == "done" and m.get("fail", -1) == 0 and m.get("ok", 0) > 0
				and str(m.get("backend_line", "")).ends_with("OK"),
				"ops_main on the CPU fallback: test-backend-ops -o %s -b RD0: OK=%d FAIL=%d not_supported=%d (%s)" % [
				_ops, m.get("ok", 0), m.get("fail", -1), m.get("unsupported", 0), m.get("backend_line", "")])
		_verdict(m.get("state", "") == "done" and m.get("required_unsupported", [null]).is_empty(),
				"ops_main: no census-required case is not supported (%d are)" % m.get("required_unsupported", [null]).size())
		var f = _results.get("ops_fault", {})
		_verdict(f.get("state", "") == "done" and f.get("fail", 0) > 0 and f.get("ok", -1) == 0,
				"control: GGML_RD_FAULT=1 (a source read one element off) fails every %s case on the fallback: FAIL=%d OK=%d (%s)" % [
				_fault_ops, f.get("fail", 0), f.get("ok", 0), f.get("backend_line", "")])
		var fm = _results.get("ops_fault_move", {})
		_verdict(fm.get("state", "") == "done" and fm.get("fail", 0) > 0 and fm.get("ok", -1) == 0,
				"control: GGML_RD_FAULT=1 fails every %s case but i32 on the fallback: FAIL=%d OK=%d (%s)" % [FAULT_MOVE_OPS,
				fm.get("fail", 0), fm.get("ok", 0), fm.get("backend_line", "")])
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
	_verdict(_probe_pass("probe_census"), "probe_census: every census row within threshold of the in-guest ggml-cpu")
	for n in ["ops_main", "ops_barrier_all"]:
		var r = _results.get(n, {})
		_verdict(r.get("state", "") == "done" and r.get("fail", -1) == 0 and r.get("ok", 0) > 0
				and str(r.get("backend_line", "")).ends_with("OK"),
				"%s: test-backend-ops -o %s -b RD0: OK=%d FAIL=%d not_supported=%d (%s)" % [n, _ops, r.get("ok", 0),
				r.get("fail", -1), r.get("unsupported", 0), r.get("backend_line", "")])
		_verdict(r.get("state", "") == "done" and r.get("required_unsupported", [null]).is_empty(),
				"%s: no census-required case is not supported (%d are)" % [n,
				r.get("required_unsupported", [null]).size()])
		var missing: Array = _required_missing(r) if r.get("state", "") == "done" else ["(run not done)"]
		_verdict(missing.is_empty(), "%s: every census-required pattern of the tested ops has an OK case (%d missing%s)" % [
				n, missing.size(), (": " + str(missing)) if missing.size() > 0 else ""])
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
	var cc = _results.get("ops_cpu_cap_control", {})
	_verdict(cc.get("state", "") == "error" and str(cc.get("error", "")).contains("killed by execution_timeout"),
			"control (rule 10): a ggml-cpu pump over its cap (%d units here; %d, ~5 min, in the other ggml-cpu runs) ends the run as FAIL: %s" % [
			CPU_CAP_CONTROL_UNITS, InferHost.GGML_CPU_TIMEOUT_UNITS, str(cc.get("error", "(not run)"))])
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
