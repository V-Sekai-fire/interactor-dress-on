# Gate 3 G3.graph and G3.cost: the apps' own graph builders
# (guest/ggml_test/app_graphs, from skin-tokens-ggml and pixal3d-ggml) on
# random weights in ggml_test.elf, ggml-rd on the GPU against the in-guest
# ggml-cpu (guest/ggml_test/probe_graph.cpp says what each run does).
#
#   godot --path project --script gate_ggml_graph.gd --rendering-driver vulkan --xr-mode off
#   ... ++ runs=graph_qwen,cost_decode   only those runs (the others' verdicts then FAIL)
#   ... ++ --dit=dit:8                    the DiT block at 8^3 tokens (a quick look;
#                                         the gate's shape is 16^3 = 4096)
#   ... ++ --out=graph-dev                results in gates/3-ggml-rd/<out>/
#
# Runs, in order (each a probe job on the pump, AGENTS.md rule 4: every
# submit's sync lands on a later frame):
#   graph_qwen    one skin-tokens Qwen3 decoder layer (decode step, f16)
#   graph_sconv   one Pixal3D shape-decoder level (27 x get_rows + mask mul + mul_mat per conv, f16)
#   cost_decode   a skin-tokens decode step (28 layers), 5 timed steps + a profiled one
#   cost_dit      a Pixal3D flow forward (30 blocks), 3 timed + a profiled one
#   graph_dit     one Pixal3D flow DiT block, 4096 x 1536, bf16 (the in-guest
#                 reference runs about 440 GFLOP twice at rv64gc, ~0.1 GFLOP/s:
#                 hours, so it runs last)
# Each must print RESULT: PASS; then rule 4 (0 same-frame syncs). Results:
# gates/3-ggml-rd/graph/results.txt (the last line is RESULT), each run's
# output in run-<name>.log. Quits on a wall clock whatever it is doing.
extends SceneTree

const InferHost := preload("res://infer_host.gd")
const WALL_S := 25200.0
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
var _results := {}
var _out_dir := "res://../gates/3-ggml-rd/graph/"
var _wall_s := WALL_S

func _clean(t: String) -> String:
	var root := ProjectSettings.globalize_path("res://").trim_suffix("/")
	root = root.get_base_dir()
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

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	var dit_arg := "dit"
	var keep: PackedStringArray = []
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--out="):
			_out_dir = "res://../gates/3-ggml-rd/" + a.trim_prefix("--out=").trim_suffix("/") + "/"
		elif a.begins_with("--dit="):
			dit_arg = a.trim_prefix("--dit=")
		elif a.begins_with("--wall="):
			_wall_s = float(a.trim_prefix("--wall="))
		elif a.begins_with("runs="):
			keep = a.trim_prefix("runs=").split(",")
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_rd = RenderingServer.create_local_rendering_device()
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_out_dir))
	_out = FileAccess.open(ProjectSettings.globalize_path(_out_dir + "results.txt"), FileAccess.WRITE)
	_say("# Gate 3 G3.graph + G3.cost, %s, Godot %s, %s, %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, OS.get_processor_name(),
			"headless: no RenderingDevice" if _rd == null else RenderingServer.get_video_adapter_name()])
	if _rd == null:
		_verdict(false, "no RenderingDevice (run with --rendering-driver vulkan, not --headless)")
		_finish()
		return
	_sb = ClassDB.instantiate("Sandbox")
	# memory_max before program= (Gate 0F): the DiT block's reference holds
	# its bf16 weights (85 MB), their f32 copy (170 MB), gallocr's compute
	# buffer and ggml-cpu's work buffer in the guest heap (0.8 x memory_max).
	_sb.memory_max = 3600
	_sb.program = load("res://ggml_test.elf")
	_sb.references_max = 65536
	# One pump runs one reference node (the CPU backend yields COOP after
	# each): the DiT block's MLP matmul is ~100 GFLOP in one vmcall.
	_sb.execution_timeout = 4000000
	_say("attach: %s" % str(_sb.vmcall("ggml_attach", _rd, TOTAL_MB)))
	_say("execution_timeout=%s memory_max=%s" % [str(_sb.execution_timeout), str(_sb.memory_max)])
	_host = InferHost.new(_sb, _rd, "ggml_pump")
	_runs = [
		["graph_qwen", "graph", "qwen"],
		["graph_sconv", "graph", "sconv"],
		["cost_decode", "cost", "decode:5"],
		["cost_dit", "cost", "dit:3"],
		["graph_dit", "graph", dit_arg],
	]
	if not keep.is_empty():
		_runs = _runs.filter(func(r): return keep.has(r[0]))
		_say("runs selected: %s" % str(keep))

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_msec() - _t0 > int(_wall_s * 1000):
		_verdict(false, "the %d s wall clock ran out%s" % [int(_wall_s), (" in run %s" % _cur[0]) if _cur != null else ""])
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
		var r := str(_sb.vmcall("ggml_probe_start", _cur[1], _cur[2], ""))
		_say("== %s: %s" % [_cur[0], r])
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
	var text := str(_sb.vmcall("ggml_output"))
	var lf := FileAccess.open(ProjectSettings.globalize_path(_out_dir + "run-%s.log" % name), FileAccess.WRITE)
	lf.store_string(_clean(text))
	lf.close()
	var stats := str(_sb.vmcall("ggml_rd_stats"))
	_say("== %s: %s in %.1f s, %s" % [name, st, ms / 1000.0, _host.summary()])
	if st == "error":
		_say("   error: %s" % _host.text)
	for l in text.split("\n"):
		if l.begins_with("PROBE") or l.begins_with("RESULT"):
			_say("   " + l)
	_say("   rd: %s" % stats)
	_results[name] = {"state": st, "ms": ms, "text": text}

func _checks() -> void:
	_say("== verdicts")
	var what := {
		"graph_qwen": "G3.graph Qwen3 decoder layer (f16): rel-L2 <= 1e-3, f32 arm <= 1e-4, elision == barrier-all bit for bit, dropped barrier detected",
		"graph_sconv": "G3.graph sparse-conv level (f16): the same criteria",
		"graph_dit": "G3.graph DiT block (bf16): the same criteria",
		"cost_decode": "G3.cost skin-tokens decode step: every step computed, finite logits",
		"cost_dit": "G3.cost Pixal3D flow forward: every forward computed, finite output",
	}
	for n in ["graph_qwen", "graph_sconv", "graph_dit", "cost_decode", "cost_dit"]:
		var r = _results.get(n, {})
		_verdict(r.get("state", "") == "done" and str(r.get("text", "")).contains("RESULT: PASS"), "%s: %s" % [n, what[n]])
	var stats := str(_sb.vmcall("ggml_rd_stats"))
	var re := RegEx.new()
	re.compile("rule4_same_frame_syncs=(\\d+)")
	var mm := re.search(stats)
	_verdict(mm != null and int(mm.get_string(1)) == 0, "rule 4: no sync in its submit's frame (%s)" % (mm.get_string(0) if mm else "?"))

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
