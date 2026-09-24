# Gate 3 G3.graph and G3.cost: the apps' own graph builders
# (guest/ggml_test/app_graphs, from skin-tokens-ggml and pixal3d-ggml) on
# random weights in ggml_test.elf, on ggml-rd ONLY (guest/ggml_test/
# probe_graph.cpp says what each run does). The guest runs no reference: the
# guest CPU is far too slow for anything beyond G3.ops' single-op cases.
# Each graph run's outputs are dumped to the host (graph_dump.gd) and the
# host-native oracle (tests/ggml_graph_oracle) rebuilds the same net from the
# same seeds and compares: ggml-vulkan on the GPU for the DiT block,
# host ggml-cpu for the Qwen layer and the sparse level, each with the other
# backend as a check of the oracle.
#
#   gates/3-ggml-rd/graph/run.sh          builds the oracle, then runs this:
#   godot --path project --script gate_ggml_graph.gd --rendering-driver vulkan --xr-mode off ++ --oracle=<exe>
#   ... ++ runs=graph_qwen,cost_decode   only those runs (verdicts for those only)
#   ... ++ --dit=dit:8                    graph_dit at 8^3 tokens (the gate's shape is 16^3 = 4096)
#   ... ++ --out=graph-dev                results in gates/3-ggml-rd/<out>/
#   ... ++ --dump=<dir>                   dumps there (default <checkout>/build/graph-dumps)
#   ... ++ --wall=<s>                     the wall clock (default 3600 s)
#   godot --path project --headless --xr-mode off --script gate_ggml_graph.gd ++ --oracle=<exe>
#                                         the graphs on ggml-rd's CPU fallback, the oracle
#                                         host ggml-cpu alone (--check=none); default
#                                         runs= graph_qwen,graph_sconv
#
# Runs, in order (each a probe job on the pump, AGENTS.md rule 4: every
# submit's sync lands on a later frame):
#   graph_qwen    one skin-tokens Qwen3 decoder layer (decode step, f16)      oracle: host ggml-cpu
#   graph_sconv   one Pixal3D shape-decoder level (27 x get_rows + mask mul + mul_mat per conv, f16)
#                                                                              oracle: host ggml-cpu
#   graph_dit8    one Pixal3D flow DiT block at 8^3 = 512 tokens, bf16       oracle: ggml-vulkan
#   graph_dit     one Pixal3D flow DiT block, 4096 x 1536, bf16              oracle: ggml-vulkan
#   graph_kimodo_denoiser  one Kimodo motion-denoiser encoder layer (1024 x 60 tokens x 3, f32)
#                                                                              oracle: host ggml-cpu
#   graph_kimodo_text      one Kimodo LLM2Vec Llama-3-8B layer (4096 x 16 tokens, bf16 base + f32 LoRA)
#                                                                              oracle: host ggml-cpu
#   cost_decode   a skin-tokens decode step (28 layers), 5 timed steps + a profiled one
#   cost_dit      a Pixal3D flow forward (30 blocks), 3 timed + a profiled one
# A graph run passes when the guest prints RESULT: PASS (RD statuses, finite,
# elision == barrier-all bit for bit, the dropped-barrier control) and the
# oracle prints RESULT: PASS (rel-L2 <= 1e-3 native, <= 1e-4 f32, the same
# input bytes). Then rule 4 (0 same-frame syncs). Results:
# gates/3-ggml-rd/graph/results.txt (the last line is RESULT), each run's
# guest output in run-<name>.log and the oracle's in oracle-<name>.txt. The
# oracle is a child process polled every frame, never waited on. Quits on a
# wall clock whatever it is doing.
extends SceneTree

const SandboxUtil := preload("res://stages/sandbox_util.gd")

const InferHost := preload("res://infer_host.gd")
const GraphDump := preload("res://graph_dump.gd")
const WALL_S := 3600.0
const ORACLE_S := 1200.0
const TOTAL_MB := 24576

var _sb = null
var _rd: RenderingDevice = null
var _host = null
var _out: FileAccess
var _t0 := 0
var _rc := 0
var _done := false
var _runs: Array = []
var _selected: Array = []
var _cur = null
var _run_t0 := 0
var _results := {}
var _out_dir := "res://../gates/3-ggml-rd/graph/"
var _dump_dir := ""
var _oracle_exe := ""
var _headless := false
var _wall_s := WALL_S
var _oracle_pid := -1
var _oracle_run = null
var _oracle_t0 := 0

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
	_dump_dir = ProjectSettings.globalize_path("res://").trim_suffix("/").get_base_dir().path_join("build/graph-dumps")
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--out="):
			_out_dir = "res://../gates/3-ggml-rd/" + a.trim_prefix("--out=").trim_suffix("/") + "/"
		elif a.begins_with("--dit="):
			dit_arg = a.trim_prefix("--dit=")
		elif a.begins_with("--wall="):
			_wall_s = float(a.trim_prefix("--wall="))
		elif a.begins_with("--oracle="):
			_oracle_exe = a.trim_prefix("--oracle=")
		elif a.begins_with("--dump="):
			_dump_dir = a.trim_prefix("--dump=")
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
	_say("oracle: %s" % (_oracle_exe.get_file() if _oracle_exe != "" else "(none: graph runs FAIL)"))
	# Headless: ggml-rd's CPU fallback stands in as RD0 (guest/ggml-rd/rd_cpu.cpp),
	# and the oracle is host ggml-cpu with no second backend (`--check=none`):
	# the Lean kernels' cpp emits in the guest against ggml's own CPU code.
	# The default headless runs are the two small graphs; the DiT block and
	# the cost runs are GPU-sized (runs= still names any of them).
	_headless = _rd == null
	SandboxUtil.enable_native_translation()
	_sb = ClassDB.instantiate("Sandbox")
	if _sb != null: _sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	# memory_max before program= (Gate 0F): the DiT block's RD runs keep
	# their outputs (block out + input, 25 MB each at 4096 tokens) for the
	# barrier-all, repeat, drop and dump comparisons in the guest heap
	# (0.8 x memory_max).
	_sb.memory_max = 3600
	_sb.program = load("res://ggml_test.elf")
	_sb.references_max = 65536
	_sb.execution_timeout = 4000000
	var attach := str(_sb.vmcall("ggml_attach", _rd, TOTAL_MB))
	_say("attach: %s" % attach)
	if _headless and not attach.contains("CPU fallback"):
		_verdict(false, "no RenderingDevice and no CPU fallback (run with --rendering-driver vulkan)")
		_finish()
		return
	_say("execution_timeout=%s memory_max=%s" % [str(_sb.execution_timeout), str(_sb.memory_max)])
	_host = InferHost.new(_sb, _rd, "ggml_pump")
	# [name, probe, arg, oracle args (graph runs)]
	_runs = [
		["graph_qwen", "graph", "qwen", ["--ref=cpu", "--check=vulkan"]],
		["graph_sconv", "graph", "sconv", ["--ref=cpu", "--check=vulkan"]],
		["graph_dit8", "graph", "dit:8", ["--ref=vulkan", "--check=cpu"]],
		["graph_dit", "graph", dit_arg, ["--ref=vulkan", "--check=cpu"]],
		["graph_kimodo_denoiser", "graph", "kimodo_denoiser", ["--ref=cpu", "--check=vulkan"]],
		["graph_kimodo_text", "graph", "kimodo_text", ["--ref=cpu", "--check=vulkan"]],
		["cost_decode", "cost", "decode:5", []],
		["cost_dit", "cost", "dit:3", []],
	]
	if _headless:
		for r in _runs:
			if r[1] == "graph":
				r[3] = ["--ref=cpu", "--check=none"]
		if keep.is_empty():
			keep = PackedStringArray(["graph_qwen", "graph_sconv"])
	if not keep.is_empty():
		_runs = _runs.filter(func(r): return keep.has(r[0]))
		_say("runs selected: %s" % str(keep))
	for r in _runs:
		_selected.append(r[0])

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_msec() - _t0 > int(_wall_s * 1000):
		if _oracle_pid >= 0:
			OS.kill(_oracle_pid)
		_verdict(false, "the %d s wall clock ran out%s" % [int(_wall_s), (" in run %s" % _cur[0]) if _cur != null else ""])
		_finish()
		return true
	if _oracle_pid >= 0:
		_poll_oracle()
		return false
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
		if (l.begins_with("PROBE") and not l.contains(" drop k=")) or l.begins_with("RESULT"):
			_say("   " + l)
	_say("   rd: %s" % stats)
	_results[name] = {"state": st, "ms": ms, "text": text, "oracle": ""}
	if _cur[1] == "graph" and st == "done":
		_start_oracle()
		if _oracle_pid >= 0:
			return
	_cur = null

# Dump the run's outputs and start the oracle on them (a child process,
# polled by _process: the frame never blocks on it).
func _start_oracle() -> void:
	var name: String = _cur[0]
	var dir := _dump_dir.path_join(name)
	var t := Time.get_ticks_msec()
	var d := GraphDump.save(_sb, dir)
	_say("   dump: %s in %.1f s" % [d, (Time.get_ticks_msec() - t) / 1000.0])
	if not d.begins_with("DUMPED") or _oracle_exe == "":
		_results[name]["oracle"] = "no oracle run (%s)" % ("no --oracle=" if _oracle_exe == "" else d)
		return
	var of := ProjectSettings.globalize_path(_out_dir + "oracle-%s.txt" % name)
	var args: Array = ["--graph=" + str(_cur[2]), "--dump=" + dir, "--out=" + of]
	args.append_array(_cur[3])
	_say("   oracle: %s %s" % [_oracle_exe.get_file(), " ".join(PackedStringArray(args))])
	_oracle_pid = OS.create_process(_oracle_exe, PackedStringArray(args))
	_oracle_t0 = Time.get_ticks_msec()
	_oracle_run = {"name": name, "file": of}
	if _oracle_pid < 0:
		_results[name]["oracle"] = "oracle would not start"

func _poll_oracle() -> void:
	var over := Time.get_ticks_msec() - _oracle_t0 > int(ORACLE_S * 1000)
	if OS.is_process_running(_oracle_pid) and not over:
		return
	if over:
		OS.kill(_oracle_pid)
	var name: String = _oracle_run["name"]
	var f := FileAccess.open(_oracle_run["file"], FileAccess.READ)
	var text := f.get_as_text() if f != null else ""
	var secs := (Time.get_ticks_msec() - _oracle_t0) / 1000.0
	_say("== %s oracle: %s in %.1f s" % [name, "killed at the %d s limit" % int(ORACLE_S) if over else "exited", secs])
	for l in text.split("\n"):
		if l.contains("SUMMARY") or l.begins_with("RESULT") or l.contains("oracle_check") or l.begins_with("ORACLE"):
			_say("   " + l)
	_results[name]["oracle"] = text
	_oracle_pid = -1
	_oracle_run = null
	_cur = null

func _checks() -> void:
	_say("== verdicts")
	var what := {
		"graph_qwen": "G3.graph Qwen3 decoder layer (f16): ggml-rd vs host ggml-cpu rel-L2 <= 1e-3, f32 arm <= 1e-4, elision == barrier-all bit for bit, dropped barrier detected",
		"graph_sconv": "G3.graph sparse-conv level (f16), vs host ggml-cpu: the same criteria",
		"graph_dit8": "G3.graph DiT block at 512 tokens (bf16), vs ggml-vulkan: the same criteria",
		"graph_dit": "G3.graph DiT block (bf16), vs ggml-vulkan: the same criteria",
		"graph_kimodo_denoiser": "G3.graph Kimodo denoiser encoder layer (f32), vs host ggml-cpu: the same criteria",
		"graph_kimodo_text": "G3.graph Kimodo LLM2Vec text layer (bf16 base + f32 LoRA), vs host ggml-cpu: the same criteria",
		"cost_decode": "G3.cost skin-tokens decode step: every step computed, finite logits",
		"cost_dit": "G3.cost Pixal3D flow forward: every forward computed, finite output",
	}
	for n in ["graph_qwen", "graph_sconv", "graph_dit8", "graph_dit", "graph_kimodo_denoiser", "graph_kimodo_text", "cost_decode", "cost_dit"]:
		if not _selected.has(n):
			continue
		var r = _results.get(n, {})
		var ok: bool = r.get("state", "") == "done" and str(r.get("text", "")).contains("RESULT: PASS")
		if n.begins_with("graph"):
			ok = ok and str(r.get("oracle", "")).contains("RESULT: PASS")
		_verdict(ok, "%s: %s" % [n, what[n]])
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
