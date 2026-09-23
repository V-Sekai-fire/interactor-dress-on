# Rule 8 for the thin root, two ways. Writes gates/8-loop/wrappers.txt; 120 s
# wall clock.
#
#   godot --path project --script tests/probe_main_wrappers.gd --rendering-driver vulkan --xr-mode off
#
# 1. Source audit (tests/wrapper_audit.gd): every ADD_API_FUNCTION /
#    add_sandbox_api_function in guest/**/main.cpp (dress_on, probes, curvenet,
#    drape, fit, ggml_test) is reached from a public main.gd function, through its stage
#    file, and every such wrapper's parameters all have defaults. FAIL names
#    each guest entry point that has none. Controls: main.gd with the
#    p_memalign delegate deleted must report exactly dress_on:p_memalign
#    missing; with rd_calls' defaults stripped, exactly its two parameters;
#    and every guest file must yield names (an empty read is not a pass).
# 2. Runtime: main.tscn's Main has every wrapper the old main.gd had
#    (279b31b), every cut-4/5/6 one, Cut 8's and Cut 3's, each callable with no
#    argument (compiled method list), and a sample answers through the stages.
extends SceneTree

const Audit := preload("res://tests/wrapper_audit.gd")
const OUT := "res://../gates/8-loop/wrappers.txt"
# main.gd at 279b31b (the thin root's input), every public method.
const OLD := ["rd_open", "rd_close", "rd_probe", "rd_bench", "rd_last_step", "avbd_fixture", "avbd_bench",
		"drape_rd_close", "drape_rd_last_step", "rd_rule4", "rd_rule4_probe", "avbd_job_start", "avbd_job_tick",
		"avbd_job_names", "p_exceptions", "p_fenv", "p_file", "p_threads", "p_spin", "p_alloc", "echo_f", "echo_i",
		"echo_b", "echo_s", "echo_pf32", "echo_pb", "f_bits", "echo_var", "p_hold", "p_release", "p_rd", "fib_start",
		"fib_pump", "sm_start", "sm_pump", "big_buffer", "refs_setup", "refs_run", "refs_usets", "refs_setup_one",
		"rid_hold", "rid_use", "set0_share", "inplace_run", "p_rd_close", "p_list_end", "f16_read", "ggml_probe",
		"zfh_probe"]
# main.gd at 8d59e6a (cut-4 curvenet, cut-5 drape, cut-6 fit merged), every
# public method, plus the entry points that had none there: dress_on.elf's
# rd_bench_quiet, rd_calls, rd_set_probe and drape.elf's drape_tick,
# drape_job_tick (main.gd's _process called those two).
const MAIN_8D59 := ["cn_reset", "cn_set_param", "cn_get_param", "cn_set_body_sphere", "pen_demo_circle",
		"pen_begin", "pen_point", "pen_end", "patch_count", "patch_vertices", "patch_indices", "mesh_patch_ids",
		"check_names", "curvenet_checks", "curvenet_check", "curvenet_build", "mesh_build", "mesh_array_mesh",
		"curvenet_extract_demo", "p_memalign",
		"drape_open", "drape_sphere_demo", "drape_scene_mesh", "drape_primitive", "drape_config", "drape_forward",
		"drape_target", "drape_backward", "drape_status", "drape_result", "drape_positions", "drape_frame",
		"drape_faces", "drape_job", "drape_job_result", "drape_job_frame", "drape_job_names", "drape_optimize",
		"drape_optimize_result", "drape_job_data", "lbfgsb_load_oracle", "drape_tick", "drape_job_tick",
		"fit_configure", "fit_configure_with", "fit_fixture_foxgirl", "foxgirl_arrays", "fit_reset", "fit_begin",
		"fit_step", "fit_run_all", "fit_status", "fit_set_skin_weights", "fit_check", "fit_result",
		"fit_result_vertices", "fit_result_vertices_f64", "fit_preview", "fit_sdf", "fit_probe_io", "fit_probe_ldlt",
		"fit_probe_exceptions", "fit_probe_io_paths", "fit_probe_ldlt8k", "fit_probe_libm", "fit_probe_stl",
		"fit_probe_instret", "fit_probe_heap", "fit_push_control", "fit_push_flat",
		"rd_bench_quiet", "rd_calls", "rd_set_probe"]
# main.gd at cut-3 f1a4bab (ggml_test.elf), every public method it added.
const CUT3 := ["p_recovery", "ggml_attach", "ggml_ops_start", "ggml_probe_start", "ggml_pump", "ggml_output",
		"ggml_rd_stats", "ggml_rd_close", "ggml_job_status", "ggml_ops_add_mul", "ggml_ops_barrier_all",
		"ggml_ops_fault", "ggml_probe_chain", "ggml_probe_independent", "ggml_probe_alias_rw", "ggml_probe_alias_ro",
		"ggml_ops_k1k5", "ggml_probe_census", "ggml_probe_census_fault", "ggml_probe_perf", "ggml_ops_move",
		"ggml_ops_move_fault", "ggml_probe_perf_move", "ggml_ops_conv", "ggml_ops_conv_fault", "ggml_probe_conv_perf",
		"ggml_ops_rows", "ggml_probe_rows_perf", "ggml_ops_mul_mat", "ggml_probe_mm_perf", "ggml_ops_flash_attn",
		"ggml_probe_fa_perf", "ggml_ops_all", "ggml_graph_qwen", "ggml_graph_sconv", "ggml_graph_dit",
		"ggml_dump_list", "ggml_graph_dump", "ggml_cost_decode", "ggml_cost_dit", "ggml_probe_files"]
const CUT8 := ["dress_on_run", "dress_on_run_drop_seam", "dress_on_run_push_vertex", "dress_on_run_opts",
		"dress_on_status", "dress_on_result", "dress_on_author_done", "dress_on_stages"]
# Properties Gate 6 sets on /root/Main (forwarded to the fit stage).
const PROPS := ["fit_config_overrides", "fit_memory_mib", "fit_elf", "fit_execution_timeout"]
# Calls whose answers are checked: [method, expected substring ("" = any)].
const CALLS := [["rd_open", ""], ["rd_probe", "PASS"], ["rd_bench", "nd=1 ns=1 barrier=true"],
		["rd_bench_quiet", "nd=1 ns=1 barrier=true"], ["rd_set_probe", "ok 552 bytes"], ["rd_calls", "ticks n=1000"],
		["rd_close", ""], ["avbd_fixture", "host_us="], ["avbd_job_names", ""], ["rd_rule4", ""], ["p_fenv", ""],
		["echo_i", "9007199254740993"], ["p_exceptions", "caught runtime_error"], ["p_memalign", ""],
		["dress_on_stages", "infer:"], ["drape_status", "IDLE"], ["drape_open", ""], ["drape_job_names", "sphere_forward"],
		["drape_sinew_align_test", "PASS"],
		["cn_reset", ""], ["cn_get_param", "0.03"], ["check_names", ""], ["fit_status", ""],
		["dress_on_status", "IDLE"], ["ggml_job_status", "IDLE"], ["ggml_rd_stats", "IDLE"]]

var _out: FileAccess
var _main: Node
var _frames := 0
var _t0 := 0
var _fails := 0

func _say(s: String) -> void:
	print(s)
	_out.store_line(s)
	_out.flush()

func _ok(pass_: bool, what: String) -> void:
	if not pass_:
		_fails += 1
	_say("%s %s" % ["PASS" if pass_ else "FAIL", what])

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT), FileAccess.WRITE)
	_main = (load("res://main.tscn") as PackedScene).instantiate()
	root.add_child(_main)

func _source_audit() -> void:
	var main_src := Audit.read("res://main.gd")
	var stages := Audit.stage_sources()
	var api := {}
	var total := 0
	for g in Audit.GUESTS:
		var names := Audit.api_names(Audit.read("res://../" + g))
		_ok(names.size() > 0, "flat control: %s has %d entry points" % [g, names.size()])
		total += names.size()
	api = Audit.api_by_stage()
	var a := Audit.audit(api, main_src, stages)
	_ok(a[0].is_empty(), "every guest entry point has a /root/Main wrapper (%d of %d; %d wrappers)%s" % [
			total - a[0].size(), total, a[2].size(), "" if a[0].is_empty() else " MISSING: " + ", ".join(a[0])])
	_ok(a[1].is_empty(), "every such wrapper's arguments have defaults%s" % [
			"" if a[1].is_empty() else " (no default: %s)" % ", ".join(a[1])])
	var bad := Audit.not_callable(a[2], load("res://main.gd"))
	_ok(bad.is_empty(), "main.gd compiles and its %d wrappers take no required argument%s" % [a[2].size(),
			"" if bad.is_empty() else " (missing or with required args: %s)" % ", ".join(bad)])
	# controls
	var lines := main_src.split("\n")
	var cut := PackedStringArray()
	for l in lines:
		if not l.begins_with("func p_memalign("):
			cut.append(l)
	var c1 := Audit.audit(api, "\n".join(cut), stages)
	_ok(c1[0] == ["dress_on:p_memalign"], "control: main.gd without the p_memalign delegate -> missing %s" % str(c1[0]))
	var no_def := main_src.replace("func rd_calls(kind: String = \"ticks\", n: int = 1000)", "func rd_calls(kind: String, n: int)")
	var c2 := Audit.audit(api, no_def, stages)
	_ok(c2[1] == ["rd_calls(kind: String)", "rd_calls(n: int)"], "control: rd_calls with its defaults stripped -> no default %s" % str(c2[1]))

func _process(_d: float) -> bool:
	_frames += 1
	if Time.get_ticks_msec() - _t0 > 120000:
		_say("RESULT: FAIL (wall clock)")
		_out.close()
		quit(1)
		return true
	if _frames < 3:
		return false
	_say("node path: %s" % str(_main.get_path()))
	_source_audit()
	var missing := []
	var have := {}
	for m in _main.get_script().get_script_method_list():
		have[m.name] = m.args.size() - m.default_args.size()
	for m in OLD + MAIN_8D59 + CUT8 + CUT3:
		if have.get(m, -1) != 0:
			missing.append(m)
	_ok(missing.is_empty(), "wrappers on /root/Main callable with no argument: %d old (279b31b) + %d main (8d59e6a + 5) + %d Cut 8 + %d Cut 3, missing %s" % [
			OLD.size(), MAIN_8D59.size(), CUT8.size(), CUT3.size(), str(missing)])
	var props := []
	for p in PROPS:
		if _main.get(p) == null:
			props.append(p)
	_ok(props.is_empty(), "properties forwarded to the fit stage: %s, missing %s" % [", ".join(PROPS), str(props)])
	for c in CALLS:
		var r := str(_main.call(c[0]))
		_ok(c[1] == "" or r.find(c[1]) >= 0, "%s -> %s" % [c[0], r.substr(0, 300).replace("\n", " | ")])
	_say("RESULT: %s" % ("PASS" if _fails == 0 else "FAIL (%d)" % _fails))
	_out.close()
	quit(0 if _fails == 0 else 1)
	return true
