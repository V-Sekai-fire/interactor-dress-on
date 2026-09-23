# Rule 8 smoke for the thin root: main.tscn's Main still has every wrapper the
# old main.gd had (and cut-4/5/6's), and a sample of them answers through the
# stages exactly as before. Writes gates/8-loop/wrappers.txt; 120 s wall clock.
#
#   godot --path project --script tests/probe_main_wrappers.gd --rendering-driver vulkan --xr-mode off
extends SceneTree

const OUT := "res://../gates/8-loop/wrappers.txt"
# main.gd at 279b31b (the thin root's input), every public method.
const OLD := ["rd_open", "rd_close", "rd_probe", "rd_bench", "rd_last_step", "avbd_fixture", "avbd_bench",
		"drape_rd_close", "drape_rd_last_step", "rd_rule4", "rd_rule4_probe", "avbd_job_start", "avbd_job_tick",
		"avbd_job_names", "p_exceptions", "p_fenv", "p_file", "p_threads", "p_spin", "p_alloc", "echo_f", "echo_i",
		"echo_b", "echo_s", "echo_pf32", "echo_pb", "f_bits", "echo_var", "p_hold", "p_release", "p_rd", "fib_start",
		"fib_pump", "sm_start", "sm_pump", "big_buffer", "refs_setup", "refs_run", "refs_usets", "refs_setup_one",
		"rid_hold", "rid_use", "set0_share", "inplace_run", "p_rd_close", "p_list_end", "f16_read", "ggml_probe",
		"zfh_probe"]
# The in-flight branches' wrappers (cut-4 curvenet, cut-5 drape, cut-6 fit) and Cut 8's.
const NEW := ["cn_reset", "cn_set_param", "cn_set_body_sphere", "pen_demo_circle", "pen_end", "patch_count",
		"curvenet_checks", "curvenet_check", "curvenet_build", "mesh_build", "mesh_array_mesh", "curvenet_extract_demo",
		"drape_open", "drape_sphere_demo", "drape_scene_mesh", "drape_primitive", "drape_config", "drape_forward",
		"drape_target", "drape_backward", "drape_status", "drape_result", "drape_positions", "drape_frame",
		"drape_faces", "drape_job", "drape_job_result", "fit_fixture_foxgirl", "fit_reset", "fit_begin", "fit_step",
		"fit_run_all", "fit_status", "fit_check", "fit_result", "fit_result_vertices", "fit_result_vertices_f64",
		"fit_preview", "fit_sdf", "fit_probe_io", "fit_probe_ldlt", "fit_probe_exceptions", "fit_probe_io_paths",
		"dress_on_run", "dress_on_run_drop_seam", "dress_on_run_push_vertex", "dress_on_run_opts", "dress_on_status",
		"dress_on_result", "dress_on_author_done", "dress_on_stages"]
# Calls whose answers are checked: [method, expected prefix or substring].
const CALLS := [["rd_open", ""], ["rd_probe", "PASS"], ["rd_bench", "nd=1 ns=1 barrier=true"], ["rd_close", ""],
		["avbd_fixture", "host_us="], ["avbd_job_names", ""], ["rd_rule4", ""], ["p_fenv", ""],
		["echo_i", "9007199254740993"], ["p_exceptions", "caught runtime_error"], ["dress_on_stages", "infer:"],
		["drape_status", "IDLE"], ["drape_open", ""], ["cn_reset", ""], ["fit_status", ""], ["dress_on_status", "IDLE"]]

var _out: FileAccess
var _main: Node
var _frames := 0
var _t0 := 0
var _fails := 0

func _say(s: String) -> void:
	print(s)
	_out.store_line(s)
	_out.flush()

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT), FileAccess.WRITE)
	_main = (load("res://main.tscn") as PackedScene).instantiate()
	root.add_child(_main)

func _process(_d: float) -> bool:
	_frames += 1
	if Time.get_ticks_msec() - _t0 > 120000:
		_say("RESULT: FAIL (wall clock)")
		quit(1)
		return true
	if _frames < 3:
		return false
	_say("node path: %s" % str(_main.get_path()))
	var missing := []
	for m in OLD + NEW:
		if not _main.has_method(m):
			missing.append(m)
	_say("%s wrappers: %d old + %d new on /root/Main, missing %s" % ["PASS" if missing.is_empty() else "FAIL",
			OLD.size(), NEW.size(), str(missing)])
	if not missing.is_empty():
		_fails += 1
	for c in CALLS:
		var r := str(_main.call(c[0]))
		var ok: bool = c[1] == "" or r.find(c[1]) >= 0
		if not ok:
			_fails += 1
		_say("%s %s -> %s" % ["PASS" if ok else "FAIL", c[0], r.substr(0, 300).replace("\n", " | ")])
	_say("RESULT: %s" % ("PASS" if _fails == 0 else "FAIL"))
	_out.close()
	quit(0 if _fails == 0 else 1)
	return true
