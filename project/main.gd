# The host root of interactor-dress-on (node "Main", so MCP reaches it at
# /root/Main in main.tscn and in xr_main.tscn). A thin root (AGENTS.md rule
# 6): one stage node per ELF, each owning its Sandbox (stages/sandbox_util.gd
# makes them the way Gate 0F says), and the pipeline that composes them.
#
#   DressOn   stages/dress_on_stage.gd   dress_on.elf (Stage 1) + probes.elf (Gate 0F)
#   Drape     stages/drape_stage.gd      drape.elf (Stage 2 AVBD jobs, Cut 5 drape)
#   Curvenet  stages/curvenet_stage.gd   curvenet.elf (Cut 4)
#   Fit       stages/fit_stage.gd        fit.elf (Cut 6)
#   Infer     stages/infer_stage.gd      infer.elf (Cut 4b / 7); fixtures until then
#   Pipeline  stages/pipeline.gd         the loop's state machine
#
# Every guest entry point keeps a no-argument wrapper here (rule 8), each a
# one-line delegate to its stage, so MCP call_method needs no argument
# marshalling and the Gate 1 / 2 / 0F / 0E drives keep working.
extends Node

const DressOnStage := preload("res://stages/dress_on_stage.gd")
const DrapeStage := preload("res://stages/drape_stage.gd")
const CurvenetStage := preload("res://stages/curvenet_stage.gd")
const FitStage := preload("res://stages/fit_stage.gd")
const InferStage := preload("res://stages/infer_stage.gd")
const Pipeline := preload("res://stages/pipeline.gd")

var dress_on = null
var drape = null
var curvenet = null
var fit = null
var infer = null
var pipeline = null

func _ready() -> void:
	dress_on = _add(DressOnStage, "DressOn")
	drape = _add(DrapeStage, "Drape")
	curvenet = _add(CurvenetStage, "Curvenet")
	fit = _add(FitStage, "Fit")
	infer = _add(InferStage, "Infer")
	pipeline = _add(Pipeline, "Pipeline")
	pipeline.setup({"infer": infer, "curvenet": curvenet, "fit": fit, "drape": drape})
	var world = get_node_or_null("World")
	if world != null and world.has_method("attach"):
		world.attach(self)

func _add(script: Script, n: String) -> Node:
	var s: Node = script.new()
	s.name = n
	add_child(s)
	return s

# --- the loop (Cut 8) ------------------------------------------------------------------
# dress_on_run starts the pipeline and returns at once; poll dress_on_status
# until it reads DONE or FAILED(reason), then dress_on_result for the record.
# allow_fixture: stages whose fixture may stand in (infer,rig by default: the
# FoxGirl body and skeleton, labelled FIXTURE).

func dress_on_run(allow_fixture: String = "infer,rig", pen: String = "scripted") -> String:
	return pipeline.start({"allow_fixture": allow_fixture, "pen": pen})

# Gate 8's control: the back seam is not drawn; must end FAILED(MESH: ...).
func dress_on_run_drop_seam(allow_fixture: String = "infer,rig") -> String:
	return pipeline.start({"allow_fixture": allow_fixture, "drop_seam": true})

# Gate 8's control: CHECK sees one garment vertex pushed inside the body.
func dress_on_run_push_vertex(allow_fixture: String = "infer,rig") -> String:
	return pipeline.start({"allow_fixture": allow_fixture, "push_vertex": true})

func dress_on_run_opts(opts: Dictionary = {}) -> String:
	return pipeline.start(opts)

func dress_on_status() -> String:
	return pipeline.status()

func dress_on_result() -> String:
	return JSON.stringify(pipeline.summary(), "  ")

# pen = xr: end the authoring (the menu button or Enter do the same).
func dress_on_author_done() -> String:
	pipeline.pen_finish()
	return "pen finished in %s" % pipeline.state

# Which stages have their ELF (and its API), and why not.
func dress_on_stages() -> String:
	var out := PackedStringArray()
	for s in [infer, curvenet, fit, drape]:
		var why: String = s.reason if not s.available() else ""
		if s == drape and why == "":
			why = drape.drape_api_missing()
		out.append("%s: %s" % [s.stage_name, "ok" if why == "" else why])
	return " | ".join(out)

# --- Stage 1: the GPU layer's own probes (dress_on.elf) ----------------------------------

func rd_open() -> String: return dress_on.rd_open()
func rd_close() -> String: return dress_on.rd_close()
func rd_probe() -> String: return dress_on.rd_probe()
func rd_bench(n_dispatch: int = 1, n_submit: int = 1, barrier: bool = true) -> String: return dress_on.rd_bench(n_dispatch, n_submit, barrier)
func rd_last_step() -> String: return dress_on.rd_last_step()
func rd_bench_quiet(n_dispatch: int = 1, n_submit: int = 1, barrier: bool = true) -> String: return dress_on.rd_bench_quiet(n_dispatch, n_submit, barrier)
func rd_set_probe() -> String: return dress_on.rd_set_probe()
# kind: ticks|limit|clock|bind|barrier|dispatch|submit|buffer|shader|shader-pba|pipeline|uset|readback|instantiate
func rd_calls(kind: String = "ticks", n: int = 1000) -> String: return dress_on.rd_calls(kind, n)

# --- Stage 2: the AVBD solver (drape.elf) --------------------------------------------------
# The one-shot calls are cpu only (rule 4); on rd use the jobs: avbd_job_start
# then avbd_job_tick once per frame.

func avbd_fixture(backend: String = "cpu") -> String: return drape.avbd_fixture(backend)
func avbd_bench(backend: String = "cpu", nx: int = 32, ny: int = 32, substeps: int = 5, iters: int = 10) -> String: return drape.avbd_bench(backend, nx, ny, substeps, iters)
func drape_rd_close() -> String: return drape.drape_rd_close()
func drape_rd_last_step() -> String: return drape.drape_rd_last_step()
func rd_rule4() -> String: return drape.rd_rule4()
func rd_rule4_probe() -> String: return drape.rd_rule4_probe()
func avbd_job_start(name: String = "fixture", backend: String = "rd") -> String: return drape.avbd_job_start(name, backend)
func avbd_job_tick() -> String: return drape.avbd_job_tick()
func avbd_job_names() -> String: return drape.avbd_job_names()

# --- Cut 5: the drape API (drape.elf from cut-5 on; FAIL with the reason before) ----------

func drape_open(backend: String = "auto") -> String: return drape.drape_open(backend)
func drape_sphere_demo(backend: String = "auto") -> String: return drape.drape_sphere_demo(backend)
func drape_scene_mesh(positions: PackedFloat32Array = PackedFloat32Array(), triangles: PackedInt32Array = PackedInt32Array(),
		pins: PackedInt32Array = PackedInt32Array(), material: PackedFloat32Array = PackedFloat32Array()) -> String:
	return drape.drape_scene_mesh(positions, triangles, pins, material)
func drape_primitive(kind: String = "clear", params: PackedFloat32Array = PackedFloat32Array()) -> String: return drape.drape_primitive(kind, params)
func drape_primitive_mesh(positions: PackedFloat32Array = PackedFloat32Array(), triangles: PackedInt32Array = PackedInt32Array(),
		params: PackedFloat32Array = PackedFloat32Array()) -> String:
	return drape.drape_primitive_mesh(positions, triangles, params)
func drape_config(key: String = "iters", value: float = 16.0) -> String: return drape.drape_config(key, value)
func drape_forward(steps: int = 100) -> String: return drape.drape_forward(steps)
func drape_target(kind: String = "trajectory", verts: PackedInt32Array = PackedInt32Array(),
		positions: PackedFloat32Array = PackedFloat32Array(), frame: int = -1) -> String:
	return drape.drape_target(kind, verts, positions, frame)
func drape_backward(loss: String = "match_trajectory", mode: String = "unrolled") -> String: return drape.drape_backward(loss, mode)
func drape_status() -> String: return drape.drape_status()
func drape_result() -> String: return drape.drape_result()
func drape_positions() -> PackedFloat32Array: return drape.drape_positions()
func drape_frame(i: int = 0) -> PackedFloat32Array: return drape.drape_frame(i)
func drape_faces() -> PackedInt32Array: return drape.drape_faces()
func drape_job(name: String = "sphere_forward", backend: String = "auto", args: String = "") -> String: return drape.drape_job(name, backend, args)
func drape_job_result() -> String: return drape.drape_job_result()
func drape_job_frame(i: int = 0) -> PackedFloat32Array: return drape.drape_job_frame(i)
func drape_job_names() -> String: return drape.drape_job_names()
func drape_job_data(key: String = "clear", text: String = "") -> String: return drape.drape_job_data(key, text)
func drape_optimize(spec: String = "params=mu mode=native", x0: PackedFloat32Array = PackedFloat32Array([0.5]),
		lb: PackedFloat32Array = PackedFloat32Array([0.01]), ub: PackedFloat32Array = PackedFloat32Array([1.0]),
		max_iter: int = 10) -> String:
	return drape.drape_optimize(spec, x0, lb, ub, max_iter)
func drape_optimize_result() -> String: return drape.drape_optimize_result()
func lbfgsb_load_oracle() -> String: return drape.lbfgsb_load_oracle()
# One drape_tick / drape_job_tick by hand (_process ticks every frame anyway).
func drape_tick() -> String: return drape.drape_tick()
func drape_job_tick() -> String: return drape.drape_job_tick()

# --- Cut 4: the curvenet stage (curvenet.elf) ---------------------------------------------

func cn_reset() -> String: return curvenet.cn_reset()
func cn_set_param(name: String = "snap_radius", value: float = 0.03) -> String: return curvenet.cn_set_param(name, value)
func cn_set_body_sphere(radius: float = 0.5) -> String: return curvenet.cn_set_body_sphere(radius)
func pen_demo_circle() -> String: return curvenet.pen_demo_circle()
func cn_get_param(name: String = "snap_radius") -> String: return curvenet.cn_get_param(name)
# The scripted pen: pen_begin loads pen_demo_circle's stroke and sends its first
# sample, pen_point the next count (0: all), pen_end(-1) ends it.
func pen_begin(samples: int = 64, pressure: float = 0.5) -> String: return curvenet.pen_begin(samples, pressure)
func pen_point(count: int = 0) -> String: return curvenet.pen_point(count)
func pen_end(id: int = -1) -> String: return curvenet.pen_end(id)
func patch_count() -> String: return curvenet.patch_count()
func patch_vertices(i: int = 0) -> String: return curvenet.patch_vertices(i)
func patch_indices(i: int = 0) -> String: return curvenet.patch_indices(i)
func mesh_patch_ids() -> String: return curvenet.mesh_patch_ids()
func check_names() -> String: return curvenet.check_names()
func curvenet_checks() -> String: return curvenet.curvenet_checks()
func curvenet_check(name: String = "pen_sphere") -> String: return curvenet.curvenet_check(name)
func curvenet_build() -> String: return curvenet.curvenet_build()
func mesh_build(target_edge_length: float = 0.02, weld_eps: float = 1e-5) -> String: return curvenet.mesh_build(target_edge_length, weld_eps)
func mesh_array_mesh() -> ArrayMesh: return curvenet.mesh_array_mesh()
func curvenet_extract_demo() -> String: return curvenet.curvenet_extract_demo()

# --- Cut 6: the fit stage (fit.elf) -------------------------------------------------------

func fit_fixture_foxgirl() -> String: return fit.fit_fixture_foxgirl()
func fit_reset() -> String: return fit.fit_reset()
func fit_begin() -> String: return fit.fit_begin()
func fit_step() -> String: return fit.fit_step()
func fit_run_all() -> String: return fit.fit_run_all()
func fit_status() -> String: return fit.fit_status()
func fit_check() -> String: return fit.fit_check()
func fit_result() -> String: return fit.fit_result()
func fit_result_vertices() -> Variant: return fit.fit_result_vertices()
func fit_result_vertices_f64() -> Variant: return fit.fit_result_vertices_f64()
func fit_preview() -> String: return fit.fit_preview()
func fit_sdf() -> String: return fit.fit_sdf()
func fit_probe_io() -> String: return fit.fit_probe_io()
func fit_probe_ldlt() -> String: return fit.fit_probe_ldlt()
func fit_probe_exceptions() -> String: return fit.fit_probe_exceptions()
func fit_probe_io_paths() -> String: return fit.fit_probe_io_paths()
func fit_probe_ldlt8k() -> String: return fit.fit_probe_ldlt8k()
func fit_probe_libm() -> String: return fit.fit_probe_libm()
func fit_probe_stl() -> String: return fit.fit_probe_stl()
func fit_probe_instret() -> String: return fit.fit_probe_instret()
func fit_probe_heap() -> String: return fit.fit_probe_heap()
func fit_push_control() -> String: return fit.fit_push_control()
func fit_push_flat() -> String: return fit.fit_push_flat()
func fit_set_skin_weights(weights: PackedFloat32Array = PackedFloat32Array()) -> String: return fit.fit_set_skin_weights(weights)
# A fresh fit Sandbox with the stage's fit_memory_mib / fit_elf / fit_execution_timeout.
func fit_configure() -> String: return fit.fit_configure()
func fit_configure_with(memory_mib: int = 2048, elf: String = "res://fit.elf", execution_timeout: int = -1) -> String:
	return fit.fit_configure_with(memory_mib, elf, execution_timeout)
func foxgirl_arrays() -> Dictionary: return fit.foxgirl_arrays()
# Gate 6 sets these on /root/Main; they live on the fit stage.
var fit_config_overrides: Dictionary:
	get:
		return fit.fit_config_overrides
	set(value):
		fit.fit_config_overrides = value
var fit_memory_mib: int:
	get:
		return fit.fit_memory_mib
	set(value):
		fit.fit_memory_mib = value
var fit_execution_timeout: int:
	get:
		return fit.fit_execution_timeout
	set(value):
		fit.fit_execution_timeout = value
var fit_elf: String:
	get:
		return fit.fit_elf
	set(value):
		fit.fit_elf = value

# --- Gate 0F: probes.elf, the sandbox runtime probes ---------------------------------------
# gate_runtime.gd is the gate; these are the no-argument wrappers, every
# argument defaulted.

func p_exceptions(do_throw: bool = true) -> String: return dress_on.pv("p_exceptions", [do_throw])
func p_fenv() -> String: return dress_on.pv("p_fenv")
func p_file(path: String = "res://project.godot") -> String: return dress_on.pv("p_file", [path])
func p_threads() -> String: return dress_on.pv("p_threads")
func p_spin(n: int = 1000000) -> String: return dress_on.pv("p_spin", [n])
func p_alloc(mb: int = 64) -> String: return dress_on.pv("p_alloc", [mb])
func p_memalign(n: int = 1000, upstream: bool = false) -> String: return dress_on.pv("p_memalign", [n, upstream])
func echo_f(x: float = 0.1) -> String: return dress_on.pv("echo_f", [x])
func echo_i(x: int = 9007199254740993) -> String: return dress_on.pv("echo_i", [x])
func echo_b(x: bool = true) -> String: return dress_on.pv("echo_b", [x])
func echo_s(s: String = "héllo") -> String: return dress_on.pv("echo_s", [s])
func echo_pf32() -> String: return dress_on.pv("echo_pf32", [PackedFloat32Array([0.1, -0.0, 1e-40])])
func echo_pb() -> String: return dress_on.pv("echo_pb", [PackedByteArray(range(256))])
func f_bits(x: float = 0.1) -> String: return dress_on.pv("f_bits", [x])
func echo_var(v = 1.5) -> String: return dress_on.pv("echo_var", [v])
func p_hold(mb: int = 64) -> String: return dress_on.pv("p_hold", [mb])
func p_release() -> String: return dress_on.pv("p_release")
func p_rd() -> String: return dress_on.pv("p_rd")
func fib_start() -> String: return dress_on.pv("fib_start")
func fib_pump() -> String: return dress_on.pv("fib_pump") # once per frame (rule 4)
func sm_start() -> String: return dress_on.pv("sm_start")
func sm_pump() -> String: return dress_on.pv("sm_pump")
func big_buffer(bytes: int = 256 << 20, direct: bool = false) -> String: return dress_on.pv("big_buffer", [bytes, direct])
func refs_setup() -> String: return dress_on.pv("refs_setup")
func refs_run(n: int = 1000, aliased: bool = false) -> String: return dress_on.pv("refs_run", [n, aliased])
func refs_usets(n: int = 16) -> String: return dress_on.pv("refs_usets", [n])
func refs_setup_one(k: int = 0) -> String: return dress_on.pv("refs_setup_one", [k])
func rid_hold(permanent: bool = true) -> String: return dress_on.pv("rid_hold", [permanent])
func rid_use() -> String: return dress_on.pv("rid_use") # a later vmcall than rid_hold
func set0_share(variant: String = "") -> String: return dress_on.pv("set0_share", [variant]) # "", "_stripped" or "_o1pp"
# mode: 0 aliased, 1 rw_only, 2 pingpong, 3 ro_then_rw, 4 rw_then_ro
func inplace_run(mode: int = 0, rounds: int = 1000) -> String: return dress_on.pv("inplace_run", [mode, rounds])
func p_rd_close() -> String: return dress_on.pv("p_rd_close")
func p_list_end() -> String: return dress_on.pv("p_list_end")
func f16_read() -> String:
	var h := PackedByteArray()
	h.resize(128)
	for i in 64:
		h.encode_u16(2 * i, 0x3C00 + i) # 1.0 upward
	return dress_on.pv("f16_read", [h])
func ggml_probe(n: int = 256) -> String: return dress_on.pv("ggml_probe", [n])
func zfh_probe() -> String: return dress_on.pv("zfh_probe")
