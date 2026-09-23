# The host side of interactor-dress-on. Owns the sandbox guests and exposes
# plain methods that an MCP client can reach with call_method. The guest APIs
# take typed buffers; wrapping here keeps those off the JSON wire. Stages are
# separate ELFs in separate Sandbox nodes, composed here; meshes cross between
# them as packed arrays.
extends Node

const MeshWire := preload("res://util/mesh_wire.gd")

var _sb = null     # dress_on.elf: the Stage 1 GPU-layer probes
var _drape = null  # drape.elf: the AVBD solver, cpu and rd
var _cn = null     # curvenet.elf: Cassie pen -> curvenet -> mesh (CPU only)
var _fit = null    # fit.elf: cloth-fit's garment solve (PolyFEM), one phase per call

const ObjIO = preload("res://util/obj_io.gd")

func _ready() -> void:
	_sb = ClassDB.instantiate("Sandbox")
	if _sb == null:
		push_error("Sandbox class not registered; is the godot_sandbox addon enabled?")
		return
	add_child(_sb)
	# Every Array, RDUniform and returned Variant is scoped to one vmcall; the
	# default cap (100) is hit by ~30 uniform sets. Stage 1 finding.
	_sb.references_max = 65536
	_sb.program = load("res://dress_on.elf")
	print("[dress-on] sandbox loaded dress_on.elf")
	_drape = ClassDB.instantiate("Sandbox")
	add_child(_drape)
	# The drape's uniform sets are kernels x colours per call.
	_drape.references_max = 65536
	_drape.program = load("res://drape.elf")
	print("[dress-on] sandbox loaded drape.elf")
	_cn = ClassDB.instantiate("Sandbox")
	add_child(_cn)
	_cn.references_max = 4096
	_cn.program = load("res://curvenet.elf")
	print("[dress-on] sandbox loaded curvenet.elf")
	_fit_load()

func _bytes(path: String) -> PackedByteArray:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return PackedByteArray()
	var b := f.get_buffer(f.get_length())
	f.close()
	return b

# --- Stage 1: the GPU layer's own probes ------------------------------------

func rd_open() -> String:
	return str(_sb.vmcall("rd_open")) if _sb != null else "FAIL: no sandbox"

func rd_close() -> String:
	return str(_sb.vmcall("rd_close")) if _sb != null else "FAIL: no sandbox"

# Gate 0A's probe, now through rd_compute. Says whether the device was held
# from an earlier call.
func rd_probe() -> String:
	if _sb == null:
		return "FAIL: no sandbox"
	var spirv := _bytes("res://probe.spv")
	if spirv.is_empty():
		return "FAIL: could not open probe.spv"
	return str(_sb.vmcall("rd_probe", spirv))

# n_submit compute lists of n_dispatch accumulate dispatches; the count must
# equal the product. host_us is the whole vmcall, boundary included.
func rd_bench(n_dispatch: int = 1, n_submit: int = 1, barrier: bool = true) -> String:
	if _sb == null:
		return "FAIL: no sandbox"
	var spirv := _bytes("res://accumulate.spv")
	if spirv.is_empty():
		return "FAIL: could not open accumulate.spv"
	var t0 := Time.get_ticks_usec()
	var r = _sb.vmcall("rd_bench", spirv, n_dispatch, n_submit, barrier)
	var dt := Time.get_ticks_usec() - t0
	return "nd=%d ns=%d barrier=%s host_us=%d %s" % [n_dispatch, n_submit, barrier, dt, str(r)]

func rd_last_step() -> String:
	return str(_sb.vmcall("rd_last_step")) if _sb != null else "FAIL: no sandbox"

# --- Stage 2: the AVBD solver (drape.elf) --------------------------------------
# The one-shot calls are cpu only (rule 4: a one-shot rd call would sync in its
# submit's frame). On rd use the jobs below: avbd_job_start("fixture", "rd"),
# avbd_job_start("bench_fwd", "rd"), then avbd_job_tick once per frame.

func avbd_fixture(backend: String = "cpu") -> String:
	if _drape == null:
		return "FAIL: no drape sandbox"
	var t0 := Time.get_ticks_usec()
	var r = _drape.vmcall("avbd_fixture", backend)
	return "host_us=%d %s" % [Time.get_ticks_usec() - t0, str(r)]

func avbd_bench(backend: String = "cpu", nx: int = 32, ny: int = 32, substeps: int = 5, iters: int = 10) -> String:
	if _drape == null:
		return "FAIL: no drape sandbox"
	var t0 := Time.get_ticks_usec()
	var r = _drape.vmcall("avbd_bench", backend, nx, ny, substeps, iters)
	var dt := Time.get_ticks_usec() - t0
	return "host_us=%d ms/substep=%.2f %s" % [dt, dt / 1000.0 / substeps, str(r)]

# Drop the job and free the drape's RenderingDevice and its permanent RID
# slots; call before freeing the drape sandbox (BUSY while a submit is in flight).
func drape_rd_close() -> String:
	return str(_drape.vmcall("rd_close")) if _drape != null else "FAIL: no drape sandbox"

func drape_rd_last_step() -> String:
	return str(_drape.vmcall("rd_last_step")) if _drape != null else "FAIL: no drape sandbox"

# Rule 4's guard: syncs that landed in their submit's process frame.
func rd_rule4() -> String:
	return str(_drape.vmcall("rd_rule4")) if _drape != null else "FAIL: no drape sandbox"

# The guard's positive control: submits and syncs in one call.
func rd_rule4_probe() -> String:
	return str(_drape.vmcall("rd_rule4_probe")) if _drape != null else "FAIL: no drape sandbox"

# --- Stage 2 gate jobs (drape.elf): one job at a time, one tick per frame ------
# Rule 4: a job's GPU submit ends its tick, so the readback lands on a later
# frame. Start a job, then call avbd_job_tick once per frame (the host clock
# goes in with it) until it stops answering "RUNNING k/N". Names: see
# avbd_job_names(). gate_avbd.gd drives the whole list.

func avbd_job_start(name: String = "fixture", backend: String = "rd") -> String:
	return str(_drape.vmcall("avbd_job_start", name, backend)) if _drape != null else "FAIL: no drape sandbox"

func avbd_job_tick() -> String:
	return str(_drape.vmcall("avbd_job_tick", Time.get_ticks_usec())) if _drape != null else "FAIL: no drape sandbox"

func avbd_job_names() -> String:
	return str(_drape.vmcall("avbd_job_names")) if _drape != null else "FAIL: no drape sandbox"

# --- Gate 0F: probes.elf, the sandbox runtime probes ---------------------------
# Its own Sandbox, created on first use. gate_runtime.gd is the gate; these are
# the no-argument wrappers (AGENTS.md rule 8), every argument defaulted.

var _probes = null

func _pv(fn: String, args: Array = []) -> String:
	if _probes == null:
		_probes = ClassDB.instantiate("Sandbox")
		if _probes == null:
			return "FAIL: no sandbox"
		add_child(_probes)
		_probes.program = load("res://probes.elf")
		_probes.references_max = 4096
	return str(_probes.callv("vmcall", [fn] + args))

func p_exceptions(do_throw: bool = true) -> String: return _pv("p_exceptions", [do_throw])
func p_fenv() -> String: return _pv("p_fenv")
func p_file(path: String = "res://project.godot") -> String: return _pv("p_file", [path])
func p_threads() -> String: return _pv("p_threads")
func p_spin(n: int = 1000000) -> String: return _pv("p_spin", [n])
func p_alloc(mb: int = 64) -> String: return _pv("p_alloc", [mb])
func p_memalign(n: int = 1000, upstream: bool = false) -> String: return _pv("p_memalign", [n, upstream])
func echo_f(x: float = 0.1) -> String: return _pv("echo_f", [x])
func echo_i(x: int = 9007199254740993) -> String: return _pv("echo_i", [x])
func echo_b(x: bool = true) -> String: return _pv("echo_b", [x])
func echo_s(s: String = "h\u00e9llo") -> String: return _pv("echo_s", [s])
func echo_pf32() -> String: return _pv("echo_pf32", [PackedFloat32Array([0.1, -0.0, 1e-40])])
func echo_pb() -> String: return _pv("echo_pb", [PackedByteArray(range(256))])
func f_bits(x: float = 0.1) -> String: return _pv("f_bits", [x])
func echo_var(v = 1.5) -> String: return _pv("echo_var", [v])
func p_hold(mb: int = 64) -> String: return _pv("p_hold", [mb])
func p_release() -> String: return _pv("p_release")
func p_rd() -> String: return _pv("p_rd")
func fib_start() -> String: return _pv("fib_start")
func fib_pump() -> String: return _pv("fib_pump") # once per frame (rule 4)
func sm_start() -> String: return _pv("sm_start")
func sm_pump() -> String: return _pv("sm_pump")
func big_buffer(bytes: int = 256 << 20, direct: bool = false) -> String: return _pv("big_buffer", [bytes, direct])
func refs_setup() -> String: return _pv("refs_setup")
func refs_run(n: int = 1000, aliased: bool = false) -> String: return _pv("refs_run", [n, aliased])
func refs_usets(n: int = 16) -> String: return _pv("refs_usets", [n])
func refs_setup_one(k: int = 0) -> String: return _pv("refs_setup_one", [k])
func rid_hold(permanent: bool = true) -> String: return _pv("rid_hold", [permanent])
func rid_use() -> String: return _pv("rid_use") # a later vmcall than rid_hold
func set0_share(variant: String = "") -> String: return _pv("set0_share", [variant]) # "", "_stripped" or "_o1pp"
# mode: 0 aliased, 1 rw_only, 2 pingpong, 3 ro_then_rw, 4 rw_then_ro
func inplace_run(mode: int = 0, rounds: int = 1000) -> String: return _pv("inplace_run", [mode, rounds])
func p_rd_close() -> String: return _pv("p_rd_close")
func p_list_end() -> String: return _pv("p_list_end")
func f16_read() -> String:
	var h := PackedByteArray()
	h.resize(128)
	for i in 64:
		h.encode_u16(2 * i, 0x3C00 + i) # 1.0 upward
	return _pv("f16_read", [h])
func ggml_probe(n: int = 256) -> String: return _pv("ggml_probe", [n])
func zfh_probe() -> String: return _pv("zfh_probe")

# --- Cut 4: the curvenet stage (curvenet.elf) ----------------------------------
# Meshes, curves and knots cross as packed arrays in util/mesh_wire.gd's
# format. No GPU: every call is a plain vmcall; host_us times it.

func _cn_call(fn: String, args: Array = []) -> String:
	if _cn == null:
		return "FAIL: no curvenet sandbox"
	var t0 := Time.get_ticks_usec()
	var r = _cn.callv("vmcall", [fn] + args)
	return "host_us=%d %s" % [Time.get_ticks_usec() - t0, str(r)]

func cn_reset() -> String:
	return _cn_call("cn_reset")

func cn_set_param(name: String = "snap_radius", value: float = 0.03) -> String:
	return _cn_call("cn_set_param", [name, value])

func cn_get_param(name: String = "snap_radius") -> String:
	return _cn_call("cn_get_param", [name])

# The demo body: an r = 0.5 SphereMesh at the origin.
func cn_set_body_sphere(radius: float = 0.5) -> String:
	var b := MeshWire.sphere(radius)
	return _cn_call("cn_set_body", [b.vertices, b.triangles])

# One closed stroke at 30 degrees latitude, 1 cm off the demo sphere: it
# snaps onto the body and closes one patch. Resets the stage first.
func pen_demo_circle() -> String:
	var b := cn_set_body_sphere()
	var r := cn_reset()
	var s := MeshWire.circle_stroke(0.51, PI / 6.0, TAU, 64)
	return "%s | %s | %s" % [b, r, _cn_call("pen_stroke", [s])]

# The scripted pen source: pen_begin/pen_point/pen_end with no arguments
# replay one stroke sample by sample, as a tracked pen would deliver it. The
# stroke is pen_demo_circle's (30 degrees latitude, 1 cm off the demo sphere,
# closed); call cn_set_body_sphere and cn_reset first for a fresh stage.
var _pen_src := PackedFloat32Array()
var _pen_next := 0
var _pen_id := -1

# Loads the scripted stroke (samples + 1 samples, the last on the first) and
# sends its first sample; the guest's stroke id is kept for pen_point/pen_end.
func pen_begin(samples: int = 64, pressure: float = 0.5) -> String:
	if _cn == null:
		return "FAIL: no curvenet sandbox"
	_pen_src = MeshWire.circle_stroke(0.51, PI / 6.0, TAU, samples, pressure)
	_pen_next = 4
	var t0 := Time.get_ticks_usec()
	_pen_id = int(_cn.vmcall("pen_begin", _pen_src[0], _pen_src[1], _pen_src[2], _pen_src[3]))
	return "host_us=%d id=%d samples=%d" % [Time.get_ticks_usec() - t0, _pen_id, _pen_src.size() / 4]

# The next `count` samples of the scripted stroke (0: all that remain).
func pen_point(count: int = 0) -> String:
	if _cn == null:
		return "FAIL: no curvenet sandbox"
	if _pen_id < 0 or _pen_next >= _pen_src.size():
		return "FAIL: no scripted stroke in progress (pen_begin first)"
	var end := _pen_src.size() if count <= 0 else mini(_pen_src.size(), _pen_next + 4 * count)
	var sent := 0
	var r = ""
	var t0 := Time.get_ticks_usec()
	while _pen_next < end:
		var i := _pen_next
		r = _cn.vmcall("pen_point", _pen_id, _pen_src[i], _pen_src[i + 1], _pen_src[i + 2], _pen_src[i + 3])
		_pen_next += 4
		sent += 1
	return "host_us=%d sent=%d left=%d %s" % [Time.get_ticks_usec() - t0, sent,
			(_pen_src.size() - _pen_next) / 4, str(r)]

# id < 0: the scripted stroke's.
func pen_end(id: int = -1) -> String:
	var r := _cn_call("pen_end", [_pen_id if id < 0 else id])
	if id < 0:
		_pen_id = -1
	return r

func patch_count() -> String:
	return _cn_call("patch_count")

# Patch i over the wire: vertex and triangle counts (the buffers are
# mesh_wire's; mesh_array_mesh has the built mesh as an ArrayMesh).
func patch_vertices(i: int = 0) -> String:
	if _cn == null:
		return "FAIL: no curvenet sandbox"
	var v: PackedFloat32Array = _cn.vmcall("patch_vertices", i)
	return "patch %d: %d vertices" % [i, v.size() / 3]

func patch_indices(i: int = 0) -> String:
	if _cn == null:
		return "FAIL: no curvenet sandbox"
	var f: PackedInt32Array = _cn.vmcall("patch_indices", i)
	return "patch %d: %d triangles" % [i, f.size() / 3]

# The built mesh's source patch per triangle (-1 after a remesh).
func mesh_patch_ids() -> String:
	if _cn == null:
		return "FAIL: no curvenet sandbox"
	var ids: PackedInt32Array = _cn.vmcall("mesh_patch_ids")
	var per := {}
	for p in ids:
		per[p] = per.get(p, 0) + 1
	return "%d triangles, per patch %s" % [ids.size(), str(per)]

func check_names() -> String:
	return _cn_call("check_names")

# Gate 4's checks in the guest, one line each. pen_sphere resets the stage
# and clears the body.
func curvenet_checks() -> String:
	return _cn_call("check_all")

func curvenet_check(name: String = "pen_sphere") -> String:
	return _cn_call("check", [name])

func curvenet_build() -> String:
	var r := _cn_call("curvenet_build")
	var c := MeshWire.curves(_cn.vmcall("curvenet_curves")) if _cn != null else []
	var k := MeshWire.knots(_cn.vmcall("curvenet_knots")) if _cn != null else []
	return "%s | wire: %d curves, %d knots" % [r, c.size(), k.size()]

# Merge + weld the active patches; target_edge_length > 0 PMP-remeshes.
func mesh_build(target_edge_length: float = 0.02, weld_eps: float = 1e-5) -> String:
	var r := _cn_call("mesh_build", [target_edge_length, weld_eps])
	if _cn == null:
		return r
	var loops := MeshWire.loops(_cn.vmcall("mesh_boundary_loops"))
	var v: PackedFloat32Array = _cn.vmcall("mesh_vertices")
	return "%s | wire: %d vertices, %d boundary loops" % [r, v.size() / 3, loops.size()]

# The built mesh as an ArrayMesh (Godot winding), for a MeshInstance3D.
func mesh_array_mesh() -> ArrayMesh:
	if _cn == null:
		return ArrayMesh.new()
	return MeshWire.to_array_mesh(_cn.vmcall("mesh_vertices"), _cn.vmcall("mesh_indices"))

# Mesh -> curvenet on a unit cube: 12 curves on 8 knots.
func curvenet_extract_demo() -> String:
	var c := MeshWire.cube()
	var r := _cn_call("curvenet_extract", [c.vertices, c.triangles, 200, 1e-3, 1e-2, 0.0])
	return r

# --- Cut 5: the drape (drape.elf), DiffCloth's sphere demo and host meshes -----
# One session, advanced one frame at a time by _process (rule 4). Every guest
# entry point has a wrapper here with defaults, so MCP call_method needs no
# argument marshalling (rule 8): drape_sphere_demo, drape_forward, then poll
# drape_status; drape_backward after a drape_target.

var _drape_status := "IDLE no session"
var _drape_job_status := ""
var _drape_job_on := false

func _process(_delta: float) -> void:
	if _drape == null:
		return
	if not _drape_status.begins_with("IDLE"):
		_drape_status = str(_drape.vmcall("drape_tick", Time.get_ticks_usec()))
	if _drape_job_on:
		_drape_job_status = str(_drape.vmcall("drape_job_tick", Time.get_ticks_usec()))
		if not _drape_job_status.begins_with("RUNNING"):
			_drape_job_on = false

func _dv(name: String, args: Array = []) -> String:
	if _drape == null:
		return "FAIL: no drape sandbox"
	return str(_drape.callv("vmcall", [name] + args))

# backend: cpu | rd | auto (rd from 160 vertices at 90 fps: Gate 5 G9).
func drape_open(backend: String = "auto") -> String:
	return _dv("drape_open", [backend])

func drape_sphere_demo(backend: String = "auto") -> String:
	var o := drape_open(backend)
	if not o.begins_with("OPENED"):
		return o
	return _dv("drape_scene_sphere_demo")

func drape_scene_mesh(positions: PackedFloat32Array = PackedFloat32Array(), triangles: PackedInt32Array = PackedInt32Array(),
		pins: PackedInt32Array = PackedInt32Array(), material: PackedFloat32Array = PackedFloat32Array()) -> String:
	return _dv("drape_scene_mesh", [positions, triangles, pins, material])

# kind: sphere [c, r, mu] | plane [c, ul, ur, mu] | capsule [b, axis, r, len, mu] | clear.
func drape_primitive(kind: String = "clear", params: PackedFloat32Array = PackedFloat32Array()) -> String:
	return _dv("drape_primitive", [kind, params])

# A triangle-mesh body collider (positions in drape units, like the capsules);
# params [skin, mu, band, depth], any prefix (0.1, 0.3, 0.1, 1.0).
func drape_primitive_mesh(positions: PackedFloat32Array = PackedFloat32Array(), triangles: PackedInt32Array = PackedInt32Array(),
		params: PackedFloat32Array = PackedFloat32Array()) -> String:
	return _dv("drape_primitive_mesh", [positions, triangles, params])

func drape_config(key: String = "iters", value: float = 16.0) -> String:
	return _dv("drape_config", [key, value])

# steps > 0 queues steps; 0 rewinds to the initial state.
func drape_forward(steps: int = 100) -> String:
	var r := _dv("drape_queue_forward", [steps])
	if r.begins_with("QUEUED"):
		_drape_status = "RUNNING"
	return r

# kind: trajectory (the recorded frames become the target) | points | clear.
func drape_target(kind: String = "trajectory", verts: PackedInt32Array = PackedInt32Array(),
		positions: PackedFloat32Array = PackedFloat32Array(), frame: int = -1) -> String:
	return _dv("drape_set_target", [kind, verts, positions, frame])

# loss: match_trajectory | target_points; mode: native | step | unrolled.
func drape_backward(loss: String = "match_trajectory", mode: String = "unrolled") -> String:
	var r := _dv("drape_queue_backward", [loss, mode])
	if r.begins_with("QUEUED"):
		_drape_status = "RUNNING"
	return r

# RUNNING k/N while the queue runs, then IDLE and the last result's first line.
func drape_status() -> String:
	return _drape_status

func drape_result() -> String:
	return _dv("drape_result")

func drape_positions() -> PackedFloat32Array:
	return _drape.vmcall("drape_positions") if _drape != null else PackedFloat32Array()

func drape_frame(i: int = 0) -> PackedFloat32Array:
	return _drape.vmcall("drape_frame", i) if _drape != null else PackedFloat32Array()

func drape_faces() -> PackedInt32Array:
	return _drape.vmcall("drape_faces") if _drape != null else PackedInt32Array()

# Gate 5 jobs: sphere_forward, sphere_backward, sim_gradcheck, bench_drape,
# inverse_min, lbfgsb_components, lbfgsb_problems, lbfgsb_replay, lbfgsb_bench.
func drape_job(name: String = "sphere_forward", backend: String = "auto", args: String = "") -> String:
	var r := _dv("drape_job_start", [name, backend, args])
	_drape_job_on = r.begins_with("STARTED")
	_drape_job_status = r
	return r

func drape_job_result() -> String:
	return _drape_job_status

func drape_job_frame(i: int = 0) -> PackedFloat32Array:
	return _drape.vmcall("drape_job_frame", i) if _drape != null else PackedFloat32Array()

func drape_job_names() -> String:
	return _dv("drape_job_names")

# L-BFGS-B over the session's parameters (drape.elf's drape_queue_optimize):
# spec "params=mu[,kTri,...] loss=match_trajectory mode=native steps=N vec=cpu
# m=10 delta=1e-3 ...", one x0/lb/ub value per parameter, max_iter 0 = to
# convergence. Poll drape_status, then drape_optimize_result.
func drape_optimize(spec: String = "params=mu mode=native", x0: PackedFloat32Array = PackedFloat32Array([0.5]),
		lb: PackedFloat32Array = PackedFloat32Array([0.01]), ub: PackedFloat32Array = PackedFloat32Array([1.0]),
		max_iter: int = 10) -> String:
	var r := _dv("drape_queue_optimize", [spec, x0, lb, ub, max_iter])
	if r.begins_with("QUEUED"):
		_drape_status = "RUNNING"
	return r

func drape_optimize_result() -> String:
	return _dv("drape_optimize_result")

# Hand the drape jobs a data file by key ("clear" drops them all).
func drape_job_data(key: String = "clear", text: String = "") -> String:
	return _dv("drape_job_data", [key, text])

# The Gate 5 oracle (gates/5-drape/oracle) into drape.elf, for the jobs
# lbfgsb_components, lbfgsb_problems and inverse_min.
func lbfgsb_load_oracle() -> String:
	var root := ProjectSettings.globalize_path("res://../gates/5-drape/oracle/")
	var r := drape_job_data("clear", "")
	for f in ["k_tri", "k_bend_density"]:
		r = drape_job_data("invmin_" + f, FileAccess.get_file_as_string(root + "inverse_min/case_" + f + ".txt"))
	for sub in [["components", ""], ["problems", "prob_"], ["traces", "trace_"]]:
		var d := DirAccess.open(root + sub[0])
		if d == null:
			return "FAIL: no " + root + sub[0]
		for f in d.get_files():
			if f.ends_with(".txt"):
				r = drape_job_data(sub[1] + f.get_basename(), FileAccess.get_file_as_string(root + sub[0] + "/" + f))
	return r

# --- The fit stage (fit.elf): cloth-fit's garment solve ---------------------------
# fit.elf is built by build.sh (BUILD_FIT=0 skips it) and is not committed, so
# it is optional here. A phase runs for seconds to minutes, so fit_step and
# fit_run_all run the vmcall on a worker Thread (Gate 0F probe 7) and return at
# once; fit_status polls (rule 4: no waits on the main thread). While a call is
# in flight every other fit call answers BUSY: one VM, one vmcall at a time.

var _fit_thread: Thread = null
var _fit_call := ""
var _fit_t0 := 0
var _fit_last := ""  # the last worker call's answer

# The fit Sandbox's limits, applied when it is (re)created (fit_configure).
# memory_max in MiB, set before program= (a lower value later is ignored,
# Gate 0F probe 6); the guest heap is 0.8 x memory_max. Gate 6.P's ladder on
# foxgirl (phases 0+1): 352 passes, 320 kills the Godot process (segfault),
# 224-256 abort the vmcall (Protection fault in the malloc ecall) although the
# heap never holds more than 124 MiB at a phase end. The default is 1.25x the
# floor, where the full four-phase run was confirmed bit for bit.
const FIT_MEMORY_FLOOR_MIB := 352
var fit_memory_mib := 440
var fit_elf := "res://fit.elf"
# execution_timeout counts 2^20-instruction units (Gate 0F probe 5; the default
# 8000 is ~8.4e9 instructions). Gate 6.P: the largest foxgirl phase retires
# 5.62e11 instructions (535,568 units, phase 3); 2,500,000 is 4.7x that, so one
# phase fits one vmcall and a runaway solve still stops (~1 h at ~0.75 G
# instructions/s).
var fit_execution_timeout := 2500000
# Top-level keys of the setup JSON replaced as text before fit_set_config
# (key -> JSON literal), e.g. {"fit_weight": "0"} for Gate 6's fit-gap control.
var fit_config_overrides := {}

func _fit_load() -> void:
	if not ResourceLoader.exists(fit_elf):
		print("[dress-on] no %s (build.sh with BUILD_FIT=1, then --import)" % fit_elf)
		return
	_fit = ClassDB.instantiate("Sandbox")
	add_child(_fit)
	_fit.memory_max = fit_memory_mib
	_fit.references_max = 4096
	# Live heap chunks: the default 10000 is exhausted inside fit_begin
	# ("Too many arena chunks (data: 10000)", from a robin_set in ipc-toolkit's
	# collision mesh); PolyFEM keeps far more small allocations alive.
	_fit.allocations_max = 4000000
	_fit.execution_timeout = fit_execution_timeout
	_fit.program = load(fit_elf)
	print("[dress-on] sandbox loaded %s (memory_max %d MiB, execution_timeout %d)" % [fit_elf, fit_memory_mib, fit_execution_timeout])

# A fresh fit Sandbox with the current fit_memory_mib / fit_elf /
# fit_execution_timeout (Gate 6.P: one Sandbox per ladder arm). Drops every
# input and the driver.
func fit_configure() -> String:
	var busy := _fit_busy() if _fit != null else ""
	if busy != "":
		return busy
	if _fit != null:
		remove_child(_fit)
		_fit.free()
		_fit = null
	_fit_last = ""
	_fit_load()
	if _fit == null:
		return "FAIL: no %s" % fit_elf
	return "OK fit sandbox %s memory_max %d MiB execution_timeout %d" % [fit_elf, fit_memory_mib, fit_execution_timeout]

func fit_configure_with(memory_mib: int, elf: String = "res://fit.elf", execution_timeout: int = -1) -> String:
	fit_memory_mib = memory_mib
	fit_elf = elf
	if execution_timeout > 0:
		fit_execution_timeout = execution_timeout
	return fit_configure()

func _apply_overrides(cfg_text: String) -> String:
	for k in fit_config_overrides:
		var re := RegEx.new()
		re.compile("(\"%s\"\\s*:\\s*)[^,}\\n]+" % k)
		if re.search(cfg_text) == null:
			cfg_text = cfg_text.replace("{", "{\"%s\": %s, " % [k, fit_config_overrides[k]])
		else:
			cfg_text = re.sub(cfg_text, "${1}" + str(fit_config_overrides[k]))
	return cfg_text

func _fit_busy() -> String:
	if _fit == null:
		return "FAIL: no fit sandbox (fit.elf not built or not imported)"
	if _fit_thread != null:
		if _fit_thread.is_alive():
			return "BUSY %s for %.1f s" % [_fit_call, (Time.get_ticks_msec() - _fit_t0) / 1000.0]
		_fit_last = str(_fit_thread.wait_to_finish())
		_fit_thread = null
	return ""

func _fit_worker(method: String) -> String:
	var t0 := Time.get_ticks_usec()
	var r = _fit.vmcall(method)
	return "host_ms=%d %s" % [(Time.get_ticks_usec() - t0) / 1000, str(r)]

func _fit_start(method: String) -> String:
	var busy := _fit_busy()
	if busy != "":
		return busy
	_fit_thread = Thread.new()
	_fit_call = method
	_fit_t0 = Time.get_ticks_msec()
	_fit_thread.start(_fit_worker.bind(method))
	return "STARTED %s (poll fit_status)" % method

func _fit_now(method: String, args: Array = []) -> String:
	var busy := _fit_busy()
	if busy != "":
		return busy
	return str(_fit.callv("vmcall", [method] + args))

func _repo_path(rel: String) -> String:
	return ProjectSettings.globalize_path("res://").path_join("..").path_join(rel).simplify_path()

# Loads foxgirl_skirt as the native oracle does (tools/native/foxgirl_oracle.json:
# the FoxGirl avatar and skeleton, LCL_Skirt_DressEvening_003 with its skeleton
# and no-fit list) through util/obj_io.gd and hands it to fit.elf. Positions go
# as float32, as fit_native rounds them by default. Then call fit_begin.
func fit_fixture_foxgirl() -> String:
	var busy := _fit_busy()
	if busy != "":
		return busy
	var a := foxgirl_arrays()
	if a.has("error"):
		return "FAIL: " + str(a["error"])
	var out := PackedStringArray()
	out.append(str(_fit.vmcall("fit_reset")))
	out.append(str(_fit.vmcall("fit_set_body", a["body_v"], a["body_f"])))
	out.append(str(_fit.vmcall("fit_set_skeletons", a["src_sk_v"], a["tgt_sk_v"], a["bones"])))
	out.append(str(_fit.vmcall("fit_set_garment", a["garment_v"], a["garment_f"], a["nofit"])))
	# The config goes as text: a GDScript round trip would turn 2 into 2.0,
	# which the spec's integer fields refuse. The guest drops the *_path keys.
	out.append(str(_fit.vmcall("fit_set_config", _apply_overrides(a["cfg_text"]))))
	if not fit_config_overrides.is_empty():
		out.append("overrides %s" % str(fit_config_overrides))
	return " | ".join(out)

# The foxgirl fixture as the packed arrays that cross the wire (Gate 6.0's
# wire check holds them to fit_native --dump-inputs bit for bit).
func foxgirl_arrays() -> Dictionary:
	var cfg_path := _repo_path("tools/native/foxgirl_oracle.json")
	var cfg_text := FileAccess.get_file_as_string(cfg_path)
	if cfg_text.is_empty():
		return {"error": "cannot read %s" % cfg_path}
	var cfg = JSON.parse_string(cfg_text)
	if typeof(cfg) != TYPE_DICTIONARY:
		return {"error": "%s is not a JSON object" % cfg_path}
	var body: Dictionary = ObjIO.read(_repo_path(cfg["avatar_mesh_path"]))
	var garment: Dictionary = ObjIO.read(_repo_path(cfg["garment_mesh_path"]))
	var src_sk: Dictionary = ObjIO.read(_repo_path(cfg["source_skeleton_path"]))
	var tgt_sk: Dictionary = ObjIO.read(_repo_path(cfg["target_skeleton_path"]))
	for m in [body, garment, src_sk, tgt_sk]:
		if m.has("error"):
			return {"error": m["error"]}
	if src_sk["l"] != tgt_sk["l"]:
		return {"error": "source and target skeletons have different bones"}
	var nofit := PackedInt32Array()
	if str(cfg.get("no_fit_spec_path", "")) != "":
		var r: Dictionary = ObjIO.read_ints(_repo_path(cfg["no_fit_spec_path"]))
		if r.has("error"):
			return {"error": r["error"]}
		nofit = r["ints"]
	return {"cfg_text": cfg_text, "body_v": body["v"], "body_f": body["f"], "garment_v": garment["v"],
			"garment_f": garment["f"], "src_sk_v": src_sk["v"], "tgt_sk_v": tgt_sk["v"], "bones": src_sk["l"],
			"nofit": nofit}

func fit_reset() -> String:
	return _fit_now("fit_reset")

func fit_begin() -> String:
	return _fit_now("fit_begin")

# One phase (AL solve or reduced solve) on the worker thread.
func fit_step() -> String:
	return _fit_start("fit_step")

# Every remaining phase in one vmcall, on the worker thread.
func fit_run_all() -> String:
	return _fit_start("fit_run_all")

# The guest's status (phase, Newton iterations, energy, io_attempts, heap) plus
# the host's view: the Sandbox heap reading and the last worker answer.
func fit_status() -> String:
	var busy := _fit_busy()
	if busy != "":
		return busy
	var host_heap := ""
	if _fit.has_method("get_heap_usage"):
		host_heap = " host_heap_usage=%d" % _fit.get_heap_usage()
	return "%s%s | last: %s" % [str(_fit.vmcall("fit_status")), host_heap, _fit_last]

# Skin weights for the target avatar (J x N, rows are joints); the FoxGirl
# fixture has none, so the default clears them.
func fit_set_skin_weights(weights: PackedFloat32Array = PackedFloat32Array()) -> String:
	return _fit_now("fit_set_skin_weights", [weights])

# Intersection check on the current state.
func fit_check() -> String:
	return _fit_now("fit_check_intersections", [PackedFloat32Array()])

# The current garment: vertex count and body-space bounds (GDScript callers
# take the arrays from fit_result_vertices / fit_result_vertices_f64).
func fit_result() -> String:
	var busy := _fit_busy()
	if busy != "":
		return busy
	var v = _fit.vmcall("fit_result_vertices")
	if typeof(v) != TYPE_PACKED_FLOAT32_ARRAY:
		return str(v)
	return "garment %d v, bounds %s" % [v.size() / 3, str(_bounds(v))]

func fit_result_vertices() -> Variant:
	var busy := _fit_busy()
	return busy if busy != "" else _fit.vmcall("fit_result_vertices")

func fit_result_vertices_f64() -> Variant:
	var busy := _fit_busy()
	return busy if busy != "" else _fit.vmcall("fit_result_vertices_f64")

# The newest preview snapshot (every Newton iteration by default).
func fit_preview() -> String:
	var busy := _fit_busy()
	if busy != "":
		return busy
	var v = _fit.vmcall("fit_preview", 0)
	if typeof(v) != TYPE_PACKED_FLOAT32_ARRAY:
		return str(v)
	return "preview %d v, bounds %s" % [v.size() / 3, str(_bounds(v))]

# The SDF (Lean kernel) at the current garment's vertices: the value range in
# voxels (the grid stores solve-frame distances, clamped to [-1, 150] voxels)
# and how many vertices are inside the body.
func fit_sdf() -> String:
	var busy := _fit_busy()
	if busy != "":
		return busy
	var g = _fit.vmcall("fit_result_vertices_f64")
	if typeof(g) != TYPE_PACKED_FLOAT64_ARRAY:
		return str(g)
	var d = _fit.vmcall("fit_sdf_dump", g)
	if typeof(d) != TYPE_PACKED_FLOAT64_ARRAY:
		return str(d)
	var n: int = d.size() / 10
	var h := 0.01
	var cfg = JSON.parse_string(FileAccess.get_file_as_string(_repo_path("tools/native/foxgirl_oracle.json")))
	if typeof(cfg) == TYPE_DICTIONARY and cfg.has("voxel_size"):
		h = float(cfg["voxel_size"])
	var lo := INF
	var hi := -INF
	var inside := 0
	for i in n:
		var x: float = d[10 * i] / h
		lo = minf(lo, x)
		hi = maxf(hi, x)
		if x < 0.0:
			inside += 1
	return "sdf at %d garment vertices: min %.4f max %.4f voxels, %d inside" % [n, lo, hi, inside]

func fit_probe_io() -> String:
	return _fit_now("fit_probe", ["io"])

func fit_probe_ldlt() -> String:
	return _fit_now("fit_probe", ["ldlt"])

func fit_probe_exceptions() -> String:
	return _fit_now("fit_probe", ["exceptions"])

func fit_probe_io_paths() -> String:
	return _fit_now("fit_probe", ["io_paths"])

# polysolve's SimplicialLDLT on an 8000-unknown 3D Laplacian, host-timed
# around the vmcall (the guest clock is not a clock).
func fit_probe_ldlt8k() -> String:
	var t0 := Time.get_ticks_usec()
	var r := _fit_now("fit_probe", ["ldlt8k"])
	return "host_us=%d %s" % [Time.get_ticks_usec() - t0, r]

# libm result hashes (fit_probes.cpp); fit_native --probe libm prints its own.
func fit_probe_libm() -> String:
	return _fit_now("fit_probe", ["libm"])

# Tie order of std::sort / nth_element / partial_sort and a sum in that order
# (fit_probes.cpp); fit_native --probe stl prints its own.
func fit_probe_stl() -> String:
	return _fit_now("fit_probe", ["stl"])

# The instret CSR advances (Gate 6.P reads a phase's instruction count with it).
func fit_probe_instret() -> String:
	return _fit_now("fit_probe", ["instret"])

func fit_probe_heap() -> String:
	return _fit_now("fit_probe", ["heap"])

# The intersection check's positive control: the garment's closest vertex
# moved 5 cm (0.05 solve units, 5 voxels) into the avatar along -grad SDF,
# through fit_check_intersections. Must say INTERSECTS.
func fit_push_control() -> String:
	return _fit_push(0.05)

# Its flat control: the same path with no push. Must say none.
func fit_push_flat() -> String:
	return _fit_push(0.0)

func _fit_push(dist: float) -> String:
	var busy := _fit_busy()
	if busy != "":
		return busy
	var cur = _fit.vmcall("fit_result_vertices")
	var pushed = _fit.vmcall("fit_push_vertex", dist)
	if typeof(pushed) != TYPE_PACKED_FLOAT32_ARRAY or typeof(cur) != TYPE_PACKED_FLOAT32_ARRAY:
		return "FAIL: %s / %s" % [str(pushed).left(200), str(cur).left(200)]
	var moved := -1
	var by := 0.0
	for i in range(0, cur.size(), 3):
		var d := Vector3(pushed[i] - cur[i], pushed[i + 1] - cur[i + 1], pushed[i + 2] - cur[i + 2]).length()
		if d > by:
			by = d
			moved = i / 3
	var r := str(_fit.vmcall("fit_check_intersections", pushed))
	return "push %.3f solve units: vertex %d moved %.5f body units -> %s" % [dist, moved, by, r]

func _bounds(v: PackedFloat32Array) -> AABB:
	if v.size() < 3:
		return AABB()
	var box := AABB(Vector3(v[0], v[1], v[2]), Vector3.ZERO)
	for i in range(0, v.size(), 3):
		box = box.expand(Vector3(v[i], v[i + 1], v[i + 2]))
	return box

func _exit_tree() -> void:
	if _fit_thread != null:
		_fit_thread.wait_to_finish()
