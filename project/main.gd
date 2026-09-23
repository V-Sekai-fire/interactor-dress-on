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
func p_recovery(on: bool = true) -> String: return _pv("p_recovery", [on])
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

# --- Cut 3: ggml_test.elf, ggml-rd under test-backend-ops ----------------------
# Its own Sandbox, created on first use, with a host-owned local
# RenderingDevice that the guest adopts. A job (test-backend-ops, or a probe)
# runs on a guest fiber; _process pumps it once per frame through
# infer_host.gd (WAIT_GPU / COOP / READ / UPLOAD / DONE / ERROR), so a sync
# always lands a frame after its submit (rule 4). gate_ggml_rd.gd is the
# gate. Every entry point has a wrapper of its own name with its arguments
# defaulted (AGENTS.md rule 8), and the gate's runs have presets; start one,
# then poll ggml_job_status() until it is not RUNNING.

const GGML_TOTAL_MB := 24576  # what ggml-rd reports as device memory (Godot has no call for it)

var _ggml = null                      # the Sandbox running ggml_test.elf
var _ggml_rd: RenderingDevice = null  # the device the guest adopted
var _ggml_host = null                 # InferHost: pumps the running job
var _ggml_pumped_frame := -1          # at most one pump per frame (rule 4)

func ggml_attach(total_mb: int = GGML_TOTAL_MB) -> String:
	if _ggml != null:
		return "ATTACHED (ggml_rd_close first to re-attach)"
	var sb = ClassDB.instantiate("Sandbox")
	if sb == null:
		return "FAIL: no Sandbox class"
	_ggml = sb
	add_child(_ggml)
	_ggml.memory_max = 2048  # before program= (Gate 0F): whole test tensors live in the heap
	_ggml.program = load("res://ggml_test.elf")
	_ggml.references_max = 65536
	_ggml.execution_timeout = 1000000  # host calls are charged against it (Gate 0F finding 6)
	_ggml_rd = RenderingServer.create_local_rendering_device()
	var r := str(_ggml.vmcall("ggml_attach", _ggml_rd, total_mb))
	_ggml_host = preload("res://infer_host.gd").new(_ggml, _ggml_rd, "ggml_pump")
	_ggml_host.state = "done"
	return r

func _ggml_started(r: String) -> String:
	if r.begins_with("STARTED"):
		_ggml_host.reset()
	return r

func ggml_ops_start(args: String = "-o ADD,MUL -b RD0", env: String = "") -> String:
	var e := "" if _ggml != null else ggml_attach()
	if _ggml == null:
		return e
	return _ggml_started(str(_ggml.vmcall("ggml_ops_start", args, env)))

func ggml_probe_start(name: String = "chain", arg: String = "256", env: String = "") -> String:
	var e := "" if _ggml != null else ggml_attach()
	if _ggml == null:
		return e
	return _ggml_started(str(_ggml.vmcall("ggml_probe_start", name, arg, env)))

# One pump, at most once per frame; _process calls it too.
func ggml_pump() -> String:
	var f := Engine.get_process_frames()
	if _ggml_host != null and _ggml_host.state == "running" and f != _ggml_pumped_frame:
		_ggml_pumped_frame = f
		_ggml_host.pump_frame()
	return ggml_job_status()

func ggml_output() -> String:
	return str(_ggml.vmcall("ggml_output")) if _ggml != null else "IDLE"

func ggml_rd_stats() -> String:
	return str(_ggml.vmcall("ggml_rd_stats")) if _ggml != null else "IDLE"

func ggml_rd_close() -> String:
	if _ggml == null:
		return "IDLE"
	var r := str(_ggml.vmcall("ggml_rd_close"))
	if r.begins_with("CLOSED"):
		_ggml.queue_free()
		_ggml = null
		_ggml_host = null
		if _ggml_rd != null:
			_ggml_rd.free()
			_ggml_rd = null
	return r

# "RUNNING ...", "DONE ..." or "ERROR ...", with the pump counters.
func ggml_job_status() -> String:
	if _ggml_host == null:
		return "IDLE"
	return "%s %s%s" % [_ggml_host.state.to_upper(), _ggml_host.summary(),
			(" " + _ggml_host.text) if _ggml_host.text != "" else ""]

# The gate's runs (gate_ggml_rd.gd), as presets.
func ggml_ops_add_mul() -> String: return ggml_ops_start("-o ADD,MUL -b RD0", "")
func ggml_ops_barrier_all() -> String: return ggml_ops_start("-o ADD,MUL -b RD0", "GGML_RD_BARRIER_ALL=1")
func ggml_ops_fault() -> String: return ggml_ops_start("-o ADD -b RD0", "GGML_RD_FAULT=1")  # must FAIL
func ggml_probe_chain() -> String: return ggml_probe_start("chain", "256", "")
func ggml_probe_independent() -> String: return ggml_probe_start("independent", "64", "")
func ggml_probe_alias_rw() -> String: return ggml_probe_start("alias", "rw", "")
func ggml_probe_alias_ro() -> String: return ggml_probe_start("alias", "ro", "")  # the control: must lose counts
# The data-movement family (CPY/DUP/CONT, GET_ROWS, CONCAT, REPEAT).
func ggml_ops_move() -> String: return ggml_ops_start("-o CPY,DUP,CONT,GET_ROWS,CONCAT,REPEAT -b RD0", "")
func ggml_ops_move_fault() -> String: return ggml_ops_start("-o DUP,CONT,GET_ROWS,CONCAT,REPEAT -p ^(?!type=i32,) -b RD0", "GGML_RD_FAULT=1")  # must FAIL
func ggml_probe_perf() -> String: return ggml_probe_start("perf", "move", "")  # GPU time per op, hot shapes
func ggml_probe_files() -> String:
	var path := ProjectSettings.globalize_path("user://ggml_upload_probe.f32")
	var f := FileAccess.open(path, FileAccess.WRITE)
	for i in 4096:
		f.store_float(0.5 * i - 7.25)
	f.close()
	return ggml_probe_start("files", path, "")

func _process(_delta: float) -> void:
	if _ggml_host != null and _ggml_host.state == "running":
		ggml_pump()
