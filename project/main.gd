# The host side of interactor-dress-on. Owns the sandbox guests and exposes
# plain methods that an MCP client can reach with call_method. The guest APIs
# take typed buffers; wrapping here keeps those off the JSON wire. Stages are
# separate ELFs in separate Sandbox nodes, composed here; meshes cross between
# them as packed arrays.
extends Node

var _sb = null     # dress_on.elf: the Stage 1 GPU-layer probes
var _drape = null  # drape.elf: the AVBD solver, cpu and rd

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

# backend: cpu | rd | auto (rd from 256 vertices).
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

# Gate 5 drape jobs: sphere_forward, sphere_backward, sim_gradcheck, bench_drape.
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
