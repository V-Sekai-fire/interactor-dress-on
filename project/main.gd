# The host side of interactor-dress-on. Owns the sandbox guests and exposes
# plain methods that an MCP client can reach with call_method. The guest APIs
# take typed buffers; wrapping here keeps those off the JSON wire. Stages are
# separate ELFs in separate Sandbox nodes, composed here; meshes cross between
# them as packed arrays.
extends Node

var _sb = null     # dress_on.elf: the Stage 1 GPU-layer probes
var _drape = null  # drape.elf: the AVBD solver, cpu and rd
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
