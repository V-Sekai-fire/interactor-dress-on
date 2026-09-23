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
func p_list_end() -> String: return _pv("p_list_end")
func f16_read() -> String:
	var h := PackedByteArray()
	h.resize(128)
	for i in 64:
		h.encode_u16(2 * i, 0x3C00 + i) # 1.0 upward
	return _pv("f16_read", [h])
func ggml_probe(n: int = 256) -> String: return _pv("ggml_probe", [n])
func zfh_probe() -> String: return _pv("zfh_probe")
