# Gate 0F runner: what the godot-sandbox runtime does, one probe at a time.
#
#   godot --path project --script gate_runtime.gd --rendering-driver vulkan --xr-mode off
#   ... -- --only=5,6            # run a subset (probe numbers)
#
# Frame-driven: each probe is a step that _process advances until it says it
# is done, so the probes that need frames (a vmcall on a Thread, the fiber's
# one-resume-per-frame job) get real frames. Every probe prints PASS / FAIL /
# INFO / DEFERRED lines with its control; results stream to
# gates/0f-runtime/results.txt (stdout is buffered when redirected). The run
# quits on a wall clock in every branch.
extends SceneTree

const RESULTS := "res://../gates/0f-runtime/results.txt"
const GGML_HOST := "res://../gates/0f-runtime/ggml_host.txt"
const WALL_MS := 30 * 60 * 1000
const LCG_A := 6364136223846793005
const LCG_C := 1442695040888963407
const LCG_SEED := -7046029254386353131 # 0x9E3779B97F4A7C15 as int64

var _out: FileAccess
var _elf: Resource
var _t0 := 0
var _steps: Array = []
var _i := 0
var _st := {}
var _only := []
var _counts := {"PASS": 0, "FAIL": 0, "INFO": 0, "DEFERRED": 0}
var _spin_per_s := 0.0
var _gpu = null # the Sandbox the GPU probes share (11, 12, 13, 14)

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _v(num: int, name: String, verdict: String, detail: String) -> void:
	_counts[verdict] = _counts.get(verdict, 0) + 1
	_say("P%02d %-14s %-8s %s" % [num, name, verdict, detail])

func _sb(refs := 4096):
	var sb = ClassDB.instantiate("Sandbox")
	sb.program = _elf
	sb.references_max = refs
	return sb

func _us() -> int:
	return Time.get_ticks_usec()

func _lcg(n: int) -> int:
	var s := LCG_SEED
	for i in n:
		s = s * LCG_A + LCG_C
	return s

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	_out = FileAccess.open(ProjectSettings.globalize_path(RESULTS), FileAccess.WRITE)
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--only="):
			for x in a.substr(7).split(","):
				_only.append(int(x))
	_say("Gate 0F -- sandbox runtime probes. Godot %s, %s" % [Engine.get_version_info().string, OS.get_processor_name()])
	_elf = load("res://probes.elf")
	if _elf == null or ClassDB.instantiate("Sandbox") == null:
		_say("FAIL: probes.elf or the Sandbox class is missing")
		_finish()
		return
	_steps = [
		[1, _p01_exceptions], [2, _p02_fenv], [3, _p03_files], [4, _p04_threads],
		[5, _p05_timeout], [9, _p09_echo], [10, _p10_heap],
		[11, _p11_fiber], [12, _p12_big_buffers], [13, _p13_refs], [14, _p14_f16],
		[15, _p15_ggml], [7, _p07_thread_vmcall], [8, _p08_two_sandboxes], [6, _p06_memory],
	]

func _process(_delta: float) -> bool:
	if Time.get_ticks_msec() - _t0 > WALL_MS:
		_say("TIMEOUT: wall clock %d s passed at step %d" % [WALL_MS / 1000, _i])
		_finish()
		return true
	while _i < _steps.size() and not _only.is_empty() and not _only.has(_steps[_i][0]):
		_i += 1
	if _i >= _steps.size():
		_finish()
		return true
	var done: bool = _steps[_i][1].call()
	if done:
		_i += 1
		_st = {}
	return false

func _finish() -> void:
	_say("SUMMARY: PASS=%d FAIL=%d INFO=%d DEFERRED=%d wall_s=%.1f" % [_counts.PASS, _counts.FAIL,
		_counts.INFO, _counts.DEFERRED, (Time.get_ticks_msec() - _t0) / 1000.0])
	if _gpu != null:
		_gpu.free()
		_gpu = null
	if _out != null:
		_out.close()
		_out = null
	quit(0)

# --- 1. exceptions ------------------------------------------------------------

func _p01_exceptions() -> bool:
	var sb = _sb()
	var thrown = sb.vmcall("p_exceptions", true)
	var ctrl = sb.vmcall("p_exceptions", false)
	var ok_t: bool = typeof(thrown) == TYPE_STRING and thrown == "caught runtime_error: negative input -7 rethrow_sum=84"
	var ok_c: bool = typeof(ctrl) == TYPE_STRING and ctrl == "returned value=11 rethrow_sum=0"
	_v(1, "exceptions", "PASS" if ok_t else "FAIL", "throw through std::function x2 + rethrow: %s" % str(thrown))
	_v(1, "exceptions", "PASS" if ok_c else "FAIL", "control, no throw: %s" % str(ctrl))
	sb.free()
	return true

# --- 2. fesetround --------------------------------------------------------------

func _p02_fenv() -> bool:
	var sb = _sb()
	var r = sb.vmcall("p_fenv")
	var s := str(r)
	var honoured := s.find("double_differs=yes") >= 0
	_v(2, "fesetround", "INFO", "%s: %s" % ["HONOURED (1/3 rounds differently up vs down)" if honoured
		else "NOT honoured (1/3 identical under FE_UPWARD and FE_DOWNWARD)", s])
	sb.free()
	return true

# --- 3. files -------------------------------------------------------------------

func _p03_files() -> bool:
	var sb = _sb()
	var ctrl := FileAccess.get_file_as_bytes("res://project.godot")
	_v(3, "files", "PASS" if ctrl.size() > 0 else "FAIL", "control, GDScript FileAccess res://project.godot bytes=%d" % ctrl.size())
	var abs_path := ProjectSettings.globalize_path("res://project.godot")
	for p in [abs_path, abs_path.replace("/", "\\"), "res://project.godot", "project.godot", "/etc/hostname"]:
		var r = sb.vmcall("p_file", p)
		var s := str(r)
		var ok := s.find("ifstream=ok bytes=%d" % ctrl.size()) >= 0 and s.find("fopen=ok") >= 0
		_v(3, "files", "PASS" if ok else "FAIL", "guest ifstream+fopen %s -> %s" % [p, s])
	sb.free()
	return true

# --- 4. threads -----------------------------------------------------------------

func _p04_threads() -> bool:
	var sb = _sb()
	var r = sb.vmcall("p_threads")
	var s := str(r)
	_v(4, "threads", "PASS" if s.find("spawn=ok join=ok value=42") >= 0 else "FAIL", s)
	sb.free()
	return true

# --- 5. execution_timeout / instructions_max ----------------------------------

func _p05_timeout() -> bool:
	var sb = _sb()
	var def_to: int = sb.execution_timeout
	var def_im: int = sb.get_instructions_max()
	sb.execution_timeout = 123
	var alias: bool = sb.get_instructions_max() == 123
	sb.execution_timeout = def_to
	_v(5, "timeout", "INFO", "defaults: execution_timeout=%d get_instructions_max=%d; execution_timeout is instructions_max: %s; jit=%s binary_translated=%s" % [
		def_to, def_im, alias, sb.is_jit(), sb.is_binary_translated()])

	# Control: a short loop at the default limit completes, exactly.
	var n := 1000000
	var tc0: int = sb.monitor_execution_timeouts
	var t := _us()
	var r = sb.vmcall("p_spin", n)
	var dt := _us() - t
	var want := _lcg(n)
	_spin_per_s = n / (dt / 1e6)
	_v(5, "timeout", "PASS" if (typeof(r) == TYPE_INT and r == want and sb.monitor_execution_timeouts == tc0) else "FAIL",
		"control: p_spin(%d) at default -> %s (GDScript LCG %d) in %d us = %.1f M iter/s, timeouts +%d" % [
		n, str(r), want, dt, _spin_per_s / 1e6, sb.monitor_execution_timeouts - tc0])

	# A loop that cannot finish, at growing limits: killed? after how long?
	var killed_all := true
	var rows := []
	for lim in [1, 10, 100, 1000]:
		sb.execution_timeout = lim
		var before: int = sb.monitor_execution_timeouts
		var t1 := _us()
		var rr = sb.vmcall("p_spin", 1000000000000)
		var d1 := _us() - t1
		var killed: bool = sb.monitor_execution_timeouts == before + 1
		killed_all = killed_all and killed
		rows.append("lim=%d killed=%s host_ms=%.1f ret=%s ~iter=%.2fe9" % [lim, killed, d1 / 1000.0, str(rr), _spin_per_s * d1 / 1e6 / 1e9])
	_v(5, "timeout", "PASS" if killed_all else "FAIL", "busy loop 1e12 at execution_timeout lim: " + "; ".join(rows))
	sb.execution_timeout = def_to
	var after = sb.vmcall("p_spin", 10)
	_v(5, "timeout", "PASS" if after == _lcg(10) else "FAIL", "sandbox usable after a kill: p_spin(10) -> %s" % str(after))
	# The default limit against ~2 s of work.
	var n2 := int(_spin_per_s * 2.0)
	var b2: int = sb.monitor_execution_timeouts
	var t2 := _us()
	var r2 = sb.vmcall("p_spin", n2)
	var d2 := _us() - t2
	_v(5, "timeout", "INFO", "default execution_timeout=%d vs p_spin(%d) (~2 s): killed=%s host_ms=%.0f" % [
		def_to, n2, sb.monitor_execution_timeouts > b2, d2 / 1000.0])
	sb.free()
	return true

# --- 6. memory_max ladder ---------------------------------------------------------

func _p06_memory() -> bool:
	if not _st.has("init"):
		_st.init = true
		var fresh = ClassDB.instantiate("Sandbox")
		var def_before: int = fresh.memory_max
		fresh.program = _elf
		var def_after: int = fresh.memory_max
		fresh.free()
		var order = ClassDB.instantiate("Sandbox")
		order.memory_max = 2048
		order.program = _elf
		var kept: int = order.memory_max
		order.free()
		_v(6, "memory_max", "INFO", "default memory_max=%d (MiB) before program=, %d after; set 2048 then program= -> reads %d (%s)" % [
			def_before, def_after, kept, "RESET by loading" if kept != 2048 else "kept"])
		_st.xs = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
		_st.k = 0
		_st.max_ok = 0
		_st.max_ok_before = 0
		_st.all_ctrl = true
		return false
	if _st.k >= _st.xs.size():
		# The ceiling: memory_max >= 4096 fails to load ("Native heap exceeds
		# 32-bit address range"), so probe just under it.
		var rows := []
		var top := 0
		for x in [3072, 3584, 3968]:
			var sb = ClassDB.instantiate("Sandbox")
			sb.program = _elf
			sb.memory_max = 4095
			var r = sb.vmcall("p_alloc", x)
			if str(r).begins_with("ok"):
				top = x
			rows.append("%d -> %s" % [x, str(r).split(" ")[0] if r != null else "null/trap"])
			sb.free()
		var tiny = ClassDB.instantiate("Sandbox")
		tiny.program = _elf
		tiny.memory_max = 32
		var tiny_r = tiny.vmcall("p_alloc", 256)
		var tiny_r2 = tiny.vmcall("p_alloc", 512)
		tiny.free()
		_v(6, "memory_max", "INFO", "at memory_max=4095: %s; at memory_max=32: 256 MiB -> %s, 512 MiB -> %s (a floor above the setting)" % [
			", ".join(rows), str(tiny_r).split(" ")[0], str(tiny_r2).split(" ")[0]])
		_v(6, "memory_max", "PASS" if _st.max_ok > 0 and _st.all_ctrl else "FAIL",
			"max usable allocation %d MiB (2X ladder, set after program=), %d MiB just under the 4 GiB ceiling; set before program= = %d MiB; X/2 arm refused for every X >= 512: %s" % [
			_st.max_ok, top, _st.max_ok_before, _st.all_ctrl])
		return true
	var x: int = _st.xs[_st.k]
	_st.k += 1
	var arms := []
	for arm in [["after", x / 2], ["after", x * 2], ["before", x * 2]]:
		var sb = ClassDB.instantiate("Sandbox")
		if arm[0] == "before":
			sb.memory_max = arm[1]
		sb.program = _elf
		if arm[0] == "after":
			sb.memory_max = arm[1]
		var t := _us()
		var r = sb.vmcall("p_alloc", x)
		var dt := _us() - t
		var ok := str(r).begins_with("ok")
		arms.append("%s lim=%d -> %s (%.0f ms)" % [arm[0], arm[1], str(r).split(" ")[0] if r != null else "null/trap", dt / 1000.0])
		if arm[1] == x / 2 and ok and x >= 512:
			_st.all_ctrl = false
		if arm[1] == x * 2 and ok and arm[0] == "after":
			_st.max_ok = x
		if arm[1] == x * 2 and ok and arm[0] == "before":
			_st.max_ok_before = x
		sb.free()
	_v(6, "memory_max", "INFO", "X=%d MiB: %s" % [x, "; ".join(arms)])
	return false

# --- 7. vmcall on a GDScript Thread -------------------------------------------------

func _p07_thread_vmcall() -> bool:
	if not _st.has("th"):
		var sb = _sb()
		var n := int(max(_spin_per_s, 1e6) * 1.5)
		_st.n = n
		# Control: the same call on the main thread; no frame can pass during it.
		var f0 := Engine.get_process_frames()
		var t := _us()
		var r = sb.vmcall("p_spin", n)
		var dt := _us() - t
		_st.want = r
		_v(7, "thread_vmcall", "PASS" if Engine.get_process_frames() - f0 == 0 else "FAIL",
			"control: main-thread vmcall p_spin(%d) %.0f ms, frames advanced %d" % [n, dt / 1000.0, Engine.get_process_frames() - f0])
		_st.sb = sb
		_st.f0 = Engine.get_process_frames()
		_st.t0 = _us()
		var th := Thread.new()
		th.start(func(): return sb.vmcall("p_spin", n))
		_st.th = th
		return false
	var th: Thread = _st.th
	if th.is_alive() and _us() - _st.t0 < 60 * 1000000:
		return false
	var r = th.wait_to_finish()
	var frames: int = Engine.get_process_frames() - _st.f0
	var dt: int = _us() - _st.t0
	var ok: bool = r == _st.want and frames > 1
	_v(7, "thread_vmcall", "PASS" if ok else "FAIL", "vmcall on a Thread: result %s (%s), main thread advanced %d frames in %.0f ms" % [
		str(r), "matches" if r == _st.want else "DIFFERS", frames, dt / 1000.0])
	_st.sb.free()
	return true

# --- 8. two sandboxes, two threads at once ---------------------------------------------

func _p08_two_sandboxes() -> bool:
	if not _st.has("ta"):
		var a = _sb()
		var b = _sb()
		var na := int(max(_spin_per_s, 1e6) * 1.0)
		var nb := int(na * 3 / 4)
		var t := _us()
		_st.wa = a.vmcall("p_spin", na)
		_st.wb = b.vmcall("p_spin", nb)
		_st.serial_ms = (_us() - t) / 1000.0
		_st.a = a
		_st.b = b
		_st.t0 = _us()
		var ta := Thread.new()
		var tb := Thread.new()
		ta.start(func(): return a.vmcall("p_spin", na))
		tb.start(func(): return b.vmcall("p_spin", nb))
		_st.ta = ta
		_st.tb = tb
		return false
	if (_st.ta.is_alive() or _st.tb.is_alive()) and _us() - _st.t0 < 60 * 1000000:
		return false
	var ra = _st.ta.wait_to_finish()
	var rb = _st.tb.wait_to_finish()
	var conc_ms: float = (_us() - _st.t0) / 1000.0
	var ok: bool = ra == _st.wa and rb == _st.wb
	_v(8, "two_sandboxes", "PASS" if ok else "FAIL",
		"A=%s B=%s (serial reference %s / %s); serial %.0f ms, concurrent %.0f ms -> speedup %.2fx" % [
		str(ra), str(rb), str(_st.wa), str(_st.wb), _st.serial_ms, conc_ms, _st.serial_ms / max(conc_ms, 0.001)])
	_st.a.free()
	_st.b.free()
	return true

# --- 9. typed argument echo --------------------------------------------------------------

func _same(a, b) -> bool:
	return typeof(a) == typeof(b) and var_to_bytes(a) == var_to_bytes(b)

func _p09_echo() -> bool:
	var i64min := -9223372036854775807 - 1
	var cases := [
		["echo_f", [0.1, -0.0, 5e-324, 1.7976931348623157e308, PI, 1.0 / 3.0, INF, -INF, NAN]],
		["echo_i", [0, -1, 1, 4611686018427387905, 9007199254740993, 9223372036854775807, i64min]],
		["echo_b", [true, false]],
		["echo_s", ["", "hello", "héllo ✓ 日本語 \U01F389"]],
		["echo_pf32", [PackedFloat32Array(), PackedFloat32Array([0.1, -0.0, 1e-40, 3.4028235e38, 1.0 / 3.0, INF, NAN])]],
		["echo_pb", [PackedByteArray(), PackedByteArray(range(256))]],
	]
	for unboxed in [true, false]:
		var sb = _sb()
		sb.unboxed_arguments = unboxed
		var rows := []
		var all_ok := true
		for c in cases:
			var good := 0
			var bad := []
			for x in c[1]:
				var r = sb.vmcall(c[0], x)
				if _same(r, x):
					good += 1
				else:
					bad.append("%s->%s" % [var_to_str(x), var_to_str(r)])
			all_ok = all_ok and bad.is_empty()
			rows.append("%s %d/%d%s" % [c[0], good, c[1].size(), "" if bad.is_empty() else " BAD " + ", ".join(bad)])
		if not unboxed:
			var good := 0
			var tot := 0
			for c in cases:
				for x in c[1]:
					tot += 1
					if _same(sb.vmcall("echo_var", x), x):
						good += 1
			rows.append("echo_var(Variant) %d/%d" % [good, tot])
		if unboxed:
			rows.append("f_bits(0.1)=%x (host %x)" % [sb.vmcall("f_bits", 0.1), var_to_bytes(0.1).decode_u64(4)])
		_v(9, "typed_echo", "PASS" if all_ok else ("FAIL" if unboxed else "INFO"),
			"unboxed_arguments=%s: %s" % [unboxed, "; ".join(rows)])
		sb.free()
	return true

# --- 10. heap readings ----------------------------------------------------------------------

func _heap(sb) -> String:
	return "heap_usage=%d chunks=%d allocs=%d deallocs=%d" % [sb.get_heap_usage(), sb.monitor_heap_chunk_count,
		sb.monitor_heap_allocation_counter, sb.monitor_heap_deallocation_counter]

func _p10_heap() -> bool:
	var sb = _sb()
	var u0: int = sb.get_heap_usage()
	var h0 := _heap(sb)
	var r1 = sb.vmcall("p_hold", 64)
	var u1: int = sb.get_heap_usage()
	var h1 := _heap(sb)
	var r2 = sb.vmcall("p_release")
	var u2: int = sb.get_heap_usage()
	var h2 := _heap(sb)
	var delta := u1 - u0
	var ok: bool = abs(delta - 64 * 1048576) <= 1048576 and abs(u2 - u0) <= 1048576 and sb.monitor_heap_usage == u2
	_v(10, "heap", "PASS" if ok else "FAIL", "before: %s | %s: %s (delta %.2f MiB) | %s: %s" % [h0, str(r1), h1, delta / 1048576.0, str(r2), h2])
	sb.free()
	return true

# --- GPU sandbox --------------------------------------------------------------------------------

func _gpu_sb():
	if _gpu == null:
		_gpu = _sb(4096)
	return _gpu

# --- 11. fiber across vmcalls ----------------------------------------------------------------------

func _p11_fiber() -> bool:
	var sb = _gpu_sb()
	if not _st.has("phase"):
		_st.phase = "fib"
		_st.frames = 0
		var s = sb.vmcall("fib_start")
		_say("P11 fiber          start: %s" % str(s))
		if not str(s).begins_with("started"):
			_v(11, "fiber", "FAIL", "fib_start: %s" % str(s))
			return true
		_st.last_frame = -1
		return false
	var fn: String = "fib_pump" if _st.phase == "fib" else "sm_pump"
	# One resume per frame: the submit of frame k is synced in frame k+1.
	if Engine.get_process_frames() == _st.last_frame:
		return false
	_st.last_frame = Engine.get_process_frames()
	_st.frames += 1
	var r := str(sb.vmcall(fn))
	if r.begins_with("WAIT_GPU") and _st.frames < 300:
		return false
	if _st.phase == "fib":
		_st.fib = r
		_st.fib_frames = _st.frames
		var ok := r.begins_with("DONE value=100 rounds=100 caught=1 escaped=no")
		_v(11, "fiber", "PASS" if ok else "FAIL", "fiber, 100 submit/yield/resume rounds over %d vmcalls (one per frame), throw/catch at round 50: %s" % [_st.frames, r])
		_st.phase = "sm"
		_st.frames = 0
		sb.vmcall("sm_start")
		return false
	var okc := r.begins_with("DONE value=100 rounds=100 caught=1")
	_v(11, "fiber", "PASS" if okc else "FAIL", "control, same job as a state machine over %d vmcalls: %s" % [_st.frames, r])
	return true

# --- 12. large RD buffers + buffer_update throughput -----------------------------------------------

func _p12_big_buffers() -> bool:
	var sb = _gpu_sb()
	var rd = sb.vmcall("p_rd")
	if not (rd is RenderingDevice):
		_v(12, "big_buffers", "FAIL", "p_rd: %s" % str(rd))
		return true
	_say("P12 big_buffers    limits: max_compute_workgroup_count_x=%d device=%s" % [
		rd.limit_get(RenderingDevice.LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X), rd.get_device_name()])
	for bytes in [256 << 20, 1 << 30, 2 << 30, (4 << 30) - 256]:
		for direct in [false, true]:
			var t := _us()
			var r := str(sb.vmcall("big_buffer", bytes, direct))
			var dt := _us() - t
			var ok := r.find("last=4.5 ") >= 0 and r.find("mid=0 ") >= 0 and r.find("first=0 ") >= 0 and r.find("clear_err=0") >= 0
			if direct:
				_v(12, "big_buffers", "INFO", "same + one direct 4-byte buffer_get_data at the end: host_ms=%.0f direct read %s" % [
					dt / 1000.0, "ok" if r.find("direct_get_last=4.5") >= 0 else "FAILED: " + r])
			else:
				_v(12, "big_buffers", "PASS" if ok else "FAIL", "%.3f GiB empty-create + buffer_clear + saxpby over all + buffer_copy readback: host_ms=%.0f %s" % [
					bytes / 1073741824.0, dt / 1000.0, r])
	# Host-side buffer_update into a guest-held device (the weight-upload path).
	var cap := 1 << 30
	var buf: RID = rd.storage_buffer_create(cap)
	if not buf.is_valid():
		_v(12, "big_buffers", "FAIL", "host storage_buffer_create(1 GiB) failed")
		return true
	var rows := []
	for mb in [64, 128, 256, 512, 1024]:
		var n: int = mb << 20
		var data := PackedByteArray()
		data.resize(n)
		data.encode_u32(n - 4, 0xC0FFEE00 + mb)
		var t := _us()
		var err: int = rd.buffer_update(buf, 0, n, data)
		var t_upd := _us() - t
		rd.submit()
		rd.sync()
		var t_all := _us() - t
		var tg := _us()
		var tail: PackedByteArray = rd.buffer_get_data(buf, n - 4, 4)
		var t_get := _us() - tg
		var ok: bool = err == OK and tail.decode_u32(0) == 0xC0FFEE00 + mb
		rows.append("%d MiB: update %.0f ms, +submit+sync %.0f ms = %.2f GB/s%s" % [mb, t_upd / 1000.0, t_all / 1000.0,
			n / (t_all / 1e6) / 1e9, "" if ok else " WRONG"])
		rows.append("4-byte buffer_get_data from the 1 GiB buffer %.0f ms" % (t_get / 1000.0))
	rd.free_rid(buf)
	_v(12, "big_buffers", "INFO", "host buffer_update throughput: " + "; ".join(rows))
	return true

# --- 13. references_max under 10k dispatches ------------------------------------------------------

func _refs_once(sb, n: int, aliased := false) -> Array:
	sb.vmcall("refs_setup")
	var to0: int = sb.monitor_execution_timeouts
	var ex0: int = sb.monitor_exceptions
	var t := _us()
	var r := str(sb.vmcall("refs_run", n, aliased))
	var dt := _us() - t
	var ok := r.find("counters=%d,%d,%d,%d " % [n / 4, n / 4, n / 4, n / 4]) >= 0
	var why := "" if ok else " killed_by=%s" % ("execution_timeout" if sb.monitor_execution_timeouts > to0 else
		("exception (references?)" if sb.monitor_exceptions > ex0 else "?"))
	if not ok:
		sb.vmcall("p_list_end") # the killed call left its compute list open
	return [ok, "n=%d host_ms=%.0f %s%s" % [n, dt / 1000.0, r, why]]

func _p13_refs() -> bool:
	var sb = _gpu_sb()
	var s = sb.vmcall("refs_setup")
	if str(s) != "ok":
		_v(13, "references", "FAIL", "refs_setup: %s" % str(s))
		return true
	var def_to: int = sb.execution_timeout
	# At the default execution_timeout: how many host calls fit in one vmcall?
	sb.references_max = 65536
	var ladder := []
	for n in [1000, 2000, 4000, 10000]:
		ladder.append(_refs_once(sb, n)[1])
	_v(13, "references", "INFO", "at default execution_timeout=%d, references_max=65536 (each dispatch = 4 host calls + a barrier per 4): %s" % [def_to, " | ".join(ladder)])
	# The references_max question proper, with the instruction budget out of the way.
	sb.execution_timeout = 1000000
	for refs in [4096, 65536]:
		sb.references_max = refs
		var res := _refs_once(sb, 10000)
		_v(13, "references", "PASS" if res[0] else "FAIL",
			"execution_timeout=1000000 references_max=%d: 10000 dispatches x 3 binds + 2499 barriers, one submit: %s" % [refs, res[1]])
	sb.references_max = 65536
	var al := _refs_once(sb, 10000, true)
	_v(13, "references", "INFO", "same, but +1 done in place (y and dst the same buffer in one set): %s -> %s" % [
		"exact" if al[0] else "COUNTS LOST across barriers", al[1]])
	sb.references_max = 100
	var c := _refs_once(sb, 10000)
	_v(13, "references", "INFO", "references_max=100 (default), same recording: %s -> %s" % [
		"trips" if not c[0] else "does NOT trip: binds, dispatches and barriers hold no references", c[1]])
	# Control: the shapes that do consume references -- buffers and uniform
	# sets created in one call (refs_setup: 12 sets, 28 buffers; refs_usets:
	# 256 sets) -- on fresh sandboxes. arm = [label, set before program=,
	# set after program=, set after refs_setup] (-1 = leave alone).
	var rows := []
	var got := {}
	for arm in [["default", -1, -1, -1], ["100 after", -1, 100, -1], ["4096 after", -1, 4096, -1],
			["4096 before", 4096, -1, -1], ["4096 then 100 live", -1, 4096, 100]]:
		var f = ClassDB.instantiate("Sandbox")
		if arm[1] > 0:
			f.references_max = arm[1]
		f.program = _elf
		if arm[2] > 0:
			f.references_max = arm[2]
		f.execution_timeout = 1000000
		var r1 := str(f.vmcall("refs_setup"))
		if r1 != "ok":
			f.vmcall("p_list_end")
		if arm[3] > 0:
			f.references_max = arm[3]
		var r2 := str(f.vmcall("refs_usets", 256)) if r1 == "ok" else "-"
		got[arm[0]] = [r1 == "ok", r2.find("made=256 of 256") >= 0]
		rows.append("%s (reads %d): setup %s, 256 sets %s" % [arm[0], f.references_max, "ok" if r1 == "ok" else "TRIPS",
			r2 if r2 == "-" else ("ok" if got[arm[0]][1] else "TRIPS")])
		f.free()
	var ctrl_ok: bool = not got["default"][0] and not got["100 after"][0] and got["4096 after"][1]
	_v(13, "references", "PASS" if ctrl_ok else "FAIL", "control, object-creating calls: " + " | ".join(rows))
	sb.references_max = 4096
	sb.execution_timeout = def_to
	return true

# --- 14. f16 storage read on the GPU ---------------------------------------------------------------

func _p14_f16() -> bool:
	var sb = _gpu_sb()
	var halves := PackedByteArray()
	halves.resize(128)
	# every binary16 class: zeros, normals, max, min normal, subnormals, inf, nan
	var vals := [0x0000, 0x8000, 0x3C00, 0xBC00, 0x3555, 0x7BFF, 0xFBFF, 0x0400, 0x0001, 0x03FF, 0x8001,
		0x7C00, 0xFC00, 0x7E00, 0x4248, 0x57D0]
	for i in 64:
		var h: int = vals[i] if i < vals.size() else ((i * 2654435761) >> 7) & 0x7BFF
		halves.encode_u16(2 * i, h)
	var r := str(sb.vmcall("f16_read", halves))
	_v(14, "f16_storage", "PASS" if r.begins_with("n=64 mismatches=0 ") else "FAIL",
		"Lean half_load (StructuredBuffer<half> -> float*2) on 64 halves vs CPU decode: %s" % r)
	return true

# --- 15. ggml-cpu in the guest ------------------------------------------------------------------------

func _fields(line: String) -> Dictionary:
	var d := {}
	for kv in line.strip_edges().split(" "):
		var p := kv.split("=")
		if p.size() == 2:
			d[p[0]] = p[1]
	return d

func _p15_ggml() -> bool:
	var sb = _sb()
	var t := _us()
	var r := str(sb.vmcall("ggml_probe", 256))
	var dt := _us() - t
	var host := FileAccess.get_file_as_string(GGML_HOST).strip_edges()
	if host.is_empty():
		_v(15, "ggml_cpu", "FAIL", "no host-native line at %s (run gates/0f-runtime/ggml_host/build.sh); guest: %s" % [GGML_HOST, r])
	else:
		var g := _fields(r)
		var h := _fields(host)
		var worst := 0.0
		var ok := true
		for k in ["sum_abs", "sum_sq", "softmax_w", "softmax_rowsum"]:
			if not (g.has(k) and h.has(k)):
				ok = false
				continue
			var gv := float(g[k])
			var hv := float(h[k])
			var rel: float = abs(gv - hv) / max(abs(hv), 1e-300)
			worst = max(worst, rel)
		ok = ok and worst <= 1e-6
		_v(15, "ggml_cpu", "PASS" if ok else "FAIL", "256^3 f16xf32 mul_mat + soft_max, 1 thread, rv64gc: worst rel diff vs host-native %s (<= 1e-6), guest host_ms=%.0f | guest %s | host %s" % [
			str(worst), dt / 1000.0, r, host])
	sb.free()
	# Control: a TU compiled with Zfh.
	var z = _sb()
	var e0: int = z.monitor_exceptions
	var zr = z.vmcall("zfh_probe")
	var trapped: bool = zr == null or z.monitor_exceptions > e0
	var right: bool = not trapped and is_equal_approx(float(zr), 4.875)
	# Seen both ways: alone (--only=15) it returns 1.5 silently; after the
	# GPU probes it traps on fmadd.h. Either way Zfh is not executed right.
	_v(15, "ggml_cpu", "PASS" if not right else "FAIL", "control, Zfh TU (1.5*2.25+1.5 via fmadd.h, want 4.875): %s (returned %s, exceptions +%d)" % [
		"TRAPS (illegal instruction)" if trapped else ("runs CORRECTLY -- libriscv executes Zfh" if right else "WRONG RESULT, no trap"),
		str(zr), z.monitor_exceptions - e0])
	z.free()
	return true
