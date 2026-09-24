# Gate 0F runner: what the godot-sandbox runtime does, one probe at a time.
#
#   godot --path project --script gate_runtime.gd --rendering-driver vulkan --xr-mode off
#   ... -- --only=5,6            # run a subset (probe numbers)
#
# Frame-driven: each probe is a step that _process advances until it says it
# is done, so the probes that need frames (a vmcall on a Thread, the fiber's
# one-resume-per-frame job) get real frames. Every probe prints PASS / FAIL /
# INFO / DEFERRED lines with its control (probe 16, set-0 sharing and in-place
# ops, was added with the corrections; probe 17, aligned allocation, with Cut
# 4's memalign fix); results stream to
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
	_say("P%02d %-14s %-8s %s" % [num, name, verdict, _clean(detail)])

# The checkout's absolute path is replaced by <project>, so the committed
# results do not name the machine or the worktree.
func _clean(t: String) -> String:
	var root := ProjectSettings.globalize_path("res://").trim_suffix("/")
	return t.replace(root, "<project>").replace(root.replace("/", "\\"), "<project>")

func _sb(refs := 4096):
	var sb = ClassDB.instantiate("Sandbox")
	if sb != null: sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
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
		[5, _p05_timeout], [9, _p09_echo], [10, _p10_heap], [17, _p17_memalign],
		[11, _p11_fiber], [11, _p11b_rid], [12, _p12_big_buffers], [13, _p13_refs], [14, _p14_f16], [16, _p16_set0],
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
		_gpu.vmcall("p_rd_close")
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
		rows.append("lim=%d killed=%s host_ms=%.1f ret=%s ~iter_at_first_call_rate=%.2fe9" % [lim, killed, d1 / 1000.0, str(rr), _spin_per_s * d1 / 1e6 / 1e9])
	_v(5, "timeout", "PASS" if killed_all else "FAIL", "busy loop 1e12 at execution_timeout lim: " + "; ".join(rows))
	sb.execution_timeout = def_to
	var after = sb.vmcall("p_spin", 10)
	_v(5, "timeout", "PASS" if after == _lcg(10) else "FAIL", "sandbox usable after a kill: p_spin(10) -> %s" % str(after))
	# The default limit against ~2 s of work. The rate above is the first
	# call's; later calls run several times faster (the first run's "~2 s"
	# took 342 ms), so measure a warm rate first.
	var nw := 50000000
	var tw := _us()
	sb.vmcall("p_spin", nw)
	var warm_per_s := nw / ((_us() - tw) / 1e6)
	_v(5, "timeout", "INFO", "rate: first call %.1f M iter/s, warm %.1f M iter/s" % [_spin_per_s / 1e6, warm_per_s / 1e6])
	var n2 := int(warm_per_s * 2.0)
	var b2: int = sb.monitor_execution_timeouts
	var t2 := _us()
	var r2 = sb.vmcall("p_spin", n2)
	var d2 := _us() - t2
	_v(5, "timeout", "INFO", "default execution_timeout=%d vs p_spin(%d) (2 s at the warm rate): killed=%s host_ms=%.0f" % [
		def_to, n2, sb.monitor_execution_timeouts > b2, d2 / 1000.0])
	sb.free()
	return true

# --- 6. memory_max ladder ---------------------------------------------------------
#
# Every arm is a fresh Sandbox with memory_max set BEFORE program=: setting it
# after program= only reloads the program when the value grows (a lower value
# is ignored), so a limit below the current one can only be tested fresh.

func _mem_arm(limit: int, mb: int, when := "before") -> String:
	var sb = ClassDB.instantiate("Sandbox")
	if sb != null: sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	if when == "before":
		sb.memory_max = limit
	sb.program = _elf
	if when == "after":
		sb.memory_max = limit
	var r = sb.vmcall("p_alloc", mb)
	sb.free()
	if r == null:
		return "null/trap"
	return str(r).split(" ")[0]

func _p06_memory() -> bool:
	if not _st.has("init"):
		_st.init = true
		var fresh = ClassDB.instantiate("Sandbox")
		if fresh != null: fresh.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
		var def_before: int = fresh.memory_max
		fresh.program = _elf
		var def_after: int = fresh.memory_max
		fresh.free()
		var order = ClassDB.instantiate("Sandbox")
		if order != null: order.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
		order.memory_max = 2048
		order.program = _elf
		var kept: int = order.memory_max
		order.free()
		_v(6, "memory_max", "INFO", "default memory_max=%d (MiB) before program=, %d after; set 2048 then program= -> reads %d (%s)" % [
			def_before, def_after, kept, "RESET by loading" if kept != 2048 else "kept"])
		_st.xs = [64, 128, 256, 512, 1024, 2048]
		_st.k = 0
		_st.max_ok = 0
		_st.all_ctrl = true
		return false
	if _st.k < _st.xs.size():
		# Allocate-and-touch X MiB under a limit of 2X (must succeed) and X/2
		# (must be refused), each in a fresh Sandbox.
		var x: int = _st.xs[_st.k]
		_st.k += 1
		var t := _us()
		var big := _mem_arm(x * 2, x)
		var dt := _us() - t
		var small := _mem_arm(x / 2, x)
		if big == "ok":
			_st.max_ok = x
		if small == "ok":
			_st.all_ctrl = false
		_v(6, "memory_max", "INFO", "X=%d MiB, fresh sandboxes, limit set before program=: limit 2X=%d -> %s (%.0f ms); control limit X/2=%d -> %s" % [
			x, x * 2, big, dt / 1000.0, x / 2, small])
		return false
	# The ceiling. The heap is 0.8 x memory_max and must end below 4 GiB, so
	# the largest limit that loads is just under 5120, and the most that can
	# be allocated is just under 0.8 x the limit.
	var rows := []
	var top := 0
	var top_lim := 0
	var rule_ok := true
	for lim in [4096, 5000, 5100, 5110, 5112, 5114, 5116, 5118, 5119, 5120, 6144]:
		var heap := int(lim * 0.8)
		var under := _mem_arm(lim, heap - 1)
		var over := _mem_arm(lim, heap + 1)
		rows.append("%d: %d MiB %s, %d MiB %s" % [lim, heap - 1, under, heap + 1, over])
		if under == "ok":
			top = max(top, heap - 1)
			top_lim = lim
		if under == "ok" and over == "ok":
			rule_ok = false
		if lim >= 5120 and under == "ok":
			rule_ok = false
	_v(6, "memory_max", "INFO", "ceiling, fresh sandbox per arm, allocating 0.8 x limit -/+ 1 MiB: " + "; ".join(rows))
	# Setting memory_max after program= : growing reloads, lowering is ignored.
	var grow := _mem_arm(2048, 1024, "after")
	var lower_before := _mem_arm(32, 256, "before")
	var lower_after := _mem_arm(32, 256, "after")
	_v(6, "memory_max", "INFO", "set after program=: 2048 then alloc 1024 MiB -> %s (growing applies); 32 then alloc 256 MiB -> %s (lowering is IGNORED: %s); 32 set before program= -> %s" % [
		grow, lower_after, "yes" if lower_after == "ok" else "no", lower_before])
	_v(6, "memory_max", "PASS" if _st.max_ok == 2048 and _st.all_ctrl and rule_ok and top > 3900 else "FAIL",
		"2X ladder ok to %d MiB; X/2 control refused at every X: %s; largest allocation %d MiB at memory_max=%d; heap = 0.8 x limit below 4 GiB holds: %s" % [
		_st.max_ok, _st.all_ctrl, top, top_lim, rule_ok])
	return true

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

# --- 17. aligned allocation ----------------------------------------------------------------------
# godot-sandbox's memalign fallback (vendor/sandbox-api native.cpp) returned
# an already-freed block when 16 malloc tries missed a > 16-byte alignment.
# 1000 blocks at each of 64/128/4096 alignment through all four aligned entry
# points, frees and reallocs interleaved: none misaligned, overlapping or
# corrupted, and the heap back to where it started. Control, in a fresh
# Sandbox: the same sequence through a copy of the old fallback must return
# freed blocks and so produce overlaps.

func _kv(line: String) -> Dictionary:
	var d := {}
	for tok in line.split(" "):
		var kv := tok.split("=")
		if kv.size() == 2:
			d[kv[0]] = int(kv[1])
	return d

func _p17_memalign() -> bool:
	var sb = _sb()
	var warm := str(sb.vmcall("p_memalign", 10, false)) # first-call statics out of the heap delta
	var u0: int = sb.get_heap_usage()
	var t := _us()
	var r := str(sb.vmcall("p_memalign", 1000, false))
	var us := _us() - t
	var u1: int = sb.get_heap_usage()
	sb.free()
	var d := _kv(r)
	var ok: bool = d.get("made", 0) == 3000 and d.get("misaligned", -1) == 0 and d.get("overlaps", -1) == 0 \
			and d.get("corrupt", -1) == 0 and d.get("nulls", -1) == 0 and u1 == u0
	_v(17, "memalign", "PASS" if ok else "FAIL", "%s | heap %d -> %d after a warm-up (%s) | %.1f ms" % [r, u0, u1, warm.get_slice(" ", 2), us / 1000.0])
	var c = _sb()
	var rc := str(c.vmcall("p_memalign", 1000, true))
	c.free()
	var dc := _kv(rc)
	var bug: bool = dc.get("exhausted", 0) > 0 and (dc.get("overlaps", 0) > 0 or dc.get("corrupt", 0) > 0)
	_v(17, "memalign", "PASS" if bug else "FAIL", "control, the old fallback: %s" % rc)
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
#
# references_max only grows: a Sandbox reserves that many scoped slots and a
# lower value set later is not applied. So every arm is a fresh Sandbox, and
# its setup goes one counter per vmcall (refs_setup_one), which keeps each
# setup call far below 100 references; only then does the 10k-dispatch
# recording meet the limit on its own.

func _refs_run(sb, n: int, aliased := false) -> Array:
	var to0: int = sb.monitor_execution_timeouts
	var ex0: int = sb.monitor_exceptions
	var t := _us()
	var r := str(sb.vmcall("refs_run", n, aliased))
	var dt := _us() - t
	var ok := r.find("counters=%d,%d,%d,%d " % [n / 4, n / 4, n / 4, n / 4]) >= 0
	var killed: bool = sb.monitor_execution_timeouts > to0 or sb.monitor_exceptions > ex0
	var why := "" if not killed else " killed_by=%s" % ("execution_timeout" if sb.monitor_execution_timeouts > to0 else
		("exception (references?)" if sb.monitor_exceptions > ex0 else "?"))
	return [ok, "n=%d host_ms=%.0f %s%s" % [n, dt / 1000.0, r, why]]

func _refs_fresh(refs: int, timeout: int):
	var f = ClassDB.instantiate("Sandbox")
	if f != null: f.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	f.program = _elf
	if refs > 0:
		f.references_max = refs
	if timeout > 0:
		f.execution_timeout = timeout
	var setup := []
	for k in 4:
		setup.append(str(f.vmcall("refs_setup_one", k)))
	return [f, setup]

func _p13_refs() -> bool:
	# At the default execution_timeout: how many host calls fit in one vmcall?
	var fr = _refs_fresh(65536, -1)
	var sb = fr[0]
	var def_to: int = sb.execution_timeout
	var ladder := []
	# rd_compute's recovery off first: the hazard as finding 2 found it.
	sb.vmcall("p_recovery", false)
	for n in [1000, 2000, 4000, 10000]:
		sb.vmcall("refs_setup")
		var res := _refs_run(sb, n)
		ladder.append(res[1])
		if not res[0]:
			# The killed call left its compute list open: every later
			# buffer_update is refused until it is ended (finding 2).
			var refused := str(sb.vmcall("refs_setup"))
			var ended := str(sb.vmcall("p_list_end"))
			var again := str(sb.vmcall("refs_setup"))
			_v(13, "references", "PASS" if refused.begins_with("FAIL buffer_update refused") and again == "ok" else "FAIL",
				"recovery off: a vmcall killed mid-recording (n=%d at the default budget) leaves its compute list open: next buffer_update -> %s | p_list_end -> %s | then -> %s" % [
				n, refused, ended, again])
			# The fix (Cut 3): recovery on, killed the same way, and the next
			# buffer_update goes through with no p_list_end.
			var r0 := str(sb.vmcall("p_recovery", true))
			var res2 := _refs_run(sb, n)
			var next := str(sb.vmcall("refs_setup"))
			var r1 := str(sb.vmcall("p_recovery", true))
			var k0 := int(r0.get_slice("recoveries=", 1))
			var k1 := int(r1.get_slice("recoveries=", 1))
			_v(13, "references", "PASS" if not res2[0] and next == "ok" and k1 == k0 + 1 else "FAIL",
				"recovery on (rdc::Device, Cut 3): killed again (%s); next buffer_update -> %s with no p_list_end; recoveries %d -> %d" % [
				res2[1], next, k0, k1])
	_v(13, "references", "INFO", "at default execution_timeout=%d, references_max=65536 (each dispatch = 4 host calls + a barrier per 4): %s" % [def_to, " | ".join(ladder)])
	# The aliased shape (saxpby, y and dst the same buffer): see probe 16.
	sb.execution_timeout = 1000000
	sb.vmcall("refs_setup")
	var al := _refs_run(sb, 10000, true)
	_v(13, "references", "INFO", "same, +1 done in place (saxpby y at b2 read-only and dst at b3 read-write, one buffer): %s -> %s" % [
		"exact" if al[0] else "COUNTS LOST across barriers", al[1]])
	sb.free()
	# The references_max question proper: fresh sandbox per arm, the budget
	# raised out of the way, setup one counter per call.
	for refs in [100, 4096, 65536]:
		var fa = _refs_fresh(refs, 1000000)
		var f = fa[0]
		var setup_ok: bool = fa[1] == ["ok", "ok", "ok", "ok"]
		var res := _refs_run(f, 10000) if setup_ok else [false, "setup: " + str(fa[1])]
		_v(13, "references", "PASS" if res[0] else "FAIL",
			"fresh sandbox, references_max=%d (reads %d), execution_timeout=1000000: 10000 dispatches x 3 binds + 2499 barriers, one submit: %s" % [
			refs, f.references_max, res[1]])
		f.free()
	# Control: the shapes that do consume references -- 4 counters' buffers and
	# uniform sets created in one call (refs_setup: 12 sets, 28 buffers), 256
	# sets in one call (refs_usets) -- on fresh sandboxes.
	# arm = [label, set before program=, set after program=, set after the setup]
	var rows := []
	var got := {}
	for arm in [["default", -1, -1, -1], ["100 after", -1, 100, -1], ["4096 after", -1, 4096, -1],
			["4096 before", 4096, -1, -1], ["65536 then 100", -1, 65536, -1], ["4096 then 100 live", -1, 4096, 100]]:
		var f = ClassDB.instantiate("Sandbox")
		if f != null: f.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
		if arm[1] > 0:
			f.references_max = arm[1]
		f.program = _elf
		if arm[2] > 0:
			f.references_max = arm[2]
		if arm[0] == "65536 then 100":
			f.references_max = 100
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
	var ctrl_ok: bool = not got["default"][0] and not got["100 after"][0] and got["4096 after"][1] and got["65536 then 100"][0]
	_v(13, "references", "PASS" if ctrl_ok else "FAIL", "control, object-creating calls in one vmcall (a lowered value is not applied): " + " | ".join(rows))
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

# --- 11b. a RID held across vmcalls -----------------------------------------------------------------

func _p11b_rid() -> bool:
	# Its own Sandbox: the non-permanent arm leaves a stray buffer behind.
	var sb = _sb(4096)
	var rows := []
	var ok_perm := false
	var bad_scoped := false
	for perm in [true, false]:
		var e0: int = sb.monitor_exceptions
		var h := str(sb.vmcall("rid_hold", perm))
		var u = sb.vmcall("rid_use")
		var us := "null (the vmcall threw)" if u == null else str(u)
		rows.append("permanent=%s: hold -> %s | next vmcall -> %s, exceptions +%d" % [perm, h, us, sb.monitor_exceptions - e0])
		if perm:
			ok_perm = h.ends_with("same_call_read=exact") and us.ends_with(" exact")
		else:
			bad_scoped = h.ends_with("same_call_read=exact") and not us.ends_with(" exact")
	sb.vmcall("p_rd_close")
	sb.free()
	_v(11, "rid_permanence", "PASS" if ok_perm and bad_scoped else "FAIL",
		"a RID kept in a guest static: exact in a later vmcall only when made permanent (rdc::Device does it); without, it resolves wrongly: %s" % " || ".join(rows))
	return true

# --- 16. set-0 uniform sets shared across pipelines; in-place ops across barriers --------------------

func _p16_set0() -> bool:
	var sb = _gpu_sb()
	var def_to: int = sb.execution_timeout
	sb.execution_timeout = 1000000
	var pres := str(sb.vmcall("set0_share", ""))
	var strip := str(sb.vmcall("set0_share", "_stripped"))
	var o1pp := str(sb.vmcall("set0_share", "_o1pp"))
	var ok_p := pres.find("add_own_set=256/256") >= 0 and pres.find("scale_under_add_set=256/256") >= 0 \
		and pres.find("inplace_b1_b4_one_dispatch=256/256") >= 0
	_v(16, "set0_shared", "PASS" if ok_p else "FAIL",
		"Lean probe_add/probe_scale (b0 params, b1-b3 sources, b4 dst), slangc -O0 -preserve-params: a set built for probe_add, bound under probe_scale: %s" % pres)
	var ok_s := strip.find("scale_under_add_set=256/256") < 0 and strip.find("scale_own_set=256/256") >= 0
	_v(16, "set0_shared", "PASS" if ok_s else "FAIL",
		"control, slangc -O0 without -preserve-params (unused sources dropped, layouts differ): %s" % strip)
	_v(16, "set0_shared", "INFO",
		"slangc -preserve-params at the default -O1 (the optimiser drops the unused sources anyway): %s -> %s" % [o1pp,
		"shared set works" if o1pp.find("scale_under_add_set=256/256") >= 0 else "shared set REFUSED, as without the flag"])
	# In place across barriers: 1000 rounds of +1 over 4096 elements, one
	# compute list, a barrier after every round, twice per shape.
	var names := ["aliased", "rw_only", "pingpong", "ro_then_rw", "rw_then_ro"]
	var exact := {}
	for mode in 5:
		var rows := []
		var all_exact := true
		for rep in 2:
			var t := _us()
			var r := str(sb.vmcall("inplace_run", mode, 1000))
			var dt := _us() - t
			rows.append("%s (%.0f ms)" % [r, dt / 1000.0])
			all_exact = all_exact and r.find("exact=4096/4096 ") >= 0
		exact[names[mode]] = all_exact
		_v(16, "inplace", "INFO", "%s: %s" % [names[mode], " | ".join(rows)])
	# The finding: a buffer seen read-only first in a compute list is tracked
	# as read-only for the whole list, so its writes order nothing after them.
	var hazard: bool = not exact["aliased"] and not exact["ro_then_rw"]
	var controls: bool = exact["rw_only"] and exact["pingpong"] and exact["rw_then_ro"]
	_v(16, "inplace", "PASS" if controls else "FAIL",
		"controls exact (same buffer bound read-write only; ping-pong; read-write binding first in the list): %s. Loss when the buffer is first bound read-only in the list (aliased b1+b4, or read by an earlier dispatch in the same list): %s" % [
		controls, "REPRODUCED" if hazard else "not reproduced"])
	sb.execution_timeout = def_to
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
	var t2 := _us()
	var r2 := str(sb.vmcall("ggml_probe", 256))
	var dt2 := _us() - t2
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
		ok = ok and r2 == r
		_v(15, "ggml_cpu", "PASS" if ok else "FAIL", "256^3 f16xf32 mul_mat + soft_max, 1 thread, rv64gc: worst rel diff vs host-native %s (<= 1e-6), guest host_ms=%.0f first call, %.0f second (same result: %s) | guest %s | host %s" % [
			str(worst), dt / 1000.0, dt2 / 1000.0, r2 == r, r, host])
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
