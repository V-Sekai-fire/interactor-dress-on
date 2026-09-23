# Gate 6G.1 runner: GPU round trips through rd_compute from a vmcall on a
# worker Thread (gates/6g-polyfem-gpu/g1-rd-worker/README.md).
#
#   godot --path project --script gate_rd_worker.gd --rendering-driver vulkan --xr-mode off [--gpu-index N] \
#         [-- --reps=300 --out=results.txt --only=A,B]
#
# One round trip (rd_worker.elf): k dispatches in one compute list (k - 1
# chained Lean saxpbys and a Lean df32 dot), then the dot's 16-byte result
# back on the CPU, checked exactly. mode 0 sync_get = submit + sync +
# buffer_get_data, 1 get = buffer_get_data alone (it flushes and stalls), 2
# sync = submit + sync with no readback. Every round trip is timed on the
# host around its vmcall (the guest clock is not a clock).
#
# Arms, in order (a step each, advanced from _process; a wall clock in every
# branch):
#   A    guest, main thread (the control that is known to work; it blocks
#        the frame while it runs)
#   FA   flat control: the same kernels and calls from GDScript, main thread
#   N1   negative: a device made on the main thread, used from a worker
#        Thread's vmcall; then the main thread again (the device must be intact)
#   FN1  its flat control: the same in GDScript, no sandbox
#   R0   baseline: the main thread renders the scene, nothing else
#   B    guest, worker Thread: open, ladder, close in one Thread, while the
#        main thread renders the scene (its frame times are kept)
#   FB   flat control on a worker Thread
#   O    does opening a device on a worker hold up the main thread? Three
#        rw_open + rw_close on one worker, then three bare GDScript
#        create_local_rendering_device + free on another (its flat control);
#        long main-thread frames are matched to the windows they overlap
#   N2   negative: a device opened by one worker Thread, used by the next
#        (stage_base.start() makes a new Thread per call)
#   Q    persistent worker: one Thread fed by a queue (Mutex + Semaphore);
#        open, rounds and close are separate vmcalls frames apart
#   C0   baseline: the main thread renders the scene and ticks drape.elf's
#        rd job (bench_fwd, one tick per frame, rule 4), no worker
#   C    C0 plus B's worker ladder running at the same time (frame times
#        while the worker runs; the drape keeps ticking until a job has passed)
#   E    frame-paced round trips on the main thread (rule 4 / a fiber that
#        yields WAIT_GPU at every submit): submit in one frame, collect in the next
#
# Results stream to gates/6g-polyfem-gpu/g1-rd-worker/<out> (stdout is
# buffered when redirected).
extends SceneTree

const SandboxUtil := preload("res://stages/sandbox_util.gd")

const OUT_DIR := "res://../gates/6g-polyfem-gpu/g1-rd-worker/"
const WALL_MS := 20 * 60 * 1000
const N := 2796 # floats per vector: 3 x 932, the loop's skirt (Gate 8)
const KS := [1, 10, 100]
const MODES := ["sync_get", "get", "sync"]
const TIMEOUT_UNITS := 4000000
# Newton-iteration time the round trips are budgeted against (README):
# foxgirl phase 0 native, 33.6 s / 41 Newton (Gate 6.P); the 932-vertex skirt
# in the guest, 1393 s / 253 Newton (Gate 8), over Gate 6.P's per-Newton
# guest/native ratio 24.6.
const FOXGIRL_NATIVE_ITER_MS := 33600.0 / 41.0
const SKIRT_GUEST_ITER_MS := 1393000.0 / 253.0
const GUEST_PER_NEWTON_RATIO := 24.6
const BUDGET := 0.10

var _out: FileAccess
var _t0 := 0
var _steps: Array = []
var _i := 0
var _st := {}
var _only := []
var _reps := 300
var _counts := {"PASS": 0, "FAIL": 0, "INFO": 0}
var _frame := 0
var _last_us := 0
var _collect_frames := false
var _frames_us := PackedInt64Array()
var _spikes: Array = [] # [start_us, end_us] of each collected frame over 25 ms
var _spin: Array = []
var _spv_sax := PackedByteArray()
var _spv_dot := PackedByteArray()
var _summary: Array = [] # [arm, k, mode, p50, p90, p99, max, mean_batch]
var _orphans: Array = [] # Sandboxes whose device no thread can free (N2)

# the persistent worker (arm Q)
var _q_mutex := Mutex.new()
var _q_sem := Semaphore.new()
var _q_jobs: Array = []
var _q_results: Array = []
var _q_quit := false

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _v(arm: String, verdict: String, detail: String) -> void:
	_counts[verdict] = _counts.get(verdict, 0) + 1
	_say("%-4s %-5s %s" % [arm, verdict, detail])

func _us() -> int:
	return Time.get_ticks_usec()

func _initialize() -> void:
	_t0 = Time.get_ticks_msec()
	var out_name := "results.txt"
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--reps="):
			_reps = int(a.substr(7))
		elif a.begins_with("--out="):
			out_name = a.substr(6)
		elif a.begins_with("--only="):
			for x in a.substr(7).split(","):
				_only.append(x)
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(OUT_DIR))
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT_DIR + out_name), FileAccess.WRITE)
	_say("Gate 6G.1 -- rd_compute from a worker Thread. Godot %s, %s, %s (%s), %d threads" % [
			Engine.get_version_info().string, OS.get_processor_name(), RenderingServer.get_video_adapter_name(),
			RenderingServer.get_video_adapter_vendor(), OS.get_processor_count()])
	_say("n=%d floats, k in %s, modes %s, %d timed round trips per cell (+5 warm-up) and one %d-round vmcall; vsync=%s; main thread id %d" % [
			N, str(KS), str(MODES), _reps, _reps, str(DisplayServer.window_get_vsync_mode()), OS.get_main_thread_id()])
	if not ResourceLoader.exists("res://rd_worker.elf") or ClassDB.instantiate("Sandbox") == null:
		_say("FAIL: rd_worker.elf or the Sandbox class is missing")
		_finish()
		return
	_build_scene()
	_steps = [
		["A", _arm_a], ["FA", _arm_fa], ["N1", _arm_n1], ["FN1", _arm_fn1], ["R0", _arm_r0], ["B", _arm_b], ["FB", _arm_fb], ["O", _arm_o],
		["N2", _arm_n2], ["Q", _arm_q], ["C0", _arm_c0], ["C", _arm_c], ["E", _arm_e],
	]

# --- the frame loop -------------------------------------------------------------------

func _process(_delta: float) -> bool:
	var now := _us()
	if _collect_frames and _last_us > 0:
		_frames_us.append(now - _last_us)
		if now - _last_us > 25000:
			_spikes.append([_last_us, now])
	_last_us = now
	_frame += 1
	for j in _spin.size():
		var mi: Node3D = _spin[j]
		mi.rotate_y(0.02 + 0.0001 * j)
	if Time.get_ticks_msec() - _t0 > WALL_MS:
		_say("TIMEOUT: wall clock %d s passed at step %d" % [WALL_MS / 1000, _i])
		_finish()
		return true
	if _frame < 5:
		return false # let the window and the scene come up
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
	_say("")
	_say("SUMMARY TABLE (us per round trip; singles host-timed around one vmcall each, batch = one vmcall of %d / %d)" % [_reps, _reps])
	_say("%-4s %4s %-9s %8s %8s %8s %8s %10s %12s %12s" % ["arm", "k", "mode", "p50", "p90", "p99", "max", "batch_mean",
			"rt/iter_skr", "rt/iter_fox"])
	var skirt_native_ms := SKIRT_GUEST_ITER_MS / GUEST_PER_NEWTON_RATIO
	for r in _summary:
		var per := float(r[7]) if r[7] > 0 else float(r[3])
		_say("%-4s %4d %-9s %8d %8d %8d %8d %10.1f %12d %12d" % [r[0], r[1], MODES[r[2]], r[3], r[4], r[5], r[6], r[7],
				int(BUDGET * skirt_native_ms * 1000.0 / per), int(BUDGET * FOXGIRL_NATIVE_ITER_MS * 1000.0 / per)])
	_say("rt/iter: round trips per Newton iteration within %d%% of its native time at the batch mean: skirt %.0f ms native (%.0f ms guest / %.1f), foxgirl %.0f ms native" % [
			int(BUDGET * 100), skirt_native_ms, SKIRT_GUEST_ITER_MS, GUEST_PER_NEWTON_RATIO, FOXGIRL_NATIVE_ITER_MS])
	_say("SUMMARY: PASS=%d FAIL=%d INFO=%d wall_s=%.1f orphaned_devices=%d" % [_counts.PASS, _counts.FAIL, _counts.INFO,
			(Time.get_ticks_msec() - _t0) / 1000.0, _orphans.size()])
	if _out != null:
		_out.close()
		_out = null
	quit(0 if _counts.FAIL == 0 else 1)

# --- the scene the main thread renders --------------------------------------------------

func _build_scene() -> void:
	var scene := Node3D.new()
	scene.name = "GateScene"
	var cam := Camera3D.new()
	cam.transform = Transform3D(Basis(), Vector3(0, 0, 26))
	scene.add_child(cam)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-50, 30, 0)
	light.shadow_enabled = true
	scene.add_child(light)
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.background_color = Color(0.12, 0.13, 0.16)
	scene.add_child(env)
	var sphere := SphereMesh.new() # 64 x 32 segments
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.8, 0.55, 0.3)
	sphere.material = mat
	for x in 20:
		for y in 20:
			var mi := MeshInstance3D.new()
			mi.mesh = sphere
			mi.position = Vector3((x - 9.5) * 1.6, (y - 9.5) * 1.1, float((x * 7 + y * 3) % 5) * -1.5)
			mi.scale = Vector3(0.7, 0.7, 0.7)
			scene.add_child(mi)
			_spin.append(mi)
	root.add_child(scene)

func _frames_begin() -> void:
	_frames_us = PackedInt64Array()
	_spikes = []
	_collect_frames = true
	_last_us = _us()

func _frames_end() -> String:
	_collect_frames = false
	var d := _stats(_frames_us)
	var over := 0
	for x in _frames_us:
		if x > 25000:
			over += 1
	return "%s, %d over 25 ms" % [_fmt(d), over]

# Which worker phase each long frame overlapped (host clock, one time base
# across threads): open (device + pipelines + buffers), ladder, close.
func _spike_phases(r: Dictionary) -> String:
	var out := []
	for sp in _spikes:
		var ph := []
		for w in [["open", r.t_open0, r.t_open1], ["ladder", r.t_open1, r.t_close0], ["close", r.t_close0, r.t_close1]]:
			if sp[0] < w[2] and sp[1] > w[1]:
				ph.append(w[0])
		if _drape_overlaps(sp):
			ph.append("drape_tick>20ms")
		out.append("%d ms[%s]" % [(sp[1] - sp[0]) / 1000, ",".join(ph) if not ph.is_empty() else "none"])
	return "long frames and the worker phase they overlap: " + (" ".join(out) if not out.is_empty() else "none")

func _drape_overlaps(sp: Array) -> bool:
	for d in _st.get("drape_long", []):
		if sp[0] < d[1] and sp[1] > d[0]:
			return true
	return false

# R0: the scene alone, for B's and C's frame times.
func _arm_r0() -> bool:
	if not _st.has("t0"):
		_st.t0 = _us()
		_frames_begin()
		return false
	if _us() - _st.t0 < 6 * 1000000:
		return false
	_v("R0", "INFO", "baseline, the scene alone (%d spheres, shadows): frame time %s; draw calls %d" % [_spin.size(), _frames_end(),
			Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)])
	return true

# --- statistics ------------------------------------------------------------------------

func _stats(a) -> Dictionary:
	var s: Array = Array(a)
	if s.is_empty():
		return {"n": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "min": 0, "mean": 0.0}
	s.sort()
	var sum := 0
	for x in s:
		sum += int(x)
	var q := func(p: float) -> int: return int(s[min(s.size() - 1, int(floor(p * (s.size() - 1) + 0.5)))])
	return {"n": s.size(), "p50": q.call(0.5), "p90": q.call(0.9), "p99": q.call(0.99), "max": int(s[-1]), "min": int(s[0]),
			"mean": float(sum) / s.size()}

func _fmt(d: Dictionary) -> String:
	return "n=%d min=%d p50=%d p90=%d p99=%d max=%d mean=%.1f us" % [d.n, d.min, d.p50, d.p90, d.p99, d.max, d.mean]

# --- the guest ladder (runs on whichever thread calls it) -------------------------------

func _sandbox():
	var r := SandboxUtil.make_sandbox(null, "res://rd_worker.elf", 0, 4096, TIMEOUT_UNITS)
	return r.sandbox

# rows: {k, mode, us (singles), bad, batch_us, batch_good, thread}
func _ladder(sb, reps: int) -> Array:
	var rows := []
	for k in KS:
		for m in 3:
			var want := 2 if m == 2 else 1
			for w in 5:
				sb.vmcall("rw_round", k, m)
			var ts := PackedInt64Array()
			var bad := 0
			for r in reps:
				var t := _us()
				var c = sb.vmcall("rw_round", k, m)
				ts.append(_us() - t)
				if c != want:
					bad += 1
			var tb := _us()
			var good = sb.vmcall("rw_rounds", k, m, reps)
			rows.append({"k": k, "mode": m, "us": ts, "bad": bad, "batch_us": _us() - tb, "batch_good": int(good),
					"thread": OS.get_thread_caller_id()})
	return rows

func _report_ladder(arm: String, rows: Array) -> bool:
	var ok := true
	for r in rows:
		var d := _stats(r.us)
		var mean_b := float(r.batch_us) / _reps
		var good: bool = r.bad == 0 and r.batch_good == _reps
		ok = ok and good
		_summary.append([arm, r.k, r.mode, d.p50, d.p90, d.p99, d.max, mean_b])
		_v(arm, "PASS" if good else "FAIL", "k=%-3d %-8s singles %s bad=%d | batch %d/%d right, %.1f us/rt" % [
				r.k, MODES[r.mode], _fmt(d), r.bad, r.batch_good, _reps, mean_b])
	return ok

# The worker body of B and C: everything on one Thread, device included.
func _worker_ladder(sb) -> Dictionary:
	var t_open0 := _us()
	var open = sb.vmcall("rw_open", N)
	var t_open1 := _us()
	var rows := _ladder(sb, _reps) if str(open).begins_with("OK") else []
	var stats = sb.vmcall("rw_stats")
	var t_close0 := _us()
	var closed = sb.vmcall("rw_close")
	var t_close1 := _us()
	return {"open": str(open), "open_us": t_open1 - t_open0, "rows": rows, "stats": str(stats), "close": str(closed),
			"close_us": t_close1 - t_close0, "thread": OS.get_thread_caller_id(), "t_open0": t_open0, "t_open1": t_open1,
			"t_close0": t_close0, "t_close1": t_close1}

# --- A: guest on the main thread ----------------------------------------------------------

func _arm_a() -> bool:
	var sb = _sandbox()
	var f0 := _frame
	var t := _us()
	var open = sb.vmcall("rw_open", N)
	var open_us := _us() - t
	_v("A", "PASS" if str(open).begins_with("OK") else "FAIL", "rw_open on the main thread in %d us: %s" % [open_us, str(open)])
	if str(open).begins_with("OK"):
		var t1 := _us()
		_report_ladder("A", _ladder(sb, _reps))
		var ladder_ms := (_us() - t1) / 1000.0
		_v("A", "INFO", "the ladder held the main thread for %.0f ms; frames advanced %d (a main-thread sync blocks the frame)" % [ladder_ms, _frame - f0])
		var stats := str(sb.vmcall("rw_stats"))
		_v("A", "INFO", "rw_stats: %s" % stats)
		_v("A", "INFO", "rule 4 on the main thread: every sync above is a same-frame sync (%s), which is what the counter is for" % _field(stats, "same_frame_syncs"))
	var closed := str(sb.vmcall("rw_close"))
	_v("A", "PASS" if closed.begins_with("closed permanent_slots_left=0") else "FAIL", "rw_close: %s" % closed)
	sb.free()
	return true

func _field(s: String, key: String) -> String:
	for w in s.split(" "):
		if w.begins_with(key + "="):
			return w
	return key + "=?"

# --- FA / FB / FN1: the flat control, GDScript on the same kernels -----------------------

func _spirv() -> bool:
	if not _spv_sax.is_empty():
		return true
	var sb = _sandbox()
	_spv_sax = sb.vmcall("rw_spirv", "saxpby")
	_spv_dot = sb.vmcall("rw_spirv", "dot_reduce")
	sb.free()
	return not _spv_sax.is_empty() and not _spv_dot.is_empty()

func _flat_shader(rd: RenderingDevice, spv: PackedByteArray) -> RID:
	var s := RDShaderSPIRV.new()
	s.set_stage_bytecode(RenderingDevice.SHADER_STAGE_COMPUTE, spv)
	return rd.shader_create_from_spirv(s)

func _flat_set(rd: RenderingDevice, sh: RID, rids: Array) -> RID:
	var us := []
	for i in rids.size():
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER if i == 0 else RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = i
		u.add_id(rids[i])
		us.append(u)
	return rd.uniform_set_create(us, sh, 0)

func _flat_open() -> Dictionary:
	var rd := RenderingServer.create_local_rendering_device()
	if rd == null:
		return {}
	var st := {"rd": rd, "cur": 0, "value": 0, "device": rd.get_device_name()}
	st.sax_sh = _flat_shader(rd, _spv_sax)
	st.dot_sh = _flat_shader(rd, _spv_dot)
	st.sax_p = rd.compute_pipeline_create(st.sax_sh)
	st.dot_p = rd.compute_pipeline_create(st.dot_sh)
	var sp := PackedByteArray()
	sp.resize(16)
	sp.encode_u32(0, N)
	sp.encode_float(4, 1.0)
	sp.encode_float(8, 1.0)
	var dp := PackedByteArray()
	dp.resize(16)
	dp.encode_u32(0, N)
	var ones := PackedFloat32Array()
	ones.resize(N)
	ones.fill(1.0)
	var zeros := PackedByteArray()
	zeros.resize(N * 4)
	var z16 := PackedByteArray()
	z16.resize(16)
	st.sax_params = rd.uniform_buffer_create(16, sp)
	st.dot_params = rd.uniform_buffer_create(16, dp)
	st.ones = rd.storage_buffer_create(N * 4, ones.to_byte_array())
	st.a = rd.storage_buffer_create(N * 4, zeros)
	st.b = rd.storage_buffer_create(N * 4, zeros)
	st.r = rd.storage_buffer_create(16, z16)
	st.set_ab = _flat_set(rd, st.sax_sh, [st.sax_params, st.ones, st.a, st.b])
	st.set_ba = _flat_set(rd, st.sax_sh, [st.sax_params, st.ones, st.b, st.a])
	st.dot_a = _flat_set(rd, st.dot_sh, [st.dot_params, st.ones, st.a, st.r])
	st.dot_b = _flat_set(rd, st.dot_sh, [st.dot_params, st.ones, st.b, st.r])
	st.ok = st.set_ab.is_valid() and st.set_ba.is_valid() and st.dot_a.is_valid() and st.dot_b.is_valid()
	return st

func _flat_round(st: Dictionary, k: int, m: int) -> int:
	var rd: RenderingDevice = st.rd
	var groups := (N + 255) / 256
	var cl := rd.compute_list_begin()
	for i in k - 1:
		rd.compute_list_bind_compute_pipeline(cl, st.sax_p)
		rd.compute_list_bind_uniform_set(cl, st.set_ab if st.cur == 0 else st.set_ba, 0)
		rd.compute_list_dispatch(cl, groups, 1, 1)
		rd.compute_list_add_barrier(cl)
		st.cur = 1 - st.cur
		st.value += 1
	rd.compute_list_bind_compute_pipeline(cl, st.dot_p)
	rd.compute_list_bind_uniform_set(cl, st.dot_a if st.cur == 0 else st.dot_b, 0)
	rd.compute_list_dispatch(cl, 1, 1, 1)
	rd.compute_list_end()
	if m != 1:
		rd.submit()
		rd.sync()
	if m == 2:
		return 2
	var b := rd.buffer_get_data(st.r, 0, 16)
	if b.size() < 8:
		return -2
	var got := b.decode_float(0) + b.decode_float(4)
	return 1 if got == float(N) * float(st.value) else 0

func _flat_ladder(st: Dictionary, reps: int) -> Array:
	var rows := []
	for k in KS:
		for m in 3:
			var want := 2 if m == 2 else 1
			for w in 5:
				_flat_round(st, k, m)
			var ts := PackedInt64Array()
			var bad := 0
			for r in reps:
				var t := _us()
				var c := _flat_round(st, k, m)
				ts.append(_us() - t)
				if c != want:
					bad += 1
			var tb := _us()
			var good := 0
			for r in reps:
				if _flat_round(st, k, m) == want:
					good += 1
			rows.append({"k": k, "mode": m, "us": ts, "bad": bad, "batch_us": _us() - tb, "batch_good": good,
					"thread": OS.get_thread_caller_id()})
	return rows

func _flat_close(st: Dictionary) -> void:
	var rd: RenderingDevice = st.rd
	for key in ["set_ab", "set_ba", "dot_a", "dot_b", "sax_params", "dot_params", "ones", "a", "b", "r", "sax_p", "dot_p",
			"sax_sh", "dot_sh"]:
		if st.has(key) and st[key].is_valid():
			rd.free_rid(st[key])
	rd.free()

func _flat_worker() -> Dictionary:
	var t := _us()
	var st := _flat_open()
	var open_us := _us() - t
	if st.is_empty() or not st.ok:
		return {"ok": false, "open_us": open_us}
	var rows := _flat_ladder(st, _reps)
	var dev: String = st.device
	_flat_close(st)
	return {"ok": true, "open_us": open_us, "rows": rows, "device": dev, "thread": OS.get_thread_caller_id()}

func _arm_fa() -> bool:
	if not _spirv():
		_v("FA", "FAIL", "no SPIR-V from rd_worker.elf")
		return true
	var r := _flat_worker()
	_v("FA", "PASS" if r.ok else "FAIL", "GDScript local device on the main thread (%s), opened in %d us" % [r.get("device", "?"), r.open_us])
	if r.ok:
		_report_ladder("FA", r.rows)
	return true

func _arm_fb() -> bool:
	if not _st.has("th"):
		if not _spirv():
			_v("FB", "FAIL", "no SPIR-V from rd_worker.elf")
			return true
		_st.th = Thread.new()
		_st.f0 = _frame
		_st.t0 = _us()
		_st.th.start(_flat_worker)
		return false
	if _st.th.is_alive() and _us() - _st.t0 < 300 * 1000000:
		return false
	if _st.th.is_alive():
		_v("FB", "FAIL", "worker still running after 300 s; left running")
		return true
	var r: Dictionary = _st.th.wait_to_finish()
	_v("FB", "PASS" if r.ok else "FAIL", "GDScript local device on a worker Thread (%s, thread %s), opened in %d us; main advanced %d frames" % [
			r.get("device", "?"), str(r.get("thread", "?")), r.open_us, _frame - _st.f0])
	if r.ok:
		_report_ladder("FB", r.rows)
	return true

# --- N1 / FN1: a device made on the main thread, used from a worker ---------------------

func _arm_n1() -> bool:
	if not _st.has("th"):
		var sb = _sandbox()
		_st.sb = sb
		var open := str(sb.vmcall("rw_open", N))
		var ctrl = sb.vmcall("rw_round", 10, 0)
		_v("N1", "PASS" if open.begins_with("OK") and ctrl == 1 else "FAIL", "control: opened and round-tripped on the main thread: open=%s round=%s" % [open.substr(0, 40), str(ctrl)])
		_st.th = Thread.new()
		_st.t0 = _us()
		# k=1: the dot alone, so a refused list leaves the chain value unchanged
		_st.th.start(func(): return [sb.vmcall("rw_round", 1, 0), sb.vmcall("rw_round", 1, 1), sb.vmcall("rw_round", 1, 0),
				str(sb.vmcall("rw_stats")), OS.get_thread_caller_id()])
		return false
	if _st.th.is_alive() and _us() - _st.t0 < 60 * 1000000:
		return false
	if _st.th.is_alive():
		_v("N1", "FAIL", "worker still running after 60 s")
		return true
	var r: Array = _st.th.wait_to_finish()
	var refused: bool = r[0] == -2 and r[1] == -2 and r[2] == -2
	_v("N1", "PASS" if refused else "FAIL", "negative: the same device from worker thread %d: codes %s (-2 = nothing came back); %s" % [
			r[4], str([r[0], r[1], r[2]]), r[3]])
	var after = _st.sb.vmcall("rw_round", 10, 0)
	_v("N1", "PASS" if after == 1 else "FAIL", "back on the main thread, the device is intact: round=%s %s" % [str(after), str(_st.sb.vmcall("rw_stats"))])
	var closed := str(_st.sb.vmcall("rw_close"))
	_v("N1", "PASS" if closed.begins_with("closed permanent_slots_left=0") else "FAIL", "rw_close on the main thread: %s" % closed)
	_st.sb.free()
	return true

func _arm_fn1() -> bool:
	if not _st.has("th"):
		if not _spirv():
			_v("FN1", "FAIL", "no SPIR-V")
			return true
		var st := _flat_open()
		_st.flat = st
		var ctrl := _flat_round(st, 10, 0)
		_v("FN1", "PASS" if ctrl == 1 else "FAIL", "control: GDScript device on the main thread, round=%d" % ctrl)
		_st.th = Thread.new()
		_st.t0 = _us()
		_st.th.start(func(): return [_flat_round(st, 1, 0), _flat_round(st, 1, 1), OS.get_thread_caller_id()])
		return false
	if _st.th.is_alive() and _us() - _st.t0 < 60 * 1000000:
		return false
	if _st.th.is_alive():
		_v("FN1", "FAIL", "worker still running after 60 s")
		return true
	var r: Array = _st.th.wait_to_finish()
	_v("FN1", "PASS" if r[0] == -2 and r[1] == -2 else "FAIL", "negative, no sandbox: the GDScript device from worker thread %d: codes %s (Godot's render-thread guard, not the sandbox)" % [r[2], str([r[0], r[1]])])
	var after := _flat_round(_st.flat, 10, 0)
	_v("FN1", "PASS" if after == 1 else "FAIL", "back on the main thread: round=%d" % after)
	_flat_close(_st.flat)
	return true

# --- B: the guest on a worker Thread ----------------------------------------------------

func _arm_b() -> bool:
	if not _st.has("th"):
		_st.sb = _sandbox()
		_st.th = Thread.new()
		_st.f0 = _frame
		_st.t0 = _us()
		_frames_begin()
		_st.th.start(_worker_ladder.bind(_st.sb))
		return false
	if _st.th.is_alive() and _us() - _st.t0 < 300 * 1000000:
		return false
	var frames := _frames_end()
	if _st.th.is_alive():
		_v("B", "FAIL", "worker still running after 300 s; left running")
		return true
	var r: Dictionary = _st.th.wait_to_finish()
	var secs: float = (_us() - _st.t0) / 1e6
	_v("B", "PASS" if r.open.begins_with("OK") else "FAIL", "rw_open on worker thread %d in %d us: %s" % [r.thread, r.open_us, r.open])
	_report_ladder("B", r.rows)
	_v("B", "INFO", "main thread advanced %d frames in %.1f s while the worker synced (%.1f fps): frame time %s" % [
			_frame - _st.f0, secs, (_frame - _st.f0) / secs, frames])
	_v("B", "INFO", _spike_phases(r))
	_v("B", "INFO", "rw_stats: %s" % r.stats)
	_v("B", "INFO", "rule 4 on a worker: %s of %s syncs landed in their submit's process frame; they block only the worker" % [
			_field(r.stats, "same_frame_syncs"), _field(r.stats, "syncs")])
	_v("B", "PASS" if r.close.begins_with("closed permanent_slots_left=0") else "FAIL", "rw_close on the same worker in %d us: %s" % [r.close_us, r.close])
	_st.sb.free()
	return true

# --- O: device creation on a worker vs the main thread's frames ----------------------------

func _open_close_guest(sb, times: int) -> Array:
	var w := []
	for i in times:
		var t0 := _us()
		var o := str(sb.vmcall("rw_open", N))
		var t1 := _us()
		var c := str(sb.vmcall("rw_close"))
		w.append(["guest_open", t0, t1, o.begins_with("OK")])
		w.append(["guest_close", t1, _us(), c.begins_with("closed")])
	return w

func _open_close_flat(times: int) -> Array:
	var w := []
	for i in times:
		var t0 := _us()
		var rd := RenderingServer.create_local_rendering_device()
		var t1 := _us()
		var ok := rd != null
		if ok:
			rd.free()
		w.append(["flat_create", t0, t1, ok])
		w.append(["flat_free", t1, _us(), ok])
	return w

func _flat_after_pause() -> Array:
	OS.delay_msec(300)
	return _open_close_flat(3)

func _arm_o() -> bool:
	if not _st.has("th"):
		_st.sb = _sandbox()
		_st.th = Thread.new()
		_st.t0 = _us()
		_st.part = 0
		_frames_begin()
		_st.th.start(_open_close_guest.bind(_st.sb, 3))
		return false
	if _st.th.is_alive() and _us() - _st.t0 < 120 * 1000000:
		return false
	if _st.th.is_alive():
		_collect_frames = false
		_v("O", "FAIL", "worker still running after 120 s; left running")
		return true
	var w: Array = _st.th.wait_to_finish()
	if _st.part == 0:
		_st.guest = w
		_st.part = 1
		_st.th = Thread.new()
		_st.t0 = _us()
		# a few quiet frames between the two halves
		_st.th.start(_flat_after_pause)
		return false
	var frames := _frames_end()
	var all: Array = _st.guest + w
	var ok := true
	var times := []
	for x in all:
		ok = ok and x[3]
		times.append("%s %d ms" % [x[0], (x[2] - x[1]) / 1000])
	var hits := []
	for sp in _spikes:
		var ph := []
		for x in all:
			if sp[0] < x[2] and sp[1] > x[1]:
				ph.append(x[0])
		hits.append("%d ms[%s]" % [(sp[1] - sp[0]) / 1000, ",".join(ph) if not ph.is_empty() else "none"])
	_v("O", "PASS" if ok else "FAIL", "on one worker: %s" % ", ".join(times))
	_v("O", "INFO", "main thread meanwhile: frame time %s; long frames: %s" % [frames, " ".join(hits) if not hits.is_empty() else "none"])
	_st.sb.free()
	return true

# --- N2: a device held across two worker Threads ----------------------------------------

func _arm_n2() -> bool:
	if not _st.has("sb"):
		_st.sb = _sandbox()
		_st.phase = 1
		_st.th = Thread.new()
		_st.t0 = _us()
		var sb = _st.sb
		_st.th.start(func(): return [str(sb.vmcall("rw_open", N)), sb.vmcall("rw_round", 10, 0), OS.get_thread_caller_id()])
		return false
	if _st.th.is_alive() and _us() - _st.t0 < 60 * 1000000:
		return false
	if _st.th.is_alive():
		_v("N2", "FAIL", "worker still running after 60 s")
		return true
	var r: Array = _st.th.wait_to_finish()
	if _st.phase == 1:
		_v("N2", "PASS" if str(r[0]).begins_with("OK") and r[1] == 1 else "FAIL", "Thread 1 (id %d) opened the device and round-tripped: round=%s" % [r[2], str(r[1])])
		_st.phase = 2
		_st.t1 = r[2]
		_st.th = Thread.new()
		_st.t0 = _us()
		var sb = _st.sb
		_st.th.start(func(): return [sb.vmcall("rw_round", 1, 0), sb.vmcall("rw_round", 1, 1), str(sb.vmcall("rw_stats")), OS.get_thread_caller_id()])
		return false
	_v("N2", "PASS" if r[0] == -2 and r[1] == -2 and r[3] != _st.t1 else "FAIL", "negative: Thread 2 (id %d, a new Thread as stage_base.start() makes per call) on Thread 1's device: codes %s; %s" % [
			r[3], str([r[0], r[1]]), r[2]])
	_v("N2", "INFO", "the device is orphaned: free_rid and its teardown are render-thread-guarded too and its thread is gone; the Sandbox is kept to the end")
	_orphans.append(_st.sb)
	return true

# --- Q: one persistent worker Thread fed by a queue -------------------------------------

func _q_loop(sb) -> void:
	while true:
		_q_sem.wait()
		_q_mutex.lock()
		var job = null if _q_jobs.is_empty() else _q_jobs.pop_front()
		var quit_now := _q_quit
		_q_mutex.unlock()
		if job == null:
			if quit_now:
				return
			continue
		var t := _us()
		var r = sb.callv("vmcall", [job[0]] + job[1])
		var dt := _us() - t
		_q_mutex.lock()
		_q_results.append([job[0], r, dt, OS.get_thread_caller_id()])
		_q_mutex.unlock()

func _q_post(fn: String, args: Array = []) -> void:
	_q_mutex.lock()
	_q_jobs.append([fn, args])
	_q_mutex.unlock()
	_q_sem.post()

func _arm_q() -> bool:
	const PLAN := [[0, "rw_open", [N]], [10, "rw_rounds", [10, 0, 100]], [20, "rw_rounds", [100, 1, 30]],
			[30, "rw_rounds", [1, 0, 100]], [40, "rw_stats", []], [50, "rw_close", []]]
	if not _st.has("th"):
		_st.sb = _sandbox()
		_st.th = Thread.new()
		_st.f0 = _frame
		_st.t0 = _us()
		_st.next = 0
		_q_results = []
		_q_quit = false
		_st.th.start(_q_loop.bind(_st.sb))
	while _st.next < PLAN.size() and _frame - _st.f0 >= PLAN[_st.next][0]:
		_q_post(PLAN[_st.next][1], PLAN[_st.next][2])
		_st.next += 1
	_q_mutex.lock()
	var got := _q_results.size()
	_q_mutex.unlock()
	if got < PLAN.size() and _us() - _st.t0 < 120 * 1000000:
		return false
	_q_mutex.lock()
	_q_quit = true
	_q_mutex.unlock()
	_q_sem.post()
	if _st.th.is_alive() and _us() - _st.t0 < 125 * 1000000:
		return false
	if _st.th.is_alive():
		_v("Q", "FAIL", "the persistent worker did not stop; left running")
		return true
	_st.th.wait_to_finish()
	var threads := {}
	var ok := got == PLAN.size()
	for r in _q_results:
		threads[r[3]] = true
		_say("Q    call  %-9s thread %d host %8d us -> %s" % [r[0], r[3], r[2], str(r[1])])
	if ok:
		ok = str(_q_results[0][1]).begins_with("OK") and _q_results[1][1] == 100 and _q_results[2][1] == 30 \
				and _q_results[3][1] == 100 and str(_q_results[5][1]).begins_with("closed permanent_slots_left=0")
	_v("Q", "PASS" if ok and threads.size() == 1 else "FAIL", "one Thread (%d distinct thread id), six vmcalls posted %d frames apart held one device from open to close: %d of %d answered right" % [
			threads.size(), 10, got, PLAN.size()])
	_st.sb.free()
	return true

# --- C0 / C: the main thread renders and runs drape.elf's rd job; C adds the worker ------

func _drape_open() -> bool:
	if _st.has("drape"):
		return _st.drape != null
	var r := SandboxUtil.make_sandbox(null, "res://drape.elf", 1024, 65536, 0)
	_st.drape = r.sandbox
	if _st.drape == null:
		_v(_st.arm, "FAIL", "drape.elf: %s" % r.reason)
		return false
	_st.jobs_pass = 0
	_st.jobs_fail = 0
	_st.ticks = 0
	_st.last_job = ""
	_st.drape_long = []
	return _drape_start()

func _drape_start() -> bool:
	var s := str(_st.drape.vmcall("avbd_job_start", "bench_fwd", "rd"))
	if not s.begins_with("STARTED"):
		_v(_st.arm, "FAIL", "avbd_job_start: %s" % s)
		return false
	return true

# One tick per frame (rule 4: the job's readbacks land a frame after its submits).
func _drape_tick() -> void:
	var t0 := _us()
	var s := str(_st.drape.vmcall("avbd_job_tick", t0))
	var t1 := _us()
	if t1 - t0 > 20000:
		_st.drape_long.append([t0, t1])
	_st.ticks += 1
	if s.begins_with("RUNNING"):
		return
	if s.begins_with("PASS"):
		_st.jobs_pass += 1
	else:
		_st.jobs_fail += 1
	_st.last_job = s.split("\n")[0]
	_drape_start()

func _drape_report(arm: String) -> void:
	var rule4 := str(_st.drape.vmcall("rd_rule4"))
	_v(arm, "PASS" if _st.jobs_fail == 0 and _st.jobs_pass > 0 and rule4.begins_with("same_frame_syncs=0 ") else "FAIL",
			"drape.elf bench_fwd on rd, main thread: %d ticks, %d jobs PASS, %d FAIL (last: %s); %s" % [
			_st.ticks, _st.jobs_pass, _st.jobs_fail, _st.last_job, rule4])
	_st.drape.vmcall("rd_close")
	_st.drape.free()

func _arm_c0() -> bool:
	_st.arm = "C0"
	if not _st.has("t0"):
		if not _drape_open():
			return true
		_st.t0 = _us()
		_frames_begin()
		return false
	_drape_tick()
	if _us() - _st.t0 < 10 * 1000000:
		return false
	var frames := _frames_end()
	var by_drape := 0
	for sp in _spikes:
		by_drape += 1 if _drape_overlaps(sp) else 0
	_v("C0", "INFO", "baseline, scene + drape job on the main thread, no worker (10 s): frame time %s (%d of them overlap a drape tick over 20 ms: %d such ticks); draw calls %d, objects %d" % [
			frames, by_drape, _st.drape_long.size(), Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME),
			Performance.get_monitor(Performance.RENDER_TOTAL_OBJECTS_IN_FRAME)])
	_drape_report("C0")
	return true

func _arm_c() -> bool:
	_st.arm = "C"
	if not _st.has("th"):
		if not _drape_open():
			return true
		_st.sb = _sandbox()
		_st.th = Thread.new()
		_st.f0 = _frame
		_st.t0 = _us()
		_frames_begin()
		_st.th.start(_worker_ladder.bind(_st.sb))
		return false
	_drape_tick()
	if _st.th.is_alive() and _us() - _st.t0 < 300 * 1000000:
		return false
	if not _st.has("frames"):
		_st.frames = _frames_end()
		_st.secs = (_us() - _st.t0) / 1e6
		_st.fcount = _frame - _st.f0
	# keep draping until a job has passed (the worker may finish first)
	if _st.jobs_pass + _st.jobs_fail == 0 and _us() - _st.t0 < 120 * 1000000:
		return false
	if _st.th.is_alive():
		_v("C", "FAIL", "worker still running after 300 s; left running")
		return true
	var r: Dictionary = _st.th.wait_to_finish()
	var secs: float = _st.secs
	_v("C", "PASS" if r.open.begins_with("OK") else "FAIL", "rw_open on worker thread %d in %d us while the main thread renders and drapes" % [r.thread, r.open_us])
	_report_ladder("C", r.rows)
	_v("C", "INFO", "main thread while the worker ran (%.1f s, %d frames, %.1f fps): frame time %s; draw calls %d" % [secs,
			_st.fcount, _st.fcount / secs, _st.frames, Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)])
	_v("C", "INFO", _spike_phases(r))
	_v("C", "INFO", "worker rw_stats: %s" % r.stats)
	_v("C", "PASS" if r.close.begins_with("closed permanent_slots_left=0") else "FAIL", "rw_close on the worker: %s" % r.close)
	_drape_report("C")
	_st.sb.free()
	return true

# --- E: frame-paced round trips on the main thread (rule 4, the fiber's WAIT_GPU) ---------

func _arm_e() -> bool:
	const ROUNDS := 120
	if not _st.has("sb"):
		_st.sb = _sandbox()
		var open := str(_st.sb.vmcall("rw_open", N))
		if not open.begins_with("OK"):
			_v("E", "FAIL", "rw_open: %s" % open)
			return true
		_st.lat = PackedInt64Array()
		_st.bad = 0
		_st.n = 0
		_st.ts = 0
		_st.sb.vmcall("rw_submit", 10)
		_st.ts = _us()
		return false
	var c = _st.sb.vmcall("rw_collect")
	_st.lat.append(_us() - _st.ts)
	if c != 1:
		_st.bad += 1
	_st.n += 1
	if _st.n < ROUNDS:
		_st.sb.vmcall("rw_submit", 10)
		_st.ts = _us()
		return false
	var stats := str(_st.sb.vmcall("rw_stats"))
	var d := _stats(_st.lat)
	_summary.append(["E", 10, 0, d.p50, d.p90, d.p99, d.max, d.mean])
	_v("E", "PASS" if _st.bad == 0 else "FAIL", "k=10 submit in frame f, sync + read in frame f+1: %d round trips, bad=%d, submit-to-result %s" % [
			ROUNDS, _st.bad, _fmt(d)])
	_v("E", "INFO", "rule 4 kept: %s of %s (rw_stats: %s)" % [_field(stats, "same_frame_syncs"), _field(stats, "syncs"), stats])
	_st.sb.vmcall("rw_close")
	_st.sb.free()
	return true
