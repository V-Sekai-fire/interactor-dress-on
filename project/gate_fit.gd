# Gate 6: fit.elf (cloth-fit's garment solve) in the guest, against its native
# twin and against upstream's 1-thread oracle. See gates/6-fit/README.md.
#
#   godot --path project --script gate_fit.gd --rendering-driver vulkan --xr-mode off -- --arm=ARM [options]
#
# Arms (one Godot process each; each builds a fresh fit Sandbox):
#   probes   6.0: exceptions, io (refused and counted), instret, ldlt, ldlt8k
#            (host-timed, 5 calls), libm hashes, and the wire check (the arrays
#            GDScript hands the guest vs fit_native --dump-inputs, bit for bit).
#   solve    fixture -> fit_begin -> fit_step x --phases (default: all), each
#            fit_step a vmcall on main.gd's worker Thread while this main
#            thread keeps drawing frames; per phase: host ms, Newton, energy,
#            instructions (instret), guest heap, and the native same-code
#            phase beside it. --mem=MiB sets memory_max (6.P ladder: a failed
#            phase is the arm's answer, not an error of the gate), --elf=res://
#            picks the ELF (6.P ISA A/B), --fitweight0 sets fit_weight = 0 (the
#            fit-gap control). With all phases: the intersection check, the
#            5 cm push control and its flat control, io_attempts, and the final
#            garment (solve frame) as runs/<tag>.garment.f64.
#   report   Gate 6's verdict from runs/*.txt: guest vs native same code, the
#            five-part criterion (fit_native --compare, fit_gap.py), the
#            fit_weight=0 control; writes gates/6-fit/results.txt.
#
# Common options: --tag=NAME (runs/NAME.txt; default the arm), --native=DIR
# (fit_native's same-code f32 output: phases.tsv, garment_final.f64,
# inputs/; default C:/b/fit-out-tbbs32), --wall=S.
#
# Results stream to gates/6-fit/runs/<tag>.txt, flushed per line (Godot
# buffers redirected stdout: poll the file). Every branch quits on the wall
# clock.
extends SceneTree

const RUNS := "res://../gates/6-fit/runs"
const RESULTS := "res://../gates/6-fit/results.txt"
const ORACLE := "C:/b/cf-up-out1/step_garment_252.obj"
const ORACLE_AVATAR := "C:/b/cf-up-out1/step_avatar_252.obj"
const NOFIT := "vendor/cloth-fit/garment-data/assets/garments/LCL_Skirt_DressEvening_003/no-fit.txt"
const VOXEL := 0.01
# The five-part criterion's constants (gates/6-fit/README.md).
const GAP_MEAN_ORACLE := 1.8467
const GAP_P95_ORACLE := 3.9207
const ENERGY_BAND := [0.9 * 0.0012964, 1.1 * 0.0014974]
const HAUS_MAX := 8.24
const VMAX_MAX := 11.63

var _m: Node = null
var _out: FileAccess
var _t0 := 0
var _state := "wait"
var _frames := 0
var _rc := 0
var _args := {}
var _arm := ""
var _tag := ""
var _wall := 0.0
var _native_dir := ""
var _phases_want := -1
var _phase_count := 0
var _phase_t0 := 0
var _last_poll := 0
var _native_phases: Array = []  # [newton, energy, wall_s] per phase
var _phase_rows: Array = []

func _say(s: String) -> void:
	# Repo-relative paths in the logs (the worktree's absolute path is noise).
	s = s.replace(ProjectSettings.globalize_path("res://") + "../", "").replace(_abs("") + "/", "")
	print(s)
	if _out != null:
		_out.store_line(s)
		_out.flush()

func _check(name: String, ok: bool, detail: String) -> void:
	if not ok:
		_rc = 1
	_say("%s %s: %s" % ["ok  " if ok else "FAIL", name, detail])

func _abs(rel: String) -> String:
	return ProjectSettings.globalize_path("res://").path_join("..").path_join(rel).simplify_path()

func _initialize() -> void:
	for a in OS.get_cmdline_user_args():
		var kv: PackedStringArray = a.trim_prefix("--").split("=", true, 1)
		_args[kv[0]] = kv[1] if kv.size() > 1 else "1"
	_arm = _args.get("arm", "probes")
	_tag = _args.get("tag", _arm)
	_native_dir = _args.get("native", "C:/b/fit-out-tbbs32")
	_phases_want = int(_args.get("phases", "-1"))
	var default_wall := {"probes": 1800.0, "solve": 4.0 * 3600.0, "report": 1800.0}
	_wall = float(_args.get("wall", str(default_wall.get(_arm, 3600.0))))
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(RUNS))
	var path := RESULTS if _arm == "report" else RUNS + "/" + _tag + ".txt"
	_out = FileAccess.open(path, FileAccess.WRITE)
	_t0 = Time.get_ticks_msec()
	_say("gate_fit %s, arm %s, tag %s, args %s" % [Time.get_datetime_string_from_system(), _arm, _tag, str(_args)])
	_m = load("res://main.gd").new()
	root.add_child(_m)

func _finish() -> void:
	_say("wall %.1f s, frames %d" % [(Time.get_ticks_msec() - _t0) / 1000.0, _frames])
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	_state = "done"
	quit(_rc)

func _process(_dt: float) -> bool:
	_frames += 1
	if _state == "done":
		return false
	if (Time.get_ticks_msec() - _t0) / 1000.0 > _wall:
		_rc = 1
		_say("FAIL wall clock %.0f s in state %s: %s" % [_wall, _state, _m.fit_status() if _m != null else "-"])
		_finish()
		return false
	match _state:
		"wait":
			if _frames > 3:
				_state = _arm
		"probes":
			_probes()
			_finish()
		"solve":
			_solve_begin()
		"step":
			_phase_t0 = Time.get_ticks_msec()
			_last_poll = _phase_t0
			var r: String = _m.fit_step()
			if not r.begins_with("STARTED"):
				_check("fit_step", false, r)
				_finish()
				return false
			_state = "poll"
		"poll":
			_poll()
		"report":
			_report()
			_finish()
		_:
			_say("FAIL unknown arm %s (probes | solve | report)" % _arm)
			_rc = 1
			_finish()
	return false

# --- 6.0 probes ---------------------------------------------------------------------

func _native_probe(name: String) -> Array:
	# fit_native --probe NAME: [line, min_ms]
	var exe := _native_exe()
	var out := []
	var code := OS.execute(exe, ["--probe", name, "--reps", "5"], out, true)
	if code != 0 and code != 1:
		return ["(fit_native failed: %d %s)" % [code, str(out)], -1.0]
	var lines: PackedStringArray = str(out[0]).strip_edges().split("\n")
	var ms := -1.0
	for l in lines:
		if l.begins_with("time_ms min "):
			ms = l.get_slice(" ", 2).to_float()
	return [lines[0].strip_edges(), ms]

func _native_exe() -> String:
	return _args.get("fit_native", "C:/b/fit-native-tbbs/fit_native.exe")

func _probes() -> void:
	_say(_m.fit_configure_with(int(_args.get("mem", "2048")), _args.get("elf", "res://fit.elf")))
	var r: String = _m.fit_probe_exceptions()
	_check("6.0 exceptions", r.begins_with("PASS"), r)
	r = _m.fit_probe_io()
	# The counter must see every refused open: "counted 3 of 3".
	_check("6.0 io", r.begins_with("PASS") and r.find("counted 3 of 3") >= 0, r)
	_say("     io_paths: " + _m.fit_probe_io_paths().replace("\n", " ;"))
	r = _m.fit_probe_instret()
	_check("6.0 instret", r.begins_with("PASS"), r)
	r = _m.fit_probe_ldlt()
	_check("6.0 ldlt (n=200)", r.begins_with("PASS"), r)
	# 8k DOF: five host-timed vmcalls, min; native: fit_native --probe ldlt8k.
	var best := INF
	var line := ""
	for i in 5:
		var s: String = _m.fit_probe_ldlt8k()
		var us := s.get_slice(" ", 0).trim_prefix("host_us=").to_float()
		best = minf(best, us / 1000.0)
		line = s.substr(s.find(" ") + 1)
	var nat := _native_probe("ldlt8k")
	var g_hash := line.get_slice("fnv1a ", 1)
	var n_hash := str(nat[0]).get_slice("fnv1a ", 1)
	_check("6.0 ldlt8k", line.begins_with("PASS"), line)
	_say("     ldlt8k guest %.1f ms (min of 5 vmcalls, host-timed), native %.1f ms (min of 5): %.1fx; solution bits %s (guest %s, native %s)" % [
			best, nat[1], best / maxf(nat[1], 1e-9), "EQUAL" if g_hash == n_hash else "DIFFER", g_hash, n_hash])
	# libm, per function.
	var gl: String = _m.fit_probe_libm()
	var nl: String = _native_probe("libm")[0]
	var same := PackedStringArray()
	var differ := PackedStringArray()
	for tok in gl.split(" ", false):
		if tok.find(":") < 0:
			continue
		if tok in nl.split(" ", false):
			same.append(tok.get_slice(":", 0))
		else:
			differ.append(tok.get_slice(":", 0))
	_say("info 6.0 libm (20000 inputs each, result bits): equal %d %s; differ %d %s" % [same.size(), ",".join(same), differ.size(), ",".join(differ)])
	_say("     guest  " + gl)
	_say("     native " + nl)
	var gs: String = _m.fit_probe_stl()
	var ns: String = _native_probe("stl")[0]
	var sd := PackedStringArray()
	for tok in gs.split(" ", false):
		if tok.find(":") >= 0 and not (tok in ns.split(" ", false)):
			sd.append(tok.get_slice(":", 0))
	_say("info 6.0 stl (tie order, 20000 elements): %s" % ("equal" if sd.is_empty() else "differ in " + ",".join(sd)))
	_say("     guest  " + gs)
	_say("     native " + ns)
	_wire_check()
	_say("     heap: " + _m.fit_probe_heap())

# The arrays GDScript hands the guest (util/obj_io.gd parse, float32) against
# the arrays fit_native gives its driver (polyfem's OBJ reader, rounded to
# float32), dumped by fit_native --dump-inputs.
func _wire_check() -> void:
	var a: Dictionary = _m.foxgirl_arrays()
	if a.has("error"):
		_check("6.0 wire", false, str(a["error"]))
		return
	var dir := _native_dir.path_join("inputs")
	var pairs := [["body_v", "avatar_v.f32"], ["body_f", "avatar_f.i32"], ["garment_v", "garment_v.f32"],
			["garment_f", "garment_f.i32"], ["src_sk_v", "skeleton_v.f32"], ["tgt_sk_v", "target_skeleton_v.f32"],
			["bones", "skeleton_b.i32"], ["nofit", "no_fit.i32"]]
	var all_ok := true
	var parts := PackedStringArray()
	for p in pairs:
		var nb := FileAccess.get_file_as_bytes(dir.path_join(p[1]))
		var gb: PackedByteArray = a[p[0]].to_byte_array()
		if nb.is_empty() and not gb.is_empty():
			parts.append("%s: no native dump" % p[0])
			all_ok = false
			continue
		var n := gb.size() / 4
		var diff := 0
		var max_ulp := 0
		for i in n:
			if i * 4 >= nb.size():
				break
			var x := gb.decode_u32(i * 4)
			var y := nb.decode_u32(i * 4)
			if x != y:
				diff += 1
				max_ulp = maxi(max_ulp, absi(x - y))
		var ok := gb.size() == nb.size() and diff == 0
		all_ok = all_ok and ok
		parts.append("%s %d%s" % [p[0], n, "" if ok else " (%d differ, max %d ulp, sizes %d/%d)" % [diff, max_ulp, gb.size(), nb.size()]])
	_check("6.0 wire (GDScript OBJ parse vs fit_native, float32 bits)", all_ok, ", ".join(parts))

# --- solve --------------------------------------------------------------------------

func _load_native_phases() -> void:
	var t := FileAccess.get_file_as_string(_native_dir.path_join("phases.tsv"))
	for l in t.split("\n", false):
		var c := l.split("\t")
		if c.size() < 11 or c[0] == "phase":
			continue
		_native_phases.append([c[4].to_int(), c[7].to_float(), c[10].to_float(), c[7]])

func _solve_begin() -> void:
	_load_native_phases()
	if _args.has("fitweight0"):
		_m.fit_config_overrides = {"fit_weight": "0"}
	var r: String = _m.fit_configure_with(int(_args.get("mem", "2048")), _args.get("elf", "res://fit.elf"),
			int(_args.get("timeout", "-1")))
	_check("fit_configure", r.begins_with("OK"), r)
	if _rc != 0:
		_finish()
		return
	# io positive control before begin: the counter sees refused opens.
	r = _m.fit_probe_io()
	_check("io positive control (before fit_begin)", r.begins_with("PASS"), r)
	var t := Time.get_ticks_usec()
	r = _m.fit_fixture_foxgirl()
	_check("fit_fixture_foxgirl (%d ms)" % ((Time.get_ticks_usec() - t) / 1000), r.begins_with("OK"), r)
	t = Time.get_ticks_usec()
	r = _m.fit_begin()
	_check("fit_begin (%d ms)" % ((Time.get_ticks_usec() - t) / 1000), r.begins_with("OK begin"), r)
	_say("     fit_status: " + _m.fit_status())
	if _rc != 0:
		_finish()
		return
	_phase_count = r.get_slice("phases ", 1).get_slice(",", 0).to_int()
	if _phases_want < 0 or _phases_want > _phase_count:
		_phases_want = _phase_count
	_state = "step"

func _poll() -> void:
	if Time.get_ticks_msec() - _last_poll < 1000:
		return
	_last_poll = Time.get_ticks_msec()
	var s: String = _m.fit_status()
	if s.begins_with("BUSY"):
		if (_last_poll - _phase_t0) % 300000 < 1000:
			_say("     %s (main thread frames %d)" % [s, _frames])
		return
	var k := _phase_rows.size()
	var last := s.get_slice("| last: ", 1)
	var ok := last.find(" OK phase") >= 0
	var host_ms := last.get_slice("host_ms=", 1).get_slice(" ", 0).to_int()
	var newton := last.get_slice("newton ", 1).get_slice(" ", 0).to_int()
	var energy_s := last.get_slice("energy ", 1).get_slice(" ", 0)
	var instr := last.get_slice("instructions ", 1).get_slice(" ", 0).to_int()
	var heap := last.get_slice("heap_used ", 1).get_slice(" ", 0).to_float()
	_phase_rows.append({"ok": ok, "ms": host_ms, "newton": newton, "energy": energy_s, "instr": instr, "heap": heap})
	if not ok:
		_rc = 1
		_say("FAIL phase %d: %s" % [k, last.left(2000)])
		_say("     status: " + s.left(600))
		_say("summary %s phases_ok %d/%d FAILED at phase %d, mem %s MiB" % [_tag, k, _phases_want, k, _args.get("mem", "2048")])
		_finish()
		return
	var nat := ""
	if k < _native_phases.size():
		var np: Array = _native_phases[k]
		var rel := absf(energy_s.to_float() - np[1]) / absf(np[1])
		nat = " | native: newton %d energy %s wall %.2f s -> newton %+d, energy rel %s, time %.1fx" % [
				np[0], np[3], np[2], newton - np[0], String.num_scientific(rel), host_ms / 1000.0 / maxf(np[2], 1e-9)]
	_say("ok   phase %d: host %.1f s, newton %d, energy %s, instructions %d (%.0f x 2^20), heap_used %.1f MiB%s" % [
			k, host_ms / 1000.0, newton, energy_s, instr, instr / 1048576.0, heap, nat])
	_say("     " + last.left(600))
	if _phase_rows.size() < _phases_want:
		_state = "step"
		return
	_solve_end()

func _solve_end() -> void:
	var tot_ms := 0
	var tot_newton := 0
	var max_instr := 0
	var max_heap := 0.0
	var per := PackedStringArray()
	for r in _phase_rows:
		tot_ms += r["ms"]
		tot_newton += r["newton"]
		max_instr = maxi(max_instr, r["instr"])
		max_heap = maxf(max_heap, r["heap"])
		per.append(str(r["newton"]))
	var status: String = _m.fit_status()
	var io := status.get_slice("io_attempts ", 1).get_slice(" ", 0).to_int()
	_check("io_attempts from fit_begin to the last phase", io == 0, "io_attempts %d (%s)" % [io, status.get_slice("io_total", 0).right(40)])
	var energy: String = _phase_rows[-1]["energy"]
	var done := _phase_rows.size() == _phase_count
	var line := "summary %s phases_ok %d/%d newton %d (%s) energy %s host_s %.1f max_phase_instructions %d max_heap_used_mib %.1f io_attempts %d mem %s elf %s fitweight0 %s" % [
			_tag, _phase_rows.size(), _phase_count, tot_newton, "/".join(per), energy, tot_ms / 1000.0, max_instr, max_heap, io,
			_args.get("mem", "2048"), _args.get("elf", "res://fit.elf"), "yes" if _args.has("fitweight0") else "no"]
	if done:
		var c: String = _m.fit_check()
		_check("intersections on the final state", c == "OK none", c)
		var push: String = _m.fit_push_control()
		_check("push control (5 cm = 0.05 solve units into the avatar) -> INTERSECTS", push.find("INTERSECTS") >= 0, push)
		var flat: String = _m.fit_push_flat()
		_check("push flat control (same path, no push) -> none", flat.ends_with("OK none"), flat)
		_say("     fit_preview: " + _m.fit_preview())
		_say("     fit_result: " + _m.fit_result())
		var v = _m.fit_result_vertices_f64()
		if typeof(v) == TYPE_PACKED_FLOAT64_ARRAY:
			var f := FileAccess.open(RUNS + "/" + _tag + ".garment.f64", FileAccess.WRITE)
			f.store_buffer(v.to_byte_array())
			f.close()
			_say("     wrote runs/%s.garment.f64 (%d vertices, solve frame)" % [_tag, v.size() / 3])
		line += " intersections %s push %s flat %s" % ["none" if c == "OK none" else "INTERSECTS",
				"INTERSECTS" if push.find("INTERSECTS") >= 0 else "none", "none" if flat.ends_with("OK none") else "INTERSECTS"]
	_say(line)
	_finish()

# --- report -------------------------------------------------------------------------

func _run_summary(tag: String) -> Dictionary:
	var t := FileAccess.get_file_as_string(RUNS + "/" + tag + ".txt")
	var out := {"text": t}
	for l in t.split("\n", false):
		if l.begins_with("summary "):
			var toks := l.split(" ", false)
			for i in range(1, toks.size() - 1):
				out[toks[i]] = toks[i + 1]
			out["line"] = l
		if l.begins_with("RESULT: "):
			out["result"] = l.trim_prefix("RESULT: ")
	return out

func _exec(exe: String, args: Array) -> String:
	var out := []
	OS.execute(exe, PackedStringArray(args), out, true)
	return str(out[0]) if out.size() > 0 else ""

func _compare(a: String, b: String) -> Dictionary:
	var s := _exec(_native_exe(), ["--compare", a, b, "--voxel", str(VOXEL)])
	var d := {"text": s.strip_edges()}
	d["max"] = s.get_slice("per-vertex: max ", 1).get_slice(" ", 0).to_float()
	d["mean"] = s.get_slice("mean ", 1).get_slice(",", 0).to_float()
	d["haus"] = s.get_slice("point to surface): ", 1).get_slice(" ", 0).to_float()
	d["max_m"] = s.get_slice("[max ", 1).get_slice(" ", 0).to_float()
	return d

func _gap(g: String) -> Dictionary:
	var s := _exec(_args.get("python", "python"), [_abs("gates/6-fit/fit_gap.py"), ORACLE_AVATAR, _abs(NOFIT), str(VOXEL), g])
	return {"text": s.strip_edges(), "mean": s.get_slice("gap mean ", 1).get_slice(" ", 0).to_float(),
			"p95": s.get_slice("p95 ", 1).get_slice(" ", 0).to_float()}

func _criterion(tag: String) -> bool:
	var run := _run_summary(tag)
	var g := ProjectSettings.globalize_path(RUNS + "/" + tag + ".garment.f64")
	if not run.has("line") or not FileAccess.file_exists(g):
		_say("FAIL %s: no complete run (runs/%s.txt / .garment.f64)" % [tag, tag])
		return false
	_say("  run: " + run["line"])
	var c1: bool = run.get("intersections", "") == "none" and run.get("push", "") == "INTERSECTS"
	var c2: bool = run.get("io_attempts", "-1") == "0"
	var gap := _gap(g)
	var c3: bool = absf(gap["mean"] - GAP_MEAN_ORACLE) <= 0.25 and absf(gap["p95"] - GAP_P95_ORACLE) <= 0.5
	var e := float(run.get("energy", "nan"))
	var c4 := e >= ENERGY_BAND[0] and e <= ENERGY_BAND[1]
	var cmp := _compare(g, ORACLE)
	var c5: bool = cmp["haus"] <= HAUS_MAX and cmp["max"] <= VMAX_MAX
	_say("  (1) intersection-free: %s (final %s, push control %s)" % ["PASS" if c1 else "FAIL", run.get("intersections", "?"), run.get("push", "?")])
	_say("  (2) no file I/O: %s (io_attempts %s; the positive control before fit_begin is in the run log)" % ["PASS" if c2 else "FAIL", run.get("io_attempts", "?")])
	_say("  (3) the fit holds: %s (gap mean %.4f vs %.4f +- 0.25, p95 %.4f vs %.4f +- 0.5 voxels)" % ["PASS" if c3 else "FAIL", gap["mean"], GAP_MEAN_ORACLE, gap["p95"], GAP_P95_ORACLE])
	_say("  (4) energy in upstream's basin: %s (%s in [%s, %s])" % ["PASS" if c4 else "FAIL", str(e), str(ENERGY_BAND[0]), str(ENERGY_BAND[1])])
	_say("  (5) coarse geometric bound: %s (vs 1t: max %.4f / mean %.4f / Hausdorff %.4f voxels; bounds %.2f / %.2f)" % ["PASS" if c5 else "FAIL", cmp["max"], cmp["mean"], cmp["haus"], VMAX_MAX, HAUS_MAX])
	_say("      " + gap["text"])
	_say("      " + cmp["text"].replace("\n", "\n      "))
	return c1 and c2 and c3 and c4 and c5

func _report() -> void:
	var tag: String = _args.get("full", "full")
	_say("== Gate 6 foxgirl, guest run runs/%s.txt ==" % tag)
	# (1) guest vs native same code
	var run := _run_summary(tag)
	var g := ProjectSettings.globalize_path(RUNS + "/" + tag + ".garment.f64")
	_load_native_phases()
	var nat_newton := PackedStringArray()
	var nat_total := 0
	for p in _native_phases:
		nat_newton.append(str(p[0]))
		nat_total += p[0]
	var gn: PackedStringArray = str(run.get("newton", "")).split(" ")
	var per_guest: PackedStringArray = str(run.get("line", "")).get_slice("(", 1).get_slice(")", 0).split("/")
	var newton_ok := per_guest.size() == _native_phases.size()
	for i in mini(per_guest.size(), _native_phases.size()):
		newton_ok = newton_ok and absi(per_guest[i].to_int() - _native_phases[i][0]) <= 2
	var cmpn := _compare(g, _native_dir.path_join("garment_final.obj"))
	var e_g := float(run.get("energy", "nan"))
	var e_n: float = _native_phases[-1][1] if _native_phases.size() > 0 else NAN
	var e_rel := absf(e_g - e_n) / absf(e_n)
	var same_ok: bool = cmpn["max_m"] <= 1e-6 and e_rel <= 1e-6 and newton_ok
	_say("(1) guest vs native same code (f32 inputs, serial TBB stand-in, guest numerics): %s" % ("PASS" if same_ok else "MISS"))
	_say("    Newton per phase guest %s vs native %s (bound +-2); energy %s vs %s (rel %s, bound 1e-6); max |dv| %s solve units (bound 1e-6)" % [
			"/".join(per_guest), "/".join(nat_newton), str(run.get("energy", "?")), _native_phases[-1][3] if _native_phases.size() > 0 else "?", String.num_scientific(e_rel), String.num_scientific(cmpn["max_m"])])
	_say("    " + cmpn["text"].replace("\n", "\n    "))
	# Where the traces part (gates/6-fit/trace_diff.py): the guest's polyfem
	# debug log against fit_native's (--native_log, default <native>.log).
	var nlog: String = _args.get("native_log", _native_dir + ".log")
	var td := _exec(_args.get("python", "python"), [_abs("gates/6-fit/trace_diff.py"),
			ProjectSettings.globalize_path(RUNS + "/" + tag + ".godot.log"), nlog, "--context", "1"])
	_say("    trace: " + td.strip_edges().replace("\n", "\n      "))
	if not same_ok:
		_say("    -> the tight bound is missed; per gates/6-fit/README.md the guest result is judged by the five-part criterion (not loosened silently)")
	_say("(2) five-part criterion, guest result vs the 1-thread oracle %s:" % ORACLE)
	var five := _criterion(tag)
	_say("    five-part: %s" % ("PASS" if five else "FAIL"))
	if not five:
		_rc = 1
	# Negative control: fit_weight = 0 must fail criterion 3.
	var fw := "full-fw0"
	_say("== control: fit_weight = 0 in the guest (runs/%s.txt) must fail (3) ==" % fw)
	var gfw := ProjectSettings.globalize_path(RUNS + "/" + fw + ".garment.f64")
	if FileAccess.file_exists(gfw):
		var gap := _gap(gfw)
		var c3: bool = absf(gap["mean"] - GAP_MEAN_ORACLE) <= 0.25 and absf(gap["p95"] - GAP_P95_ORACLE) <= 0.5
		_say("    %s (3) %s: gap mean %.4f p95 %.4f voxels" % ["ok  " if not c3 else "FAIL", "fails as it must" if not c3 else "PASSES (control broken)", gap["mean"], gap["p95"]])
		_say("    " + str(_run_summary(fw).get("line", "")))
		if c3:
			_rc = 1
	else:
		_say("    FAIL no runs/%s.garment.f64" % fw)
		_rc = 1
	# 6.P: the full run again at 1.25x the heap floor (--confirm, default
	# full-440) must finish and give the same garment, bit for bit.
	var cf: String = _args.get("confirm", "full-440")
	_say("== 6.P confirm: runs/%s.txt (1.25x the smallest memory_max that ran phases 0+1) ==" % cf)
	var gc := ProjectSettings.globalize_path(RUNS + "/" + cf + ".garment.f64")
	if FileAccess.file_exists(gc) and FileAccess.file_exists(g):
		var same := FileAccess.get_file_as_bytes(gc) == FileAccess.get_file_as_bytes(g)
		_say("    %s the garment is %s the %s run's" % ["ok  " if same else "FAIL", "bitwise" if same else "NOT bitwise", tag])
		_say("    " + str(_run_summary(cf).get("line", "")))
		if not same:
			_rc = 1
	else:
		_say("    FAIL no runs/%s.garment.f64" % cf)
		_rc = 1
	# 6.P control: execution_timeout below one phase must stop the vmcall.
	var tc := "timeout-ctl"
	var tlog := FileAccess.get_file_as_string(RUNS + "/" + tc + ".godot.log")
	var tsum := _run_summary(tc)
	var stopped: bool = tlog.find("Sandbox: Timeout for") >= 0 and str(tsum.get("line", "")).find("FAILED at phase 0") >= 0
	_say("== 6.P control: runs/%s.txt, execution_timeout 100000 units (< phase 0's ~463,000) must stop phase 0 ==" % tc)
	_say("    %s %s" % ["ok  " if stopped else "FAIL", "the sandbox timed the vmcall out (\"Sandbox: Timeout for fit_step\")" if stopped else "no timeout seen"])
	if not stopped:
		_rc = 1
	# Everything else the runs recorded (6.0, 6.P). The ladder arms below the
	# floor and the timeout control FAIL by design.
	_say("== runs ==")
	var d := DirAccess.open(ProjectSettings.globalize_path(RUNS))
	var names := []
	for f in d.get_files():
		if f.ends_with(".txt"):
			names.append(f)
	names.sort()
	for f in names:
		var s := _run_summary(f.trim_suffix(".txt"))
		var last: String = s.get("line", "")
		if last == "":
			var ls: PackedStringArray = str(s["text"]).strip_edges().split("\n")
			last = ls[-1].strip_edges() if ls.size() > 0 else ""
		_say("  %-18s %-5s %s" % [f.trim_suffix(".txt"), s.get("result", "-"), last.left(400)])
