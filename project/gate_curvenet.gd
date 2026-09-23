# Gate 4: curvenet.elf -- Cassie's pen -> curvenet -> mesh and mesh ->
# curvenet in the guest, against the same checks run natively.
#
#   tests/native/curvenet/build.sh   (writes gates/4-curvenet/native-checks.log)
#   godot --path project --script gate_curvenet.gd --rendering-driver vulkan --xr-mode off > ../gates/4-curvenet/run.log 2>&1
#
# 1. checks: every guest/curvenet/checks.cpp check in the guest, one vmcall
#    each, host-timed, heap after; each must PASS.
# 2. flat control: the same lines from the native cassie_checks.exe
#    (native-checks.log). Verdicts and integer outputs must match exactly;
#    float signatures (FNV-1a over float32 bit patterns) are compared as hex.
# 2b. determinism: every check again in the same guest; each line must be
#    byte-identical to the first pass (Delaunay's BRIO shuffle used to seed
#    from std::random_device; README).
# 3. the FAIL paths answer "FAIL: ..." instead of unwinding into the host.
# 4. the pen through the API as a host drives it: a Godot SphereMesh body
#    (rewound by util/mesh_wire.gd), one closed stroke sample by sample;
#    pen_end, mesh_build and curvenet_build host-timed; the wire buffers
#    decoded (1 patch, 1 loop, CCW-outward, 2 curves, 2 knots).
# 5. mesh -> curvenet on a cube through the wire (12 curves, 8 knots).
# 6. llvm-nm -C curvenet.elf: 0 Eigen:: symbols (control: Cassie's own
#    symbols are there, so the table was read).
# 7. rule 8: every ADD_API_FUNCTION in guest/curvenet/main.cpp has a
#    project/main.gd wrapper (a delegate into stages/curvenet_stage.gd),
#    all its arguments defaulted (with controls).
#
# No GPU: the stage never opens a RenderingDevice. One step per frame; quits
# on a 300 s wall clock whatever step it is in. Results stream to
# gates/4-curvenet/results.txt; the last line is RESULT: PASS or RESULT: FAIL.
extends SceneTree

const MeshWire := preload("res://util/mesh_wire.gd")
const OUT_DIR := "res://../gates/4-curvenet/"
const WALL_S := 300.0
const STEPS := ["checks", "compare", "repeat", "failpaths", "pen", "extract", "nm", "wrappers"]

var _sb = null
var _out: FileAccess
var _t_start := 0
var _rc := 0
var _step := 0
var _done := false
var _guest := {}
var _names: PackedStringArray

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _fail(why: String) -> void:
	_rc = 1
	_say("FAIL: " + why)

func _check(ok: bool, what: String) -> bool:
	_say("%s %s" % ["PASS" if ok else "FAIL", what])
	if not ok:
		_rc = 1
	return ok

func _finish() -> void:
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	if _out != null:
		_out.close()
		_out = null
	if _sb != null:
		_sb.free()
		_sb = null
	_done = true
	quit(_rc)

# A vmcall, host-timed (the guest clock is not a clock): [result, us].
func _call(fn: String, args: Array = []) -> Array:
	var t := Time.get_ticks_usec()
	var r = _sb.callv("vmcall", [fn] + args)
	return [r, Time.get_ticks_usec() - t]

func _heap() -> int:
	return int(_sb.get_heap_usage())

func _initialize() -> void:
	_t_start = Time.get_ticks_usec()
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT_DIR + "results.txt"), FileAccess.WRITE)
	_say("# Gate 4 (curvenet.elf), %s, Godot %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string])
	_sb = ClassDB.instantiate("Sandbox")
	if _sb == null:
		_fail("Sandbox class not registered")
		_finish()
		return
	_sb.references_max = 4096
	var t := Time.get_ticks_usec()
	_sb.program = load("res://curvenet.elf")
	_say("load curvenet.elf: %.1f ms; execution_timeout=%s memory_max=%s heap=%d" % [
			(Time.get_ticks_usec() - t) / 1000.0, str(_sb.execution_timeout), str(_sb.memory_max), _heap()])

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_usec() - _t_start > int(WALL_S * 1e6):
		_fail("the %d s wall clock ran out at step %s" % [int(WALL_S), STEPS[mini(_step, STEPS.size() - 1)]])
		_finish()
		return true
	if _step >= STEPS.size():
		_finish()
		return true
	var s: String = STEPS[_step]
	_step += 1
	_say("--- " + s)
	call("_" + s)
	return _done

# --- 1. the checks in the guest -------------------------------------------------

func _checks() -> void:
	_names = str(_call("check_names")[0]).split(" ", false)
	_check(_names.size() == 9, "check_names: %d checks (%s)" % [_names.size(), " ".join(_names)])
	for n in _names:
		var r := _call("check", [n])
		var line := str(r[0])
		_guest[n] = line
		_say("guest %-22s %9.1f ms heap=%9d  %s" % [n, r[1] / 1000.0, _heap(), line])
		if not line.begins_with("PASS "):
			_rc = 1

# --- 2. guest vs native ------------------------------------------------------------

func _parse(line: String) -> Dictionary:
	var re := RegEx.new()
	re.compile("^(PASS|FAIL) (\\S+) ints=(\\S*) fsig=([0-9a-f]{16})/(\\d+)")
	var m := re.search(line)
	if m == null:
		return {}
	return {"verdict": m.get_string(1), "name": m.get_string(2), "ints": m.get_string(3),
			"fsig": m.get_string(4), "nf": m.get_string(5).to_int()}

func _compare() -> void:
	var f := FileAccess.open(ProjectSettings.globalize_path(OUT_DIR + "native-checks.log"), FileAccess.READ)
	if f == null:
		_fail("no native-checks.log (run tests/native/curvenet/build.sh first)")
		return
	var native := {}
	for line in f.get_as_text().split("\n"):
		var d := _parse(line.strip_edges())
		if not d.is_empty():
			native[d.name] = d
	f.close()
	_check(native.size() == _names.size(), "native-checks.log: %d check lines, guest %d" % [native.size(), _names.size()])
	var ints_ok := 0
	var fsig_ok := 0
	for n in _names:
		var g := _parse(str(_guest.get(n, "")))
		var nv: Dictionary = native.get(n, {})
		if g.is_empty() or nv.is_empty():
			_fail("%s: unparsable (guest %s, native %s)" % [n, not g.is_empty(), not nv.is_empty()])
			continue
		var same_i: bool = g.verdict == nv.verdict and g.ints == nv.ints
		var same_f: bool = g.fsig == nv.fsig and g.nf == nv.nf
		ints_ok += 1 if same_i else 0
		fsig_ok += 1 if same_f else 0
		_say("%s %-22s verdict %s/%s ints %s fsig guest %s/%d native %s/%d%s" % [
				"same " if same_i and same_f else ("INTS " if not same_i else "FLOAT"), n, g.verdict, nv.verdict,
				g.ints if same_i else "guest %s native %s" % [g.ints, nv.ints], g.fsig, g.nf, nv.fsig, nv.nf,
				"" if same_f else "  <- float signature differs"])
	_check(ints_ok == _names.size(), "guest vs native: verdicts and integer outputs identical in %d/%d checks" % [ints_ok, _names.size()])
	_check(fsig_ok == _names.size(), "guest vs native: float signatures identical in %d/%d checks" % [fsig_ok, _names.size()])

# --- 2b. the same checks again, same process -------------------------------------

func _repeat() -> void:
	var same := 0
	for n in _names:
		var line := str(_call("check", [n])[0])
		if line == str(_guest.get(n, "")):
			same += 1
		else:
			_say("differs on repeat: %s\n  first:  %s\n  second: %s" % [n, _guest.get(n, ""), line])
	_check(same == _names.size(), "repeat in the same guest: %d/%d check lines byte-identical" % [same, _names.size()])

# --- 3. FAIL paths ---------------------------------------------------------------------

func _failpaths() -> void:
	var a := str(_call("cn_set_param", ["no_such_param", 1.0])[0])
	_check(a.begins_with("FAIL: unknown param"), "cn_set_param(no_such_param) -> %s" % a)
	var b := str(_call("curvenet_extract", [PackedFloat32Array([0, 0, 0]), PackedInt32Array([0, 1, 2]), 200, 1e-3, 1e-2, 0.0])[0])
	_check(b.begins_with("FAIL: triangles[1]"), "curvenet_extract(out-of-range index) -> %s" % b)
	var c := str(_call("mesh_build", [0.0, 1e-5])[0])
	_check(c.begins_with("ok patches=0"), "mesh_build with no patches -> %s" % c)

# --- 4. the pen as a host drives it ------------------------------------------------

func _signed_volume(v: PackedFloat32Array, f: PackedInt32Array) -> float:
	var vol := 0.0
	for t in range(0, f.size(), 3):
		var a := Vector3(v[3 * f[t]], v[3 * f[t] + 1], v[3 * f[t] + 2])
		var b := Vector3(v[3 * f[t + 1]], v[3 * f[t + 1] + 1], v[3 * f[t + 1] + 2])
		var c := Vector3(v[3 * f[t + 2]], v[3 * f[t + 2] + 1], v[3 * f[t + 2] + 2])
		vol += a.dot(b.cross(c)) / 6.0
	return vol

func _area_normal(v: PackedFloat32Array, f: PackedInt32Array) -> Vector3:
	var n := Vector3.ZERO
	for t in range(0, f.size(), 3):
		var a := Vector3(v[3 * f[t]], v[3 * f[t] + 1], v[3 * f[t] + 2])
		var b := Vector3(v[3 * f[t + 1]], v[3 * f[t + 1] + 1], v[3 * f[t + 1] + 2])
		var c := Vector3(v[3 * f[t + 2]], v[3 * f[t + 2] + 1], v[3 * f[t + 2] + 2])
		n += (b - a).cross(c - a)
	return n

func _pen() -> void:
	var body := MeshWire.sphere(0.5)
	var vol := _signed_volume(body.vertices, body.triangles)
	_check(vol > 0.5, "body: SphereMesh r=0.5 rewound to wire winding, %d vertices %d triangles, signed volume %.4f (4/3 pi r^3 = %.4f; > 0 is CCW-outward)" % [
			body.vertices.size() / 3, body.triangles.size() / 3, vol, 4.0 / 3.0 * PI * 0.125])
	var h0 := _heap()
	var sb := _call("cn_set_body", [body.vertices, body.triangles])
	_say("cn_set_body: %.1f ms  %s" % [sb[1] / 1000.0, sb[0]])
	var pen_end_us := []
	var last := ""
	for rep in 3:
		_call("cn_reset")
		var s := MeshWire.circle_stroke(0.51, PI / 6.0, TAU, 64)
		var t := Time.get_ticks_usec()
		var id := int(_call("pen_begin", [s[0], s[1], s[2], s[3]])[0])
		for i in range(4, s.size(), 4):
			_sb.vmcall("pen_point", id, s[i], s[i + 1], s[i + 2], s[i + 3])
		var points_us := Time.get_ticks_usec() - t
		var e := _call("pen_end", [id])
		pen_end_us.append(e[1])
		last = str(e[0])
		_say("stroke %d: 65 samples, pen_begin+64 pen_point %.1f ms (%.1f us/sample), pen_end %.1f ms, heap %d  %s" % [
				rep, points_us / 1000.0, points_us / 65.0, e[1] / 1000.0, _heap(), last])
	pen_end_us.sort()
	_say("pen_end latency (host-timed, 3 strokes): min %.1f ms, median %.1f ms, max %.1f ms" % [
			pen_end_us[0] / 1000.0, pen_end_us[1] / 1000.0, pen_end_us[2] / 1000.0])
	_check(last.contains("closed=1") and last.contains(" patches=1"), "closed stroke -> 1 patch (%s)" % last)
	var pc := int(_call("patch_count")[0])
	var pv: PackedFloat32Array = _call("patch_vertices", [0])[0]
	var pf: PackedInt32Array = _call("patch_indices", [0])[0]
	_check(pc == 1 and pv.size() > 0 and pf.size() > 0, "patch 0 over the wire: %d vertices, %d triangles" % [pv.size() / 3, pf.size() / 3])
	for target in [0.0, 0.02]:
		var m := _call("mesh_build", [target, 1e-5])
		var v: PackedFloat32Array = _call("mesh_vertices")[0]
		var f: PackedInt32Array = _call("mesh_indices")[0]
		var loops := MeshWire.loops(_call("mesh_boundary_loops")[0])
		var ids: PackedInt32Array = _call("mesh_patch_ids")[0]
		var n := _area_normal(v, f)
		var am := MeshWire.to_array_mesh(v, f)
		_say("mesh_build(%.2f): %.1f ms, heap %d  %s" % [target, m[1] / 1000.0, _heap(), m[0]])
		_check(str(m[0]).begins_with("ok") and loops.size() == 1 and ids.size() == f.size() / 3 and am.get_surface_count() == 1,
				"mesh_build(%.2f): %d vertices, %d triangles, %d boundary loop (%d vertices), %d patch ids, ArrayMesh with %d surface" % [
				target, v.size() / 3, f.size() / 3, loops.size(), loops[0].size() if loops.size() > 0 else 0, ids.size(), am.get_surface_count()])
		_check(n.normalized().y > 0.99, "mesh_build(%.2f): CCW-outward, area-weighted normal %s (the cap's outward is +Y)" % [target, n.normalized()])
	var cb := _call("curvenet_build")
	var curves := MeshWire.curves(_call("curvenet_curves")[0])
	var knots := MeshWire.knots(_call("curvenet_knots")[0])
	_say("curvenet_build: %.1f ms  %s" % [cb[1] / 1000.0, cb[0]])
	var ends_ok := true
	for c in curves:
		ends_ok = ends_ok and c.knots.x >= 0 and c.knots.y >= 0 and c.points.size() >= 2
	_check(curves.size() == 2 and knots.size() == 2 and ends_ok, "curvenet over the wire: %d curves (both ends on knots: %s), %d knots of degree %s; Curve3D length %.3f m" % [
			curves.size(), ends_ok, knots.size(), [knots[0].degree, knots[1].degree] if knots.size() == 2 else [],
			MeshWire.to_curve3d(curves[0]).get_baked_length() + MeshWire.to_curve3d(curves[1]).get_baked_length() if curves.size() == 2 else 0.0])
	_say("heap: %d before the body, %d after the pen, mesh and curvenet steps" % [h0, _heap()])
	_call("cn_set_body", [PackedFloat32Array(), PackedInt32Array()])
	_call("cn_reset")

# --- 5. mesh -> curvenet --------------------------------------------------------------

func _extract() -> void:
	var cube := MeshWire.cube()
	var r := _call("curvenet_extract", [cube.vertices, cube.triangles, 200, 1e-3, 1e-2, 0.0])
	var curves := MeshWire.curves(_call("curvenet_curves")[0])
	var knots := MeshWire.knots(_call("curvenet_knots")[0])
	var deg3 := 0
	for k in knots:
		deg3 += 1 if k.degree >= 3 and k.is_intersection else 0
	_say("curvenet_extract(cube): %.1f ms  %s" % [r[1] / 1000.0, r[0]])
	_check(curves.size() == 12 and knots.size() == 8 and deg3 == 8, "cube over the wire: %d curves, %d knots, %d intersections of degree >= 3" % [
			curves.size(), knots.size(), deg3])

# --- 6. no Eigen in curvenet.elf -----------------------------------------------------

func _nm() -> void:
	var elf := ProjectSettings.globalize_path("res://curvenet.elf")
	var out := []
	var code := OS.execute("llvm-nm", ["-C", elf], out, true)
	if code != 0 or out.is_empty():
		_fail("llvm-nm -C %s exited %d (llvm-nm on PATH?)" % [elf, code])
		return
	var total := 0
	var eigen := 0
	var cassie := 0
	for line in str(out[0]).split("\n"):
		if line.is_empty():
			continue
		total += 1
		eigen += 1 if line.contains("Eigen::") else 0
		cassie += 1 if line.contains("CassieSketcher::") else 0
	_check(eigen == 0 and cassie > 0, "llvm-nm -C curvenet.elf: %d symbols, %d Eigen:: (must be 0), %d CassieSketcher:: (control: > 0)" % [total, eigen, cassie])


# --- 7. rule 8 --------------------------------------------------------------------------
# Every ADD_API_FUNCTION in guest/curvenet/main.cpp must be reached by a public
# project/main.gd function, and every such wrapper's parameters must all have
# defaults. Since Cut 8 main.gd is a thin root: a wrapper is a delegate to
# stages/curvenet_stage.gd, and the guest name's string literal ("name") is in
# the stage function it calls or in one that calls in turn
# (tests/wrapper_audit.gd). Read from the source text, so a wrapper that only
# exists as a helper call is not counted. Controls: the same audit on main.gd
# with the cn_get_param delegate deleted, and with one default stripped, must
# each report exactly that.

const Audit := preload("res://tests/wrapper_audit.gd")

func _wrappers() -> void:
	var api := Audit.api_by_stage({"guest/curvenet/main.cpp": "curvenet"})
	var gd := Audit.read("res://main.gd")
	var stages := {"curvenet": Audit.read(Audit.STAGE_FILES["curvenet"])}
	var n: int = api["curvenet"].size()
	if not _check(n > 0 and not gd.is_empty() and not stages.curvenet.is_empty(),
			"rule 8: %d ADD_API_FUNCTION in guest/curvenet/main.cpp, main.gd %d bytes, curvenet_stage.gd %d bytes" % [
			n, gd.length(), stages.curvenet.length()]):
		return
	var a := Audit.audit(api, gd, stages)
	_check(a[0].is_empty(), "rule 8: every curvenet entry point has a main.gd wrapper (%d/%d; %d wrappers)%s" % [
			n - a[0].size(), n, a[2].size(), "" if a[0].is_empty() else " missing: %s" % ", ".join(a[0])])
	_check(a[1].is_empty(), "rule 8: every wrapper argument has a default%s" % ["" if a[1].is_empty() else " (no default: %s)" % ", ".join(a[1])])
	# controls: delete the cn_get_param delegate; strip patch_vertices' default
	var kept := PackedStringArray()
	for l in gd.split("\n"):
		if not l.begins_with("func cn_get_param("):
			kept.append(l)
	var c1 := Audit.audit(api, "\n".join(kept), stages)
	_check(c1[0] == ["curvenet:cn_get_param"], "rule 8 control: main.gd without the cn_get_param delegate -> missing %s" % str(c1[0]))
	var no_def := gd.replace("func patch_vertices(i: int = 0)", "func patch_vertices(i: int)")
	var c2 := Audit.audit(api, no_def, stages)
	_check(c2[1] == ["patch_vertices(i: int)"], "rule 8 control: patch_vertices(i: int) -> no default %s" % str(c2[1]))
	# the wrappers actually load: every one is a method of the compiled script
	var bad := Audit.not_callable(a[2], load("res://main.gd"))
	_check(bad.is_empty(), "project/main.gd compiles; its %d wrappers take no required argument: %s%s" % [a[2].size(), ", ".join(a[2]),
			"" if bad.is_empty() else " (missing or with required args: %s)" % ", ".join(bad)])
