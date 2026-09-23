# Gate 8 unit tests, no ELF needed: skeleton15 (identity on FoxGirl, a
# permuted rig by topology, a permuted named rig with extra joints, and
# refusals), the scripted pen (deterministic, rings meet, the drop-seam
# control), mesh_topo on an open tube.
#
#   godot --path project --script tests/test_loop_units.gd --rendering-driver vulkan --xr-mode off
#
# Results: gates/8-loop/units.txt, last line RESULT: PASS | FAIL. Quits on a
# 60 s wall clock.
extends SceneTree

const ObjIO := preload("res://util/obj_io.gd")
const Skeleton15 := preload("res://util/skeleton15.gd")
const PenSource := preload("res://xr/pen_source_scripted.gd")
const MeshTopo := preload("res://util/mesh_topo.gd")
const MeshWire := preload("res://util/mesh_wire.gd")
const OUT := "res://../gates/8-loop/units.txt"

var _out: FileAccess
var _fails := 0
var _n := 0

func _say(s: String) -> void:
	print(s)
	if _out != null:
		_out.store_line(s)
		_out.flush()

func _ok(name: String, cond: bool, detail: String = "") -> void:
	_n += 1
	if not cond:
		_fails += 1
	_say("%s %s%s" % ["PASS" if cond else "FAIL", name, (": " + detail) if detail != "" else ""])

func _initialize() -> void:
	create_timer(60.0).timeout.connect(func(): _say("RESULT: FAIL (60 s wall clock)"); quit(1))
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(OUT).get_base_dir())
	_out = FileAccess.open(ProjectSettings.globalize_path(OUT), FileAccess.WRITE)
	_say("Gate 8 units, Godot %s" % Engine.get_version_info().string)
	_skeleton()
	_pen()
	_topo()
	_fit_budget()
	_say("%d checks, %d failed" % [_n, _fails])
	_say("RESULT: %s" % ("PASS" if _fails == 0 else "FAIL"))
	_out.close()
	quit(0 if _fails == 0 else 1)

func _perm(n: int, seed: int) -> PackedInt32Array:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var p := PackedInt32Array(range(n))
	for i in range(n - 1, 0, -1):
		var j := rng.randi_range(0, i)
		var t := p[i]
		p[i] = p[j]
		p[j] = t
	return p

# new index of old joint i is inv[i]; rig joint k holds old joint perm[k].
func _permute(pos: PackedFloat32Array, bones: PackedInt32Array, perm: PackedInt32Array, names := PackedStringArray()) -> Dictionary:
	var inv := PackedInt32Array()
	inv.resize(perm.size())
	for k in perm.size():
		inv[perm[k]] = k
	var p2 := PackedFloat32Array()
	var n2 := PackedStringArray()
	for k in perm.size():
		p2.append_array([pos[3 * perm[k]], pos[3 * perm[k] + 1], pos[3 * perm[k] + 2]])
		if not names.is_empty():
			n2.append(names[perm[k]])
	var b2 := PackedInt32Array()
	for b in bones:
		b2.append(inv[b])
	return {"positions": p2, "bones": b2, "names": n2, "inv": inv}

func _skeleton() -> void:
	var sk := ObjIO.read("res://fixtures/foxgirl/skeleton.obj")
	_ok("fixture skeleton reads", not sk.has("error") and sk.v.size() == 45 and sk.l.size() == 28,
			"%d floats, %d bone ids" % [sk.v.size(), sk.l.size()])
	var a := Skeleton15.adapt({"positions": sk.v, "bones": sk.l})
	_ok("FoxGirl -> identity", a.error == "" and a.map == PackedInt32Array(range(15)) and a.joints == sk.v,
			"method %s map %s" % [a.get("method", "?"), str(Array(a.get("map", [])))])
	_ok("layout bones == FoxGirl's own", Skeleton15.layout_bones() == sk.l, str(Array(sk.l)))
	var gs := ObjIO.read("res://fixtures/foxgirl/garment_skeleton.obj")
	var ga := Skeleton15.adapt({"positions": gs.v, "bones": gs.l})
	_ok("garment skeleton -> identity", ga.error == "" and ga.map == PackedInt32Array(range(15)), str(Array(ga.get("map", []))))
	# A permuted rig (topology): every slot must land on the original joint.
	for seed in [1, 7, 12345]:
		var pr := _permute(sk.v, sk.l, _perm(15, seed))
		var r := Skeleton15.adapt(pr)
		var want := PackedInt32Array()
		for s in 15:
			want.append(pr.inv[s])
		_ok("permuted rig seed %d (topology)" % seed, r.error == "" and r.map == want and r.joints == sk.v,
				"map %s want %s" % [str(Array(r.get("map", []))), str(Array(want))])
	# A named humanoid rig with a spine joint, clavicles, toes and a head end,
	# permuted: names decide.
	var names := PackedStringArray(["Hips", "UpperChest", "Head", "LeftUpperArm", "LeftLowerArm", "LeftHand",
			"RightUpperArm", "RightHand", "RightLowerArm", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
			"RightUpperLeg", "RightLowerLeg", "RightFoot", "Spine", "LeftShoulder", "RightShoulder", "LeftToes", "RightToes"])
	var pos: PackedFloat32Array = sk.v.duplicate()
	var extra := [Vector3(0, 1.0, 0), Vector3(0.05, 1.25, 0), Vector3(-0.05, 1.25, 0), Vector3(0.17, 0.0, 0.1), Vector3(-0.17, 0.0, 0.1)]
	for e in extra:
		pos.append_array([e.x, e.y, e.z])
	var bones := PackedInt32Array([0, 15, 15, 1, 1, 2, 1, 16, 16, 3, 3, 4, 4, 5, 1, 17, 17, 6, 6, 8, 7, 8, 0, 9, 9, 10,
			10, 11, 11, 18, 0, 12, 12, 13, 13, 14, 14, 19])
	var pn := _permute(pos, bones, _perm(20, 99), names)
	var rn := Skeleton15.adapt(pn)
	var wantn := PackedInt32Array()
	for s in 15:
		wantn.append(pn.inv[s])
	_ok("named rig, 20 joints permuted (names)", rn.error == "" and rn.method == "names" and rn.map == wantn,
			"method %s map %s" % [rn.get("method", "?"), str(Array(rn.get("map", [])))])
	# The same rig without names: topology skips the clavicles, toes and spine.
	pn.names = PackedStringArray()
	var rt := Skeleton15.adapt(pn)
	_ok("same rig unnamed (topology: clavicle, toes, spine skipped)", rt.error == "" and rt.map == wantn,
			"method %s map %s" % [rt.get("method", "?"), str(Array(rt.get("map", [])))])
	# Refusals: a one-armed rig, a chain.
	var one_arm := Skeleton15.adapt({"positions": sk.v.slice(0, 42), "bones": PackedInt32Array([0, 1, 1, 2, 1, 3, 3, 4, 4, 5,
			1, 6, 6, 8, 7, 8, 0, 9, 9, 10, 10, 11, 0, 12, 12, 13])})
	_ok("control: 14 joints refused", one_arm.error != "", one_arm.error)
	var chain := Skeleton15.adapt({"positions": sk.v, "bones": PackedInt32Array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
			7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14])})
	_ok("control: a chain refused", chain.error != "", chain.error)

func _pen() -> void:
	var body := ObjIO.read("res://fixtures/foxgirl/avatar.obj")
	var sk := ObjIO.read("res://fixtures/foxgirl/skeleton.obj")
	_ok("fixture body reads", not body.has("error") and body.v.size() == 5128 * 3 and body.f.size() == 10171 * 3,
			"%d v %d f" % [body.v.size() / 3, body.f.size() / 3])
	var box := AABB(Vector3(body.v[0], body.v[1], body.v[2]), Vector3.ZERO)
	for i in range(0, body.v.size(), 3):
		box = box.expand(Vector3(body.v[i], body.v[i + 1], body.v[i + 2]))
	_ok("fixture body is 1.7 m tall, feet on y=0", absf(box.size.y - 1.7) < 1e-4 and absf(box.position.y) < 1e-4, str(box))
	var a := PenSource.make(body.v, sk.v)
	var b := PenSource.make(body.v, sk.v)
	_ok("scripted pen made", a.error == "", "waist %s | hem %s" % [str(a.get("waist", {}).get("radius")), str(a.get("hem", {}).get("radius"))])
	_ok("deterministic", JSON.stringify(a.events) == JSON.stringify(b.events), "%d events" % a.events.size())
	var names := []
	for s in a.strokes:
		names.append(s.name)
	_ok("6 strokes, none closed", a.strokes.size() == 6, str(names))
	# The rings are boundary strokes (their cycles are openings), the seams not;
	# the begin event carries the mark to curvenet's pen mode.
	var marks := []
	for s in a.strokes:
		marks.append("%s=%s" % [s.name, str(s.boundary)])
	var ring_marked: bool = a.strokes.all(func(s): return bool(s.boundary) == (str(s.name).begins_with("waist")
			or str(s.name).begins_with("hem")))
	var begins: Array = a.events.filter(func(e): return e.kind == "begin")
	_ok("rings are boundary strokes, seams are not", ring_marked and begins.size() == 6
			and begins.all(func(e): return e.boundary == a.strokes[e.stroke].boundary), ", ".join(marks))
	var nb := PenSource.make(body.v, sk.v, {"no_boundary": true})
	_ok("control: no_boundary marks no stroke", nb.strokes.all(func(s): return not s.boundary))
	# Endpoints: the 4 knots, each shared by exactly 3 stroke ends.
	var ends := {}
	for s in a.strokes:
		var pts: PackedVector3Array = s.points
		_ok("stroke %s open" % s.name, not pts[0].is_equal_approx(pts[-1]))
		for p in [pts[0], pts[-1]]:
			var key := "%.6f,%.6f,%.6f" % [p.x, p.y, p.z]
			ends[key] = ends.get(key, 0) + 1
	_ok("4 knots of degree 3", ends.size() == 4 and ends.values().all(func(c): return c == 3), str(ends.values()))
	_ok("waist above hem", a.waist.center.y > a.hem.center.y + 0.3, "%.3f vs %.3f" % [a.waist.center.y, a.hem.center.y])
	_ok("ring radii torso-sized (hands left out)", a.waist.radius > 0.1 and a.waist.radius < 0.25 and a.hem.radius > 0.1
			and a.hem.radius < 0.3, "waist %.4f hem %.4f" % [a.waist.radius, a.hem.radius])
	_say("INFO straight-cone clearance to the body between the rings: %.4f m before growing, grown by %.4f m" % [
			a.clearance_before_grow, a.grow])
	_ok("the cone between the rings clears the body by 1 cm", a.min_clearance >= 0.01 - 1e-6, "%.4f m" % a.min_clearance)
	var d := PenSource.make(body.v, sk.v, {"drop_seam": true})
	_ok("control: drop_seam leaves 5 strokes", d.strokes.size() == 5 and d.strokes[-1].name == "seam_front")
	var c := PenSource.make(body.v, sk.v, {"closed_rings": true})
	var closed := 0
	for s in c.strokes:
		var pts: PackedVector3Array = s.points
		if pts[0] == pts[-1]:
			closed += 1
	_ok("closed_rings: two closed strokes", closed == 2, "%d closed of %d" % [closed, c.strokes.size()])

func _topo() -> void:
	var cyl := CylinderMesh.new()
	cyl.cap_top = false
	cyl.cap_bottom = false
	cyl.radial_segments = 16
	cyl.rings = 3
	var w := MeshWire.from_godot_arrays(cyl.get_mesh_arrays())
	# Godot's cylinder duplicates the seam column; weld by position first.
	var weld := _weld(w.vertices, w.triangles)
	var loops := MeshTopo.boundary_loops(weld.triangles)
	var hi := MeshTopo.highest_loop(weld.vertices, loops)
	_ok("open tube: 2 boundary loops of 16", loops.size() == 2 and loops[0].size() == 16 and loops[1].size() == 16,
			str(loops.map(func(l): return l.size())))
	_ok("open tube: the top loop is highest", hi >= 0 and absf(MeshTopo.mean_y(weld.vertices, loops[hi]) - 1.0) < 1e-5,
			"mean y %.4f" % (MeshTopo.mean_y(weld.vertices, loops[hi]) if hi >= 0 else NAN))
	_ok("open tube: one component", MeshTopo.components(weld.vertices.size() / 3, weld.triangles) == 1)
	var two: PackedInt32Array = weld.triangles.duplicate()
	var shift: int = weld.vertices.size() / 3
	var v2: PackedFloat32Array = weld.vertices.duplicate()
	v2.append_array(weld.vertices)
	for t in weld.triangles:
		two.append(t + shift)
	_ok("control: two tubes -> two components, four loops", MeshTopo.components(v2.size() / 3, two) == 2
			and MeshTopo.boundary_loops(two).size() == 4)

# The fit budget's text edits on fixtures/foxgirl/fit_config.json (pipeline
# _fit_budget): the result parses, carries each edit once, and leaves every
# integer an integer (a GDScript JSON round trip would not).
func _fit_budget() -> void:
	var Pipeline = load("res://stages/pipeline.gd")
	var cfg := FileAccess.get_file_as_string("res://fixtures/foxgirl/fit_config.json")
	_ok("fit_config.json reads", not cfg.is_empty())
	var p = Pipeline.new()
	p.opts = Pipeline.DEFAULTS.duplicate(true)
	var r: Dictionary = p._fit_budget(cfg)
	_ok("default budget edits apply", not r.has("error"), str(r.get("error", r.get("note", ""))))
	var j = JSON.parse_string(str(r.get("text", "")))
	_ok("edited config parses", typeof(j) == TYPE_DICTIONARY)
	if typeof(j) != TYPE_DICTIONARY:
		return
	var al: Dictionary = j.solver.augmented_lagrangian
	_ok("incremental_steps 1, an integer literal", int(j.incremental_steps) == 1
			and str(r.text).find('"incremental_steps": 1,') >= 0)
	_ok("solver.nonlinear.Newton: force_psd_projection (the AL's nonlinear inherits it)",
			j.solver.nonlinear.Newton.get("force_psd_projection", false) == true
			and j.solver.nonlinear.Newton.use_psd_projection == true, str(j.solver.nonlinear.Newton))
	_ok("AL nonlinear untouched", not al.nonlinear.has("Newton") and al.nonlinear.grad_norm == 1
			and al.nonlinear.max_iterations == 50, str(al.nonlinear))
	_ok("max_iterations stay integer literals", str(r.text).find('"max_iterations": 50') >= 0
			and str(r.text).find('"max_iterations": 5000') >= 0 and str(r.text).find('"max_iterations": 200') >= 0)
	p.opts.fit_grad_norm = 0.03
	var rg: Dictionary = p._fit_budget(cfg)
	var jg = JSON.parse_string(str(rg.get("text", "")))
	_ok("fit_grad_norm 0.03: the reduced solve's grad_norm, not the AL's", not rg.has("error") and typeof(jg) == TYPE_DICTIONARY
			and is_equal_approx(float(jg.solver.nonlinear.grad_norm), 0.03)
			and float(jg.solver.augmented_lagrangian.nonlinear.grad_norm) == 1.0, str(rg.get("note", rg.get("error", ""))))
	p.opts.fit_grad_norm = -1.0
	p.opts.fit_force_psd = false
	var r0: Dictionary = p._fit_budget(cfg)
	_ok("control: fit_force_psd off leaves Newton as fit_config.json has it", not r0.has("error")
			and str(r0.text).find("force_psd_projection") < 0 and str(r0.text) == cfg.replace('"incremental_steps": 2', '"incremental_steps": 1'))
	p.free()

func _weld(v: PackedFloat32Array, f: PackedInt32Array) -> Dictionary:
	var at := {}
	var remap := PackedInt32Array()
	var out := PackedFloat32Array()
	for i in v.size() / 3:
		var key := "%.5f,%.5f,%.5f" % [v[3 * i], v[3 * i + 1], v[3 * i + 2]]
		if not at.has(key):
			at[key] = out.size() / 3
			out.append_array([v[3 * i], v[3 * i + 1], v[3 * i + 2]])
		remap.append(at[key])
	var tris := PackedInt32Array()
	for t in f:
		tris.append(remap[t])
	return {"vertices": out, "triangles": tris}
