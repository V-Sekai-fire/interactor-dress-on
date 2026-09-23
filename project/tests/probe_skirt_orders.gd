# Scratch probe (integration copy only): which stroke order / layout makes
# curvenet.elf patch the skirt's panels rather than its waist/hem caps.
extends SceneTree

const ObjIO := preload("res://util/obj_io.gd")
const PenSource := preload("res://xr/pen_source_scripted.gd")
const MeshWire := preload("res://util/mesh_wire.gd")
const MeshTopo := preload("res://util/mesh_topo.gd")
const SandboxUtil := preload("res://stages/sandbox_util.gd")

var _out: FileAccess

func _say(s: String) -> void:
	print(s)
	_out.store_line(s)
	_out.flush()

func _initialize() -> void:
	_out = FileAccess.open(ProjectSettings.globalize_path("res://../gates/8-loop/orders.txt"), FileAccess.WRITE)
	var body := ObjIO.read("res://fixtures/foxgirl/avatar.obj")
	var sk := ObjIO.read("res://fixtures/foxgirl/skeleton.obj")
	var src := PenSource.make(body.v, sk.v)
	var S := {}
	for s in src.strokes:
		S[s.name] = s.points
	var orders := {
		"rings_then_seams": ["waist_left", "waist_right", "hem_left", "hem_right", "seam_front", "seam_back"],
		"seams_first": ["seam_front", "seam_back", "waist_left", "hem_left", "waist_right", "hem_right"],
		"panel_by_panel": ["seam_front", "waist_left", "seam_back", "hem_left", "waist_right", "hem_right"],
		"left_then_right_last_both": ["seam_front", "seam_back", "waist_left", "hem_left", "hem_right", "waist_right"],
	}
	S["waist_right_rev"] = _rev(S["waist_right"])
	S["hem_right_rev"] = _rev(S["hem_right"])
	var sf := ["seam_front", "seam_back", "waist_left", "hem_left", "waist_right", "hem_right"]
	_run("seams_first merge 0.05", sf, S, body, {"merge_eps": 0.05})
	_run("seams_first merge 0.1", sf, S, body, {"merge_eps": 0.1})
	_run("seams_first right reversed", ["seam_front", "seam_back", "waist_left", "hem_left", "waist_right_rev", "hem_right_rev"], S, body, {})
	_run("rings_then_seams merge 0.05", ["waist_left", "waist_right", "hem_left", "hem_right", "seam_front", "seam_back"], S, body, {"merge_eps": 0.05})
	quit(0)

func _rev(p: PackedVector3Array) -> PackedVector3Array:
	var o := PackedVector3Array()
	for i in range(p.size() - 1, -1, -1):
		o.append(p[i])
	return o

func _run(name: String, order: Array, S: Dictionary, body, params: Dictionary) -> void:
	var r := SandboxUtil.make_sandbox(null, "res://curvenet.elf", 1024, 4096, 1 << 24)
	var sb = r.sandbox
	if sb == null:
		_say("no curvenet: " + r.reason)
		return
	sb.vmcall("cn_reset")
	for k in params:
		sb.vmcall("cn_set_param", k, params[k])
	if body != null:
		sb.vmcall("cn_set_body", body.v, body.f)
	var ends := []
	for sname in order:
		var pts: PackedVector3Array = S[sname]
		var id: int = sb.vmcall("pen_begin", pts[0].x, pts[0].y, pts[0].z, 0.5)
		for i in range(1, pts.size()):
			sb.vmcall("pen_point", id, pts[i].x, pts[i].y, pts[i].z, 0.5)
		ends.append("%s: %s" % [sname, str(sb.vmcall("pen_end", id))])
	var mb := str(sb.vmcall("mesh_build", 0.03, 0.005))
	var v: PackedFloat32Array = sb.vmcall("mesh_vertices")
	var f: PackedInt32Array = sb.vmcall("mesh_indices")
	var loops := MeshTopo.boundary_loops(f)
	var ys := []
	for l in loops:
		ys.append("%.3f" % MeshTopo.mean_y(v, l))
	var box := AABB()
	if v.size() >= 3:
		box = AABB(Vector3(v[0], v[1], v[2]), Vector3.ZERO)
		for i in range(0, v.size(), 3):
			box = box.expand(Vector3(v[i], v[i + 1], v[i + 2]))
	_say("== %s %s" % [name, str(params)])
	for e in ends:
		_say("   " + e)
	_say("   mesh_build: %s | comps %d loops %d mean_y %s bounds %s" % [mb, MeshTopo.components(v.size() / 3, f), loops.size(), str(ys), str(box)])
	sb.free()
