# curvenet_stage -- curvenet.elf (Cut 4): Cassie's pen -> curvenet -> mesh.
# The pipeline's AUTHOR state feeds it pen_begin / pen_point / pen_end in the
# body-local frame; MESH calls mesh_build and reads the welded mesh and its
# boundary loops (util/mesh_wire.gd's format). CPU only.
#
# The wrappers below mirror cut-4's main.gd (same names, same defaults) so MCP
# drives keep working after the merge; main.gd delegates to them.
extends "res://stages/stage_base.gd"

const MeshWire := preload("res://util/mesh_wire.gd")
const REQUIRED := ["cn_reset", "cn_set_param", "cn_set_body", "pen_begin", "pen_point", "pen_end", "patch_count",
		"curvenet_build", "curvenet_curves", "curvenet_knots", "mesh_build", "mesh_vertices", "mesh_indices",
		"mesh_boundary_loops"]

func _ready() -> void:
	stage_name = "curvenet"
	# A PMP remesh of a whole garment runs far past the default 8000 x 2^20
	# instructions (Gate 0F probe 5); 2^24 units is effectively unbounded.
	open_sandbox("res://curvenet.elf", 1024, 4096, 1 << 24, {}, PackedStringArray(REQUIRED))

func _cn_call(fn: String, args: Array = []) -> String:
	if sandbox == null:
		return "FAIL: no curvenet sandbox (%s)" % reason
	var t0 := Time.get_ticks_usec()
	var r = call_now(fn, args)
	return "host_us=%d %s" % [Time.get_ticks_usec() - t0, str(r)]

# --- pipeline calls (raw guest answers) ---------------------------------------------

func set_body(v: PackedFloat32Array, f: PackedInt32Array) -> String:
	return str(call_now("cn_set_body", [v, f]))

func reset() -> String:
	return str(call_now("cn_reset"))

func set_param(name: String, value: float) -> String:
	return str(call_now("cn_set_param", [name, value]))

func pen_begin(p: Vector3, pressure: float) -> int:
	var r = call_now("pen_begin", [p.x, p.y, p.z, pressure])
	return int(r) if typeof(r) == TYPE_INT or typeof(r) == TYPE_FLOAT else -1

func pen_point(id: int, p: Vector3, pressure: float) -> String:
	return str(call_now("pen_point", [id, p.x, p.y, p.z, pressure]))

func pen_end_raw(id: int) -> String:
	return str(call_now("pen_end", [id]))

func patches() -> int:
	var r = call_now("patch_count")
	return int(r) if typeof(r) == TYPE_INT or typeof(r) == TYPE_FLOAT else -1

func build_curvenet() -> String:
	return str(call_now("curvenet_build"))

func curves():
	return call_now("curvenet_curves")

func knots():
	return call_now("curvenet_knots")

# mesh_build on the worker thread (a remesh can take seconds); poll().
func start_mesh_build(target_edge_length: float, weld_eps: float) -> String:
	return start("mesh_build", [target_edge_length, weld_eps])

func mesh_arrays() -> Dictionary:
	var v = call_now("mesh_vertices")
	var f = call_now("mesh_indices")
	var l = call_now("mesh_boundary_loops")
	if typeof(v) != TYPE_PACKED_FLOAT32_ARRAY or typeof(f) != TYPE_PACKED_INT32_ARRAY or typeof(l) != TYPE_PACKED_INT32_ARRAY:
		return {"error": "mesh arrays: %s / %s / %s" % [type_string(typeof(v)), type_string(typeof(f)), type_string(typeof(l))]}
	return {"vertices": v, "triangles": f, "loops": MeshWire.loops(l)}

# --- MCP wrappers (cut-4's, rule 8) ---------------------------------------------------

func cn_reset() -> String:
	return _cn_call("cn_reset")

func cn_set_param(name: String = "snap_radius", value: float = 0.03) -> String:
	return _cn_call("cn_set_param", [name, value])

# The demo body: an r = 0.5 SphereMesh at the origin.
func cn_set_body_sphere(radius: float = 0.5) -> String:
	var b := MeshWire.sphere(radius)
	return _cn_call("cn_set_body", [b.vertices, b.triangles])

# One closed stroke at 30 degrees latitude, 1 cm off the demo sphere: it
# snaps onto the body and closes one patch. Resets the stage first.
func pen_demo_circle() -> String:
	var b := cn_set_body_sphere()
	var r := cn_reset()
	var s := MeshWire.circle_stroke(0.51, PI / 6.0, TAU, 64)
	return "%s | %s | %s" % [b, r, _cn_call("pen_stroke", [s])]

func pen_end(id: int = 1) -> String:
	return _cn_call("pen_end", [id])

func patch_count() -> String:
	return _cn_call("patch_count")

# Gate 4's checks in the guest, one line each. pen_sphere resets the stage
# and clears the body.
func curvenet_checks() -> String:
	return _cn_call("check_all")

func curvenet_check(name: String = "pen_sphere") -> String:
	return _cn_call("check", [name])

func curvenet_build() -> String:
	var r := _cn_call("curvenet_build")
	var c := MeshWire.curves(call_now("curvenet_curves")) if sandbox != null else []
	var k := MeshWire.knots(call_now("curvenet_knots")) if sandbox != null else []
	return "%s | wire: %d curves, %d knots" % [r, c.size(), k.size()]

# Merge + weld the active patches; target_edge_length > 0 PMP-remeshes.
func mesh_build(target_edge_length: float = 0.02, weld_eps: float = 1e-5) -> String:
	var r := _cn_call("mesh_build", [target_edge_length, weld_eps])
	if sandbox == null:
		return r
	var loops := MeshWire.loops(call_now("mesh_boundary_loops"))
	var v: PackedFloat32Array = call_now("mesh_vertices")
	return "%s | wire: %d vertices, %d boundary loops" % [r, v.size() / 3, loops.size()]

# The built mesh as an ArrayMesh (Godot winding), for a MeshInstance3D.
func mesh_array_mesh() -> ArrayMesh:
	if sandbox == null:
		return ArrayMesh.new()
	return MeshWire.to_array_mesh(call_now("mesh_vertices"), call_now("mesh_indices"))

# Mesh -> curvenet on a unit cube: 12 curves on 8 knots.
func curvenet_extract_demo() -> String:
	var c := MeshWire.cube()
	return _cn_call("curvenet_extract", [c.vertices, c.triangles, 200, 1e-3, 1e-2, 0.0])
