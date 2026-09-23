# Rule 8 smoke: instantiate project/main.gd as the MCP host does and call the
# curvenet stage's no-argument wrappers once each, printing what an MCP
# call_method would get back. Quits on a 120 s wall clock.
#
#   godot --path project --script probe_curvenet_wrappers.gd --rendering-driver vulkan --xr-mode off > ../gates/4-curvenet/wrappers.log 2>&1
extends SceneTree

# The scripted pen (cn_reset, pen_begin, pen_point, pen_end) replays
# pen_demo_circle's stroke sample by sample on the body pen_demo_circle set.
# curvenet_checks runs last: its pen_sphere check resets the stage.
const WRAPPERS := ["check_names", "cn_get_param", "curvenet_extract_demo", "pen_demo_circle", "patch_count",
		"patch_vertices", "patch_indices", "mesh_build", "mesh_patch_ids", "mesh_array_mesh", "curvenet_build",
		"cn_reset", "pen_begin", "pen_point", "pen_end", "patch_count", "curvenet_checks"]

var _main: Node
var _t0 := 0
var _i := -1

func _initialize() -> void:
	_t0 = Time.get_ticks_usec()
	_main = load("res://main.gd").new()
	root.add_child(_main)

func _process(_d: float) -> bool:
	if Time.get_ticks_usec() - _t0 > 120 * 1000000:
		print("FAIL: wall clock")
		quit(1)
		return true
	_i += 1
	if _i == 0:
		return false # let _ready run
	if _i > WRAPPERS.size():
		quit(0)
		return true
	var w: String = WRAPPERS[_i - 1]
	var r = _main.call(w)
	if r is ArrayMesh:
		r = "ArrayMesh, %d surface(s), %d vertices" % [r.get_surface_count(),
				r.surface_get_array_len(0) if r.get_surface_count() > 0 else 0]
	print("%s() -> %s" % [w, str(r)])
	return false
