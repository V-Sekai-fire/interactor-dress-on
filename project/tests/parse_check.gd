# Loads every Gate 8 script and scene so a parse error shows as FAIL here
# rather than as a silent null later. Writes gates/8-loop/parse_check.txt.
#   godot --path project --script tests/parse_check.gd --rendering-driver vulkan --xr-mode off
extends SceneTree

const OUT := "res://../gates/8-loop/parse_check.txt"
const PATHS := ["res://main.gd", "res://gate_loop.gd", "res://stages/sandbox_util.gd", "res://stages/stage_base.gd",
		"res://stages/dress_on_stage.gd", "res://stages/drape_stage.gd", "res://stages/curvenet_stage.gd",
		"res://stages/fit_stage.gd", "res://stages/infer_stage.gd", "res://stages/pipeline.gd",
		"res://util/skeleton15.gd", "res://util/mesh_topo.gd", "res://util/mesh_wire.gd", "res://util/obj_io.gd",
		"res://xr/pen_source_scripted.gd", "res://xr/pen_bridge.gd", "res://xr/xr_world.gd",
		"res://tests/test_loop_units.gd", "res://xr_main.tscn", "res://main.tscn"]

func _initialize() -> void:
	var f := FileAccess.open(ProjectSettings.globalize_path(OUT), FileAccess.WRITE)
	var bad := 0
	for p in PATHS:
		var r = load(p)
		var ok: bool = r != null and (not (r is GDScript) or r.can_instantiate())
		if not ok:
			bad += 1
		f.store_line("%s %s" % ["ok  " if ok else "FAIL", p])
		f.flush()
	f.store_line("RESULT: %s" % ("PASS" if bad == 0 else "FAIL"))
	f.close()
	quit(bad)
