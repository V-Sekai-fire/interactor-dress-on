# The Cut 8 scripted skirt on the FoxGirl fixture through curvenet.elf, the
# rings drawn as boundary strokes (cn_set_param("boundary", 1)), in the stroke
# orders cut-8's project/tests/probe_skirt_orders.gd tried, at merge_eps 0.02
# and 0.05; controls: the rings as ordinary strokes (the caps come back) and
# the back seam dropped (no panel). Writes gates/4-curvenet/skirt-foxgirl.txt.
#
# It needs cut-8's project/fixtures/foxgirl/, project/xr/pen_source_scripted.gd
# and project/util/skeleton15.gd, which this branch does not carry: copy this
# file and those into a checkout that has them (cut-8 once this branch is
# merged into it), as project/tests/, then
#   godot --path project --script tests/skirt_foxgirl_probe.gd --rendering-driver vulkan --xr-mode off
# Quits on a 400 s wall clock whatever it is doing.
extends SceneTree

const ObjIO := preload("res://util/obj_io.gd")
const PenSource := preload("res://xr/pen_source_scripted.gd")
const MeshWire := preload("res://util/mesh_wire.gd")
const WALL_S := 400.0

var _out: FileAccess
var _t0 := 0

func _say(s: String) -> void:
	print(s)
	_out.store_line(s)
	_out.flush()

func _initialize() -> void:
	_t0 = Time.get_ticks_usec()
	_out = FileAccess.open(ProjectSettings.globalize_path("res://../gates/4-curvenet/skirt-foxgirl.txt"), FileAccess.WRITE)
	_say("# skirt on FoxGirl through curvenet.elf, %s, Godot %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string])
	var body := ObjIO.read("res://fixtures/foxgirl/avatar.obj")
	var sk := ObjIO.read("res://fixtures/foxgirl/skeleton.obj")
	var src := PenSource.make(body.v, sk.v)
	if str(src.get("error", "")) != "":
		_say("FAIL pen source: " + str(src.error))
		quit(1)
		return
	_say("body %d vertices %d triangles; waist r %.4f at y %.4f, hem r %.4f at y %.4f (grow %.4f)" % [body.v.size() / 3,
			body.f.size() / 3, src.waist.radius, src.waist.center.y, src.hem.radius, src.hem.center.y, src.grow])
	var S := {}
	for s in src.strokes:
		S[s.name] = s.points
	var orders := {
		"pen_source (rings, then seams)": ["waist_left", "waist_right", "hem_left", "hem_right", "seam_front", "seam_back"],
		"seams_first": ["seam_front", "seam_back", "waist_left", "hem_left", "waist_right", "hem_right"],
		"panel_by_panel": ["seam_front", "waist_left", "seam_back", "hem_left", "waist_right", "hem_right"],
		"left_then_right_last_both": ["seam_front", "seam_back", "waist_left", "hem_left", "hem_right", "waist_right"],
	}
	var fails := 0
	for name in orders:
		for m in [0.02, 0.05]:
			fails += _run("%s merge_eps %.2f" % [name, m], orders[name], S, body, m, true, true)
	var all: Array = orders["pen_source (rings, then seams)"]
	fails += _run("CONTROL rings not boundary", all, S, body, 0.02, false, false)
	fails += _run("CONTROL back seam dropped", all.filter(func(n): return n != "seam_back"), S, body, 0.02, true, false)
	_say("RESULT: %s (%d runs failed their expectation)" % ["PASS" if fails == 0 else "FAIL", fails])
	_out.close()
	quit(0 if fails == 0 else 1)

func _process(_d: float) -> bool:
	if Time.get_ticks_usec() - _t0 > int(WALL_S * 1e6):
		_say("FAIL: the %d s wall clock ran out" % int(WALL_S))
		quit(1)
	return false

# One run in a fresh sandbox; answers 1 when the run misses its expectation
# (the skirt: 4 knots of degree 3, 6 curves, 2 cycles + 2 openings, 2 patches,
# 1 component, 2 loops, Euler 0; the controls: see below).
func _run(title: String, order: Array, S: Dictionary, body: Dictionary, merge: float, boundary: bool, skirt: bool) -> int:
	var sb = ClassDB.instantiate("Sandbox")
	sb.references_max = 4096
	sb.program = load("res://curvenet.elf")
	sb.vmcall("cn_set_param", "merge_eps", merge)
	sb.vmcall("cn_set_body", body.v, body.f)
	sb.vmcall("cn_reset")
	_say("== " + title)
	var last := ""
	for n in order:
		var pts: PackedVector3Array = S[n]
		var ring: bool = not n.begins_with("seam")
		sb.vmcall("cn_set_param", "boundary", 1.0 if (boundary and ring) else 0.0)
		var id: int = sb.vmcall("pen_begin", pts[0].x, pts[0].y, pts[0].z, 0.5)
		for i in range(1, pts.size()):
			sb.vmcall("pen_point", id, pts[i].x, pts[i].y, pts[i].z, 0.5)
		var t := Time.get_ticks_usec()
		last = str(sb.vmcall("pen_end", id))
		_say("   %-11s boundary=%d pen_end %6.1f ms  %s" % [n, 1 if (boundary and ring) else 0,
				(Time.get_ticks_usec() - t) / 1000.0, last])
	sb.vmcall("cn_set_param", "boundary", 0.0)
	var cb := str(sb.vmcall("curvenet_build"))
	var degrees := []
	for k in MeshWire.knots(sb.vmcall("curvenet_knots")):
		degrees.append(k.degree)
	var patches: int = sb.vmcall("patch_count")
	var spans := []
	for i in patches:
		var pv: PackedFloat32Array = sb.vmcall("patch_vertices", i)
		var mx := 0.0
		var ymin := INF
		var ymax := -INF
		for k in range(0, pv.size(), 3):
			mx += pv[k]
			ymin = minf(ymin, pv[k + 1])
			ymax = maxf(ymax, pv[k + 1])
		spans.append("%s y %.3f..%.3f" % ["+x" if mx > 0 else "-x", ymin, ymax])
	_say("   curvenet: %s, knot degrees %s; patches: %s" % [cb, str(degrees), str(spans)])
	var good := true
	for spec in [[0.03, 0.005], [0.0, 1e-5]]:
		var mb := str(sb.vmcall("mesh_build", spec[0], spec[1]))
		var v: PackedFloat32Array = sb.vmcall("mesh_vertices")
		var loops := MeshWire.loops(sb.vmcall("mesh_boundary_loops"))
		var ids: PackedInt32Array = sb.vmcall("mesh_patch_ids")
		var per := {}
		for p in ids:
			per[p] = per.get(p, 0) + 1
		var ys := []
		for l in loops:
			var y := 0.0
			for i in l:
				y += v[3 * i + 1]
			ys.append(snappedf(y / maxf(1.0, l.size()), 0.001))
		_say("   mesh_build(%.2f, %s): %s | loops at mean y %s, patch ids %s" % [spec[0], str(spec[1]), mb, str(ys), str(per)])
		if skirt:
			good = good and mb.contains(" components=1 ") and mb.ends_with(" euler=0") and loops.size() == 2 and not per.has(-1)
	sb.free()
	var ok := false
	if skirt:
		ok = good and last.contains(" patches=2 ") and last.contains(" nodes=4 ") and last.contains(" edges=6 ") \
				and last.contains(" cycles=2 ") and last.contains(" openings=2") and degrees == [3, 3, 3, 3] and cb.contains(" curves=6 ")
	elif boundary:
		ok = last.contains(" patches=0 ") # back seam dropped: no panel, the caps are openings
	else:
		ok = last.contains(" patches=4 ") and last.contains(" openings=0") # unmarked: 2 panels + 2 caps
	_say("   %s" % ("as expected" if ok else "NOT AS EXPECTED"))
	return 0 if ok else 1
