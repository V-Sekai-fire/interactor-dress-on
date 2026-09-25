# Gate C: cassie_graph.elf -- CASSIE's sketch graph (godot-cassie's
# CassieSketchGraph at a4e33d895f, vendor/cassie-graph) runs in the sandbox,
# and gives a sketch's expected cycles the way the module's pipeline bench does.
#
#   godot --path project --headless --xr-mode off --script gate_cassie_graph.gd
#       -> gates/C-cassie/results.txt ("expected <sketch> cycles=N ..." lines, RESULT: last)
#
# 1. Controls with known answers: an open polyline has no cycle; a closed
#    square of 4 strokes has one; removing a side, or opening a gap wider than
#    merge_epsilon (0.02 m), leaves none; a cube's 12 edges bound 6 faces.
# 2. Mirroring a sketch (the Unity -> Body Z flip) does not change its cycles.
# 3. dress.usda (gates/S-strokes/inputs), un-flipped to CASSIE's own canvas
#    points, gives dress's expected cycles; the named train sketches too.
extends SceneTree

const CgStage := preload("res://stages/cassie_graph_stage.gd")
const UsdStage := preload("res://stages/usd_stage.gd")
const StrokesUsd := preload("res://util/strokes_usd.gd")
const GATE := "res://../gates/C-cassie/"

var _cg = null
var _usd = null
var _out: FileAccess
var _rc := 0

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _check(ok: bool, what: String) -> bool:
	_say("%s %s" % ["PASS" if ok else "FAIL", what])
	if not ok:
		_rc = 1
	return ok

func _finish() -> void:
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	if _out != null:
		_out.close()
	quit(_rc)

static func _seg(a: Vector3, b: Vector3, n: int = 8) -> Dictionary:
	var p := PackedVector3Array()
	for i in n + 1:
		p.append(a.lerp(b, float(i) / n))
	return {"points": p}

static func _square(gap: float = 0.0, drop: int = -1) -> Array:
	var c := [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0), Vector3(0, 1, 0)]
	var out := []
	for i in 4:
		if i == drop:
			continue
		var b: Vector3 = c[(i + 1) % 4]
		if i == 3:
			b += Vector3(gap, 0, 0) # the last side stops short of the first corner
		out.append(_seg(c[i], b))
	return out

static func _cube() -> Array:
	var out := []
	for a in 8:
		for bit in [1, 2, 4]:
			var b: int = a | bit
			if b != a:
				out.append(_seg(Vector3(a & 1, (a >> 1) & 1, (a >> 2) & 1), Vector3(b & 1, (b >> 1) & 1, (b >> 2) & 1)))
	return out

static func _mirror(strokes: Array) -> Array:
	var out := []
	for s in strokes:
		var p := PackedVector3Array()
		for v in s.points:
			p.append(Vector3(v.x, v.y, -v.z))
		out.append({"points": p})
	return out

func _count(label: String, strokes: Array) -> int:
	var r: Dictionary = _cg.cycles(strokes)
	if r.has("error"):
		_check(false, "%s: %s" % [label, r.error])
		return -1
	_say("  %s: strokes %d/%d added, nodes %d, edges %d, cycles %d, sizes %s" % [label, r.strokes_added, r.strokes_in,
			r.nodes, r.edges, r.cycles, str(Array(r.cycle_sizes))])
	return int(r.cycles)

func _controls() -> void:
	_check(_count("open polyline", [_seg(Vector3.ZERO, Vector3(1, 0, 0)), _seg(Vector3(1, 0, 0), Vector3(1, 1, 0))]) == 0,
			"control: an open polyline has no cycle")
	var closed := _count("square", _square())
	_check(closed >= 1, "a closed square of 4 strokes has a cycle (%d)" % closed)
	_check(_count("square, a side dropped", _square(0.0, 2)) == 0, "control: dropping a side leaves no cycle")
	_check(_count("square, a 0.05 m gap", _square(0.05)) == 0, "control: a gap wider than merge_epsilon leaves no cycle")
	var cube := _count("cube", _cube())
	_check(cube == 6, "a cube's 12 edges bound 6 faces (%d)" % cube)
	_check(_count("cube, mirrored", _mirror(_cube())) == cube, "property: mirroring does not change the cycles")

func _sketches() -> void:
	var r := StrokesUsd.from_file(_usd, "res://../gates/S-strokes/inputs/dress.usda")
	if not _check(not r.has("error"), "dress.usda reads through usd.elf"):
		return
	var canvas := _mirror(r.strokes) # undo the Z flip: CASSIE's own canvas points, as the bench reads them
	var n := _count("dress (canvas)", canvas)
	var m := _count("dress (Body frame)", r.strokes)
	_check(n >= 0 and n == m, "property: dress has the same cycles in canvas and Body frames (%d, %d)" % [n, m])
	if n >= 0:
		_say("expected dress.curves cycles=%d strokes=%d source_rev=%s" % [n, r.strokes.size(), str(r.meta.get("source_rev", "")).left(10)])
		_say("merge_epsilon sweep on dress (CASSIE's default 0.02 m; reference: credit card 0.76 mm, penny 1.52 mm, pencil 7 mm, AA 14.5 mm, nickel 21.2 mm, golf ball 42.7 mm):")
		for eps in [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
			var q: Dictionary = _cg.cycles(canvas, eps)
			var two := 0
			for sz in q.cycle_sizes:
				two += int(sz == 2)
			_say("  eps %4.1f mm: nodes %3d, cycles %2d, of them 2-edge %d, sizes %s" % [eps * 1000.0, q.nodes, q.cycles, two, str(Array(q.cycle_sizes))])

func _initialize() -> void:
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(GATE))
	_out = FileAccess.open(ProjectSettings.globalize_path(GATE + "results.txt"), FileAccess.WRITE)
	_say("# Gate C (CASSIE's sketch graph in cassie_graph.elf), %s, Godot %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string])
	_cg = CgStage.new()
	root.add_child(_cg)
	_cg.ensure()
	_usd = UsdStage.new()
	root.add_child(_usd)
	_usd.ensure()
	if not _check(_cg.sandbox != null and _usd.sandbox != null, "cassie_graph.elf and usd.elf load (%s %s)" % [_cg.reason, _usd.reason]):
		_finish()
		return
	_controls()
	_sketches()
	_finish()
