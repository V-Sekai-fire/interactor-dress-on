# Gate S: pen strokes <-> OpenUSD through usd.elf (util/strokes_usd.gd), both
# directions, checked against the host's OpenUSD (gates/S-strokes/host_oracle.py).
#
#   pixi run python gates/S-strokes/host_oracle.py make    -> inputs/ + host-make.log
#   godot --path project --headless --xr-mode off --script gate_strokes_usd.gd
#       -> gates/S-strokes/results.txt, out/*.usda + out/guest.log
#   pixi run python gates/S-strokes/host_oracle.py check   -> host-check.log
#
# 1. Round trips: seeded random sketches -> usd_write_curves -> usd_open give
#    back the same stroke count, names and boundary marks, the points bit for
#    bit, and the guest's BLAKE3 equals the sum of the bytes that crossed.
# 2. Controls, each must fail: counts that disagree with the points, a
#    one-point stroke, a bad prim name, a boundary index out of range, a
#    truncated layer, a layer whose curveVertexCounts were edited.
# 3. Host-written dress.usda (tools/curves_usd.py from CASSIE's dress.curves)
#    reads here with the host's per-stroke sums; the guest-written layers go to
#    out/ for host_oracle.py check.
extends SceneTree

const UsdStage := preload("res://stages/usd_stage.gd")
const StrokesUsd := preload("res://util/strokes_usd.gd")
const GATE := "res://../gates/S-strokes/"
const SKETCHES := 40
const SEED := 20260925

var _stage = null
var _out: FileAccess
var _log: FileAccess
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
	for f in [_out, _log]:
		if f != null:
			f.close()
	quit(_rc)

func _b3_floats(p: PackedVector3Array) -> String:
	var f := PackedFloat32Array()
	for v in p:
		f.append_array([v.x, v.y, v.z])
	return str(_stage.call_now("usd_blake3", [f.to_byte_array()])).left(12)

func _sketch(rng: RandomNumberGenerator, k: int) -> Array:
	var out := []
	for s in rng.randi_range(1, 8):
		var p := PackedVector3Array()
		for i in rng.randi_range(2, 24):
			p.append(Vector3(rng.randf_range(-1, 1), rng.randf_range(0, 2), rng.randf_range(-1, 1)))
		out.append({"name": "s%d_%d" % [k, s], "boundary": rng.randf() < 0.3, "points": p})
	return out

func _same(a: Array, b: Array) -> String:
	if a.size() != b.size():
		return "%d strokes, wrote %d" % [b.size(), a.size()]
	for i in a.size():
		if str(a[i].name) != str(b[i].name):
			return "stroke %d named %s, wrote %s" % [i, b[i].name, a[i].name]
		if bool(a[i].boundary) != bool(b[i].boundary):
			return "stroke %d boundary %s, wrote %s" % [i, b[i].boundary, a[i].boundary]
		if a[i].points != b[i].points:
			return "stroke %d points differ" % i
		if str(b[i].blake3_points).left(12) != _b3_floats(b[i].points):
			return "stroke %d: guest sum %s, crossed %s" % [i, str(b[i].blake3_points).left(12), _b3_floats(b[i].points)]
	return ""

func _round_trips() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = SEED
	var bad := 0
	for k in SKETCHES:
		var sk := _sketch(rng, k)
		var w := StrokesUsd.to_usda(_stage, sk, {"gate": "S", "sketch": str(k)})
		if w.has("error"):
			_check(false, "sketch %d write: %s" % [k, w.error])
			bad += 1
			continue
		var r := StrokesUsd.from_usda(_stage, w.text.to_utf8_buffer())
		var why: String = r.error if r.has("error") else _same(sk, r.strokes)
		if why == "" and str(r.meta.get("sketch", "")) != str(k):
			why = "customLayerData sketch=%s" % str(r.meta.get("sketch", ""))
		if why != "":
			_check(false, "sketch %d: %s" % [k, why])
			bad += 1
			continue
		if k < 4:
			var path := GATE + "out/sketch_%02d.usda" % k
			var f := FileAccess.open(ProjectSettings.globalize_path(path), FileAccess.WRITE)
			f.store_string(w.text)
			f.close()
			for i in r.strokes.size():
				_log.store_line("sketch_%02d.usda %d %s %s %d" % [k, i, r.strokes[i].name, str(r.strokes[i].blake3_points).left(12),
						int(r.strokes[i].boundary)])
	_check(bad == 0, "%d seeded sketches round-trip through usd.elf: names, boundary marks, points bit for bit, guest sums = crossed sums" % SKETCHES)

func _refused(label: String, text: String) -> void:
	_check(text.begins_with("ERR") or text.begins_with("FAIL"), "control, %s: refused (%s)" % [label, text.left(90)])

func _controls() -> void:
	var p := PackedFloat32Array([0, 0, 0, 1, 0, 0, 1, 1, 0])
	_refused("counts sum 4 over 3 points", _stage.write_curves(p, PackedInt32Array([4]), PackedInt32Array(), PackedStringArray(), {}))
	_refused("a one-point stroke", _stage.write_curves(p, PackedInt32Array([1, 2]), PackedInt32Array(), PackedStringArray(), {}))
	_refused("a bad prim name", _stage.write_curves(p, PackedInt32Array([3]), PackedInt32Array(), PackedStringArray(["no spaces"]), {}))
	_refused("boundary index 5 of 1 curve", _stage.write_curves(p, PackedInt32Array([3]), PackedInt32Array([5]), PackedStringArray(), {}))
	var good: String = _stage.write_curves(p, PackedInt32Array([3]), PackedInt32Array([0]), PackedStringArray(), {})
	_check(good.begins_with("#usda"), "control baseline: the same stroke with good counts writes")
	var marked := StrokesUsd.from_usda(_stage, good.to_utf8_buffer())
	_check(not marked.has("error") and marked.strokes.size() == 1 and marked.strokes[0].boundary,
			"control baseline: the boundary mark reads back")
	_refused("counts edited to 4", _stage.open_package(good.replace("curveVertexCounts = [3]", "curveVertexCounts = [4]").to_utf8_buffer()))
	_refused("truncated layer", _stage.open_package(good.left(good.length() / 2).to_utf8_buffer()))
	var after: String = _stage.open_package(good.to_utf8_buffer())
	_check(after.begins_with("ok ") and _stage.curve_count() == 1, "the guest answers after the refused opens: " + after.left(60))

func _dress() -> void:
	var src := GATE + "inputs/dress.usda"
	var log_path := ProjectSettings.globalize_path(GATE + "host-make.log")
	if not FileAccess.file_exists(ProjectSettings.globalize_path(src)) or not FileAccess.file_exists(log_path):
		_check(false, "dress.usda or host-make.log missing (run host_oracle.py make first)")
		return
	var r := StrokesUsd.from_file(_stage, src)
	if not _check(not r.has("error"), "dress.usda opens: %s" % (r.error if r.has("error") else r.open.left(80))):
		return
	var host := {}
	var f := FileAccess.open(log_path, FileAccess.READ)
	while not f.eof_reached():
		var parts := f.get_line().split(" ", false)
		if parts.size() >= 4 and parts[0] == "dress.usda":
			host[int(parts[1])] = parts[3]
	var mismatch := 0
	for i in r.strokes.size():
		if host.get(i, "") != str(r.strokes[i].blake3_points).left(12):
			mismatch += 1
	_check(r.strokes.size() == host.size() and mismatch == 0,
			"dress.usda: %d strokes read (host %d), %d per-stroke sums differ from the host's; source %s rev %s" % [
			r.strokes.size(), host.size(), mismatch, r.meta.get("source", "?"), str(r.meta.get("source_rev", "?")).left(7)])
	var marks := 0
	for s in r.strokes:
		marks += int(s.boundary)
	_check(marks == 0, "dress.usda marks no stroke boundary (.curves has no flag; curvenet decides)")

func _initialize() -> void:
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(GATE + "out"))
	_out = FileAccess.open(ProjectSettings.globalize_path(GATE + "results.txt"), FileAccess.WRITE)
	_log = FileAccess.open(ProjectSettings.globalize_path(GATE + "out/guest.log"), FileAccess.WRITE)
	_say("# Gate S (strokes <-> OpenUSD via usd.elf), %s, Godot %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string])
	_stage = UsdStage.new()
	root.add_child(_stage)
	_stage.ensure()
	if not _check(_stage.sandbox != null, "usd.elf loads: %s" % _stage.reason):
		_finish()
		return
	_check(str(_stage.init()).begins_with("ok"), "usd_init")
	_round_trips()
	_dress()
	_controls() # last: refused opens are what Finding 1 is about
	_finish()
