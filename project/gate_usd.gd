# Gate 0G: usd_probe.elf -- OpenUSD 26.05 (static, riscv64) opens a stage from
# bytes in the guest, no filesystem, against the host's OpenUSD on the same
# bytes (gates/0g-openusd/host_control.py -> native-control.log).
#
#   python gates/0g-openusd/host_control.py
#   godot --path project --script gate_usd.gd --rendering-driver vulkan --xr-mode off > ../gates/0g-openusd/run.log 2>&1
#
# 1. init: embedded plugInfo.json + generatedSchema.usda registered from
#    memory, the in-memory resolver found; host-timed, heap after.
# 2. every case of native-control.log: the guest's line must equal the host's
#    (counts + checksum) for good inputs; for corrupt ones both must be an
#    "ERR: fmt=<same>" line and the guest must still answer afterwards.
#    USDA goes through ImportFromString (mode 0) and again through the
#    in-memory resolver (mode 1); USDC only through the resolver.
# 3. repeat: the heaviest case again, the same line (the layer registry holds
#    no stale state), host-timed.
# One vmcall per frame; quits on a 300 s wall clock whatever step it is in.
# Results stream to gates/0g-openusd/results.txt; last line RESULT: PASS|FAIL.
extends SceneTree

const GATE := "res://../gates/0g-openusd/"
const EXT := "C:/contract-manifest/3-interactor/datasource-flow-project/morph_stress_test.usda"
const TMP := "C:/b/g0g-inputs/"
const WALL_S := 300.0
const MEM_MB := 1024

var _sb = null
var _out: FileAccess
var _t_start := 0
var _rc := 0
var _done := false
var _queue: Array = [] # [label, bytes, mode, expected]
var _init_done := false

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
		_out = null
	if _sb != null:
		_sb.free()
		_sb = null
	_done = true
	quit(_rc)

func _call(fn: String, args: Array = []) -> Array:
	var t := Time.get_ticks_usec()
	var r = _sb.callv("vmcall", [fn] + args)
	return [r, Time.get_ticks_usec() - t]

func _heap() -> int:
	return int(_sb.get_heap_usage())

func _bytes(name: String) -> PackedByteArray:
	for p in [GATE + "inputs/" + name, TMP + name]:
		var g := ProjectSettings.globalize_path(p)
		if FileAccess.file_exists(g):
			return FileAccess.get_file_as_bytes(g)
	if name == EXT.get_file():
		return FileAccess.get_file_as_bytes(EXT)
	return PackedByteArray()

func _cases() -> void:
	var f := FileAccess.open(ProjectSettings.globalize_path(GATE + "native-control.log"), FileAccess.READ)
	if f == null:
		_check(false, "native-control.log missing (run host_control.py first)")
		return
	_say("host control: " + f.get_line())
	var good := _bytes("blendshape_test.usdc")
	while not f.eof_reached():
		var line := f.get_line().strip_edges()
		if line.is_empty():
			continue
		var name := line.get_slice(" ", 0)
		var expect := line.substr(line.find(" ", line.find("bytes=")) + 1)
		var b := PackedByteArray()
		if name == "corrupt.usda":
			b = "#usda 1.0\ndef Mesh \"M\" {\n  point3f[] points = [(1, 2,\n".to_utf8_buffer()
		elif name == "corrupt.usdc":
			b = good.slice(0, good.size() / 3)
		elif name == "garbage.bin":
			for k in 4:
				for i in 256:
					b.append(i)
		else:
			b = _bytes(name)
		if b.is_empty():
			_check(false, "%s: no bytes" % name)
			continue
		if not _check(line.get_slice(" ", 1) == "bytes=%d" % b.size(), "%s: %d bytes, as the host read" % [name, b.size()]):
			continue
		_queue.append([name + " mode0", b, 0, expect])
		if name.ends_with(".usda"):
			_queue.append([name + " mode1", b, 1, expect])

func _initialize() -> void:
	_t_start = Time.get_ticks_usec()
	_out = FileAccess.open(ProjectSettings.globalize_path(GATE + "results.txt"), FileAccess.WRITE)
	_say("# Gate 0G (usd_probe.elf), %s, Godot %s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string])
	_sb = ClassDB.instantiate("Sandbox")
	if _sb != null: _sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	if _sb == null:
		_check(false, "Sandbox class not registered")
		_finish()
		return
	_sb.memory_max = MEM_MB # before program= (Gate 0F)
	_sb.references_max = 4096
	var t := Time.get_ticks_usec()
	_sb.program = load("res://usd_probe.elf")
	_say("load usd_probe.elf: %.1f ms; execution_timeout=%s memory_max=%s heap=%d" % [
			(Time.get_ticks_usec() - t) / 1000.0, str(_sb.execution_timeout), str(_sb.memory_max), _heap()])
	_cases()

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_usec() - _t_start > int(WALL_S * 1e6):
		_check(false, "the %d s wall clock ran out (%d cases left)" % [int(WALL_S), _queue.size()])
		_finish()
		return true
	if not _init_done:
		_init_done = true
		var r := _call("usd_init")
		var s := str(r[0])
		_check(s.begins_with("ok ") and not s.contains("first_error") and s.contains("resolvers="),
				"usd_init %.1f ms heap=%d: %s" % [r[1] / 1000.0, _heap(), s])
		if not s.begins_with("ok "):
			_finish()
		return _done
	if _queue.is_empty():
		_finish()
		return true
	var c: Array = _queue.pop_front()
	var r := _call("usd_load", [c[1], c[2]])
	var got := str(r[0])
	var exp: String = c[3]
	var ok := false
	if exp.begins_with("ok "):
		ok = got == exp
	else:
		ok = got.begins_with(exp.strip_edges()) # "ERR: fmt=x": a clean error, same format
	_check(ok, "%-26s %9.1f ms heap=%10d  guest: %s%s" % [c[0], r[1] / 1000.0, _heap(), got,
			"" if ok else "  | host: " + exp])
	if c[0] == "morph_stress_test.usdc mode0" and ok:
		# 3. repeat the heaviest case once.
		_queue.push_front(["morph_stress_test.usdc repeat", c[1], c[2], c[3]])
	return _done
