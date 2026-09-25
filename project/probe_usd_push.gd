# Gate U probe: which path crashes on a 14.86 MB package? One arm a process
# (a guest fault below the heap floor can kill Godot, AGENTS.md).
#   godot --path project --script probe_usd_push.gd --rendering-driver vulkan --xr-mode off -- --arm=open|push8|push4 [--file=...] [--mem=320]
# Writes gates/U-usd/ladder/probe-<arm>.txt.
extends SceneTree

const UsdStage := preload("res://stages/usd_stage.gd")
var _arm := "open"
var _file := "C:/b/gu-inputs/real-t2048.usdz"
var _mem := 320
var _seq := PackedStringArray()
var _stage = null
var _out: FileAccess
var _step := 0

func _say(s: String) -> void:
	print(s)
	_out.store_line(s)
	_out.flush()

func _initialize() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--arm="):
			_arm = a.substr(6)
		elif a.begins_with("--file="):
			_file = a.substr(7)
		elif a.begins_with("--mem="):
			_mem = int(a.substr(6))
		elif a.begins_with("--seq="):
			_seq = a.substr(6).split(",")
			_arm = "seq"
	_out = FileAccess.open(ProjectSettings.globalize_path("res://../gates/U-usd/ladder/probe-%s.txt" % _arm), FileAccess.WRITE)
	_stage = UsdStage.new()
	_stage.mem_mb = _mem
	root.add_child(_stage)
	_stage.ensure()
	_say("arm=%s file=%s mem=%d sandbox=%s" % [_arm, _file, _mem, str(_stage.sandbox != null)])

func _process(_d: float) -> bool:
	_step += 1
	if _step < 2:
		return false
	var b := FileAccess.get_file_as_bytes(_file)
	_say("bytes=%d init=%s heap=%d" % [b.size(), _stage.init(), _stage.heap()])
	var t0 := Time.get_ticks_usec()
	var r := ""
	if _arm == "seq":
		for f in _seq:
			var fb := FileAccess.get_file_as_bytes("C:/b/gu-inputs/" + f) if not f.begins_with("res://") else FileAccess.get_file_as_bytes(ProjectSettings.globalize_path(f))
			var t1 := Time.get_ticks_usec()
			var o: String = _stage.open_package(fb)
			var doc: Dictionary = _stage.document()
			var sb = _stage.sandbox
			var stats := ""
			for k in ["get_heap_chunk_count", "get_heap_allocation_counter", "get_heap_deallocation_counter"]:
				if sb.has_method(k):
					stats += " %s=%s" % [k.replace("get_heap_", ""), str(sb.call(k))]
			_say("seq %s: %d bytes open %.1f ms heap=%d -> %s; meshes=%d; close=%s heap=%d%s" % [f, fb.size(), (Time.get_ticks_usec() - t1) / 1000.0,
					_stage.heap(), o.left(60), doc.meshes.size(), _stage.close(), _stage.heap(), stats])
		r = "seq done"
	elif _arm == "open":
		r = str(_stage.call_now("usd_open", [b]))
	else:
		var chunk: int = (8 << 20) if _arm == "push8" else (4 << 20)
		var at := 0
		while at < b.size():
			var n: int = mini(chunk, b.size() - at)
			var p := str(_stage.call_now("usd_push", [b.slice(at, at + n)]))
			_say("push %d+%d -> %s heap=%d" % [at, n, p, _stage.heap()])
			at += n
		r = str(_stage.call_now("usd_open_staged"))
	_say("%s: %.1f ms heap=%d -> %s" % [_arm, (Time.get_ticks_usec() - t0) / 1000.0, _stage.heap(), r.left(200)])
	_say("count=%d" % _stage.count())
	_say("RESULT: DONE")
	_out.close()
	quit(0)
	return true
