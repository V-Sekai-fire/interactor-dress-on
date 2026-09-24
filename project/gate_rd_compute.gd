# Stage 1 runner: rd_compute on stock Godot + the godot_sandbox addon.
#
#   godot --path project --script gate_rd_compute.gd --rendering-driver vulkan --xr-mode off
#
# Prints the held-device probe and the bench table, exits 0 only if every
# count came back exact. The same calls are reachable over MCP through
# main.gd; this script is the flat, no-MCP control.
extends SceneTree

func _bytes(path: String) -> PackedByteArray:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return PackedByteArray()
	var b := f.get_buffer(f.get_length())
	f.close()
	return b

func _init() -> void:
	var rc := 0
	var sb = ClassDB.instantiate("Sandbox")
	if sb != null: sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	if sb == null:
		print("FAIL: Sandbox class not registered; is the addon enabled?")
		quit(1)
		return
	var elf = load("res://dress_on.elf")
	if elf == null:
		print("FAIL: could not load dress_on.elf")
		quit(1)
		return
	sb.program = elf
	sb.references_max = 4096 # see main.gd

	var probe := _bytes("res://probe.spv")
	var acc := _bytes("res://accumulate.spv")
	if probe.is_empty() or acc.is_empty():
		print("FAIL: missing probe.spv / accumulate.spv")
		quit(1)
		return

	# Held device: open in one vmcall, use in the next.
	print("open:  ", sb.vmcall("rd_open"))
	var p = sb.vmcall("rd_probe", probe)
	print("probe: ", p)
	if not (typeof(p) == TYPE_STRING and p.begins_with("PASS")):
		print("last step attempted: ", sb.vmcall("rd_last_step"))
		rc = 1

	# Boundary cost. (1,1) is the floor; (n,1) isolates per-dispatch cost
	# inside one list; (1,n) isolates per-submit cost. The barrier=true arm
	# must be exact. The barrier=false arm is a question, not a requirement:
	# does Godot's render graph order same-buffer dispatches by itself? It
	# does not (first run: 7/16, 23/64, 76/256), and that is reported, not
	# failed.
	var unordered := 0
	for cfg in [[1, 1], [16, 1], [64, 1], [256, 1], [1, 16], [1, 64], [16, 16]]:
		for barrier in [true, false]:
			var t0 := Time.get_ticks_usec()
			var r = sb.vmcall("rd_bench", acc, cfg[0], cfg[1], barrier)
			var dt := Time.get_ticks_usec() - t0
			print("bench nd=%4d ns=%3d barrier=%-5s host_us=%7d  %s" % [cfg[0], cfg[1], barrier, dt, str(r)])
			var ok: bool = typeof(r) == TYPE_STRING and r.begins_with("OK")
			if barrier and not ok:
				rc = 1
			if not barrier and not ok:
				unordered += 1

	print("no-barrier arm: %d of 7 counts wrong -> %s" % [unordered,
		"the render graph does NOT order same-buffer dispatches; barrier() is mandatory" if unordered > 0
		else "the render graph ordered them; barrier() is optional here"])
	print("close: ", sb.vmcall("rd_close"))
	sb.free()
	print("RESULT: ", "PASS" if rc == 0 else "FAIL")
	quit(rc)
