# Boundary-cost instrument: host-timed per-call cost of each call kind the
# GPU layer makes, from inside the guest. Three rounds, to see drift.
#   godot --path project --script probe_rd_calls.gd --rendering-driver vulkan --xr-mode off
extends SceneTree

func _bytes(path: String) -> PackedByteArray:
	var f := FileAccess.open(path, FileAccess.READ)
	var b := f.get_buffer(f.get_length())
	f.close()
	return b

func _init() -> void:
	var sb = ClassDB.instantiate("Sandbox")
	if sb != null: sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	if sb != null: sb.memory_max = 512 # the Windows addon's default, which the recorded runs used; Linux's is 32 MiB
	sb.program = load("res://dress_on.elf")
	sb.references_max = 4096 # the default 100 fails the uset kind
	print("set_probe: ", sb.vmcall("rd_set_probe", _bytes("res://probe.spv")))
	print("open: ", sb.vmcall("rd_open"))
	for round in 3:
		for kind in ["ticks", "clock", "limit", "bind", "dispatch", "barrier", "submit", "buffer", "shader", "shader-pba", "pipeline", "uset", "readback", "instantiate"]:
			var n := 1000 if kind in ["ticks", "clock", "limit", "bind", "dispatch", "barrier"] else 32
			var t0 := Time.get_ticks_usec()
			var r = sb.vmcall("rd_calls", kind, n)
			var dt := Time.get_ticks_usec() - t0
			print("round %d %-8s n=%4d host_us=%8d  us_per_call=%8.2f  %s" % [round, kind, n, dt, float(dt) / float(n), str(r)])
	print("close: ", sb.vmcall("rd_close"))
	sb.free()
	quit(0)
