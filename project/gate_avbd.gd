# Stage 2 gate: the AVBD solver in the guest, CPU and GPU backends from the
# same Lean-emitted kernels, host-timed.
#
#   godot --path project --script gate_avbd.gd --rendering-driver vulkan --xr-mode off
#
# Results stream to gates/2-avbd/results.txt as they happen. The fixture must
# PASS on both backends; the bench table gives ms per substep for cpu, rd
# (one submit per iteration) and rd-batched (one submit per substep) at
# growing panel sizes, which is where the CPU/GPU threshold comes from.
extends SceneTree

const RESULTS := "res://../gates/2-avbd/results.txt"
const SIZES := [[8, 8], [16, 16], [32, 32], [64, 64]]
const SUBSTEPS := 5
const ITERS := 10

var _out: FileAccess

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _init() -> void:
	var rc := 0
	_out = FileAccess.open(ProjectSettings.globalize_path(RESULTS), FileAccess.WRITE)
	var sb = ClassDB.instantiate("Sandbox")
	if sb == null:
		_say("FAIL: Sandbox class not registered")
		quit(1)
		return
	sb.program = load("res://dress_on.elf")
	# Every uniform set costs ~10 scoped references for the call; a coloured
	# panel needs kernels x colours of them.
	sb.references_max = 65536

	for backend in ["cpu", "rd"]:
		var t0 := Time.get_ticks_usec()
		var r = sb.vmcall("avbd_fixture", backend)
		var dt := Time.get_ticks_usec() - t0
		_say("fixture %-3s host_us=%8d  %s" % [backend, dt, str(r)])
		if not (typeof(r) == TYPE_STRING and r.begins_with("PASS")):
			rc = 1
			_say("last rd step: " + str(sb.vmcall("rd_last_step")))

	for sz in SIZES:
		var nx: int = sz[0]
		var ny: int = sz[1]
		for backend in ["cpu", "rd", "rd-batched"]:
			if backend == "cpu" and nx * ny > 1024:
				continue # the interpreter is not the point of the CPU path at this size
			var t0 := Time.get_ticks_usec()
			var r = sb.vmcall("avbd_bench", backend, nx, ny, SUBSTEPS, ITERS)
			var dt := Time.get_ticks_usec() - t0
			var ok: bool = typeof(r) == TYPE_STRING and not r.begins_with("FAIL") and r.find("finite=yes") >= 0
			_say("bench %-10s %3dx%-3d host_us=%9d  ms/substep=%9.2f  %s" % [backend, nx, ny, dt, dt / 1000.0 / SUBSTEPS, str(r)])
			if not ok:
				rc = 1

	_say("RESULT: %s" % ("PASS" if rc == 0 else "FAIL"))
	if _out != null:
		_out.close()
	sb.free()
	quit(rc)
