# GATE 0A runner. Stock Godot + the godot-sandbox addon and nothing else --
# that is the point of the gate, so resist adding anything here.
extends SceneTree

func _init() -> void:
	var rc := 1
	var sb = ClassDB.instantiate("Sandbox")
	if sb == null:
		print("FAIL: Sandbox class not registered; is the addon enabled?")
		quit(1)
		return

	var elf = load("res://rdprobe.elf")
	if elf == null:
		print("FAIL: could not load rdprobe.elf")
		quit(1)
		return
	sb.program = elf

	# The SPIR-V is passed in rather than embedded, so the gate tests the
	# RenderingDevice path and not our ability to bake bytes into an ELF.
	var f := FileAccess.open("res://probe.spv", FileAccess.READ)
	if f == null:
		print("FAIL: could not open probe.spv")
		quit(1)
		return
	var spirv := f.get_buffer(f.get_length())
	f.close()
	print("spirv bytes: ", spirv.size())

	var result = sb.vmcall("rd_probe", spirv)
	print(result)
	if typeof(result) == TYPE_STRING and result.begins_with("PASS"):
		rc = 0
	else:
		print("last step attempted: ", sb.vmcall("rd_last_step"))
	quit(rc)
