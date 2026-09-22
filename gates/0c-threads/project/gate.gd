extends SceneTree
func _init() -> void:
	var sb = ClassDB.instantiate("Sandbox")
	if sb == null: print("FAIL: no Sandbox class"); quit(1); return
	sb.program = load("res://threadprobe.elf")
	var r = sb.vmcall("thread_probe", 4)
	print(r)
	quit(0 if (typeof(r) == TYPE_STRING and r.begins_with("PASS")) else 1)
