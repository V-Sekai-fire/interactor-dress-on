# What the loaded addon can do about native translation: the JIT and
# binary-translation features it was built with, and whether a loaded
# program is translated. Headless, one Sandbox, no vmcall.
#   godot --path project --headless --xr-mode off --script probe_bintr.gd
extends SceneTree

func _initialize() -> void:
	var sb = ClassDB.instantiate("Sandbox")
	print("has_feature_jit=%s has_feature_binary_translation=%s jit_enabled=%s" % [
			str(sb.call("has_feature_jit")), str(sb.call("has_feature_binary_translation")),
			str(sb.call("is_jit_enabled")) if sb.has_method("is_jit_enabled") else "?"])
	print("bintr project setting enabled=%s cache_dir=%s" % [
			str(ProjectSettings.get_setting("sandbox/binary_translation/enabled", null)),
			str(ProjectSettings.get_setting("sandbox/binary_translation/cache_dir", null))])
	if OS.get_environment("BINTR_PROBE_ENABLE") == "1":
		ProjectSettings.set_setting("sandbox/binary_translation/enabled", true)
		print("lookup enabled for this process")
	sb.allocations_max = 1000000
	sb.memory_max = 512
	sb.program = load("res://ggml_test.elf")
	print("program loaded: is_binary_translated=%s is_jit=%s translation_hash=%s" % [
			str(sb.call("is_binary_translated")), str(sb.call("is_jit")),
			str(sb.call("get_translation_hash")) if sb.has_method("get_translation_hash") else "?"])
	var methods := []
	for m in sb.get_method_list():
		if str(m.name).contains("translat") or str(m.name).contains("jit"):
			methods.append(str(m.name))
	print("methods: %s" % str(methods))
	quit(0)
