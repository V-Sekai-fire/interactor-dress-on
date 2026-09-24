# Emit the native translation of every guest ELF in project/ as C99.
#
#   GODOT_SANDBOX_BINTR_EMIT=<dir> godot --path project --headless --xr-mode off --script tools/bintr_emit.gd
#
# With the lookup on for this process, the org's addon build (godot-sandbox
# feat/bintr-emit) writes <dir>/bintr-<hash>.c as it loads each program; the
# hash names the execute segment plus the translation flags, and the
# Sandbox's memory_max is part of those flags, so each ELF is loaded with the
# memory_max its stage uses (stages/*.gd, the gates). tools/build.exs
# compiles the sources into bintr-<HASH>.so beside them, which the addon
# loads when sandbox/binary_translation/enabled is on. Prints one line per
# ELF: name, memory_max, hash, translated-after-load.
extends SceneTree

# ELF -> the memory_max (MiB) its Sandbox is made with (sandbox_util / gates).
const MEMORY_MAX := {
	"dress_on": 512, "probes": 512, "rd_worker": 512,
	"drape": 1024, "curvenet": 1024, "fit": 2048, "ggml_test": 2048,
}

func _initialize() -> void:
	var dir := OS.get_environment("GODOT_SANDBOX_BINTR_EMIT")
	if dir == "":
		printerr("bintr_emit: set GODOT_SANDBOX_BINTR_EMIT=<dir>")
		quit(2)
		return
	ProjectSettings.set_setting("sandbox/binary_translation/enabled", true)
	var probe = ClassDB.instantiate("Sandbox")
	if probe == null or not probe.call("has_feature_binary_translation"):
		printerr("bintr_emit: this addon has no binary translation")
		quit(2)
		return
	var n := 0
	for name in MEMORY_MAX:
		var path := "res://%s.elf" % name
		if not ResourceLoader.exists(path):
			continue
		var sb = ClassDB.instantiate("Sandbox")
		sb.allocations_max = 1000000
		sb.memory_max = MEMORY_MAX[name]
		sb.references_max = 4096
		var t := Time.get_ticks_msec()
		sb.program = load(path)
		print("bintr_emit: %s memory_max=%d hash=%08x translated=%s (%d ms)" % [name, MEMORY_MAX[name],
				int(sb.call("get_translation_hash")), str(sb.call("is_binary_translated")),
				Time.get_ticks_msec() - t])
		sb.free()
		n += 1
	print("bintr_emit: %d programs, sources under %s" % [n, dir])
	quit(0)
