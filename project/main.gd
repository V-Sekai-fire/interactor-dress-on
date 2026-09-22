# The host side of interactor-dress-on. Owns the sandbox guest and exposes
# plain no-arg methods that an MCP client can reach with call_method. The
# guest API itself takes typed buffers; wrapping here keeps those off the
# JSON wire.
extends Node

var _sb = null

func _ready() -> void:
	_sb = ClassDB.instantiate("Sandbox")
	if _sb == null:
		push_error("Sandbox class not registered; is the godot_sandbox addon enabled?")
		return
	add_child(_sb)
	_sb.program = load("res://rdprobe.elf")
	print("[dress-on] sandbox loaded rdprobe.elf")

# Gate 0E: the Gate 0A probe, callable over MCP with no arguments.
func rd_probe() -> String:
	if _sb == null:
		return "FAIL: no sandbox"
	var f := FileAccess.open("res://probe.spv", FileAccess.READ)
	if f == null:
		return "FAIL: could not open probe.spv"
	var spirv := f.get_buffer(f.get_length())
	f.close()
	var r = _sb.vmcall("rd_probe", spirv)
	return str(r)

func rd_last_step() -> String:
	return str(_sb.vmcall("rd_last_step")) if _sb != null else "FAIL: no sandbox"
