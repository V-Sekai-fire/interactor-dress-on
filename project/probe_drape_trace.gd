# Gate 5 diagnostic (not a gate): one drape job with every step's statistics
# (sphere_forward stats=all) and, for a forward, every frame, dumped to a file
# for gates/5-drape/trace/compare_steps.py and loss_attribution.py.
#   godot --path project --script probe_drape_trace.gd --rendering-driver vulkan --xr-mode off -- out=<file> backend=rd steps=350 mu=0.5397701956236457
#   (job=<name> runs another drape job with the remaining key=value args; the wall clock quits after 900 s)
extends SceneTree

var _sb = null
var _t0 := 0
var _out := ""
var _backend := "rd"
var _args := ""
var _steps := 120
var _started := false
var _ticks := 0
var _jobname := "sphere_forward"

func _initialize() -> void:
	_t0 = Time.get_ticks_usec()
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	var extra := []
	for a in OS.get_cmdline_user_args():
		if a.begins_with("out="):
			_out = a.substr(4)
		elif a.begins_with("job="):
			_jobname = a.substr(4)
		elif a.begins_with("backend="):
			_backend = a.substr(8)
		else:
			if a.begins_with("steps="):
				_steps = int(a.substr(6))
			extra.append(a)
	_args = " ".join(PackedStringArray(extra)) + (" stats=all" if _jobname == "sphere_forward" else "")
	_sb = ClassDB.instantiate("Sandbox")
	if _sb != null: _sb.allocations_max = 1000000 # the Linux addon's 4000 default runs out (stages/sandbox_util.gd)
	_sb.references_max = 65536
	_sb.program = load("res://drape.elf")

func _process(_d: float) -> bool:
	if Time.get_ticks_usec() - _t0 > 900 * 1000000:
		print("wall clock")
		quit(2)
		return true
	if not _started:
		print(_sb.vmcall("drape_job_start", _jobname, _backend, _args))
		_started = true
		return false
	var r := str(_sb.vmcall("drape_job_tick", Time.get_ticks_usec()))
	_ticks += 1
	if r.begins_with("RUNNING"):
		return false
	var f := FileAccess.open(_out, FileAccess.WRITE)
	f.store_string(r + "\n")
	for i in range(_steps + 1):
		var a: PackedFloat32Array = _sb.vmcall("drape_job_frame", i)
		var parts := PackedStringArray()
		for x in a:
			parts.append("%.9f" % x)
		f.store_line("FRAME %d %s" % [i, " ".join(parts)])
	f.close()
	print("done ticks=%d" % _ticks)
	_sb.vmcall("rd_close")
	quit(0)
	return true
