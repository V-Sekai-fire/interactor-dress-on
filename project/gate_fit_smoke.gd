# fit.elf smoke: the guest ELF loads, its probes pass, the foxgirl fixture goes
# in through main.gd's no-argument wrappers (rule 8), fit_begin runs, and the
# first phase runs on the worker thread while the main thread keeps ticking.
# Not the Gate 6 guest-vs-native comparison; that runs every phase.
#
#   godot --path project --script gate_fit_smoke.gd --rendering-driver vulkan --xr-mode off
#
# Results stream to gates/6-fit/elf/smoke.txt (flushed per line: Godot buffers
# redirected stdout). Quits on a WALL_S wall clock in every branch.
extends SceneTree

const OUT := "res://../gates/6-fit/elf/smoke.txt"
const WALL_S := 1800.0
const PHASES := 1

var _m: Node = null
var _out: FileAccess
var _t0 := 0
var _state := "wait"
var _frames := 0
var _rc := 0
var _phase_t0 := 0
var _phases_done := 0
var _last_poll := 0

func _say(s: String) -> void:
	print(s)
	if _out != null:
		_out.store_line(s)
		_out.flush()

func _check(name: String, r: String, want_prefix: String) -> void:
	var ok := r.begins_with(want_prefix)
	if not ok:
		_rc = 1
	_say("%s %s: %s" % ["ok  " if ok else "FAIL", name, r])

# Phase 0 of the native same-code run with float32 inputs
# (gates/6-fit/foxgirl/run-sdf-guest-f32.log; the fit form is off in phase 0,
# so the sampler does not enter). The Gate 6 guest-vs-native bound: Newton
# within 2, energy within 1e-6 relative.
const NATIVE_P0_NEWTON := 41
const NATIVE_P0_ENERGY := 0.00033678666696946268

func _vs_native(s: String) -> void:
	var newton := s.get_slice("newton ", 1).get_slice(" ", 0).to_int()
	var energy_s := s.get_slice("energy ", 1).get_slice(" ", 0)
	var rel := absf(energy_s.to_float() - NATIVE_P0_ENERGY) / NATIVE_P0_ENERGY
	var ok := absi(newton - NATIVE_P0_NEWTON) <= 2 and rel <= 1e-6
	# Reported, not gated here: the smoke asks whether fit.elf runs. The
	# guest-vs-native bound is Gate 6's, over every phase.
	_say("%s phase 0 vs native f32: newton %d (native %d), energy %s (native %s), rel %s" % [
			"info" if ok else "INFO-MISS", newton, NATIVE_P0_NEWTON, energy_s, str(NATIVE_P0_ENERGY), str(rel)])

func _initialize() -> void:
	_out = FileAccess.open(OUT, FileAccess.WRITE)
	_t0 = Time.get_ticks_msec()
	_say("fit.elf smoke, %s" % Time.get_datetime_string_from_system())
	_m = load("res://main.gd").new()
	root.add_child(_m)

func _finish() -> void:
	_say("wall %.1f s, frames %d" % [(Time.get_ticks_msec() - _t0) / 1000.0, _frames])
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	_state = "done"
	quit(_rc)

func _process(_dt: float) -> bool:
	_frames += 1
	if _state == "done":
		return false
	if (Time.get_ticks_msec() - _t0) / 1000.0 > WALL_S:
		_rc = 1
		_say("FAIL wall clock %.0f s in state %s: %s" % [WALL_S, _state, _m.fit_status()])
		_finish()
		return false
	match _state:
		"wait":
			if _frames > 3:
				_state = "probes"
		"probes":
			_check("fit_probe_exceptions", _m.fit_probe_exceptions(), "PASS")
			_check("fit_probe_io", _m.fit_probe_io(), "PASS")
			_check("fit_probe_ldlt", _m.fit_probe_ldlt(), "PASS")
			_say("     fit_probe_io_paths: " + _m.fit_probe_io_paths())
			var t := Time.get_ticks_usec()
			var r: String = _m.fit_fixture_foxgirl()
			_check("fit_fixture_foxgirl (%d ms)" % ((Time.get_ticks_usec() - t) / 1000), r, "OK")
			t = Time.get_ticks_usec()
			r = _m.fit_begin()
			_check("fit_begin (%d ms)" % ((Time.get_ticks_usec() - t) / 1000), r, "OK begin")
			_say("     fit_status: " + _m.fit_status())
			if _rc != 0:
				_finish()
				return false
			_state = "step"
		"step":
			_phase_t0 = Time.get_ticks_msec()
			_last_poll = _phase_t0
			_check("fit_step", _m.fit_step(), "STARTED")
			_state = "poll"
		"poll":
			if Time.get_ticks_msec() - _last_poll < 1000:
				return false
			_last_poll = Time.get_ticks_msec()
			var s: String = _m.fit_status()
			if s.begins_with("BUSY"):
				if (_last_poll - _phase_t0) % 60000 < 1000:
					_say("     " + s + " (main thread frames %d)" % _frames)
				return false
			_phases_done += 1
			_check("phase %d" % (_phases_done - 1), s, "phase")
			if s.find("last: host_ms=") < 0 or s.find("FAIL") >= 0:
				_rc = 1
			if _phases_done == 1:
				_vs_native(s)
			if _phases_done < PHASES:
				_state = "step"
				return false
			_check("fit_check", _m.fit_check(), "OK none")
			_say("     fit_result: " + _m.fit_result())
			_say("     fit_preview: " + _m.fit_preview())
			_say("     fit_sdf: " + _m.fit_sdf())
			_finish()
	return false
