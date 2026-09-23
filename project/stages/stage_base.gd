# stage_base -- what every stage node shares: its one Sandbox (made by
# sandbox_util), the rule that at most one vmcall is in flight on it, and a
# worker Thread for calls that run for seconds (Gate 0F probe 7: a vmcall on a
# GDScript Thread works and the main thread keeps its frames; probe 8: two
# Sandboxes on two Threads at once are fine).
#
#   call_now(fn, args)   a vmcall on the calling thread; "BUSY ..." while a
#                        worker call is in flight on this sandbox
#   start(fn, args)      a vmcall on the worker Thread; poll() until it is done
#   poll()               {done: bool, result: Variant, host_ms: int}
#
# Two kinds of worker. By default every start() makes a Thread of its own and
# it ends with the call. With persistent_worker set (the fit stage), one
# Thread lives for the session and takes the calls from a queue: Gate 6G.1
# found that a local RenderingDevice is bound to the OS thread that created
# it, so a guest that keeps a device across vmcalls (fit.elf's GPU Hessian)
# must see the same thread every time.
#
# Rule 4: nothing here waits. The pipeline advances from _process and reads a
# worker result only once the call has ended (Thread.is_alive() false, or the
# persistent worker's done flag); the persistent Thread is joined only at
# _exit_tree, after it was told to quit.
extends Node

const SandboxUtil := preload("res://stages/sandbox_util.gd")

var sandbox = null
var reason := ""        # why sandbox is null
var stage_name := ""
var vm_us := 0          # host-timed vmcall time since the last take_vm_us()
var persistent_worker := false

var _thread: Thread = null
var _call := ""
var _t0 := 0
var _result = null
var _result_ms := 0

# The persistent worker: a job posted under _pmutex, woken by _psem.
var _pthread: Thread = null
var _psem := Semaphore.new()
var _pmutex := Mutex.new()
var _pjob := {}
var _pdone := true
var _pquit := false
var _pout = null

func open_sandbox(elf: String, mem_mb: int, refs: int, timeout_units: int, extra: Dictionary = {},
		required: PackedStringArray = PackedStringArray()) -> bool:
	if sandbox != null:
		return true
	var r := SandboxUtil.make_sandbox(self, elf, mem_mb, refs, timeout_units, extra, required)
	sandbox = r.sandbox
	reason = r.reason
	if sandbox != null:
		print("[dress-on] %s: sandbox loaded %s" % [stage_name, elf.get_file()])
	else:
		print("[dress-on] %s: %s" % [stage_name, reason])
	return sandbox != null

func available() -> bool:
	return sandbox != null

func busy() -> bool:
	if persistent_worker:
		_pmutex.lock()
		var b := not _pdone
		_pmutex.unlock()
		return b
	return _thread != null and _thread.is_alive()

func busy_text() -> String:
	return "BUSY %s for %.1f s" % [_call, (Time.get_ticks_msec() - _t0) / 1000.0]

func heap() -> int:
	return SandboxUtil.heap(sandbox)

func take_vm_us() -> int:
	var v := vm_us
	vm_us = 0
	return v

func call_now(fn: String, args: Array = []):
	if sandbox == null:
		return "FAIL: %s missing (%s)" % [stage_name, reason]
	if busy():
		return busy_text()
	_reap()
	var t0 := Time.get_ticks_usec()
	var r = sandbox.callv("vmcall", [fn] + args)
	vm_us += Time.get_ticks_usec() - t0
	return r

func start(fn: String, args: Array = []) -> String:
	if sandbox == null:
		return "FAIL: %s missing (%s)" % [stage_name, reason]
	if busy():
		return busy_text()
	_reap()
	_call = fn
	_t0 = Time.get_ticks_msec()
	_result = null
	if persistent_worker:
		if _pthread == null:
			_pthread = Thread.new()
			_pthread.start(_ploop)
		_pmutex.lock()
		_pjob = {"fn": fn, "args": args}
		_pdone = false
		_pout = null
		_pmutex.unlock()
		_psem.post()
		return "STARTED %s" % fn
	_thread = Thread.new()
	_thread.start(_worker.bind(fn, args))
	return "STARTED %s" % fn

func _worker(fn: String, args: Array):
	var t0 := Time.get_ticks_usec()
	var r = sandbox.callv("vmcall", [fn] + args)
	return [r, Time.get_ticks_usec() - t0]

# The persistent worker's loop: one vmcall per posted job, until quit.
func _ploop() -> void:
	while true:
		_psem.wait()
		_pmutex.lock()
		var quit := _pquit
		var job: Dictionary = _pjob
		_pmutex.unlock()
		if quit:
			return
		var t0 := Time.get_ticks_usec()
		var r = sandbox.callv("vmcall", [job.fn] + job.args)
		var dt := Time.get_ticks_usec() - t0
		_pmutex.lock()
		_pout = [r, dt]
		_pdone = true
		_pmutex.unlock()

func _reap() -> void:
	if persistent_worker:
		_pmutex.lock()
		var out = _pout if _pdone else null
		_pout = null
		_pmutex.unlock()
		if out != null:
			_result = out[0]
			_result_ms = int(out[1] / 1000)
			vm_us += int(out[1])
		return
	if _thread != null and not _thread.is_alive():
		var out = _thread.wait_to_finish()
		_thread = null
		_result = out[0]
		_result_ms = int(out[1] / 1000)
		vm_us += int(out[1])

# {done, result, host_ms, call}. done is true once the worker call has ended
# (and stays true, with the same result, until the next start()).
func poll() -> Dictionary:
	if busy():
		return {"done": false, "result": null, "host_ms": Time.get_ticks_msec() - _t0, "call": _call}
	_reap()
	return {"done": true, "result": _result, "host_ms": _result_ms, "call": _call}

func _exit_tree() -> void:
	if _thread != null:
		_thread.wait_to_finish()
		_thread = null
	if _pthread != null:
		_pmutex.lock()
		_pquit = true
		_pmutex.unlock()
		_psem.post()
		_pthread.wait_to_finish()
		_pthread = null
