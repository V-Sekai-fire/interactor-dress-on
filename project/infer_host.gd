# The host side of the pump protocol (guest/pump/pump.h).
#
# A guest job runs on a fiber in its Sandbox; the host advances it with
# pump_frame() once per frame. pump_frame() calls the guest's pump method,
# serves what it asks for, and stops for the frame at:
#   WAIT_GPU  work is submitted; the guest syncs on the NEXT frame (rule 4)
#   COOP      the guest gives the frame back (long setup loops)
#   DONE      the job returned
#   ERROR     the job failed; text says why
# READ and UPLOAD are served in the same frame, until this frame's bytes
# pass cap_bytes (~512 MB); an UPLOAD larger than the rest of the budget is
# carried on the next frame before the guest is resumed:
#   READ    [2, off, n]           text=path: n bytes at off (n = 0: to the
#                                 end; the guest asks at most 16 MiB at a
#                                 time), handed to the next pump call
#   UPLOAD  [3, foff, n, doff]    text=path rid=RD buffer: n bytes at foff of
#                                 the file -> rd.buffer_update(rid, doff, ...)
# The guest cannot open files (Gate 0F probe 3), so every byte it reads comes
# through here, and weights never enter its heap: UPLOAD goes from the file
# to the RenderingDevice, which the host owns and the guest adopted. The
# guest yields UPLOAD only while nothing is submitted.
extends RefCounted

const NONE := 0
const WAIT_GPU := 1
const READ := 2
const UPLOAD := 3
const COOP := 4
const DONE := 5
const ERROR := 6
const KIND_NAMES := ["NONE", "WAIT_GPU", "READ", "UPLOAD", "COOP", "DONE", "ERROR"]
const UPLOAD_CHUNK := 64 * 1024 * 1024

var sb = null                  # the Sandbox running the job
var rd: RenderingDevice = null # the device the guest adopted (may be null)
var pump_method := "ggml_pump"
var cap_bytes := 512 * 1024 * 1024

# "running", "done" or "error"; text is the ERROR reason.
var state := "running"
var text := ""
var last_kind := NONE

# Counters since the last reset().
var frames := 0
var pumps := 0
var vm_us := 0
var waits := 0
var coops := 0
var reads := 0
var read_bytes := 0
var uploads := 0
var upload_bytes := 0

var _feed := PackedByteArray()  # a READ's bytes, for the next pump call
var _up := {}                   # an UPLOAD being carried across frames

func _init(p_sb, p_rd: RenderingDevice, p_method := "ggml_pump") -> void:
	sb = p_sb
	rd = p_rd
	pump_method = p_method

func reset() -> void:
	state = "running"
	text = ""
	last_kind = NONE
	frames = 0
	pumps = 0
	vm_us = 0
	waits = 0
	coops = 0
	reads = 0
	read_bytes = 0
	uploads = 0
	upload_bytes = 0
	_feed = PackedByteArray()
	_up = {}

func summary() -> String:
	return "frames=%d pumps=%d vm_ms=%.1f waits=%d coops=%d reads=%d read_bytes=%d uploads=%d upload_bytes=%d" % [
			frames, pumps, vm_us / 1000.0, waits, coops, reads, read_bytes, uploads, upload_bytes]

# Advance the job by one frame's worth. Returns state.
func pump_frame() -> String:
	if state != "running":
		return state
	frames += 1
	var budget := cap_bytes
	if not _up.is_empty():
		budget = _carry_upload(budget)
		if not _up.is_empty() or state != "running":
			return state
	while true:
		var t0 := Time.get_ticks_usec()
		var r = sb.vmcall(pump_method, _feed)
		vm_us += Time.get_ticks_usec() - t0
		pumps += 1
		_feed = PackedByteArray()
		if typeof(r) != TYPE_ARRAY or r.size() < 3:
			return _fail("%s returned %s (the vmcall failed or was killed)" % [pump_method, str(r)])
		var hdr: PackedInt64Array = r[0]
		last_kind = hdr[0]
		if last_kind == WAIT_GPU:
			waits += 1
			return state
		elif last_kind == COOP:
			coops += 1
			return state
		elif last_kind == DONE:
			state = "done"
			return state
		elif last_kind == ERROR:
			return _fail(str(r[1]))
		elif last_kind == READ:
			var data = _read(str(r[1]), hdr[1], hdr[2])
			if data == null:
				return state
			_feed = data
			reads += 1
			read_bytes += _feed.size()
			budget -= _feed.size()
		elif last_kind == UPLOAD:
			uploads += 1
			_up = {"path": str(r[1]), "foff": hdr[1], "n": hdr[2], "doff": hdr[3], "rid": r[2], "done": 0}
			budget = _carry_upload(budget)
			if not _up.is_empty() or state != "running":
				return state
		else:
			return _fail("unknown request kind %d" % last_kind)
		if budget <= 0:
			return state
	return state

func _fail(why: String) -> String:
	state = "error"
	text = why
	return state

func _read(path: String, off: int, n: int):
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		_fail("READ %s: %s" % [path, error_string(FileAccess.get_open_error())])
		return null
	var flen := f.get_length()
	if n <= 0:
		n = max(flen - off, 0)
	f.seek(off)
	var b := f.get_buffer(n)
	f.close()
	return b

# Serve the current UPLOAD within `budget` bytes; returns what is left.
func _carry_upload(budget: int) -> int:
	if rd == null:
		_fail("UPLOAD without a RenderingDevice")
		return budget
	var f := FileAccess.open(_up.path, FileAccess.READ)
	if f == null:
		_fail("UPLOAD %s: %s" % [_up.path, error_string(FileAccess.get_open_error())])
		return budget
	while _up.done < _up.n and budget > 0:
		var n: int = min(_up.n - _up.done, UPLOAD_CHUNK, budget)
		f.seek(_up.foff + _up.done)
		var b := f.get_buffer(n)
		if b.size() != n:
			f.close()
			_fail("UPLOAD %s: short read at %d (%d of %d bytes)" % [_up.path, _up.foff + _up.done, b.size(), n])
			return budget
		var err := rd.buffer_update(_up.rid, _up.doff + _up.done, n, b)
		if err != OK:
			f.close()
			_fail("UPLOAD buffer_update at %d: %s" % [_up.doff + _up.done, error_string(err)])
			return budget
		_up.done += n
		upload_bytes += n
		budget -= n
	f.close()
	if _up.done >= _up.n:
		_up = {}
	return budget
