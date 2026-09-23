# fit_stage -- fit.elf (Cut 6): cloth-fit's garment retargeting, one phase per
# vmcall. A phase runs for seconds to minutes, so fit_begin and fit_step go to
# the worker Thread (Gate 0F probe 7) and the pipeline polls between jobs;
# every other fit call answers BUSY meanwhile (one VM, one vmcall at a time).
#
# The MCP wrappers mirror cut-6's main.gd (same names) so its drives keep
# working after the merge; main.gd delegates to them.
extends "res://stages/stage_base.gd"

const ObjIO := preload("res://util/obj_io.gd")
const REQUIRED := ["fit_reset", "fit_set_body", "fit_set_skeletons", "fit_set_garment", "fit_set_config", "fit_begin",
		"fit_step", "fit_status", "fit_result_vertices", "fit_check_intersections"]

var _last := "" # the last worker call's answer

func _ready() -> void:
	stage_name = "fit"
	# Before program=: a lower memory_max later is ignored (Gate 0F probe 6).
	# The native run peaks at 214 MB; the heap is 0.8 x memory_max.
	# allocations_max: the default 10000 live chunks is exhausted inside
	# fit_begin (cut-6: "Too many arena chunks"). execution_timeout: a phase
	# is far past the default 8000 x 2^20 instructions.
	open_sandbox("res://fit.elf", 2048, 4096, 1 << 30, {"allocations_max": 4000000}, PackedStringArray(REQUIRED))

# --- pipeline calls -------------------------------------------------------------------

# Hands the whole problem over (small, synchronous). "" on success, else the
# first guest answer that was not OK.
func setup(body_v: PackedFloat32Array, body_f: PackedInt32Array, src_skel: PackedFloat32Array,
		tgt_skel: PackedFloat32Array, bones: PackedInt32Array, garment_v: PackedFloat32Array,
		garment_f: PackedInt32Array, nofit: PackedInt32Array, config_text: String) -> String:
	var calls := [
		["fit_reset", []],
		["fit_set_body", [body_v, body_f]],
		["fit_set_skeletons", [src_skel, tgt_skel, bones]],
		["fit_set_garment", [garment_v, garment_f, nofit]],
		["fit_set_config", [config_text]],
	]
	for c in calls:
		var r := str(call_now(c[0], c[1]))
		if not r.begins_with("OK"):
			return "%s: %s" % [c[0], r]
	return ""

func start_begin() -> String:
	return start("fit_begin")

func start_step() -> String:
	return start("fit_step")

func status_raw() -> String:
	return str(call_now("fit_status"))

func result_vertices():
	return call_now("fit_result_vertices")

# override: empty checks the solver's own state; else that garment (body space).
func check_intersections(override_v: PackedFloat32Array = PackedFloat32Array()) -> String:
	return str(call_now("fit_check_intersections", [override_v]))

# --- MCP wrappers (cut-6's, rule 8) --------------------------------------------------

func _fit_busy() -> String:
	if sandbox == null:
		return "FAIL: no fit sandbox (%s)" % reason
	if busy():
		return busy_text()
	var p := poll()
	if p.result != null:
		_last = "host_ms=%d %s" % [p.host_ms, str(p.result)]
	return ""

func _fit_now(method: String, args: Array = []) -> String:
	var b := _fit_busy()
	return b if b != "" else str(call_now(method, args))

func _fit_start(method: String) -> String:
	var b := _fit_busy()
	if b != "":
		return b
	var r := start(method)
	return r + " (poll fit_status)" if r.begins_with("STARTED") else r

func _repo_path(rel: String) -> String:
	return ProjectSettings.globalize_path("res://").path_join("..").path_join(rel).simplify_path()

# cut-6's oracle fixture (tools/native/foxgirl_oracle.json and
# vendor/cloth-fit/garment-data, both from cut-6), then call fit_begin.
func fit_fixture_foxgirl() -> String:
	var b := _fit_busy()
	if b != "":
		return b
	var cfg_path := _repo_path("tools/native/foxgirl_oracle.json")
	var cfg_text := FileAccess.get_file_as_string(cfg_path)
	if cfg_text.is_empty():
		return "FAIL: cannot read %s" % cfg_path
	var cfg = JSON.parse_string(cfg_text)
	if typeof(cfg) != TYPE_DICTIONARY:
		return "FAIL: %s is not a JSON object" % cfg_path
	var body: Dictionary = ObjIO.read(_repo_path(cfg["avatar_mesh_path"]))
	var garment: Dictionary = ObjIO.read(_repo_path(cfg["garment_mesh_path"]))
	var src_sk: Dictionary = ObjIO.read(_repo_path(cfg["source_skeleton_path"]))
	var tgt_sk: Dictionary = ObjIO.read(_repo_path(cfg["target_skeleton_path"]))
	for m in [body, garment, src_sk, tgt_sk]:
		if m.has("error"):
			return "FAIL: " + str(m["error"])
	if src_sk["l"] != tgt_sk["l"]:
		return "FAIL: source and target skeletons have different bones"
	var nofit := PackedInt32Array()
	if str(cfg.get("no_fit_spec_path", "")) != "":
		var r: Dictionary = ObjIO.read_ints(_repo_path(cfg["no_fit_spec_path"]))
		if r.has("error"):
			return "FAIL: " + str(r["error"])
		nofit = r["ints"]
	var e := setup(body["v"], body["f"], src_sk["v"], tgt_sk["v"], src_sk["l"], garment["v"], garment["f"], nofit, cfg_text)
	return "OK fixture foxgirl_skirt" if e == "" else "FAIL: " + e

func fit_reset() -> String:
	return _fit_now("fit_reset")

func fit_begin() -> String:
	return _fit_now("fit_begin")

# One phase (AL solve or reduced solve) on the worker thread.
func fit_step() -> String:
	return _fit_start("fit_step")

# Every remaining phase in one vmcall, on the worker thread.
func fit_run_all() -> String:
	return _fit_start("fit_run_all")

func fit_status() -> String:
	var b := _fit_busy()
	if b != "":
		return b
	return "%s host_heap_usage=%d | last: %s" % [status_raw(), heap(), _last]

func fit_check() -> String:
	return _fit_now("fit_check_intersections", [PackedFloat32Array()])

func fit_result() -> String:
	var b := _fit_busy()
	if b != "":
		return b
	var v = call_now("fit_result_vertices")
	if typeof(v) != TYPE_PACKED_FLOAT32_ARRAY:
		return str(v)
	return "garment %d v, bounds %s" % [v.size() / 3, str(bounds(v))]

func fit_result_vertices() -> Variant:
	var b := _fit_busy()
	return b if b != "" else call_now("fit_result_vertices")

func fit_result_vertices_f64() -> Variant:
	var b := _fit_busy()
	return b if b != "" else call_now("fit_result_vertices_f64")

func fit_preview() -> String:
	var b := _fit_busy()
	if b != "":
		return b
	var v = call_now("fit_preview", [0])
	if typeof(v) != TYPE_PACKED_FLOAT32_ARRAY:
		return str(v)
	return "preview %d v, bounds %s" % [v.size() / 3, str(bounds(v))]

func fit_sdf() -> String:
	var b := _fit_busy()
	if b != "":
		return b
	var g = call_now("fit_result_vertices_f64")
	if typeof(g) != TYPE_PACKED_FLOAT64_ARRAY:
		return str(g)
	var d = call_now("fit_sdf_dump", [g])
	if typeof(d) != TYPE_PACKED_FLOAT64_ARRAY:
		return str(d)
	var n: int = d.size() / 10
	var lo := INF
	var hi := -INF
	var inside := 0
	for i in n:
		var x: float = d[10 * i]
		lo = minf(lo, x)
		hi = maxf(hi, x)
		if x < 0.0:
			inside += 1
	return "sdf at %d garment vertices: min %.4g max %.4g voxels, %d inside" % [n, lo, hi, inside]

func fit_probe_io() -> String:
	return _fit_now("fit_probe", ["io"])

func fit_probe_ldlt() -> String:
	return _fit_now("fit_probe", ["ldlt"])

func fit_probe_exceptions() -> String:
	return _fit_now("fit_probe", ["exceptions"])

func fit_probe_io_paths() -> String:
	return _fit_now("fit_probe", ["io_paths"])

static func bounds(v: PackedFloat32Array) -> AABB:
	if v.size() < 3:
		return AABB()
	var box := AABB(Vector3(v[0], v[1], v[2]), Vector3.ZERO)
	for i in range(0, v.size(), 3):
		box = box.expand(Vector3(v[i], v[i + 1], v[i + 2]))
	return box
