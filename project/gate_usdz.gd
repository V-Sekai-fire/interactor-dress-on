# Gate U: usd.elf -- a Pixal3D .usdz opens from bytes inside the sandbox and
# its arrays equal the host's OpenUSD reading of the same package
# (gates/U-usd/host_oracle.py -> host-oracle.log).
#
#   timeout 300 python gates/U-usd/host_oracle.py > gates/U-usd/host-oracle.log
#   godot --path project --script gate_usdz.gd --rendering-driver vulkan --xr-mode off
#       -> gates/U-usd/results.txt (RESULT: PASS|FAIL last)
#   godot --path project --script gate_usdz.gd --rendering-driver vulkan --xr-mode off -- --rung=128
#       -> gates/U-usd/ladder/128.txt: one memory_max rung in its own process
#          (below the floor the guest faults or Godot dies, so one rung a process)
#
# 1. init: the embedded plugins and the in-memory resolver come up.
# 2. every case of host-oracle.log: the guest's mesh count, per-mesh point
#    and triangle counts and BLAKE3 sums, material count and texture
#    sizes + sums equal the host's; the wiring (diffuse rgb, metallic b,
#    roughness g) is read; the textures decode in Godot (Image); the arrays
#    that crossed are re-summed on the host and must equal the guest's own
#    sum (the transfer, not just the guest's reading). Corrupt inputs: a clean ERR:, and the guest answers
#    afterwards. Host-timed open and extract; instret of the open.
# 3. usd_nodes: an ArrayMesh + StandardMaterial3D + Node3D from the arrays.
# One case per frame; quits on a 300 s wall clock in every branch.
extends SceneTree

const UsdStage := preload("res://stages/usd_stage.gd")
const UsdNodes := preload("res://util/usd_nodes.gd")
const GATE := "res://../gates/U-usd/"
const EXT := "C:/b/gu-inputs/"
const WALL_S := 300.0

var _stage = null
var _out: FileAccess
var _t_start := 0
var _rc := 0
var _done := false
var _queue: Array = [] # [name, bytes, expected line]
var _step := 0
var _rung := 0
var _max_instr := 0
var _timings := []
var _mem := 0 # --mem=MiB overrides the stage's MEM_MB for the whole run
var _skip := PackedStringArray() # --skip=mesh,images,nodes,extract (bisecting a crash)

func _say(line: String) -> void:
	print(line)
	if _out != null:
		_out.store_line(line)
		_out.flush()

func _check(ok: bool, what: String) -> bool:
	_say("%s %s" % ["PASS" if ok else "FAIL", what])
	if not ok:
		_rc = 1
	return ok

func _finish() -> void:
	_say("RESULT: %s" % ("PASS" if _rc == 0 else "FAIL"))
	if _out != null:
		_out.close()
		_out = null
	_done = true
	quit(_rc)

func _bytes(name: String) -> PackedByteArray:
	for p in [GATE + "inputs/" + name, EXT + name]:
		var g := ProjectSettings.globalize_path(p)
		if FileAccess.file_exists(g):
			return FileAccess.get_file_as_bytes(g)
	return PackedByteArray()

# One checksum everywhere: BLAKE3, compared as its first 12 hex digits. The
# guest sums what it read, this sums what crossed, and host_oracle.py sums
# what usd-core read. Godot's HashingContext has no BLAKE3, so what crossed
# goes back through usd_blake3: a byte lost either way changes the sum.
func _b3(b: PackedByteArray) -> String:
	return str(_stage.call_now("usd_blake3", [b])).left(12)

# The ELF's identity, as build.sh's sha256sum prints it (a file id, not a check).
static func _elf_sha256(b: PackedByteArray) -> String:
	var c := HashingContext.new()
	c.start(HashingContext.HASH_SHA256)
	c.update(b)
	return c.finish().hex_encode().substr(0, 12)

static func _field(line: String, key: String, def: String = "") -> String:
	return UsdStage._field(line, key, def)

func _cases() -> void:
	var f := FileAccess.open(ProjectSettings.globalize_path(GATE + "host-oracle.log"), FileAccess.READ)
	if f == null:
		_check(false, "host-oracle.log missing (run host_oracle.py first)")
		return
	_say("host control: " + f.get_line())
	while not f.eof_reached():
		var line := f.get_line().strip_edges()
		if line.is_empty() or line.ends_with(" missing"):
			continue
		var name := line.get_slice(" ", 0)
		var b := _bytes(name)
		if b.is_empty():
			_check(false, "%s: no bytes" % name)
			continue
		if not _check(line.get_slice(" ", 1) == "bytes=%d" % b.size(), "%s: %d bytes, as the host read" % [name, b.size()]):
			continue
		_queue.append([name, b, line])
	# an empty package: nothing the host could read either
	_queue.append(["empty", PackedByteArray(), "empty bytes=0 blake3=- ERR: empty package"])

func _initialize() -> void:
	_t_start = Time.get_ticks_usec()
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--rung="):
			_rung = int(a.substr(7))
		elif a.begins_with("--mem="):
			_mem = int(a.substr(6))
		elif a.begins_with("--skip="):
			_skip = a.substr(7).split(",")
	var out_path := GATE + ("ladder/%d.txt" % _rung if _rung > 0 else "results.txt")
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(GATE + "ladder"))
	_out = FileAccess.open(ProjectSettings.globalize_path(out_path), FileAccess.WRITE)
	_say("# Gate U (usd.elf), %s, Godot %s%s" % [Time.get_datetime_string_from_system(true),
			Engine.get_version_info().string, " rung memory_max=%d" % _rung if _rung > 0 else ""])
	_stage = UsdStage.new()
	if _rung > 0:
		_stage.mem_mb = _rung
	elif _mem > 0:
		_stage.mem_mb = _mem
	var t := Time.get_ticks_usec()
	root.add_child(_stage)
	_stage.ensure() # the root is not ready yet, so _ready() would wait for the first frame
	if _stage.sandbox == null:
		_check(false, "usd.elf: %s" % _stage.reason)
		_finish()
		return
	var elf := FileAccess.get_file_as_bytes(ProjectSettings.globalize_path("res://usd.elf"))
	_say("load usd.elf: %.1f ms; %d bytes sha256=%s memory_max=%s execution_timeout=%s allocations_max=%s heap=%d" % [
			(Time.get_ticks_usec() - t) / 1000.0, elf.size(), elf.slice(0, 0).hex_encode() if elf.is_empty() else _elf_sha256(elf),
			str(_stage.sandbox.memory_max), str(_stage.sandbox.execution_timeout), str(_stage.sandbox.allocations_max), _stage.heap()])
	if _rung > 0:
		var big := "real-t2048.usdz" if not _bytes("real-t2048.usdz").is_empty() else "real-asset.usdz"
		_queue.append([big, _bytes(big), ""])
	else:
		_cases()

# One case: open, compare with the host line, extract, re-sum, decode textures.
func _case(name: String, bytes: PackedByteArray, exp: String) -> void:
	var r: String = _stage.open_package(bytes)
	var open_us: int = _stage.last_open_us
	if _stage.last_load_us > 0:
		_say("     (usd.elf reloaded in a fresh Sandbox: %.1f ms)" % (_stage.last_load_us / 1000.0))
	var heap: int = _stage.heap()
	var instr := int(_field(r, "instr", "0"))
	_max_instr = maxi(_max_instr, instr)
	var exp_ok := exp.find(" ok ") > 0
	if _rung > 0:
		exp_ok = true
	if not exp_ok:
		# a clean error, and the guest still answers
		var ok: bool = r.begins_with("ERR:") and _stage.count() == 0
		_check(ok, "%-18s %8.1f ms heap=%9d  guest: %s" % [name, open_us / 1000.0, heap, r.left(220)])
		_stage.close()
		return
	if not _check(r.begins_with("ok "), "%-18s %8.1f ms heap=%9d  guest: %s" % [name, open_us / 1000.0, heap, r.left(260)]):
		return
	var t0 := Time.get_ticks_usec()
	if _skip.has("extract"):
		_stage.close()
		_check(true, "%s: extract skipped, closed" % name)
		return
	var doc: Dictionary = _stage.document()
	var extract_us := Time.get_ticks_usec() - t0
	var meshes: int = _stage.count()
	var host_meshes := int(_field(exp, "meshes", "-1"))
	if _rung == 0:
		_check(meshes == host_meshes, "%s: meshes=%d (host %d), materials=%d (host %s), textures=%d (host %s); open %.1f ms (%d instr = %d units), extract %.1f ms, heap %d" % [
				name, meshes, host_meshes, _stage.material_count(), _field(exp, "materials"), _stage.texture_count(),
				_field(exp, "textures"), open_us / 1000.0, instr, instr >> 20, extract_us / 1000.0, _stage.heap()])
	else:
		_check(meshes >= 0, "%s: meshes=%d open %.1f ms extract %.1f ms heap %d" % [name, meshes, open_us / 1000.0, extract_us / 1000.0, _stage.heap()])
	_timings.append([name, open_us, extract_us, instr])
	for i in meshes:
		var m: Dictionary = doc.meshes[i]
		if m.has("error"):
			_check(false, "%s mesh%d: %s" % [name, i, m.error])
			continue
		var pts: PackedFloat32Array = m.points
		var idx: PackedInt32Array = m.indices
		var n: int = m.point_count
		var host := ""
		if _rung == 0:
			# the host's fields for this mesh sit between mesh<i>= and the next mesh or materials=
			var at := exp.find(" mesh%d=" % i)
			var end := exp.find(" mesh%d=" % (i + 1))
			if end < 0:
				end = exp.find(" materials=")
			host = exp.substr(at, end - at) if at >= 0 else ""
		var b3_p := _b3(pts.to_byte_array())
		var b3_i := _b3(idx.to_byte_array())
		var guest_p := str(m.blake3_points).left(12)
		var guest_i := str(m.blake3_indices).left(12)
		var line := "%s mesh%d=%s points=%d triangles=%d material=%d blake3_points=%s blake3_indices=%s normals=%d uvs=%d indexed=%s" % [
				name, i, m.path, n, m.triangles, m.material, b3_p, b3_i, m.normals.size() / 3, m.uvs.size() / 2, str(m.indexed)]
		var ok: bool = pts.size() == 3 * n and idx.size() == 3 * int(m.triangles)
		# what crossed == what the guest read
		ok = ok and guest_p == b3_p and guest_i == b3_i
		if guest_p != b3_p or guest_i != b3_i:
			line += " guest_blake3_points=%s guest_blake3_indices=%s" % [guest_p, guest_i]
		if _rung == 0:
			ok = ok and host == " mesh%d=%s points=%d triangles=%d material=%d blake3_points=%s blake3_indices=%s" % [
					i, m.path, n, m.triangles, m.material, b3_p, b3_i]
		_check(ok, line + ("" if ok else "  | host:%s" % host))
		# the arrays make a surface
		if _skip.has("mesh"):
			continue
		var am := UsdNodes.to_array_mesh(m)
		_check(am.get_surface_count() == 1 and am.surface_get_array_len(0) == n and am.surface_get_array_index_len(0) == 3 * int(m.triangles),
				"%s mesh%d: ArrayMesh %d vertices %d indices, aabb %s" % [name, i, am.surface_get_array_len(0), am.surface_get_array_index_len(0), str(am.get_aabb())])
		var xf: PackedFloat32Array = m.xform
		_say("     xform rows: [%s] [%s] [%s] origin [%s]" % [_row(xf, 0), _row(xf, 4), _row(xf, 8), _row(xf, 12)])
	# materials: wiring and textures
	var wired := []
	var texs := []
	var cache := {}
	for k in doc.materials.size():
		var mat: Dictionary = doc.materials[k]
		if mat.has("error"):
			_check(false, "%s material%d: %s" % [name, k, mat.error])
			continue
		for inp in ["diffuseColor:diffuse", "metallic:metallic", "roughness:roughness", "opacity:opacity", "normal:normal"]:
			var usd_name: String = inp.get_slice(":", 0)
			var key: String = inp.get_slice(":", 1)
			var t: int = int(mat.get(key + "_texture", -1))
			if _skip.has("images"):
				continue
			# connected is wired, whether or not its texture resolved (the host lists both)
			if not str(mat.get(key + "_channel", "")).is_empty():
				wired.append("%s:%s:%s" % [usd_name, mat.get(key + "_file", ""), mat.get(key + "_channel", "")])
			if t >= 0:
				if not cache.has(t):
					var b: PackedByteArray = mat.get(key + "_bytes", PackedByteArray())
					var img := UsdNodes.to_image(b)
					cache[t] = "%s:%d:%s" % [mat.get(key + "_file", ""), b.size(), _b3(b)]
					_check(img != null, "%s texture %d (%s): %d bytes, decodes to %s" % [name, t, mat.get(key + "_file", ""), b.size(),
							"nothing" if img == null else "%dx%d %s" % [img.get_width(), img.get_height(), str(img.get_format())]])
		if _skip.has("images"):
			continue
		var sm := UsdNodes.to_material(mat)
		_say("     material%d %s shader=%s diffuse=%s metallic=%.3f roughness=%.3f opacity=%.3f -> albedo_texture=%s metallic_texture=%s(ch %d) roughness_texture=%s(ch %d)" % [
				k, mat.path, mat.shader_id, str(mat.diffuse), mat.metallic, mat.roughness, mat.opacity,
				sm.albedo_texture != null, sm.metallic_texture != null, sm.metallic_texture_channel, sm.roughness_texture != null, sm.roughness_texture_channel])
	if _rung == 0:
		var host_tex := _field(exp, "tex", "-")
		var host_wired := _field(exp, "wired", "-")
		var got_tex := ",".join(cache.values()) if not cache.is_empty() else "-"
		var got_wired := ",".join(wired) if not wired.is_empty() else "-"
		_check(got_tex == host_tex, "%s textures as the host read: %s%s" % [name, got_tex, "" if got_tex == host_tex else " | host " + host_tex])
		_check(got_wired == host_wired, "%s wiring as the host read: %s%s" % [name, got_wired, "" if got_wired == host_wired else " | host " + host_wired])
	# the node tree
	if _skip.has("nodes"):
		_stage.close()
		return
	var node := UsdNodes.to_node(doc)
	var mi := 0
	for c in node.get_children():
		if c is MeshInstance3D and c.mesh != null and c.mesh.get_surface_count() > 0:
			mi += 1
	_check(mi == meshes, "%s: usd_nodes -> Node3D with %d MeshInstance3D (%d meshes)" % [name, mi, meshes])
	node.free()
	_stage.close()
	_check(_stage.count() == 0, "%s: closed, count=%d heap=%d" % [name, _stage.count(), _stage.heap()])

static func _row(xf: PackedFloat32Array, at: int) -> String:
	if xf.size() < at + 3:
		return "?"
	return "%.4f %.4f %.4f" % [xf[at], xf[at + 1], xf[at + 2]]

func _process(_delta: float) -> bool:
	if _done:
		return true
	if Time.get_ticks_usec() - _t_start > int(WALL_S * 1e6):
		_check(false, "the %d s wall clock ran out (%d cases left)" % [int(WALL_S), _queue.size()])
		_finish()
		return true
	if _step == 0:
		_step = 1
		var t := Time.get_ticks_usec()
		var s: String = _stage.init()
		_check(s.begins_with("ok ") and not s.contains("first_error"), "usd_init %.1f ms heap=%d: %s" % [
				(Time.get_ticks_usec() - t) / 1000.0, _stage.heap(), s])
		if not s.begins_with("ok "):
			_finish()
		return _done
	if _queue.is_empty():
		if _rung == 0:
			_say("largest open: %d instructions = %d units of 2^20 (execution_timeout in use: %s)" % [
					_max_instr, _max_instr >> 20, str(_stage.sandbox.execution_timeout)])
			# a second open of the real package after everything else: no stale state
			var b := _bytes("real-asset.usdz")
			if not b.is_empty():
				var r: String = _stage.open_package(b)
				_check(r.begins_with("ok ") and _stage.count() == 1, "real-asset.usdz again after every case: %.1f ms %s" % [_stage.last_open_us / 1000.0, r.left(120)])
				_stage.close()
		_finish()
		return true
	var c: Array = _queue.pop_front()
	_case(c[0], c[1], c[2])
	return _done
