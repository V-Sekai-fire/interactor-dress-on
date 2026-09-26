# The skateboard (issue 5): the whole character clip end to end, crude first.
# The FoxGirl body and its 15-joint skeleton (Gate 8's infer and rig
# fixtures), the draped garment (the fixture garment, or --garment=<obj> such as
# a Gate 8 run's <out>.fitted.obj), a 3-second wave keyed on the right arm,
# rendered at 1920 x 1080, 24 fps.
#
#   godot --path project --rendering-driver vulkan --xr-mode off \
#       --write-movie ../gates/skateboard/wave.avi --fixed-fps 24 --resolution 1920x1080 \
#       --script skateboard.gd -- --out=../gates/skateboard/results.txt
#   (CineForm on Windows: --write-movie wave.cfhd)
#
# Stand-ins, each named in the results: the body and rig are fixtures; the
# skin weights are nearest-bone by distance (two bones, inverse distance), not
# a rig's; the garment is rigid (it does not follow the arm). Rotations are 3x3
# matrices built from an axis and an angle (AGENTS.md rule 11: no Euler angles,
# no quaternions); skinning is linear blend on the CPU, 5128 vertices a frame.
#
# The wave is keyed, not simulated: shoulder raise 0 -> 1.25 rad over 0.5 s,
# the forearm swinging +-0.45 rad at 2 Hz from 0.5 s to 2.5 s, back down by 3 s.
# wave_key(t) is the key function; skateboard_wave() (below) is the no-argument
# entry an MCP call_method reaches (rule 8's pattern).
extends SceneTree

const FIX := "res://fixtures/foxgirl/"
const ObjIo := preload("res://util/obj_io.gd")
const SECONDS := 3.0
const FPS := 24

# Joint indices in skeleton.obj (0-based): 0 hips, 1 spine, 2 head,
# 3/4/5 one arm (shoulder, elbow, wrist; +x), 6/8/7 the other, 9-11 and 12-14 legs.
const SHOULDER := 3
const ELBOW := 4
const WRIST := 5

var _out: FileAccess
var _body: MeshInstance3D
var _rest := PackedVector3Array()
var _normals_rest := PackedVector3Array()
var _tris := PackedInt32Array()
var _bone_of := PackedInt32Array()   # per vertex: 0 none, 1 upper arm, 2 forearm
var _w := PackedFloat32Array()       # per vertex: weight of that bone (rest is identity)
var _joints := PackedVector3Array()
var _frame := 0
var _frames := int(SECONDS * FPS)
var _t0 := 0

func _say(s: String) -> void:
	print(s)
	if _out:
		_out.store_line(s)
		_out.flush()

func _initialize() -> void:
	var out := "res://../gates/skateboard/results.txt"
	var garment := FIX + "garment.obj"
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--out="):
			out = a.trim_prefix("--out=")
		elif a.begins_with("--garment="):
			garment = a.trim_prefix("--garment=")
	var dir := ProjectSettings.globalize_path(out).get_base_dir()
	DirAccess.make_dir_recursive_absolute(dir)
	_out = FileAccess.open(ProjectSettings.globalize_path(out), FileAccess.WRITE)
	_t0 = Time.get_ticks_msec()

	var body := ObjIo.read(ProjectSettings.globalize_path(FIX + "avatar.obj"))
	var skel := ObjIo.read(ProjectSettings.globalize_path(FIX + "skeleton.obj"))
	var gar := ObjIo.read(ProjectSettings.globalize_path(garment) if garment.begins_with("res://") else garment)
	if body.has("error") or skel.has("error") or gar.has("error"):
		_say("FAIL reading fixtures: %s %s %s" % [body.get("error", ""), skel.get("error", ""), gar.get("error", "")])
		quit(1)
		return
	_joints = _vec3(skel.v)
	_rest = _vec3(body.v)
	_tris = PackedInt32Array(body.f)
	_weights()

	var root := Node3D.new()
	get_root().add_child(root)
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.background_color = Color(0.13, 0.14, 0.16)
	env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.environment.ambient_light_color = Color(0.45, 0.47, 0.5)
	root.add_child(env)
	var sun := DirectionalLight3D.new()
	sun.transform = Transform3D(Basis(Vector3(1, 0, 0), -0.8) * Basis(Vector3(0, 1, 0), 0.5), Vector3.ZERO)
	root.add_child(sun)
	var cam := Camera3D.new()
	cam.position = Vector3(0.0, 1.0, 2.9)
	cam.fov = 45.0
	root.add_child(cam)
	cam.make_current()

	_body = MeshInstance3D.new()
	_body.material_override = _mat(Color(0.86, 0.72, 0.62))
	root.add_child(_body)
	var g := MeshInstance3D.new()
	g.mesh = _mesh(_vec3(gar.v), PackedInt32Array(gar.f))
	g.material_override = _mat(Color(0.25, 0.42, 0.72))
	root.add_child(g)
	_pose(0.0)

	var n1 := 0
	var n2 := 0
	for b in _bone_of:
		n1 += 1 if b == 1 else 0
		n2 += 1 if b == 2 else 0
	_say("skateboard: body %d v %d f (fixture), skeleton %d joints (fixture), garment %d v (%s, rigid)" % [
			_rest.size(), _tris.size() / 3, _joints.size(), gar.v.size() / 3, garment])
	_say("skin (stand-in, nearest-bone): upper arm %d v, forearm %d v, the rest static" % [n1, n2])
	_say("wave: %d frames at %d fps (%.1f s), movie %s" % [_frames, FPS, SECONDS,
			"on (--write-movie)" if OS.get_cmdline_args().has("--write-movie") else "off (no --write-movie)"])

func _process(_delta: float) -> bool:
	var t := float(_frame) / FPS
	_pose(t)
	_frame += 1
	if _frame > _frames:
		var bad := 0
		var m := _body.mesh as ArrayMesh
		for v in m.surface_get_arrays(0)[Mesh.ARRAY_VERTEX]:
			bad += 0 if (is_finite(v.x) and is_finite(v.y) and is_finite(v.z)) else 1
		_say("RESULT: %s (%d frames posed, %d non-finite vertices, %d ms)" % [
				"PASS" if bad == 0 else "FAIL", _frame, bad, Time.get_ticks_msec() - _t0])
		_out.close()
		quit(0 if bad == 0 else 1)
		return true
	return false

# --- the wave -----------------------------------------------------------------

# (shoulder raise, forearm swing) in radians at time t: the keyed curve.
static func wave_key(t: float) -> Vector2:
	var up := clampf(t / 0.5, 0.0, 1.0) * clampf((SECONDS - t) / 0.5, 0.0, 1.0)
	up = up * up * (3.0 - 2.0 * up)
	var swing := 0.0
	if t > 0.5 and t < SECONDS - 0.5:
		swing = 0.45 * sin(TAU * 2.0 * (t - 0.5))
	return Vector2(1.25 * up, swing * up)

func _pose(t: float) -> void:
	var k := wave_key(t)
	var fwd := Vector3(0, 0, 1)
	var s := _joints[SHOULDER]
	var e := _joints[ELBOW]
	# Raise the arm about the forward axis at the shoulder; swing the forearm
	# about the same axis at the (moved) elbow. 3x3 matrices, rule 11.
	var r_up := Basis(fwd, k.x)
	var t_up := Transform3D(r_up, s - r_up * s)
	var e2 := t_up * e
	var r_sw := Basis(fwd, k.y)
	var t_fore := Transform3D(r_sw, e2 - r_sw * e2) * t_up
	var v := PackedVector3Array()
	v.resize(_rest.size())
	for i in _rest.size():
		var p := _rest[i]
		match _bone_of[i]:
			1: p = p.lerp(t_up * p, _w[i])
			2: p = (t_up * p).lerp(t_fore * p, _w[i])
		v[i] = p
	_body.mesh = _mesh(v, _tris)

# Nearest-bone weights for the arm: a vertex nearer the forearm segment than
# any other bone follows the forearm (blended toward the upper arm near the
# elbow), one nearer the upper-arm segment follows the upper arm (blended to
# static near the shoulder). A stand-in for a rig's weights.
func _weights() -> void:
	var bones := [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [1, 6], [6, 8], [7, 8], [0, 9], [9, 10], [10, 11],
			[0, 12], [12, 13], [13, 14]]
	_bone_of.resize(_rest.size())
	_w.resize(_rest.size())
	for i in _rest.size():
		var p := _rest[i]
		var best := -1
		var bd := INF
		for bi in bones.size():
			var d := _seg_dist(p, _joints[bones[bi][0]], _joints[bones[bi][1]])
			if d < bd:
				bd = d
				best = bi
		var a := _joints[SHOULDER]
		var b := _joints[ELBOW]
		var c := _joints[WRIST]
		if best == 4:   # forearm (elbow -> wrist)
			_bone_of[i] = 2
			_w[i] = clampf(_seg_t(p, b, c) * 4.0, 0.0, 1.0)
		elif best == 3:  # upper arm (shoulder -> elbow)
			_bone_of[i] = 1
			_w[i] = clampf(_seg_t(p, a, b) * 4.0, 0.0, 1.0)
		else:
			_bone_of[i] = 0
			_w[i] = 0.0

static func _seg_t(p: Vector3, a: Vector3, b: Vector3) -> float:
	var ab := b - a
	return clampf((p - a).dot(ab) / maxf(ab.length_squared(), 1e-12), 0.0, 1.0)

static func _seg_dist(p: Vector3, a: Vector3, b: Vector3) -> float:
	return p.distance_to(a.lerp(b, _seg_t(p, a, b)))

static func _vec3(f) -> PackedVector3Array:
	var v := PackedVector3Array()
	for i in range(0, f.size(), 3):
		v.append(Vector3(f[i], f[i + 1], f[i + 2]))
	return v

static func _mesh(v: PackedVector3Array, tris: PackedInt32Array) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for p in v:
		st.add_vertex(p)
	for i in tris:
		st.add_index(i)
	st.generate_normals()
	return st.commit()

static func _mat(c: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = c
	m.roughness = 0.8
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	return m
