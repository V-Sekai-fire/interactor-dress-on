extends SceneTree

func v3(v: Vector3) -> String:
	return "%.6f %.6f %.6f" % [v.x, v.y, v.z]

func dump(name: String, c: Curve3D) -> void:
	var L := c.get_baked_length()
	print("%s length %.6f" % [name, L])
	print("%s baked_points %d" % [name, c.get_baked_points().size()])
	for f in [0.1, 0.25, 0.5, 0.75, 0.9]:
		print("%s sample_baked %.2f lin %s" % [name, f, v3(c.sample_baked(L * f, false))])
		print("%s sample_baked %.2f cub %s" % [name, f, v3(c.sample_baked(L * f, true))])
		print("%s up %.2f %s" % [name, f, v3(c.sample_baked_up_vector(L * f))])
	print("%s closest_offset %.6f" % [name, c.get_closest_offset(Vector3(30, 20, 10))])
	print("%s closest_point %s" % [name, v3(c.get_closest_point(Vector3(30, 20, 10)))])
	print("%s tessellate %d" % [name, c.tessellate().size()])
	print("%s tessellate_even %d" % [name, c.tessellate_even_length().size()])

func _init() -> void:
	var a := Curve3D.new()
	a.add_point(Vector3(0, 0, 0), Vector3(), Vector3(50, 0, 0))
	a.add_point(Vector3(0, 50, 0), Vector3(-50, 0, -50), Vector3())
	dump("bezier", a)
	var b := Curve3D.new()
	b.bake_interval = 0.05
	b.add_point(Vector3(0, 0, 0), Vector3(), Vector3(0.5, 0, 0))
	b.add_point(Vector3(1, 1, 0), Vector3(0, -0.5, 0), Vector3(0, 0.5, 0.2))
	b.add_point(Vector3(0, 2, 0.5), Vector3(0.4, 0, 0), Vector3())
	b.closed = true
	dump("closed", b)
	quit()
