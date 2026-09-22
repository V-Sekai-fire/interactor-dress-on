# Stage 1 control: the same bench in plain GDScript on a local
# RenderingDevice, no sandbox. Separates "the boundary costs this" from
# "Godot costs this". Run:
#   godot --path project --script control_rd_compute.gd --rendering-driver vulkan --xr-mode off
extends SceneTree

const STORAGE := RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER

func _bytes(path: String) -> PackedByteArray:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return PackedByteArray()
	var b := f.get_buffer(f.get_length())
	f.close()
	return b

func _bench(rd: RenderingDevice, spirv: PackedByteArray, n_dispatch: int, n_submit: int, barrier: bool) -> String:
	var t0 := Time.get_ticks_usec()
	var sp := RDShaderSPIRV.new()
	sp.set_stage_bytecode(RenderingDevice.SHADER_STAGE_COMPUTE, spirv)
	var shader := rd.shader_create_from_spirv(sp)
	var buf := rd.storage_buffer_create(4, PackedByteArray([0, 0, 0, 0]))
	var un := RDUniform.new()
	un.uniform_type = STORAGE
	un.binding = 0
	un.add_id(buf)
	var uset := rd.uniform_set_create([un], shader, 0)
	var pipe := rd.compute_pipeline_create(shader)
	var t1 := Time.get_ticks_usec()
	for s in n_submit:
		var cl := rd.compute_list_begin()
		for i in n_dispatch:
			rd.compute_list_bind_compute_pipeline(cl, pipe)
			rd.compute_list_bind_uniform_set(cl, uset, 0)
			rd.compute_list_dispatch(cl, 1, 1, 1)
			if barrier and i + 1 < n_dispatch:
				rd.compute_list_add_barrier(cl)
		rd.compute_list_end()
		rd.submit()
		rd.sync()
	var t2 := Time.get_ticks_usec()
	var out := rd.buffer_get_data(buf)
	var t3 := Time.get_ticks_usec()
	rd.free_rid(uset)
	rd.free_rid(pipe)
	rd.free_rid(shader)
	rd.free_rid(buf)
	var got := out.decode_u32(0)
	var expected := n_dispatch * n_submit
	var loop := t2 - t1
	return "%s value=%d expected=%d setup_us=%d loop_us=%d read_us=%d us_per_dispatch=%.1f us_per_submit=%.1f" % [
		"OK" if got == expected else "WRONG", got, expected, t1 - t0, loop, t3 - t2,
		float(loop) / float(n_dispatch * n_submit), float(loop) / float(n_submit)]

func _init() -> void:
	var rd := RenderingServer.create_local_rendering_device()
	if rd == null:
		print("FAIL: no local RenderingDevice (headless?)")
		quit(1)
		return
	var acc := _bytes("res://accumulate.spv")
	for cfg in [[1, 1], [16, 1], [64, 1], [256, 1], [1, 16], [1, 64], [16, 16]]:
		for barrier in [true, false]:
			var t0 := Time.get_ticks_usec()
			var r := _bench(rd, acc, cfg[0], cfg[1], barrier)
			var dt := Time.get_ticks_usec() - t0
			print("control nd=%4d ns=%3d barrier=%-5s host_us=%7d  %s" % [cfg[0], cfg[1], barrier, dt, r])
	rd.free()
	quit(0)
