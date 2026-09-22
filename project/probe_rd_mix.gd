# Interleave the GDScript control and the guest bench in ONE process, all
# host-timed, to separate "the GPU/driver state at that moment" from "the
# sandbox path". Same (16,1,barrier) shape everywhere.
extends SceneTree

const STORAGE := RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER

func _bytes(path: String) -> PackedByteArray:
	var f := FileAccess.open(path, FileAccess.READ)
	var b := f.get_buffer(f.get_length())
	f.close()
	return b

func _gd(rd: RenderingDevice, spirv: PackedByteArray, nd: int, ns: int, barrier: bool) -> int:
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
	for s in ns:
		var cl := rd.compute_list_begin()
		for i in nd:
			rd.compute_list_bind_compute_pipeline(cl, pipe)
			rd.compute_list_bind_uniform_set(cl, uset, 0)
			rd.compute_list_dispatch(cl, 1, 1, 1)
			if barrier and i + 1 < nd:
				rd.compute_list_add_barrier(cl)
		rd.compute_list_end()
		rd.submit()
		rd.sync()
	var out := rd.buffer_get_data(buf)
	rd.free_rid(uset)
	rd.free_rid(pipe)
	rd.free_rid(shader)
	rd.free_rid(buf)
	assert(out.decode_u32(0) == nd * ns)
	return Time.get_ticks_usec() - t0

func _init() -> void:
	var acc := _bytes("res://accumulate.spv")
	var rd := RenderingServer.create_local_rendering_device()
	var sb = ClassDB.instantiate("Sandbox")
	sb.program = load("res://dress_on.elf")
	sb.references_max = 4096
	sb.vmcall("rd_open")
	for shape in [[16, 1, true], [1, 16, true], [64, 1, true]]:
		var nd: int = shape[0]
		var ns: int = shape[1]
		var barrier: bool = shape[2]
		for phase in ["gdscript", "guest", "guest-quiet", "gdscript", "guest-quiet", "guest"]:
			var line := "shape nd=%3d ns=%3d %-12s" % [nd, ns, phase]
			for k in 3:
				var dt := 0
				if phase == "gdscript":
					dt = _gd(rd, acc, nd, ns, barrier)
				else:
					var t0 := Time.get_ticks_usec()
					var r = sb.vmcall("rd_bench_quiet" if phase == "guest-quiet" else "rd_bench", acc, nd, ns, barrier)
					dt = Time.get_ticks_usec() - t0
					assert(str(r).begins_with("OK"))
				line += " %8d" % dt
			print(line, "   (host_us x3)")
	sb.vmcall("rd_close")
	sb.free()
	rd.free()
	quit(0)
