// GATE 0A — can a godot-sandbox guest drive RenderingDevice compute?
//
// Everything in interactor-dress-on rests on this. The guest is required to be
// independent of engine modules, so the GPU can only be reached through Godot's
// RenderingDevice, called out over the sandbox boundary. ggml's inference
// backend and AVBD's 24 SPIR-V kernels both sit on top of that one path. If it
// does not work, no amount of porting helps and the plan is void.
//
// The probe reports the step it REACHED, not merely pass/fail. A failure that
// says "created the device but shader_create_from_spirv returned a null RID" is
// actionable; a bare false is not.
//
// It ends by reading back 0x00C0FFEE. That constant is deliberate: zero or a
// small integer could be uninitialised memory or a zeroed buffer, and would not
// distinguish "the kernel ran" from "nothing happened but the readback
// succeeded".

#include <api.hpp>

#include <string>

static std::string g_step = "nothing attempted";

// RID::RID(const Variant&) is declared in the API headers but never defined in
// libsandbox_api, so it links only if unused. Go through the conversion
// operator instead, which is.
static ::RID as_rid(const Variant &v) {
	return v.operator ::RID();
}

static Variant fail(const char *why) {
	return Variant(String(std::string("FAIL at ") + g_step + ": " + why));
}

// Godot's RenderingDevice.ShaderStage. COMPUTE is 4 in Godot 4.x
// (VERTEX=0, FRAGMENT=1, TESSELATION_CONTROL=2, TESSELATION_EVALUATION=3).
static constexpr int SHADER_STAGE_COMPUTE = 4;
// RenderingDevice.UniformType.UNIFORM_TYPE_STORAGE_BUFFER. Not 6 -- that is
// IMAGE_BUFFER, and the validator says so in as many words.
static constexpr int UNIFORM_TYPE_STORAGE_BUFFER = 8;

static Variant rd_probe(PackedByteArray spirv) {
	g_step = "Object(\"RenderingServer\")";
	Object rs("RenderingServer");

	g_step = "create_local_rendering_device";
	Variant rdv = rs.call("create_local_rendering_device");
	if (rdv.get_type() != Variant::OBJECT) {
		return fail("not an Object");
	}
	Object rd = rdv;

	g_step = "ClassDB::instantiate(RDShaderSPIRV)";
	Object sp = ClassDB::instantiate("RDShaderSPIRV");

	g_step = "set_stage_bytecode";
	sp.call("set_stage_bytecode", SHADER_STAGE_COMPUTE, spirv);

	g_step = "shader_create_from_spirv";
	Variant shv = rd.call("shader_create_from_spirv", sp);
	RID shader = as_rid(shv);
	if (shader.index == 0) {
		return fail("null RID");
	}

	g_step = "storage_buffer_create";
	PackedByteArray zeros;
	{
		uint8_t z[4] = { 0, 0, 0, 0 };
		zeros = PackedByteArray(z, 4);
	}
	Variant bufv = rd.call("storage_buffer_create", 4, zeros);
	RID buf = as_rid(bufv);
	if (buf.index == 0) {
		return fail("null RID");
	}

	g_step = "RDUniform + uniform_set_create";
	Object un = ClassDB::instantiate("RDUniform");
	un.call("set_uniform_type", UNIFORM_TYPE_STORAGE_BUFFER);
	un.call("set_binding", 0);
	un.call("add_id", buf);
	Array uniforms = Array::Create(0);
	uniforms.push_back(un);
	Variant usv = rd.call("uniform_set_create", uniforms, shader, 0);
	RID uset = as_rid(usv);
	if (uset.index == 0) {
		return fail("null RID");
	}

	g_step = "compute_pipeline_create";
	Variant pipev = rd.call("compute_pipeline_create", shader);
	RID pipe = as_rid(pipev);
	if (pipe.index == 0) {
		return fail("null RID");
	}

	g_step = "compute_list dispatch";
	Variant clv = rd.call("compute_list_begin");
	rd.call("compute_list_bind_compute_pipeline", clv, pipe);
	rd.call("compute_list_bind_uniform_set", clv, uset, 0);
	rd.call("compute_list_dispatch", clv, 1, 1, 1);
	rd.call("compute_list_end");

	g_step = "submit + sync";
	rd.call("submit");
	rd.call("sync");

	g_step = "buffer_get_data";
	Variant outv = rd.call("buffer_get_data", buf);
	PackedByteArray out = outv.as_byte_array();
	if (out.size() < 4) {
		return fail("short read");
	}
	auto bytes = out.fetch();
	const uint32_t got = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
			(uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);

	if (got == 0x00C0FFEEu) {
		return Variant(String(std::string("PASS: GPU compute reached from the guest")));
	}
	return Variant(String(std::string("FAIL: dispatched but read back a wrong value")));
}

// Where did it get to? Called after rd_probe when that returns a FAIL, so the
// report names a step rather than a boolean.
static Variant rd_last_step() {
	return Variant(String(g_step));
}

int main() {
	ADD_API_FUNCTION(rd_probe, "String", "PackedByteArray spirv",
			"Drive a trivial SPIR-V compute dispatch through RenderingDevice");
	ADD_API_FUNCTION(rd_last_step, "String", "",
			"The last RenderingDevice step attempted");
	halt();
}
