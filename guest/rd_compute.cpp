#include "rd_compute.h"

namespace rdc {

// RID::RID(const Variant&) is declared in the API headers but never defined
// in the guest library, so it links only while unused. The conversion
// operator is defined. Every Variant->RID goes through here.
static ::RID as_rid(const Variant &v) {
	return v.operator ::RID();
}

bool Device::fail(const char *why) {
	err_ = std::string("FAIL at ") + step_ + ": " + why;
	return false;
}

::RID Device::rid_or_fail(const Variant &v) {
	::RID r = as_rid(v);
	if (r.index == 0) {
		fail("null RID");
	}
	return r;
}

bool Device::open() {
	if (rd_.is_valid()) {
		return true;
	}
	err_.clear();
	step_ = "Object(\"RenderingServer\")";
	Object rs("RenderingServer");

	step_ = "create_local_rendering_device";
	Variant v = rs.call("create_local_rendering_device");
	if (v.get_type() != Variant::OBJECT) {
		return fail("not an Object (headless? run with --rendering-driver vulkan)");
	}
	rd_ = v.as_object();
	owned_ = true;
	if (!rd_.is_valid()) {
		return fail("null device");
	}
	return true;
}

void Device::adopt(const Object &rd) {
	rd_ = rd;
	owned_ = false;
	err_.clear();
}

void Device::close() {
	if (rd_.is_valid() && owned_) {
		step_ = "free";
		rd_.voidcall("free");
	}
	rd_ = Object(uint64_t(0));
	owned_ = false;
}

// --- resources -------------------------------------------------------------

::RID Device::shader_from_spirv(const PackedByteArray &spirv) {
	step_ = "ClassDB::instantiate(RDShaderSPIRV)";
	Object sp = ClassDB::instantiate("RDShaderSPIRV");
	step_ = "set_stage_bytecode";
	sp.voidcall("set_stage_bytecode", SHADER_STAGE_COMPUTE, spirv);
	step_ = "shader_create_from_spirv";
	return rid_or_fail(rd_.call("shader_create_from_spirv", sp));
}

::RID Device::shader_from_spirv(const uint8_t *spirv, size_t bytes) {
	return shader_from_spirv(PackedByteArray(spirv, bytes));
}

::RID Device::compute_pipeline(::RID shader) {
	step_ = "compute_pipeline_create";
	return rid_or_fail(rd_.call("compute_pipeline_create", shader));
}

static PackedByteArray bytes_or_zeros(size_t bytes, const void *data) {
	if (data) {
		return PackedByteArray(static_cast<const uint8_t *>(data), bytes);
	}
	std::vector<uint8_t> zeros(bytes, 0);
	return PackedByteArray(zeros.data(), zeros.size());
}

::RID Device::storage_buffer(size_t bytes, const void *data) {
	step_ = "storage_buffer_create";
	return rid_or_fail(rd_.call("storage_buffer_create", int64_t(bytes), bytes_or_zeros(bytes, data)));
}

::RID Device::uniform_buffer(size_t bytes, const void *data) {
	step_ = "uniform_buffer_create";
	return rid_or_fail(rd_.call("uniform_buffer_create", int64_t(bytes), bytes_or_zeros(bytes, data)));
}

bool Device::buffer_update(::RID buffer, size_t offset, size_t bytes, const void *data) {
	step_ = "buffer_update";
	PackedByteArray pba(static_cast<const uint8_t *>(data), bytes);
	Variant err = rd_.call("buffer_update", buffer, int64_t(offset), int64_t(bytes), pba);
	if (int64_t(err) != GODOT_OK) {
		return fail("Godot Error != OK");
	}
	return true;
}

std::vector<uint8_t> Device::buffer_get(::RID buffer, size_t offset, size_t bytes) {
	step_ = "buffer_get_data";
	Variant v = rd_.call("buffer_get_data", buffer, int64_t(offset), int64_t(bytes));
	return v.as_byte_array().fetch();
}

::RID Device::uniform_set(::RID shader, const std::vector<Binding> &bindings, uint32_t set) {
	step_ = "RDUniform";
	Array uniforms = Array::Create(0);
	for (const Binding &b : bindings) {
		Object un = ClassDB::instantiate("RDUniform");
		un.voidcall("set_uniform_type", b.type);
		un.voidcall("set_binding", int64_t(b.binding));
		un.voidcall("add_id", b.rid);
		uniforms.push_back(un);
	}
	step_ = "uniform_set_create";
	return rid_or_fail(rd_.call("uniform_set_create", uniforms, shader, int64_t(set)));
}

void Device::free_rid(::RID rid) {
	if (rid.index == 0) {
		return;
	}
	step_ = "free_rid";
	rd_.voidcall("free_rid", rid);
}

int64_t Device::limit_get(int limit) {
	step_ = "limit_get";
	return int64_t(rd_.call("limit_get", limit));
}

// --- recording -------------------------------------------------------------

bool Device::list_begin() {
	step_ = "compute_list_begin";
	list_ = rd_.call("compute_list_begin");
	return true;
}

void Device::bind_pipeline(::RID pipeline) {
	step_ = "compute_list_bind_compute_pipeline";
	rd_.voidcall("compute_list_bind_compute_pipeline", list_, pipeline);
}

void Device::bind_uniform_set(::RID uniform_set, uint32_t set) {
	step_ = "compute_list_bind_uniform_set";
	rd_.voidcall("compute_list_bind_uniform_set", list_, uniform_set, int64_t(set));
}

void Device::dispatch(uint32_t gx, uint32_t gy, uint32_t gz) {
	step_ = "compute_list_dispatch";
	rd_.voidcall("compute_list_dispatch", list_, int64_t(gx), int64_t(gy), int64_t(gz));
}

void Device::barrier() {
	step_ = "compute_list_add_barrier";
	rd_.voidcall("compute_list_add_barrier", list_);
}

void Device::list_end() {
	step_ = "compute_list_end";
	rd_.voidcall("compute_list_end");
}

void Device::submit() {
	step_ = "submit";
	rd_.voidcall("submit");
}

void Device::sync() {
	step_ = "sync";
	rd_.voidcall("sync");
}

} // namespace rdc
