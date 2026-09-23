#include "rd_compute.h"

#include <syscalls.h>

#include <algorithm>
#include <cstring>
#include <string_view>

// Assigning a non-scoped Variant over a Variant that holds a permanent slot
// releases the slot (godot-sandbox api_vstore_global). The guest API has no
// wrapper for it; this is the only release path for a permanent Variant.
MAKE_SYSCALL(ECALL_VSTORE_GLOBAL, void, sys_vstore_global, Variant *, const Variant *);

namespace rdc {

// --- method names at addresses that do not collide in the host's name cache ---
//
// godot-sandbox resolves an Object call's method name through a 32-entry
// direct-mapped cache keyed by the guest ADDRESS of the name
// (Sandbox::cached_guest_name: slot = ((address * 2654435761) >> 8) & 31). A
// miss rebuilds the entry, and that drops the MethodBind cached beside it, so
// the next call re-resolves it through ClassDB.class_get_method_list on
// RenderingDevice: 2-5 ms. Two string literals the linker happened to place in
// one slot evict each other on every call, so a recording loop over them runs
// at milliseconds per call. The layout decides it, not the code: HEAD's
// dress_on.elf had no hot pair in one slot, and ELFs built later did
// (gates/2-avbd/README.md, perf-bisect.log). So every RenderingDevice method
// name this class calls is copied once into a static pool, each at an offset
// that gives it a slot of its own. The host still reads the name from guest
// memory and compares the text, so the copies behave as the literals did.
namespace {

enum Name : int {
	N_LIST_BEGIN,
	N_BIND_PIPELINE,
	N_BIND_UNIFORM_SET,
	N_DISPATCH,
	N_BARRIER,
	N_LIST_END,
	N_SUBMIT,
	N_SYNC,
	N_PROCESS_FRAMES,
	N_BUFFER_UPDATE,
	N_BUFFER_COPY,
	N_BUFFER_CLEAR,
	N_BUFFER_GET_DATA,
	N_UNIFORM_SET_IS_VALID,
	N_FREE_RID,
	N_SHADER_FROM_SPIRV,
	N_SET_STAGE_BYTECODE,
	N_PIPELINE_CREATE,
	N_STORAGE_BUFFER_CREATE,
	N_UNIFORM_BUFFER_CREATE,
	N_SET_UNIFORM_TYPE,
	N_SET_BINDING,
	N_ADD_ID,
	N_UNIFORM_SET_CREATE,
	N_LIMIT_GET,
	N_CREATE_LOCAL_DEVICE,
	N_FREE,
	N_GET_MEMORY_USAGE,
	N_GET_DEVICE_NAME,
	N_COUNT
};

// Hot names (the recording loop) first, so they are placed first.
const char *const kNameText[N_COUNT] = {
	"compute_list_begin",
	"compute_list_bind_compute_pipeline",
	"compute_list_bind_uniform_set",
	"compute_list_dispatch",
	"compute_list_add_barrier",
	"compute_list_end",
	"submit",
	"sync",
	"get_process_frames",
	"buffer_update",
	"buffer_copy",
	"buffer_clear",
	"buffer_get_data",
	"uniform_set_is_valid",
	"free_rid",
	"shader_create_from_spirv",
	"set_stage_bytecode",
	"compute_pipeline_create",
	"storage_buffer_create",
	"uniform_buffer_create",
	"set_uniform_type",
	"set_binding",
	"add_id",
	"uniform_set_create",
	"limit_get",
	"create_local_rendering_device",
	"free",
	"get_memory_usage",
	"get_device_name",
};

constexpr unsigned kHostSlots = 32;

unsigned host_slot(uintptr_t address) {
	return unsigned(((uint64_t(address) * 2654435761u) >> 8) & (kHostSlots - 1));
}

struct NameTable {
	char pool[2048];
	const char *at[N_COUNT];
	size_t len[N_COUNT];
	unsigned slot[N_COUNT];
	int distinct = 0;

	NameTable() {
		static_assert(N_COUNT <= int(kHostSlots), "more names than host cache slots");
		bool used[kHostSlots] = {};
		size_t cursor = 0;
		for (int i = 0; i < N_COUNT; ++i) {
			const size_t n = std::strlen(kNameText[i]);
			len[i] = n;
			at[i] = kNameText[i]; // fallback: the literal, wherever it landed
			for (size_t off = cursor; off + n + 1 <= sizeof(pool); ++off) {
				const unsigned s = host_slot(reinterpret_cast<uintptr_t>(pool + off));
				if (!used[s]) {
					used[s] = true;
					std::memcpy(pool + off, kNameText[i], n + 1);
					at[i] = pool + off;
					cursor = off + n + 1;
					++distinct;
					break;
				}
			}
			slot[i] = host_slot(reinterpret_cast<uintptr_t>(at[i]));
		}
	}
};

const NameTable &names() {
	static NameTable t;
	return t;
}

std::string_view nm(Name n) {
	const NameTable &t = names();
	return std::string_view(t.at[n], t.len[n]);
}

} // namespace

std::string name_slots() {
	const NameTable &t = names();
	return std::to_string(t.distinct) + " of " + std::to_string(int(N_COUNT)) +
			" RenderingDevice method names in slots of their own (host name cache: " +
			std::to_string(kHostSlots) + " slots)";
}

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
	if (v.get_type() != Variant::RID) {
		fail("not an RID");
		return ::RID();
	}
	// A returned RID is a per-vmcall scoped Variant; kept, its index would
	// name some other Variant in a later call. Moving it to permanent storage
	// gives it an index (negative) valid until free_rid()/forget().
	Variant p = v;
	if (permanent_rids_) {
		p.make_permanent();
	}
	::RID r = as_rid(p);
	if (r.index == 0) {
		fail("null RID");
	} else if (r.index < 0) {
		++permanent_live_;
	}
	return r;
}

void Device::forget(::RID rid) {
	if (rid.index >= 0) {
		return; // null, or a per-call index that dies with the call anyway
	}
	Variant held(rid);
	Variant nil;
	sys_vstore_global(&held, &nil);
	--permanent_live_;
}

bool Device::uniform_set_valid(::RID rid) {
	if (rid.index == 0) {
		return false;
	}
	step_ = "uniform_set_is_valid";
	return bool(rd_.call(nm(N_UNIFORM_SET_IS_VALID), rid));
}

bool Device::open() {
	if (rd_.is_valid()) {
		return true;
	}
	err_.clear();
	step_ = "Object(\"RenderingServer\")";
	Object rs("RenderingServer");

	step_ = "create_local_rendering_device";
	Variant v = rs.call(nm(N_CREATE_LOCAL_DEVICE));
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
	list_open_ = false;
	submitted_ = false;
	err_.clear();
}

void Device::close() {
	if (staging_.index != 0) {
		if (rd_.is_valid()) {
			free_rid(staging_);
		} else {
			forget(staging_);
		}
		staging_ = ::RID();
		staging_bytes_ = 0;
	}
	if (rd_.is_valid() && owned_) {
		step_ = "free";
		rd_.voidcall(nm(N_FREE));
	}
	rd_ = Object(uint64_t(0));
	owned_ = false;
	list_open_ = false;
	submitted_ = false;
}

// --- resources -------------------------------------------------------------

::RID Device::shader_from_spirv(const PackedByteArray &spirv) {
	step_ = "ClassDB::instantiate(RDShaderSPIRV)";
	Object sp = ClassDB::instantiate("RDShaderSPIRV");
	step_ = "set_stage_bytecode";
	sp.voidcall(nm(N_SET_STAGE_BYTECODE), SHADER_STAGE_COMPUTE, spirv);
	step_ = "shader_create_from_spirv";
	return rid_or_fail(rd_.call(nm(N_SHADER_FROM_SPIRV), sp));
}

::RID Device::shader_from_spirv(const uint8_t *spirv, size_t bytes) {
	return shader_from_spirv(PackedByteArray(spirv, bytes));
}

::RID Device::compute_pipeline(::RID shader) {
	step_ = "compute_pipeline_create";
	return rid_or_fail(rd_.call(nm(N_PIPELINE_CREATE), shader));
}

static PackedByteArray bytes_or_zeros(size_t bytes, const void *data) {
	if (data) {
		return PackedByteArray(static_cast<const uint8_t *>(data), bytes);
	}
	std::vector<uint8_t> zeros(bytes, 0);
	return PackedByteArray(zeros.data(), zeros.size());
}

::RID Device::storage_buffer(size_t bytes, const void *data) {
	if (bytes > kMaxTransferBytes) {
		// Too big to cross in one PackedByteArray: create empty, then fill.
		::RID r = storage_buffer_empty(bytes, data == nullptr);
		if (r.index != 0 && data != nullptr && !buffer_update(r, 0, bytes, data)) {
			free_rid(r);
			return ::RID();
		}
		return r;
	}
	step_ = "storage_buffer_create";
	return rid_or_fail(rd_.call(nm(N_STORAGE_BUFFER_CREATE), int64_t(bytes), bytes_or_zeros(bytes, data)));
}

::RID Device::storage_buffer_uninit(size_t bytes) {
	step_ = "storage_buffer_create";
	return rid_or_fail(rd_.call(nm(N_STORAGE_BUFFER_CREATE), int64_t(bytes), PackedByteArray(std::vector<uint8_t>{})));
}

::RID Device::storage_buffer_empty(size_t bytes, bool clear) {
	bytes = (bytes + 3) & ~size_t(3);
	::RID r = storage_buffer_uninit(bytes);
	if (r.index != 0 && clear && !buffer_clear(r, 0, bytes)) {
		free_rid(r);
		return ::RID();
	}
	return r;
}

::RID Device::uniform_buffer(size_t bytes, const void *data) {
	step_ = "uniform_buffer_create";
	return rid_or_fail(rd_.call(nm(N_UNIFORM_BUFFER_CREATE), int64_t(bytes), bytes_or_zeros(bytes, data)));
}

bool Device::buffer_update(::RID buffer, size_t offset, size_t bytes, const void *data) {
	recover();
	step_ = "buffer_update";
	// One call per kMaxTransferBytes (a single call when bytes == 0, as before).
	const uint8_t *src = static_cast<const uint8_t *>(data);
	size_t done = 0;
	do {
		const size_t n = std::min(bytes - done, kMaxTransferBytes);
		PackedByteArray pba(src + done, n);
		Variant err = rd_.call(nm(N_BUFFER_UPDATE), buffer, int64_t(offset + done), int64_t(n), pba);
		if (int64_t(err) != GODOT_OK) {
			return fail("Godot Error != OK");
		}
		done += n;
	} while (done < bytes);
	return true;
}

std::vector<uint8_t> Device::buffer_get(::RID buffer, size_t offset, size_t bytes) {
	recover();
	step_ = "buffer_get_data";
	Variant v = rd_.call(nm(N_BUFFER_GET_DATA), buffer, int64_t(offset), int64_t(bytes));
	return v.as_byte_array().fetch();
}

bool Device::buffer_get_into(::RID buffer, size_t offset, size_t bytes, void *out) {
	if (submitted_) {
		step_ = "buffer_get_into";
		return fail("the device is submitted; sync() first");
	}
	if (bytes == 0) {
		return true;
	}
	// Grow the staging buffer to the read (rounded to 256), up to the cap.
	const size_t want = std::min(staging_max_, (bytes + 255) & ~size_t(255));
	if (staging_bytes_ < want) {
		if (staging_.index != 0) {
			free_rid(staging_);
			staging_ = ::RID();
			staging_bytes_ = 0;
		}
		staging_ = storage_buffer_empty(want, false);
		if (staging_.index == 0) {
			return false;
		}
		staging_bytes_ = want;
	}
	uint8_t *dst = static_cast<uint8_t *>(out);
	size_t done = 0;
	while (done < bytes) {
		const size_t n = std::min(bytes - done, staging_bytes_);
		// buffer_copy takes whole 4-byte words; copy the words that cover
		// [offset+done, offset+done+n) and take the bytes wanted out of them.
		const size_t lo = (offset + done) & ~size_t(3);
		const size_t skew = (offset + done) - lo;
		const size_t span = std::min(staging_bytes_, (skew + n + 3) & ~size_t(3));
		const size_t take = std::min(n, span - skew);
		if (!buffer_copy(buffer, staging_, span, lo, 0)) {
			return false;
		}
		step_ = "buffer_get_data (staging)";
		Variant v = rd_.call(nm(N_BUFFER_GET_DATA), staging_, int64_t(0), int64_t(span));
		std::vector<uint8_t> got = v.as_byte_array().fetch();
		if (got.size() < span) {
			return fail("short staging read");
		}
		std::memcpy(dst + done, got.data() + skew, take);
		done += take;
	}
	return true;
}

bool Device::buffer_copy(::RID src, ::RID dst, size_t bytes, size_t src_offset, size_t dst_offset) {
	recover();
	// RenderingDevice.buffer_copy(src_buffer, dst_buffer, src_offset, dst_offset, size)
	step_ = "buffer_copy";
	Variant err = rd_.call(nm(N_BUFFER_COPY), src, dst, int64_t(src_offset), int64_t(dst_offset), int64_t(bytes));
	if (int64_t(err) != GODOT_OK) {
		return fail("Godot Error != OK");
	}
	return true;
}

bool Device::buffer_clear(::RID buffer, size_t offset, size_t bytes) {
	recover();
	// RenderingDevice.buffer_clear(buffer, offset, size_bytes)
	step_ = "buffer_clear";
	Variant err = rd_.call(nm(N_BUFFER_CLEAR), buffer, int64_t(offset), int64_t(bytes));
	if (int64_t(err) != GODOT_OK) {
		return fail("Godot Error != OK");
	}
	return true;
}

::RID Device::uniform_set(::RID shader, const std::vector<Binding> &bindings, uint32_t set) {
	step_ = "RDUniform";
	Array uniforms = Array::Create(0);
	for (const Binding &b : bindings) {
		Object un = ClassDB::instantiate("RDUniform");
		un.voidcall(nm(N_SET_UNIFORM_TYPE), b.type);
		un.voidcall(nm(N_SET_BINDING), int64_t(b.binding));
		un.voidcall(nm(N_ADD_ID), b.rid);
		uniforms.push_back(un);
	}
	step_ = "uniform_set_create";
	return rid_or_fail(rd_.call(nm(N_UNIFORM_SET_CREATE), uniforms, shader, int64_t(set)));
}

void Device::free_rid(::RID rid) {
	if (rid.index == 0) {
		return;
	}
	step_ = "free_rid";
	rd_.voidcall(nm(N_FREE_RID), rid);
	forget(rid);
}

int64_t Device::limit_get(int limit) {
	step_ = "limit_get";
	return int64_t(rd_.call(nm(N_LIMIT_GET), limit));
}

int64_t Device::memory_usage() {
	step_ = "get_memory_usage";
	return int64_t(rd_.call(nm(N_GET_MEMORY_USAGE), MEMORY_TOTAL));
}

std::string Device::device_name() {
	step_ = "get_device_name";
	return rd_.call(nm(N_GET_DEVICE_NAME)).as_std_string();
}

// --- recording -------------------------------------------------------------

bool Device::recover() {
	if (!list_open_ || !recovery_) {
		return false;
	}
	list_end();
	++recoveries_;
	return true;
}

bool Device::list_begin() {
	recover();
	step_ = "compute_list_begin";
	// The flag goes up before the call: if this vmcall dies anywhere after
	// here, the next one ends the list.
	list_open_ = true;
	list_ = rd_.call(nm(N_LIST_BEGIN));
	return true;
}

void Device::bind_pipeline(::RID pipeline) {
	step_ = "compute_list_bind_compute_pipeline";
	rd_.voidcall(nm(N_BIND_PIPELINE), list_, pipeline);
}

void Device::bind_uniform_set(::RID uniform_set, uint32_t set) {
	step_ = "compute_list_bind_uniform_set";
	rd_.voidcall(nm(N_BIND_UNIFORM_SET), list_, uniform_set, int64_t(set));
}

void Device::dispatch(uint32_t gx, uint32_t gy, uint32_t gz) {
	step_ = "compute_list_dispatch";
	rd_.voidcall(nm(N_DISPATCH), list_, int64_t(gx), int64_t(gy), int64_t(gz));
}

void Device::barrier() {
	step_ = "compute_list_add_barrier";
	rd_.voidcall(nm(N_BARRIER), list_);
}

void Device::list_end() {
	step_ = "compute_list_end";
	rd_.voidcall(nm(N_LIST_END));
	list_open_ = false;
}

int64_t Device::process_frame() {
	if (!engine_.is_valid()) {
		engine_ = Object("Engine");
	}
	return int64_t(engine_.call(nm(N_PROCESS_FRAMES)));
}

void Device::submit() {
	recover();
	step_ = "submit";
	rd_.voidcall(nm(N_SUBMIT));
	submitted_ = true;
	submit_frame_ = process_frame();
	++submits_;
}

void Device::sync() {
	step_ = "sync";
	if (process_frame() == submit_frame_) {
		++same_frame_syncs_;
	}
	++syncs_;
	rd_.voidcall(nm(N_SYNC));
	submitted_ = false;
}

} // namespace rdc
