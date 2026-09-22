// rd_compute -- the one GPU layer in the guest.
//
// Everything that runs on the GPU here (AVBD's kernels, ggml's) goes through
// this class, and this class goes through exactly one thing: Godot's
// RenderingDevice, called over the sandbox boundary. It is Gate 0A's working
// call sequence factored into a class, with the enums pinned in rd_enums.h
// and every Variant->handle conversion wrapped once.
//
// Two facts about the boundary shape the interface:
//
//  - Every method call crosses guest -> host. A submit+fence on the Vulkan
//    backend already costs ~245 us regardless of work; here each *call* costs
//    too. So the recording API is a compute list you fill with as many
//    dispatches as you like and submit once. Batch. rd_bench measures it.
//
//  - The device can be held between vmcalls. godot-sandbox hands the guest an
//    engine instance id for a plain Object (a local RenderingDevice is one),
//    resolved live on each use. RefCounted helpers (RDUniform, RDShaderSPIRV)
//    are only held for the duration of a call, so nothing here retains them:
//    what persists is RIDs, which are integers.
//
// Errors: methods return a null RID (index 0) or false, and error() names the
// step that failed, because "which call returned null" is what makes a GPU
// failure actionable from outside the sandbox.
#pragma once

#include <api.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rd_enums.h"

namespace rdc {

// One entry of a uniform set: a buffer RID at a binding slot, of a type from
// rd_enums.h.
struct Binding {
	uint32_t binding;
	int type;
	::RID rid;
};

class Device {
public:
	// Create a local RenderingDevice through RenderingServer. Requires a real
	// renderer (--rendering-driver vulkan); under --headless the server hands
	// back null and this fails at "create_local_rendering_device". Idempotent.
	bool open();
	// Use a device the host owns instead. It is not freed by close().
	void adopt(const Object &rd);
	bool ok() const { return rd_.is_valid(); }
	// Free the device if open() created it.
	void close();

	// --- resources (each is one host call; RIDs are plain integers) ---
	::RID shader_from_spirv(const PackedByteArray &spirv);
	::RID shader_from_spirv(const uint8_t *spirv, size_t bytes);
	::RID compute_pipeline(::RID shader);
	// Storage/uniform buffers are always created with contents: zeros when
	// `data` is null. Godot leaves an uninitialised buffer's contents undefined
	// and a readback of "whatever was there" is not a result.
	::RID storage_buffer(size_t bytes, const void *data = nullptr);
	::RID uniform_buffer(size_t bytes, const void *data = nullptr);
	// Refused by Godot while a compute list is being recorded; call it before
	// list_begin(). (That is why params blocks live in per-site buffers.)
	bool buffer_update(::RID buffer, size_t offset, size_t bytes, const void *data);
	// bytes == 0 reads to the end.
	std::vector<uint8_t> buffer_get(::RID buffer, size_t offset = 0, size_t bytes = 0);
	::RID uniform_set(::RID shader, const std::vector<Binding> &bindings, uint32_t set = 0);
	void free_rid(::RID rid);
	// RenderingDevice.limit_get: the cheapest RD call there is, for timing the
	// boundary.
	int64_t limit_get(int limit);

	// --- recording: one compute list, then one submit + sync ---
	bool list_begin();
	void bind_pipeline(::RID pipeline);
	void bind_uniform_set(::RID uniform_set, uint32_t set = 0);
	void dispatch(uint32_t groups_x, uint32_t groups_y = 1, uint32_t groups_z = 1);
	// Between dependent dispatches. rd_bench also runs without it, to learn
	// whether Godot's render graph orders same-buffer dispatches on its own.
	void barrier();
	void list_end();
	void submit();
	void sync();

	// Workgroups for `threads` invocations at `threadgroup` per group. Every
	// AVBD kernel is numthreads(64,1,1).
	static uint32_t groups_for(uint32_t threads, uint32_t threadgroup = 64) {
		return (threads + threadgroup - 1) / threadgroup;
	}

	// The last step attempted, and the error text if one failed.
	const std::string &step() const { return step_; }
	const std::string &error() const { return err_; }

private:
	bool fail(const char *why);
	::RID rid_or_fail(const Variant &v);

	Object rd_{ uint64_t(0) };
	bool owned_ = false;
	Variant list_;
	std::string step_;
	std::string err_;
};

} // namespace rdc
