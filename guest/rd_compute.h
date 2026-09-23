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
//    are only held for the duration of a call, so nothing here retains them.
//    An RID is NOT an integer to the guest: it is a scoped Variant whose index
//    is valid for one vmcall only (godot-sandbox GuestVariant::
//    is_scoped_variant lists RID), so a kept RID names some other Variant in
//    the next call. Every RID this class returns is therefore moved to
//    permanent storage (Variant::make_permanent: a negative index, valid
//    across calls) and its slot is released by free_rid() or forget().
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

// The most bytes one PackedByteArray may carry across the guest boundary,
// either way (made from guest memory, or fetched into it): 16 MiB works and
// 17 MiB is a protection fault in the host's view of guest memory (Gate 3).
// buffer_update, storage_buffer and buffer_get_into split at this size.
constexpr size_t kMaxTransferBytes = size_t(16) << 20;

// One entry of a uniform set: a buffer RID at a binding slot, of a type from
// rd_enums.h.
struct Binding {
	uint32_t binding;
	int type;
	::RID rid;
};

// How many of this class's RenderingDevice method names got a host name-cache
// slot of their own (rd_compute.cpp explains the cache). Evidence for gates.
std::string name_slots();

class Device {
public:
	// Create a local RenderingDevice through RenderingServer. Requires a real
	// renderer (--rendering-driver vulkan); under --headless the server hands
	// back null and this fails at "create_local_rendering_device". Idempotent.
	bool open();
	// Use a device the host owns instead. It is not freed by close().
	void adopt(const Object &rd);
	bool ok() const { return rd_.is_valid(); }
	// The device as an Object, to hand to the host (a plain Object's id is
	// valid across vmcalls; Gate 0F times host buffer_update on it).
	const Object &object() const { return rd_; }
	// Free the device if open() created it. The RIDs made on it are the
	// owners' to free first (free_rid releases their permanent slots too);
	// permanent_slots() says how many are still held.
	void close();

	// --- resources (each is one host call; each returned RID holds a
	// permanent slot until free_rid/forget) ---
	::RID shader_from_spirv(const PackedByteArray &spirv);
	::RID shader_from_spirv(const uint8_t *spirv, size_t bytes);
	::RID compute_pipeline(::RID shader);
	// Storage/uniform buffers are always created with contents: zeros when
	// `data` is null. Godot leaves an uninitialised buffer's contents undefined
	// and a readback of "whatever was there" is not a result.
	::RID storage_buffer(size_t bytes, const void *data = nullptr);
	// The one exception: a storage buffer created WITHOUT contents, for sizes
	// whose zero-filled PackedByteArray cannot exist in the guest heap (a
	// 4 GiB buffer). Its contents are undefined until buffer_clear() or a
	// write covers them (Gate 0F probe 12).
	::RID storage_buffer_uninit(size_t bytes);
	// storage_buffer_uninit, then zeroed on the GPU by buffer_clear when
	// `clear` is set (the size rounded up to 4 bytes, buffer_clear's unit):
	// how ggml-rd makes every buffer, up to 4 GiB - 256, with no zero array
	// in the guest heap.
	::RID storage_buffer_empty(size_t bytes, bool clear = true);
	::RID uniform_buffer(size_t bytes, const void *data = nullptr);
	// Refused by Godot while a compute list is being recorded; call it before
	// list_begin(). (That is why params blocks live in per-site buffers.)
	// Larger than kMaxTransferBytes goes up in several calls.
	bool buffer_update(::RID buffer, size_t offset, size_t bytes, const void *data);
	// bytes == 0 reads to the end. RenderingDevice.buffer_get_data stages the
	// WHOLE buffer on the host whatever `bytes` is (Gate 0F finding 3), and the
	// result must fit kMaxTransferBytes; for anything but a small buffer use
	// buffer_get_into.
	std::vector<uint8_t> buffer_get(::RID buffer, size_t offset = 0, size_t bytes = 0);
	// Read `bytes` at `offset` into `out` through a staging buffer: a
	// buffer_copy into it, then buffer_get_data of the staging buffer only.
	// The staging buffer grows to at most staging_max() bytes and is reused;
	// larger reads go in chunks, each a flush-and-stall of its own copy. Only
	// while nothing is submitted (after sync): a local device drops whatever
	// is recorded between submit() and sync().
	bool buffer_get_into(::RID buffer, size_t offset, size_t bytes, void *out);
	size_t staging_max() const { return staging_max_; }
	void set_staging_max(size_t bytes) {
		staging_max_ = bytes < 256 ? 256 : (bytes > kMaxTransferBytes ? kMaxTransferBytes : bytes);
	}
	// GPU-side copy, recorded into the graph; never inside a compute list.
	bool buffer_copy(::RID src, ::RID dst, size_t bytes, size_t src_offset = 0, size_t dst_offset = 0);
	// GPU-side zero fill (bytes a multiple of 4), recorded into the graph;
	// never inside a compute list.
	bool buffer_clear(::RID buffer, size_t offset, size_t bytes);
	::RID uniform_set(::RID shader, const std::vector<Binding> &bindings, uint32_t set = 0);
	// Frees the GPU object and releases the RID's permanent slot.
	void free_rid(::RID rid);
	// Releases the permanent slot only: for an RID Godot already freed (a
	// uniform set goes with its buffers). Permanent slots are finite (65534).
	void forget(::RID rid);
	bool uniform_set_valid(::RID rid);
	// RenderingDevice.limit_get: the cheapest RD call there is, for timing the
	// boundary.
	int64_t limit_get(int limit);
	// RenderingDevice.get_memory_usage(MEMORY_TOTAL): bytes the device holds.
	int64_t memory_usage();
	// RenderingDevice.get_device_name(): the adapter, for descriptions.
	std::string device_name();

	// --- recording: one compute list, then one submit + sync ---
	// A vmcall killed mid-recording (execution_timeout, references_max, a
	// trap) leaves its compute list open, and Godot then refuses every later
	// list_begin, buffer_update, buffer_clear and buffer_copy on the device
	// (Gate 0F finding 2). This class knows a list is open from its own flag,
	// which lives in guest memory and so survives the killed call: those
	// calls, and submit(), first end the orphaned list (recover()). The
	// recovered list's dispatches are ended, not undone.
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

	// The rule-4 guard (AGENTS.md rule 4): submit() records the engine's
	// process frame, sync() counts itself as same-frame when that frame has
	// not advanced. A frame-driven host keeps same_frame_syncs() at 0.
	int64_t same_frame_syncs() const { return same_frame_syncs_; }
	// Engine.get_process_frames(), through this class's method-name table.
	int64_t process_frame();
	int64_t syncs() const { return syncs_; }
	// Permanent Variant slots this device's RIDs hold now (made by every
	// returned RID, released by free_rid/forget). A leak shows up here.
	int64_t permanent_slots() const { return permanent_live_; }
	// Gate hook, never for production: false returns RIDs as the host scoped
	// them, valid for the current vmcall only, which is what Gate 0F probe 11b
	// shows failing across calls. Default true.
	void set_permanent_rids(bool on) { permanent_rids_ = on; }
	int64_t submits() const { return submits_; }
	// submit() has run and sync() has not: the device is processing, and
	// nothing may be recorded until sync() (a local RenderingDevice's
	// _begin_frame clears whatever the graph holds).
	bool submitted() const { return submitted_; }
	// A compute list is open (between list_begin and list_end).
	bool list_open() const { return list_open_; }
	// End a compute list left open by a killed vmcall; true if there was one.
	bool recover();
	int64_t recoveries() const { return recoveries_; }
	// Gate hook, never for production: false turns recover() off, so a list
	// a killed vmcall left open stays open and Godot refuses the next
	// buffer_update, as Gate 0F found (probe 13's control arm). Default true.
	void set_recovery(bool on) { recovery_ = on; }

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
	// The Engine singleton, resolved once and held like rd_ (a plain Object).
	Object engine_{ uint64_t(0) };
	int64_t submit_frame_ = -1;
	int64_t same_frame_syncs_ = 0, syncs_ = 0, submits_ = 0;
	int64_t permanent_live_ = 0;
	bool permanent_rids_ = true;
	bool list_open_ = false;
	bool submitted_ = false;
	bool recovery_ = true;
	int64_t recoveries_ = 0;
	::RID staging_;
	size_t staging_bytes_ = 0;
	size_t staging_max_ = kMaxTransferBytes;
};

} // namespace rdc
