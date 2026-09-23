// ggml-rd internals shared by ggml-rd.cpp, rd_graph.cpp and rd_kernels.cpp.
// The op packers (ops/*.cpp) see only rd_pack.h, which has no RenderingDevice
// in it, so the host test harness (tests/ggml_rd_kernels) builds them too.
// How an op is added: gates/3-ggml-rd/README.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-rd.h"
#include "rd_compute.h"
#include "rd_pack.h"

namespace ggml_rd {

// One ggml buffer: one RD storage buffer, and the fake base ggml sees.
struct Buffer {
	::RID rid;
	size_t size = 0; // bytes of the RD buffer (>= the ggml size, multiple of 256)
	uintptr_t base = 0;
	uint32_t index = 0;
};

struct Stats {
	int64_t graphs = 0, nodes = 0, dispatches = 0, skipped = 0, barriers = 0;
	int64_t pipelines = 0, set0_created = 0, slots_created = 0, coop_yields = 0;
	int64_t waits = 0, faults = 0, failed_graphs = 0;
	int64_t params_bytes = 0, set_bytes = 0, get_bytes = 0, copies = 0, clears = 0;
	int64_t buffers_live = 0, bytes_live = 0;
	// The last graph.
	int64_t last_nodes = 0, last_dispatches = 0, last_barriers = 0, last_skipped = 0;
};

struct Ctx {
	rdc::Device *dev = nullptr;
	size_t total_bytes = 0;
	ggml_rd_hooks hooks{};
	bool pending = false; // a graph is submitted and not yet synced
	uint32_t next_buffer_index = 1;
	uint64_t dispatch_serial = 0; // every dispatch ever recorded (GGML_RD_FAULT)
	// GPU timestamps around each graph's compute list (ggml_backend_rd_set_timestamps
	// or GGML_RD_TIMESTAMPS=1): armed at submit, read after the sync.
	bool timestamps = false;
	bool ts_armed = false;
	int64_t last_gpu_ns = -1;
	Stats st;
	std::string last_error;
};

Ctx &ctx();
bool device_ok();
// If a graph is in flight, run the wait hook (a later frame), then sync.
void ensure_idle();
// A cooperative yield point (the COOP hook), for long setup loops.
void coop();
void set_error(const std::string &e);

// The RD buffer holding `t` (a view's too), or null if it is not in one.
Buffer *buffer_of(const ggml_tensor *t);
// Byte offset of t's data in its RD buffer.
uint64_t byte_offset(const ggml_tensor *t, const Buffer *b);
ggml_backend_buffer_type_t buft();

// --- rd_graph.cpp -----------------------------------------------------------
ggml_status graph_compute(ggml_cgraph *g);

// --- rd_kernels.cpp: pipelines, the params table, uniform-set caches --------
::RID kernel_pipeline(int k); // created on first use; null on failure
// The params table buffer, grown (and every set-0 set with it) to `slots`.
::RID params_buffer(uint32_t slots);
// Slot i's set-1 uniform set (a 16-byte UBO holding base = 64 * i).
::RID slot_set(uint32_t i);
// The set-0 uniform set of (params, s0, s1, s2, dst), cached by their RIDs.
::RID tensor_set(::RID params, ::RID s0, ::RID s1, ::RID s2, ::RID dst);
// Godot frees a buffer's uniform sets with it; drop them from the cache.
void forget_sets_of(::RID buffer);
void release_kernels();

} // namespace ggml_rd
