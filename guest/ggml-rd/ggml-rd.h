// ggml-rd -- a ggml backend whose only way to the GPU is Godot's
// RenderingDevice, through guest/rd_compute (AGENTS.md).
//
//   registry "RD": one device, "RD0": the GPU, or, when no RenderingDevice
//                  was attached (headless), the CPU fallback: the same
//                  packers, the same params words, and the kernels' slangc
//                  cpp emits (kernels/ggml/cpp, the second target of every
//                  Lean kernel, AGENTS.md rule 2) run on guest memory, one
//                  dispatch after another (guest/ggml-rd/rd_cpu.cpp).
//                  GGML_RD_CPU_FALLBACK=0 before the registry is read turns
//                  the fallback off: then there is no device;
//   buffer type:   one ggml buffer = one RD storage buffer, created empty and
//                  cleared on the GPU; the tensor addresses ggml sees are a
//                  fake base per buffer (0x1000 + (index << 40)) plus the
//                  offset, never dereferenced;
//   backend:       graph_compute packs every node's params into one table
//                  (one buffer_update), records one compute list with a
//                  barrier only where a dispatch touches a byte range an
//                  earlier dispatch since the last barrier wrote (or writes
//                  one it read), submits, and returns. It never syncs in the
//                  frame it submits (AGENTS.md rule 4): the next call that
//                  needs the results runs the wait hook, which on a guest
//                  fiber yields WAIT_GPU to the host, and syncs when resumed.
//
// The kernels are Lean-generated (lean/Ggml, kernels/ggml); each op's packer
// lives in its own ops/<op>.cpp and registers itself with GGML_RD_OP, so the
// op families never edit a shared file.
//
// Environment (read on every graph): GGML_RD_BARRIER_ALL=1 puts a barrier
// after every dispatch; GGML_RD_FAULT=<n> corrupts the params of every n-th
// dispatch (a source offset +1 element), the control that shows a wrong
// kernel result is caught; GGML_RD_DROP_BARRIER=<k> leaves out the k-th
// barrier (1-based) that elision places in each graph, the control that
// shows a missing barrier is caught (Gate 3 G3.graph); GGML_RD_PROFILE=<1|2>
// times graph_compute on the host clock (ggml_backend_rd_last_profile).
#pragma once

#include <api.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"

namespace rdc {
class Device;
}

// The registry's name; its one device (and backend) is GGML_RD_NAME "0".
#define GGML_RD_NAME "RD"

// Where a backend may give up the frame, and how it reaches host files. The
// first two default to "do nothing": then a wait syncs immediately (in the
// submitting frame, which rd_compute's rule-4 counter records).
struct ggml_rd_hooks {
	// Called when the backend must wait for submitted work. Return on a later
	// frame (a fiber yields WAIT_GPU); the backend then syncs.
	void (*wait_gpu)(void *user) = nullptr;
	// Called every 256 uniform sets created, to give the frame back (COOP).
	void (*coop)(void *user) = nullptr;
	// Have the host copy `bytes` at `file_offset` of the host file `path` into
	// the RD buffer `rid` at `dst_offset` (a fiber yields UPLOAD); true once
	// done. Only ggml_backend_rd_tensor_upload calls it, with the device idle.
	bool (*upload)(void *user, const std::string &path, uint64_t file_offset, uint64_t bytes, ::RID rid,
			uint64_t dst_offset) = nullptr;
	// The CPU fallback's counterpart: copy `bytes` at `file_offset` of the
	// host file `path` to `dst` in guest memory (a fiber yields READ). Only
	// ggml_backend_rd_tensor_upload calls it, for a tensor in a CPU buffer.
	bool (*read)(void *user, const std::string &path, uint64_t file_offset, uint64_t bytes, void *dst) = nullptr;
	void *user = nullptr;
};

// The registry. Register it with ggml_backend_register() after
// ggml_backend_rd_attach(): the device list is read at registration.
ggml_backend_reg_t ggml_backend_rd_reg(void);

// Hand the backend the guest's device (normally adopted from the host) and
// the device memory the host reports; `dev` may be null (no device). The
// backend never frees the device.
void ggml_backend_rd_attach(rdc::Device *dev, size_t total_bytes);
void ggml_backend_rd_set_hooks(const ggml_rd_hooks &hooks);

bool ggml_backend_is_rd(ggml_backend_t backend);
bool ggml_backend_buffer_is_rd(ggml_backend_buffer_t buffer);
// The RD storage buffer behind a ggml buffer (for UPLOAD requests), and a
// tensor's byte offset in it.
::RID ggml_backend_rd_buffer_rid(ggml_backend_buffer_t buffer);
size_t ggml_backend_rd_tensor_offset(const ggml_tensor *tensor);

// Wait for (and sync) submitted work, if any. The host may only
// buffer_update the device while it is idle.
void ggml_backend_rd_ensure_idle(void);

// Fill `bytes` of `tensor`, from byte `offset` of its data, with the bytes at
// `file_offset` of the host file `path`, through the upload hook: the host
// reads the file and buffer_updates the RD buffer, so the bytes never enter
// the guest heap (how weights load). Waits for submitted work first. False,
// with ggml_backend_rd_last_error() set, if the tensor is not in an RD
// buffer, the range is past its end, or there is no hook.
bool ggml_backend_rd_tensor_upload(const ggml_tensor *tensor, size_t offset, const std::string &path,
		uint64_t file_offset, size_t bytes);

// Counters since start, and the last graph's, as one line.
std::string ggml_backend_rd_stats(void);
// The last graph's dispatches and barriers.
void ggml_backend_rd_last_graph(int64_t *dispatches, int64_t *barriers);
// GPU timestamps around every graph's compute list (also GGML_RD_TIMESTAMPS=1),
// and the last synced graph's GPU time between them in ns (-1 if none).
// Measurement only (Gate 3's perf probe): two host calls per graph.
void ggml_backend_rd_set_timestamps(bool on);
int64_t ggml_backend_rd_last_gpu_ns(void);
std::string ggml_backend_rd_last_error(void);

// Host-clock profile of the last graph_compute (rdc::host_usec, one host call
// per reading; the guest clock is not a clock). Level 1 times the phases (7
// readings per graph); level 2 also each dispatch's packing and its
// recording (bind, set, barrier, dispatch: 3 more readings per dispatch).
// Level 0 (the default) reads the clock not at all. GGML_RD_PROFILE=<level>
// overrides the level set here. Measurement only (Gate 3 G3.cost).
struct ggml_rd_profile_dispatch {
	const ggml_tensor *node = nullptr;
	int kernel = -1;
	bool barrier = false; // a barrier was recorded before it
	int64_t pack_us = 0, record_us = 0;
};
struct ggml_rd_profile {
	int level = 0;
	int64_t nodes = 0, dispatches = 0, barriers = 0, skipped = 0;
	// pack: every node packed; prepare: pipelines, slot sets and set-0 sets
	// (and any COOP yields they took); upload: the params table; record: the
	// compute list; submit: submit(). total: graph_compute from entry to return.
	int64_t us_pack = 0, us_prepare = 0, us_upload = 0, us_record = 0, us_submit = 0, us_total = 0;
	std::vector<ggml_rd_profile_dispatch> per_dispatch; // level 2
};
void ggml_backend_rd_set_profile(int level);
const ggml_rd_profile &ggml_backend_rd_last_profile(void);
// The kernel a dispatch ran, by the id in ggml_rd_profile_dispatch.
const char *ggml_backend_rd_kernel_name(int kernel);
// Under GGML_RD_DROP_BARRIER=<k>: the dispatch the dropped barrier preceded
// in the last graph ("" if the graph had fewer than k barriers).
std::string ggml_backend_rd_last_dropped(void);

// Free every pipeline, shader, uniform set and the params and slot buffers
// (the ggml buffers are their owners' to free first).
void ggml_backend_rd_release(void);
