// ggml-rd -- a ggml backend whose only way to the GPU is Godot's
// RenderingDevice, through guest/rd_compute (AGENTS.md).
//
//   registry "RD": one device, "RD0" (GPU), or none when no RenderingDevice
//                  was attached (headless);
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
// kernel result is caught.
#pragma once

#include <api.hpp>

#include <cstddef>
#include <string>

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

// Free every pipeline, shader, uniform set and the params and slot buffers
// (the ggml buffers are their owners' to free first).
void ggml_backend_rd_release(void);
