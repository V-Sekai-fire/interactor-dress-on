// The pump protocol: a guest job on a fiber, advanced by the host once per
// pump call, asking the host for what only the host can do.
//
// The job is straight-line code (ggml, a model's forward pass) on a
// guest/fiber stack. Whenever it needs the host it yields a request, and the
// host's next pump() call resumes it:
//
//   WAIT_GPU                 GPU work is submitted; resume me on a LATER frame
//                            (then the job syncs: AGENTS.md rule 4)
//   READ    path off n       read n bytes at off of a host file; pass them to
//                            the next pump(data) call (n <= 16 MiB: read()
//                            splits, since a PackedByteArray crossing into
//                            the guest faults above 16 MiB)
//   UPLOAD  path foff n rid doff
//                            read n bytes at foff of a host file and
//                            rd.buffer_update(rid, doff, ...) them into the
//                            guest's RenderingDevice (which the host owns):
//                            weights never enter the guest heap
//   COOP                     nothing to wait for; give the frame back
//   DONE                     the job returned
//   ERROR   text             the job failed (an escaped exception, a ggml
//                            abort); it is never resumed
//
// pump() returns Array[PackedInt64Array header, String text, RID rid]:
//   header = [kind, a0, a1, a2]; READ: [2, off, n, 0] text=path;
//   UPLOAD: [3, file_off, n, dst_off] text=path rid=the RD buffer;
//   ERROR: text = the reason.
// The guest cannot open files (Gate 0F probe 3: EBADF on every path), so
// READ and UPLOAD are the only way bytes reach it. project/infer_host.gd is
// the host side: it serves READ and UPLOAD in the same frame up to a byte
// cap, and stops the frame at WAIT_GPU, COOP, DONE and ERROR.
//
// An UPLOAD must be yielded only while the device is idle (nothing
// submitted): a local RenderingDevice drops whatever is recorded between
// submit() and sync().
#pragma once

#include <api.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace pump {

enum Kind : int64_t {
	NONE = 0,
	WAIT_GPU = 1,
	READ = 2,
	UPLOAD = 3,
	COOP = 4,
	DONE = 5,
	ERROR = 6,
};

const char *kind_name(int64_t k);

using Job = void (*)(void *arg);

// Start `job(arg)` on a fresh fiber (it does not run until the first pump).
// False if a job is still running.
bool start(Job job, void *arg, size_t stack_bytes = size_t(4) << 20);
// Resume the job once. `in` is what a READ asked for (empty otherwise).
Variant step(const PackedArray<uint8_t> &in);
bool running();
// The last request's kind (DONE or ERROR once finished).
int64_t last_kind();
// Requests yielded since the process started, by kind (index = Kind). A
// WAIT_GPU or COOP ends the host's frame, so their sum counts frames.
int64_t yields(int64_t kind);
const std::string &last_text();

// --- inside the job -------------------------------------------------------------
// True while the job runs (resumed and not yet yielded). The calls below
// yield only then; outside the job (e.g. a host call that closes the
// backend after the job returned) wait_gpu and coop return at once, read
// returns nothing and upload returns false.
bool inside();
void wait_gpu();
void coop();
std::vector<uint8_t> read(const std::string &path, uint64_t offset, uint64_t bytes);
bool upload(const std::string &path, uint64_t file_offset, uint64_t bytes, ::RID rid, uint64_t dst_offset);
// Report ERROR; inside the job it never returns (the fiber is abandoned),
// outside it records the error and returns.
void fail(const std::string &text);

} // namespace pump
