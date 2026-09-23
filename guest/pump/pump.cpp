#include "pump.h"

#include <algorithm>

#include "fiber/fiber.h"

namespace pump {

namespace {

struct State {
	Fiber *fiber = nullptr;
	Job job = nullptr;
	void *arg = nullptr;
	int64_t kind = NONE;
	int64_t a[3] = { 0, 0, 0 };
	std::string text;
	::RID rid;
	std::vector<uint8_t> in;
	bool failed = false; // fail() was called: the fiber is abandoned
	bool inside = false; // the job is running (between resume and yield)
};

State &S() {
	static State s;
	return s;
}

void body(Fiber &, void *) {
	State &s = S();
	s.job(s.arg);
}

void request(int64_t kind) {
	State &s = S();
	s.kind = kind;
	s.inside = false;
	s.fiber->yield(); // the host's next pump() lands here
	s.inside = true;
}

Variant result() {
	State &s = S();
	std::vector<int64_t> h = { s.kind, s.a[0], s.a[1], s.a[2] };
	Array r = Array::Create();
	r.push_back(Variant(PackedArray<int64_t>(h)));
	r.push_back(Variant(String(s.text)));
	r.push_back(Variant(s.rid));
	return Variant(r);
}

} // namespace

const char *kind_name(int64_t k) {
	switch (k) {
		case NONE: return "NONE";
		case WAIT_GPU: return "WAIT_GPU";
		case READ: return "READ";
		case UPLOAD: return "UPLOAD";
		case COOP: return "COOP";
		case DONE: return "DONE";
		case ERROR: return "ERROR";
		default: return "?";
	}
}

bool start(Job job, void *arg, size_t stack_bytes) {
	State &s = S();
	if (s.fiber != nullptr && !s.fiber->done() && !s.failed) {
		return false;
	}
	// An abandoned (failed) fiber's stack is freed with it; whatever its
	// frames owned leaks, which is what abandoning means.
	delete s.fiber;
	s = State{};
	s.job = job;
	s.arg = arg;
	s.fiber = new Fiber(&body, nullptr, stack_bytes);
	if (s.fiber->done()) {
		s.kind = ERROR;
		s.text = "no memory for the fiber stack";
		s.failed = true;
		return false;
	}
	return true;
}

bool running() {
	const State &s = S();
	return s.fiber != nullptr && !s.fiber->done() && !s.failed;
}

int64_t last_kind() {
	return S().kind;
}

const std::string &last_text() {
	return S().text;
}

Variant step(const PackedArray<uint8_t> &in) {
	State &s = S();
	if (s.fiber == nullptr) {
		s.kind = ERROR;
		s.text = "no job started";
		return result();
	}
	if (s.failed || s.fiber->done()) {
		return result(); // DONE / ERROR stay put
	}
	s.in = in.fetch();
	s.kind = NONE;
	s.a[0] = s.a[1] = s.a[2] = 0;
	s.text.clear();
	s.rid = ::RID();
	s.inside = true;
	const bool more = s.fiber->resume();
	s.inside = false;
	if (!more && !s.failed) {
		if (s.fiber->escaped()) {
			s.kind = ERROR;
			s.text = "an exception escaped the job";
		} else {
			s.kind = DONE;
		}
	}
	return result();
}

bool inside() {
	return S().inside;
}

// Outside the job (a host call such as a close, after the job returned)
// there is no frame to give back: wait_gpu and coop return at once, and the
// caller syncs in this call.
void wait_gpu() {
	if (inside()) {
		request(WAIT_GPU);
	}
}

void coop() {
	if (inside()) {
		request(COOP);
	}
}

std::vector<uint8_t> read(const std::string &path, uint64_t offset, uint64_t bytes) {
	// One READ answers at most kReadChunk bytes: a PackedByteArray fetched
	// into the guest faults above 16 MiB (Gate 3). bytes == 0 reads to the
	// end: chunks until one comes back short.
	constexpr uint64_t kReadChunk = uint64_t(16) << 20;
	State &s = S();
	std::vector<uint8_t> out;
	if (!s.inside) {
		return out; // only a job can ask the host for bytes
	}
	for (;;) {
		const uint64_t want = bytes == 0 ? kReadChunk : std::min<uint64_t>(bytes - out.size(), kReadChunk);
		if (want == 0) {
			break;
		}
		s.text = path;
		s.a[0] = int64_t(offset + out.size());
		s.a[1] = int64_t(want);
		request(READ);
		const size_t got = s.in.size();
		out.insert(out.end(), s.in.begin(), s.in.end());
		s.in.clear();
		if (got < want) {
			break; // end of file
		}
	}
	return out;
}

bool upload(const std::string &path, uint64_t file_offset, uint64_t bytes, ::RID rid, uint64_t dst_offset) {
	State &s = S();
	if (!s.inside) {
		return false; // only a job can ask the host for an upload
	}
	s.text = path;
	s.a[0] = int64_t(file_offset);
	s.a[1] = int64_t(bytes);
	s.a[2] = int64_t(dst_offset);
	s.rid = rid;
	request(UPLOAD);
	return true;
}

void fail(const std::string &text) {
	State &s = S();
	if (!s.inside) {
		// Not on the job's fiber: nothing to abandon. Record it and let the
		// caller's own failure path run (ggml_abort then aborts the vmcall).
		s.kind = ERROR;
		s.text = text;
		return;
	}
	s.failed = true;
	s.text = text;
	for (;;) {
		request(ERROR);
	}
}

} // namespace pump
