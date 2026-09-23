// rd_worker.elf -- Gate 6G.1: rd_compute from a worker Thread's vmcall.
//
// PolyFEM's Newton loop is synchronous: a line search needs the energy of the
// trial step before it picks the next one, a CCD pass needs its step bound,
// a CG iteration its dot products. If fit.elf moves that arithmetic to the
// GPU, each of those is a GPU round trip made from inside a vmcall that runs
// on the fit stage's worker Thread (project/stages/stage_base.gd). This ELF
// answers whether such a round trip works from there and what it costs.
//
// One round trip: k dispatches in one compute list, k - 1 of them a saxpby
// chain and the last a df32 dot of the chain's result, then the dot's 16-byte
// result buffer back on the CPU. The kernels are Lean's (kernels/drape:
// saxpby is Cloth.SlangCodegen.Saxpby, dot_reduce the df32 dot of the
// L-BFGS-B line search), embedded by kernels/drape/gen.sh; nothing here is
// hand-written GPU code (AGENTS.md rule 2).
//
// The chain is exact by construction, so every readback is checked: A and B
// start at 0, each saxpby writes other = 1 * ones + 1 * cur (a ping-pong,
// never in place: Gate 0F finding 4), so after D saxpbys every element of the
// current buffer is D and dot(ones, cur) = n * D, an integer the df32 sum
// holds exactly.
//
// Three ways to finish a round trip (mode):
//   0 sync_get  list_end, submit, sync, buffer_get_data of the 16-byte result
//   1 get       list_end, buffer_get_data alone (it flushes and stalls itself)
//   2 sync      list_end, submit, sync, no readback (the fence alone)
// and a frame-paced one for the rule-4 comparison: rw_submit records and
// submits, rw_collect (a later frame) syncs and reads.
//
// Every RenderingDevice call goes through rdc::Device (its method names have
// host name-cache slots of their own; AGENTS.md). Nothing is timed in the
// guest: the host times each vmcall.

#include <api.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "drape_kernels.inc" // saxpby, dot_reduce: Lean -> Slang -> SPIR-V (kernels/drape/gen.sh)
#include "rd_compute.h"

static Variant text(const std::string &s) {
	return Variant(String(s));
}

static rdc::Device g_dev;

// saxpby: set 0 = { b0 params UBO {n, alpha, beta}, b1 x, b2 y, b3 dst }.
struct SaxParams {
	uint32_t n;
	float alpha;
	float beta;
	uint32_t pad;
};
// dot_reduce: set 0 = { b0 params UBO {n}, b1 a, b2 b, b3 dst[2] = (hi, lo) }.
struct DotParams {
	uint32_t n;
	uint32_t pad[3];
};

struct State {
	bool ok = false;
	uint32_t n = 0;
	::RID sax_shader, sax_pipe, dot_shader, dot_pipe;
	::RID sax_params, dot_params, ones, a, b, r;
	::RID set_ab, set_ba, dot_a, dot_b;
	int cur = 0; // 0: A holds the chain value, 1: B
	int64_t value = 0; // every element of the current buffer
	int64_t exact = 0, wrong = 0, empty = 0, unread = 0, rounds = 0, dispatches = 0;
	double last = -1.0, last_want = -1.0;
	bool pending = false; // rw_submit made, rw_collect not yet
	std::string open_note;
};
static State g;

static const uint8_t *kernel(const char *name, size_t &bytes) {
	const drape_kernels::Entry *e = drape_kernels::find(name);
	if (!e) {
		bytes = 0;
		return nullptr;
	}
	bytes = e->size;
	return e->bytes;
}

// n floats per vector (the fit's 3 x vertices). Creates the device on the
// calling thread: Godot binds a RenderingDevice to the thread that made it.
static Variant rw_open(int64_t n) {
	if (g.ok) {
		return text("OK already open n=" + std::to_string(g.n));
	}
	if (n < 1 || n > (1 << 24)) {
		return text("FAIL n must be in [1, 2^24]");
	}
	if (!g_dev.open()) {
		return text(g_dev.error());
	}
	size_t sb = 0, db = 0;
	const uint8_t *sax = kernel("saxpby", sb);
	const uint8_t *dot = kernel("dot_reduce", db);
	if (!sax || !dot) {
		return text("FAIL saxpby or dot_reduce not embedded");
	}
	State s;
	s.n = uint32_t(n);
	s.sax_shader = g_dev.shader_from_spirv(sax, sb);
	s.sax_pipe = s.sax_shader.index ? g_dev.compute_pipeline(s.sax_shader) : ::RID();
	s.dot_shader = g_dev.shader_from_spirv(dot, db);
	s.dot_pipe = s.dot_shader.index ? g_dev.compute_pipeline(s.dot_shader) : ::RID();
	if (!s.sax_pipe.index || !s.dot_pipe.index) {
		return text("FAIL pipelines: " + g_dev.error());
	}
	const SaxParams sp{ s.n, 1.0f, 1.0f, 0 };
	const DotParams dp{ s.n, { 0, 0, 0 } };
	const std::vector<float> ones(s.n, 1.0f);
	s.sax_params = g_dev.uniform_buffer(sizeof sp, &sp);
	s.dot_params = g_dev.uniform_buffer(sizeof dp, &dp);
	s.ones = g_dev.storage_buffer(ones.size() * 4, ones.data());
	s.a = g_dev.storage_buffer(size_t(s.n) * 4);
	s.b = g_dev.storage_buffer(size_t(s.n) * 4);
	s.r = g_dev.storage_buffer(16);
	if (!s.sax_params.index || !s.dot_params.index || !s.ones.index || !s.a.index || !s.b.index || !s.r.index) {
		return text("FAIL buffers: " + g_dev.error());
	}
	const int U = rdc::UNIFORM_TYPE_UNIFORM_BUFFER, S = rdc::UNIFORM_TYPE_STORAGE_BUFFER;
	s.set_ab = g_dev.uniform_set(s.sax_shader, { { 0, U, s.sax_params }, { 1, S, s.ones }, { 2, S, s.a }, { 3, S, s.b } });
	s.set_ba = g_dev.uniform_set(s.sax_shader, { { 0, U, s.sax_params }, { 1, S, s.ones }, { 2, S, s.b }, { 3, S, s.a } });
	s.dot_a = g_dev.uniform_set(s.dot_shader, { { 0, U, s.dot_params }, { 1, S, s.ones }, { 2, S, s.a }, { 3, S, s.r } });
	s.dot_b = g_dev.uniform_set(s.dot_shader, { { 0, U, s.dot_params }, { 1, S, s.ones }, { 2, S, s.b }, { 3, S, s.r } });
	if (!s.set_ab.index || !s.set_ba.index || !s.dot_a.index || !s.dot_b.index) {
		return text("FAIL uniform sets: " + g_dev.error());
	}
	s.ok = true;
	s.open_note = "n=" + std::to_string(s.n) + " saxpby_groups=" + std::to_string(rdc::Device::groups_for(s.n, 256)) +
			" spirv_bytes=" + std::to_string(sb) + "+" + std::to_string(db);
	g = s;
	return text("OK " + g.open_note + " permanent_slots=" + std::to_string(g_dev.permanent_slots()) + " " +
			rdc::name_slots());
}

// k dispatches in one list: k - 1 chained saxpbys, a barrier after each (the
// next one reads what it wrote), then the dot of the current buffer.
static void record(int64_t k) {
	g_dev.list_begin();
	const uint32_t groups = rdc::Device::groups_for(g.n, 256);
	for (int64_t i = 0; i + 1 < k; ++i) {
		g_dev.bind_pipeline(g.sax_pipe);
		g_dev.bind_uniform_set(g.cur == 0 ? g.set_ab : g.set_ba);
		g_dev.dispatch(groups);
		g_dev.barrier();
		g.cur ^= 1;
		g.value += 1;
	}
	g_dev.bind_pipeline(g.dot_pipe);
	g_dev.bind_uniform_set(g.cur == 0 ? g.dot_a : g.dot_b);
	g_dev.dispatch(1); // one workgroup of 256 strides over n
	g_dev.list_end();
	g.dispatches += k;
	++g.rounds;
}

// 1 exact, 0 wrong value, -2 nothing came back (a refused buffer_get_data).
static int64_t read_check() {
	const std::vector<uint8_t> v = g_dev.buffer_get(g.r, 0, 16);
	if (v.size() < 8) {
		++g.empty;
		g.last = -1.0;
		return -2;
	}
	float hi = 0.0f, lo = 0.0f;
	std::memcpy(&hi, v.data(), 4);
	std::memcpy(&lo, v.data() + 4, 4);
	g.last = double(hi) + double(lo);
	g.last_want = double(g.n) * double(g.value);
	if (g.last == g.last_want) {
		++g.exact;
		return 1;
	}
	++g.wrong;
	return 0;
}

// One round trip. Returns the check (1 / 0 / -2), 2 for mode 2 (not read),
// -3 no device, -4 a frame-paced round trip is pending, -5 bad mode.
static int64_t round_once(int64_t k, int64_t mode) {
	if (!g.ok) {
		return -3;
	}
	if (g.pending) {
		return -4;
	}
	if (k < 1 || mode < 0 || mode > 2) {
		return -5;
	}
	record(k);
	if (mode == 0) {
		g_dev.submit();
		g_dev.sync();
		return read_check();
	}
	if (mode == 1) {
		return read_check();
	}
	g_dev.submit();
	g_dev.sync();
	++g.unread;
	return 2;
}

static Variant rw_round(int64_t k, int64_t mode) {
	return Variant(round_once(k, mode));
}

// reps round trips in one vmcall; the answer counts the ones that came back
// as expected (1, or 2 in mode 2).
static Variant rw_rounds(int64_t k, int64_t mode, int64_t reps) {
	int64_t good = 0;
	const int64_t want = mode == 2 ? 2 : 1;
	for (int64_t i = 0; i < reps; ++i) {
		if (round_once(k, mode) == want) {
			++good;
		}
	}
	return Variant(good);
}

// The frame-paced round trip of a rule-4 host (and of a fiber that yields
// WAIT_GPU at every submit): record + submit in one frame ...
static Variant rw_submit(int64_t k) {
	if (!g.ok) {
		return Variant(int64_t(-3));
	}
	if (g.pending) {
		return Variant(int64_t(-4));
	}
	record(k);
	g_dev.submit();
	g.pending = true;
	return Variant(int64_t(1));
}

// ... sync and read in a later one.
static Variant rw_collect() {
	if (!g.pending) {
		return Variant(int64_t(-4));
	}
	g.pending = false;
	g_dev.sync();
	return Variant(read_check());
}

static Variant rw_stats() {
	char b[640];
	std::snprintf(b, sizeof b,
			"open=%s exact=%lld wrong=%lld empty=%lld unread=%lld rounds=%lld dispatches=%lld chain=%lld "
			"last=%.1f want=%.1f same_frame_syncs=%lld syncs=%lld submits=%lld permanent_slots=%lld",
			g.ok ? "yes" : "no", (long long)g.exact, (long long)g.wrong, (long long)g.empty, (long long)g.unread,
			(long long)g.rounds, (long long)g.dispatches, (long long)g.value, g.last, g.last_want,
			(long long)g_dev.same_frame_syncs(), (long long)g_dev.syncs(), (long long)g_dev.submits(),
			(long long)g_dev.permanent_slots());
	return text(std::string(b) + " step=" + g_dev.step() + (g_dev.error().empty() ? "" : " err=" + g_dev.error()));
}

// Frees every RID and the device. Must run on the thread that opened it:
// free_rid and the device's own teardown are render-thread-guarded too.
static Variant rw_close() {
	if (!g_dev.ok()) {
		return text("no device");
	}
	const ::RID sets[] = { g.set_ab, g.set_ba, g.dot_a, g.dot_b };
	for (const ::RID &r : sets) {
		g_dev.free_rid(r);
	}
	const ::RID rest[] = { g.sax_params, g.dot_params, g.ones, g.a, g.b, g.r, g.sax_pipe, g.dot_pipe, g.sax_shader,
		g.dot_shader };
	for (const ::RID &r : rest) {
		g_dev.free_rid(r);
	}
	const int64_t left = g_dev.permanent_slots();
	g_dev.close();
	g = State{};
	return text("closed permanent_slots_left=" + std::to_string(left));
}

// A kernel's SPIR-V, for the host's flat control (GDScript on the same
// kernels, no sandbox).
static Variant rw_spirv(String name_s) {
	const std::string name = name_s.utf8();
	size_t bytes = 0;
	const uint8_t *p = kernel(name.c_str(), bytes);
	const PackedArray<uint8_t> out(p ? std::vector<uint8_t>(p, p + bytes) : std::vector<uint8_t>{});
	return Variant(out);
}

int main() {
	ADD_API_FUNCTION(rw_open, "String", "int n", "Open a local RenderingDevice on this thread; saxpby + df32 dot over n floats");
	ADD_API_FUNCTION(rw_round, "int", "int k, int mode", "One GPU round trip of k dispatches (mode 0 sync_get, 1 get, 2 sync)");
	ADD_API_FUNCTION(rw_rounds, "int", "int k, int mode, int reps", "reps round trips in one vmcall; how many came back right");
	ADD_API_FUNCTION(rw_submit, "int", "int k", "Record k dispatches and submit (frame-paced round trip, first half)");
	ADD_API_FUNCTION(rw_collect, "int", "", "Sync and read back (frame-paced round trip, second half)");
	ADD_API_FUNCTION(rw_stats, "String", "", "Checks, chain value and rdc::Device counters");
	ADD_API_FUNCTION(rw_close, "String", "", "Free every RID and the device (on the thread that opened it)");
	ADD_API_FUNCTION(rw_spirv, "PackedByteArray", "String name", "An embedded kernel's SPIR-V, for the flat control");
	halt();
}
