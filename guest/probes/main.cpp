// probes.elf -- Gate 0F: what the godot-sandbox runtime actually does.
//
// Each entry point answers one question that could break a later cut
// (exceptions, rounding modes, files, threads, limits, argument marshalling,
// the heap, a fiber across vmcalls, large RD buffers, references_max, ggml).
// The host side is project/gate_runtime.gd; it runs every probe with a
// control and writes gates/0f-runtime/results.txt. Nothing in here decides a
// verdict: the guest reports what it saw and the host compares.

#include <api.hpp>

#include <atomic>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "fiber/fiber.h"
#include "rd_compute.h"
#include "probes_kernels.inc" // saxpby, embedded by kernels/probes/gen.sh

#ifdef PROBES_WITH_GGML
#include "ggml_probe.h"
#endif
#include "zfh_probe.h"

static Variant text(const std::string &s) {
	return Variant(String(s));
}

// ---------------------------------------------------------------------------
// 1. Exceptions through std::function

static Variant p_exceptions(bool do_throw) {
	std::function<int(int)> inner = [](int x) -> int {
		if (x < 0) {
			throw std::runtime_error("negative input " + std::to_string(x));
		}
		return x * 2;
	};
	std::function<int(int)> outer = [&](int x) -> int { return inner(x) + 1; };
	std::string r;
	try {
		const int v = outer(do_throw ? -7 : 5);
		r = "returned value=" + std::to_string(v);
	} catch (const std::runtime_error &e) {
		r = std::string("caught runtime_error: ") + e.what();
	} catch (...) {
		r = "caught unknown";
	}
	// A second, differently typed throw through a rethrow, so the unwinder's
	// type matching is exercised as well as the landing pad.
	int nested = 0;
	try {
		try {
			if (do_throw) {
				throw 42;
			}
		} catch (int v) {
			nested = v;
			throw;
		}
	} catch (int v) {
		nested += v;
	}
	return text(r + " rethrow_sum=" + std::to_string(nested));
}

// ---------------------------------------------------------------------------
// 2. fesetround

static volatile double g_one = 1.0;
static volatile double g_three = 3.0;
static volatile float g_onef = 1.0f;
static volatile float g_threef = 3.0f;

static uint64_t bits(double d) {
	uint64_t u;
	std::memcpy(&u, &d, 8);
	return u;
}
static uint32_t bitsf(float f) {
	uint32_t u;
	std::memcpy(&u, &f, 4);
	return u;
}

static Variant p_fenv() {
	const int r_up = fesetround(FE_UPWARD);
	const int got_up = fegetround();
	const double up = g_one / g_three;
	const float upf = g_onef / g_threef;
	const int r_dn = fesetround(FE_DOWNWARD);
	const int got_dn = fegetround();
	const double dn = g_one / g_three;
	const float dnf = g_onef / g_threef;
	fesetround(FE_TONEAREST);
	const double nr = g_one / g_three;
	char b[400];
	std::snprintf(b, sizeof b,
			"set_up=%d get_up=%d(FE_UPWARD=%d) set_dn=%d get_dn=%d(FE_DOWNWARD=%d) "
			"up=%016llx dn=%016llx nearest=%016llx upf=%08x dnf=%08x double_differs=%s float_differs=%s",
			r_up, got_up, FE_UPWARD, r_dn, got_dn, FE_DOWNWARD, (unsigned long long)bits(up),
			(unsigned long long)bits(dn), (unsigned long long)bits(nr), bitsf(upf), bitsf(dnf),
			bits(up) != bits(dn) ? "yes" : "no", bitsf(upf) != bitsf(dnf) ? "yes" : "no");
	return text(b);
}

// ---------------------------------------------------------------------------
// 3. File I/O

static Variant p_file(String path_s) {
	const std::string path = path_s.utf8();
	std::string r = "path=" + path;
	{
		std::ifstream f(path, std::ios::binary);
		if (f.is_open()) {
			std::stringstream ss;
			ss << f.rdbuf();
			const std::string all = ss.str();
			r += " ifstream=ok bytes=" + std::to_string(all.size()) + " head=[" + all.substr(0, 24) + "]";
		} else {
			r += " ifstream=FAIL";
		}
	}
	errno = 0;
	FILE *fp = std::fopen(path.c_str(), "rb");
	if (fp) {
		char buf[4096];
		size_t total = 0, n;
		while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) {
			total += n;
		}
		std::fclose(fp);
		r += " fopen=ok bytes=" + std::to_string(total);
	} else {
		const int e = errno;
		r += " fopen=FAIL errno=" + std::to_string(e) + "(" + std::strerror(e) + ")";
	}
	return text(r);
}

// ---------------------------------------------------------------------------
// 4. Threads

static Variant p_threads() {
	std::string r = "hardware_concurrency=" + std::to_string(std::thread::hardware_concurrency());
	std::atomic<int> v{ 0 };
	try {
		std::thread t([&] { v.store(42); });
		t.join();
		r += " spawn=ok join=ok value=" + std::to_string(v.load());
	} catch (const std::exception &e) {
		r += std::string(" spawn=FAIL(") + e.what() + ")";
	} catch (...) {
		r += " spawn=FAIL(unknown)";
	}
	return text(r);
}

// ---------------------------------------------------------------------------
// 5, 7, 8. A deterministic busy loop: n LCG steps, returns the state.

static Variant p_spin(int64_t n) {
	uint64_t s = 0x9E3779B97F4A7C15ull;
	for (int64_t i = 0; i < n; ++i) {
		s = s * 6364136223846793005ull + 1442695040888963407ull;
	}
	return Variant(int64_t(s));
}

// ---------------------------------------------------------------------------
// 6. Allocate and touch mb MiB, then free.

static Variant p_alloc(int64_t mb) {
	const size_t bytes = size_t(mb) << 20;
	uint8_t *p = static_cast<uint8_t *>(std::malloc(bytes));
	if (!p) {
		return text("null mb=" + std::to_string(mb));
	}
	uint64_t sum = 0;
	for (size_t i = 0; i < bytes; i += 4096) {
		p[i] = uint8_t(i >> 12);
	}
	for (size_t i = 0; i < bytes; i += 4096) {
		sum += p[i];
	}
	std::free(p);
	return text("ok mb=" + std::to_string(mb) + " pages=" + std::to_string(bytes / 4096) + " sum=" + std::to_string(sum));
}

// ---------------------------------------------------------------------------
// 9. Typed argument echo. Every value goes through guest memory and back.

static Variant echo_f(double x) {
	return Variant(x);
}
static Variant echo_i(int64_t x) {
	return Variant(x);
}
static Variant echo_b(bool x) {
	return Variant(x);
}
static Variant echo_s(String s) {
	return text(s.utf8());
}
static Variant echo_pf32(PackedFloat32Array a) {
	const PackedFloat32Array out(a.fetch());
	return Variant(out);
}
static Variant echo_pb(PackedByteArray a) {
	const PackedArray<uint8_t> out(a.fetch());
	return Variant(out);
}
// The bits the guest actually saw, for when a float comes back wrong.
static Variant f_bits(double x) {
	return Variant(int64_t(bits(x)));
}
// Boxed form: one Variant in, the same Variant out (for unboxed_arguments=false).
static Variant echo_var(const Variant &v) {
	return v;
}

// ---------------------------------------------------------------------------
// 10. Heap: hold an allocation across vmcalls.

static uint8_t *g_hold = nullptr;
static size_t g_hold_bytes = 0;

static Variant p_hold(int64_t mb) {
	std::free(g_hold);
	g_hold_bytes = size_t(mb) << 20;
	g_hold = static_cast<uint8_t *>(std::malloc(g_hold_bytes));
	if (!g_hold) {
		return text("null");
	}
	std::memset(g_hold, 0x5A, g_hold_bytes);
	return text("held " + std::to_string(mb) + " MiB");
}
static Variant p_release() {
	std::free(g_hold);
	g_hold = nullptr;
	return text("released");
}

// ---------------------------------------------------------------------------
// The RenderingDevice, opened by the guest and held across vmcalls. p_rd()
// hands the same Object to the host so GDScript can time buffer_update on it
// (the path weights will take: host FileAccess -> rd.buffer_update).

static Object g_rd{ uint64_t(0) };
static rdc::Device g_dev;

static bool ensure_rd(std::string &err) {
	if (g_rd.is_valid()) {
		return true;
	}
	Object rs("RenderingServer");
	Variant v = rs.call("create_local_rendering_device");
	if (v.get_type() != Variant::OBJECT) {
		err = "FAIL create_local_rendering_device: not an Object (headless?)";
		return false;
	}
	g_rd = v.as_object();
	g_dev.adopt(g_rd);
	return true;
}

static Variant p_rd() {
	std::string err;
	if (!ensure_rd(err)) {
		return text(err);
	}
	return Variant(g_rd);
}

// A RID the host returns is a *scoped* Variant index, valid for the current
// vmcall only; held across vmcalls it names whatever the next call scoped at
// that index (probe 11's first run: "RID idx=3 is not known/scoped", then a
// PackedByteArray where a RID was expected). Moving it to permanent storage
// is what lets a guest static hold GPU objects between vmcalls.
static ::RID keep(::RID r) {
	if (!r.index) {
		return r;
	}
	Variant v(r);
	v.make_permanent();
	return v.operator ::RID();
}

// saxpby: set 0 = { b0 params UBO {n, alpha, beta}, b1 x, b2 y, b3 dst }.
struct Saxpby {
	::RID shader, pipeline;
	bool ok = false;
};
static Saxpby g_sax;

static bool ensure_saxpby(std::string &err) {
	if (g_sax.ok) {
		return true;
	}
	if (!ensure_rd(err)) {
		return false;
	}
	const avbd_kernels::Entry *k = avbd_kernels::find("saxpby");
	if (!k) {
		err = "FAIL saxpby not embedded";
		return false;
	}
	g_sax.shader = keep(g_dev.shader_from_spirv(k->bytes, k->size));
	g_sax.pipeline = g_sax.shader.index ? keep(g_dev.compute_pipeline(g_sax.shader)) : ::RID();
	if (!g_sax.pipeline.index) {
		err = g_dev.error();
		return false;
	}
	g_sax.ok = true;
	return true;
}

struct SaxParams {
	uint32_t n;
	float alpha;
	float beta;
	uint32_t pad;
};

// A "+1 counter": dst[0] = 1 * ones[0] + 1 * dst[0].
struct Counter {
	::RID params, ones, dst, set;
};
static bool make_counter(Counter &c, std::string &err) {
	const SaxParams p{ 1, 1.0f, 1.0f, 0 };
	const float one = 1.0f;
	c.params = keep(g_dev.uniform_buffer(sizeof p, &p));
	c.ones = keep(g_dev.storage_buffer(4, &one));
	c.dst = keep(g_dev.storage_buffer(4));
	if (!c.params.index || !c.ones.index || !c.dst.index) {
		err = g_dev.error();
		return false;
	}
	c.set = keep(g_dev.uniform_set(g_sax.shader,
			{ { 0, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, c.params }, { 1, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.ones },
					{ 2, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.dst }, { 3, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.dst } }));
	if (!c.set.index) {
		err = g_dev.error();
		return false;
	}
	return true;
}
static void free_counter(Counter &c) {
	g_dev.free_rid(c.set);
	g_dev.free_rid(c.params);
	g_dev.free_rid(c.ones);
	g_dev.free_rid(c.dst);
	c = Counter{};
}
static float read_f32(::RID buf, size_t offset = 0) {
	std::vector<uint8_t> b = g_dev.buffer_get(buf, offset, 4);
	float f = -1.0f;
	if (b.size() >= 4) {
		std::memcpy(&f, b.data(), 4);
	}
	return f;
}

// ---------------------------------------------------------------------------
// 11. A fiber across vmcalls, and the same job as an explicit state machine.
//
// Job: 100 rounds of { record a +1 dispatch, submit, yield WAIT_GPU }. The
// next vmcall resumes the job, which syncs (a frame later: rule 4) and goes
// on. At round 50 it throws and catches inside the fiber. The final readback
// must be 100.

enum { ST_WAIT_GPU = 1, ST_DONE = 2 };
static const int kRounds = 100;

struct Job {
	Counter c;
	int rounds_done = 0;
	int caught = 0;
	float value = -1.0f;
	std::string err;
};

static void record_plus_one(const Counter &c) {
	g_dev.list_begin();
	g_dev.bind_pipeline(g_sax.pipeline);
	g_dev.bind_uniform_set(c.set);
	g_dev.dispatch(1);
	g_dev.list_end();
	g_dev.submit();
}

static Job g_fjob;
static Fiber *g_fiber = nullptr;

static void fiber_body(Fiber &self, void *arg) {
	Job &j = *static_cast<Job *>(arg);
	for (int i = 0; i < kRounds; ++i) {
		record_plus_one(j.c);
		self.status = ST_WAIT_GPU;
		self.yield(); // the host's next vmcall lands here
		g_dev.sync();
		++j.rounds_done;
		if (i == kRounds / 2) {
			try {
				std::function<void()> f = [&] { throw std::runtime_error("inside the fiber at round 50"); };
				f();
			} catch (const std::runtime_error &) {
				++j.caught;
			}
		}
	}
	j.value = read_f32(j.c.dst);
	self.status = ST_DONE;
}

static Variant fib_start() {
	std::string err;
	if (!ensure_saxpby(err)) {
		return text(err);
	}
	delete g_fiber;
	if (g_fjob.c.set.index) {
		free_counter(g_fjob.c);
	}
	g_fjob = Job{};
	if (!make_counter(g_fjob.c, err)) {
		return text(err);
	}
	g_fiber = new Fiber(&fiber_body, &g_fjob, 256 * 1024);
	return text("started stack=" + std::to_string(g_fiber->stack_bytes()));
}

// One resume per vmcall. "WAIT_GPU n" while running; "DONE ..." at the end.
static Variant fib_pump() {
	if (!g_fiber) {
		return text("FAIL no fiber");
	}
	if (g_fiber->done()) {
		return text("FAIL resumed after done");
	}
	const bool more = g_fiber->resume();
	if (more) {
		return text("WAIT_GPU " + std::to_string(g_fjob.rounds_done));
	}
	char b[160];
	std::snprintf(b, sizeof b, "DONE value=%g rounds=%d caught=%d escaped=%s", g_fjob.value, g_fjob.rounds_done,
			g_fjob.caught, g_fiber->escaped() ? "yes" : "no");
	return text(b);
}

// Control: the same job with the state in a struct instead of on a stack.
static Job g_sjob;
static int g_sm_state = 0; // 0 = submit next, 1 = waiting, 2 = done

static Variant sm_start() {
	std::string err;
	if (!ensure_saxpby(err)) {
		return text(err);
	}
	if (g_sjob.c.set.index) {
		free_counter(g_sjob.c);
	}
	g_sjob = Job{};
	if (!make_counter(g_sjob.c, err)) {
		return text(err);
	}
	g_sm_state = 0;
	return text("started");
}

static Variant sm_pump() {
	if (g_sm_state == 1) {
		g_dev.sync();
		++g_sjob.rounds_done;
		if (g_sjob.rounds_done == kRounds / 2 + 1) {
			try {
				throw std::runtime_error("state machine at round 50");
			} catch (const std::runtime_error &) {
				++g_sjob.caught;
			}
		}
		g_sm_state = g_sjob.rounds_done < kRounds ? 0 : 2;
	}
	if (g_sm_state == 0) {
		record_plus_one(g_sjob.c);
		g_sm_state = 1;
		return text("WAIT_GPU " + std::to_string(g_sjob.rounds_done));
	}
	g_sjob.value = read_f32(g_sjob.c.dst);
	char b[128];
	std::snprintf(b, sizeof b, "DONE value=%g rounds=%d caught=%d", g_sjob.value, g_sjob.rounds_done, g_sjob.caught);
	return text(b);
}

// ---------------------------------------------------------------------------
// 12. A large storage buffer created empty, cleared, written at its end by a
// whole-buffer saxpby, read back at its end.

static Variant big_buffer(int64_t bytes, bool direct) {
	std::string err;
	if (!ensure_saxpby(err)) {
		return text(err);
	}
	if (bytes < 1024 || bytes % 256 != 0 || bytes > int64_t(0xFFFFFFFFll)) {
		return text("FAIL bytes must be a multiple of 256 in [1 KiB, 4 GiB)");
	}
	const uint32_t n = uint32_t(bytes / 4);
	std::string r = "bytes=" + std::to_string(bytes);

	Variant vb = g_rd.call("storage_buffer_create", bytes, PackedByteArray(std::vector<uint8_t>{}));
	::RID buf = vb.operator ::RID();
	if (!buf.index) {
		return text(r + " FAIL storage_buffer_create (empty) returned null");
	}
	Variant e = g_rd.call("buffer_clear", buf, int64_t(0), bytes);
	r += " clear_err=" + std::to_string(int64_t(e));

	// Seed the last element; the kernel then writes it as 2*x + 1*y = 3*seed.
	const float seed = 1.5f;
	if (!g_dev.buffer_update(buf, size_t(bytes - 4), 4, &seed)) {
		g_dev.free_rid(buf);
		return text(r + " FAIL buffer_update at the end");
	}
	const SaxParams p{ n, 2.0f, 1.0f, 0 };
	::RID params = g_dev.uniform_buffer(sizeof p, &p);
	::RID set = g_dev.uniform_set(g_sax.shader,
			{ { 0, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, params }, { 1, rdc::UNIFORM_TYPE_STORAGE_BUFFER, buf },
					{ 2, rdc::UNIFORM_TYPE_STORAGE_BUFFER, buf }, { 3, rdc::UNIFORM_TYPE_STORAGE_BUFFER, buf } });
	if (!set.index) {
		g_dev.free_rid(params);
		g_dev.free_rid(buf);
		return text(r + " FAIL uniform_set: " + g_dev.error());
	}
	const uint32_t groups = rdc::Device::groups_for(n, 256);
	g_dev.list_begin();
	g_dev.bind_pipeline(g_sax.pipeline);
	g_dev.bind_uniform_set(set);
	g_dev.dispatch(groups);
	g_dev.list_end();
	g_dev.submit();
	g_dev.sync(); // a probe, not a frame loop: timed as one call by the host

	// Read back through a 16-byte staging buffer: buffer_copy the words out on
	// the GPU, then buffer_get_data the small buffer. (Godot's buffer_get_data
	// stages the *whole* source buffer however few bytes are asked for; the
	// direct read below is kept to show it.)
	::RID stage = g_dev.storage_buffer(16);
	const int64_t offs[3] = { bytes - 4, (bytes / 2) & ~int64_t(3), 0 };
	for (int i = 0; i < 3; ++i) {
		g_rd.call("buffer_copy", buf, stage, offs[i], int64_t(4 * i), int64_t(4));
	}
	const float last = read_f32(stage, 0);
	const float mid = read_f32(stage, 4);
	const float first = read_f32(stage, 8);
	const float direct_last = direct ? read_f32(buf, size_t(bytes - 4)) : -2.0f;
	g_dev.free_rid(stage);
	g_dev.free_rid(set);
	g_dev.free_rid(params);
	g_dev.free_rid(buf);
	char b[240];
	std::snprintf(b, sizeof b, " groups=%u last=%g (expect 4.5) mid=%g first=%g (expect 0) direct_get_last=%g", groups, last,
			mid, first, direct_last);
	return text(r + b);
}

// ---------------------------------------------------------------------------
// 13. references_max under a long recording: n dispatches, each with three
// binds (pipeline + its set + the set again), a barrier after every 4th.
// Four counters in rotation, so the 4 dispatches between two barriers are
// independent: each counter must read n/4.
//
// Two shapes of "+1". aliased: y and dst are the same buffer (bindings 2 and
// 3), a read-modify-write in place. pingpong: two buffers per counter,
// b = ones + a then a = ones + b, so no dispatch reads what it writes. The
// first run of this probe lost counts in the aliased shape only.

struct PingPong {
	::RID params, ones, a, b, set_ab, set_ba;
};
static Counter g_rc[4];
static PingPong g_pp[4];
static bool g_rc_ok = false;

static bool make_pingpong(PingPong &c, std::string &err) {
	const SaxParams p{ 1, 1.0f, 1.0f, 0 };
	const float one = 1.0f;
	c.params = keep(g_dev.uniform_buffer(sizeof p, &p));
	c.ones = keep(g_dev.storage_buffer(4, &one));
	c.a = keep(g_dev.storage_buffer(4));
	c.b = keep(g_dev.storage_buffer(4));
	c.set_ab = keep(g_dev.uniform_set(g_sax.shader,
			{ { 0, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, c.params }, { 1, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.ones },
					{ 2, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.a }, { 3, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.b } }));
	c.set_ba = keep(g_dev.uniform_set(g_sax.shader,
			{ { 0, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, c.params }, { 1, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.ones },
					{ 2, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.b }, { 3, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.a } }));
	if (!c.set_ab.index || !c.set_ba.index) {
		err = g_dev.error();
		return false;
	}
	return true;
}

static Variant refs_setup() {
	std::string err;
	if (!ensure_saxpby(err)) {
		return text(err);
	}
	if (!g_rc_ok) {
		for (int k = 0; k < 4; ++k) {
			if (!make_counter(g_rc[k], err) || !make_pingpong(g_pp[k], err)) {
				return text(err);
			}
		}
		g_rc_ok = true;
	}
	const float z = 0.0f;
	for (int k = 0; k < 4; ++k) {
		g_dev.buffer_update(g_rc[k].dst, 0, 4, &z);
		g_dev.buffer_update(g_pp[k].a, 0, 4, &z);
		g_dev.buffer_update(g_pp[k].b, 0, 4, &z);
	}
	return text("ok");
}

static Variant refs_run(int64_t n, bool aliased) {
	if (!g_rc_ok) {
		return text("FAIL refs_setup first");
	}
	g_dev.list_begin();
	for (int64_t i = 0; i < n; ++i) {
		const int k = int(i % 4);
		const ::RID set = aliased ? g_rc[k].set : ((i / 4) % 2 == 0 ? g_pp[k].set_ab : g_pp[k].set_ba);
		g_dev.bind_pipeline(g_sax.pipeline);
		g_dev.bind_uniform_set(set);
		g_dev.bind_uniform_set(set);
		g_dev.dispatch(1);
		if (k == 3 && i + 1 < n) {
			g_dev.barrier();
		}
	}
	g_dev.list_end();
	g_dev.submit();
	g_dev.sync();
	// pingpong: after an even number of rounds the count is in a, else in b.
	const bool in_a = ((n / 4) % 2) == 0;
	std::string r = std::string(aliased ? "aliased" : "pingpong") + " n=" + std::to_string(n) + " counters=";
	for (int k = 0; k < 4; ++k) {
		const ::RID buf = aliased ? g_rc[k].dst : (in_a ? g_pp[k].a : g_pp[k].b);
		r += std::to_string(int64_t(read_f32(buf))) + (k < 3 ? "," : "");
	}
	return text(r + " expect_each=" + std::to_string(n / 4));
}

// ---------------------------------------------------------------------------
// 14. f16 storage read on the GPU: the Lean-emitted half_load kernel
// (o[i] = float(w[i]) * 2) over every binary16 class, checked bit-exactly
// against a CPU decode. 64 values, one workgroup (the kernel has no bound).

static float half_to_float(uint16_t h) {
	const uint32_t sign = uint32_t(h >> 15) << 31;
	const int exp = (h >> 10) & 0x1F;
	uint32_t mant = h & 0x3FF;
	uint32_t f;
	if (exp == 0x1F) {
		f = sign | 0x7F800000u | (mant << 13);
	} else if (exp != 0) {
		f = sign | (uint32_t(exp - 15 + 127) << 23) | (mant << 13);
	} else if (mant == 0) {
		f = sign;
	} else {
		int e = -1;
		do {
			++e;
			mant <<= 1;
		} while ((mant & 0x400) == 0);
		f = sign | (uint32_t(127 - 15 - e) << 23) | ((mant & 0x3FF) << 13);
	}
	float out;
	std::memcpy(&out, &f, 4);
	return out;
}

static Variant f16_read(PackedByteArray halves_pba) {
	std::string err;
	if (!ensure_rd(err)) {
		return text(err);
	}
	const std::vector<uint8_t> hb = halves_pba.fetch();
	if (hb.size() != 128) {
		return text("FAIL need 64 halves (128 bytes)");
	}
	const avbd_kernels::Entry *k = avbd_kernels::find("half_load");
	if (!k) {
		return text("FAIL half_load not embedded");
	}
	::RID shader = g_dev.shader_from_spirv(k->bytes, k->size);
	if (!shader.index) {
		return text("FAIL shader_create_from_spirv(half_load): " + g_dev.error());
	}
	::RID pipe = g_dev.compute_pipeline(shader);
	::RID w = g_dev.storage_buffer(hb.size(), hb.data());
	::RID o = g_dev.storage_buffer(64 * 4);
	::RID set = pipe.index ? g_dev.uniform_set(shader,
									 { { 0, rdc::UNIFORM_TYPE_STORAGE_BUFFER, w }, { 1, rdc::UNIFORM_TYPE_STORAGE_BUFFER, o } })
						   : ::RID();
	if (!pipe.index || !set.index) {
		return text("FAIL " + g_dev.error());
	}
	g_dev.list_begin();
	g_dev.bind_pipeline(pipe);
	g_dev.bind_uniform_set(set);
	g_dev.dispatch(1);
	g_dev.list_end();
	g_dev.submit();
	g_dev.sync();
	const std::vector<uint8_t> ob = g_dev.buffer_get(o);
	g_dev.free_rid(set);
	g_dev.free_rid(o);
	g_dev.free_rid(w);
	g_dev.free_rid(pipe);
	g_dev.free_rid(shader);
	if (ob.size() < 256) {
		return text("FAIL short readback");
	}
	int bad = 0, nan_ok = 0;
	std::string first_bad;
	for (int i = 0; i < 64; ++i) {
		uint16_t h;
		std::memcpy(&h, &hb[2 * i], 2);
		const float want = half_to_float(h) * 2.0f;
		float got;
		std::memcpy(&got, &ob[4 * i], 4);
		const bool both_nan = want != want && got != got;
		if (both_nan) {
			++nan_ok;
			continue;
		}
		if (bitsf(want) != bitsf(got)) {
			if (bad == 0) {
				char b[96];
				std::snprintf(b, sizeof b, " first_bad=[%d] h=%04x want=%08x got=%08x", i, h, bitsf(want), bitsf(got));
				first_bad = b;
			}
			++bad;
		}
	}
	return text("n=64 mismatches=" + std::to_string(bad) + " nan_pairs=" + std::to_string(nan_ok) + first_bad);
}

// The reference-hungry shape, for the references_max control: n uniform sets
// created (and freed) in one call. Each creation instantiates RDUniform
// objects and returns a RID, and those are scoped to the call.
static Variant refs_usets(int64_t n) {
	if (!g_rc_ok) {
		return text("FAIL refs_setup first");
	}
	int64_t made = 0;
	for (int64_t i = 0; i < n; ++i) {
		::RID s = g_dev.uniform_set(g_sax.shader,
				{ { 0, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, g_pp[0].params }, { 1, rdc::UNIFORM_TYPE_STORAGE_BUFFER, g_pp[0].ones },
						{ 2, rdc::UNIFORM_TYPE_STORAGE_BUFFER, g_pp[0].a }, { 3, rdc::UNIFORM_TYPE_STORAGE_BUFFER, g_pp[0].b } });
		if (!s.index) {
			break;
		}
		g_dev.free_rid(s);
		++made;
	}
	return text("uniform_sets made=" + std::to_string(made) + " of " + std::to_string(n));
}

// Recovery: a vmcall killed mid-recording (references_max, a trap) leaves
// the compute list open, and every later list_begin/buffer_update on the
// device is refused until it is ended.
static Variant p_list_end() {
	if (!g_rd.is_valid()) {
		return text("no device");
	}
	g_dev.list_end();
	return text("list ended");
}

// ---------------------------------------------------------------------------
// 15. ggml-cpu in the guest, and the Zfh control.

static Variant ggml_probe(int64_t n) {
#ifdef PROBES_WITH_GGML
	return text(ggml_probe_run(int(n)));
#else
	(void)n;
	return text("SKIP probes.elf built without ggml (GGML_SRC not found)");
#endif
}

static Variant zfh_probe() {
	return Variant(double(zfh_probe_run()));
}

int main() {
	ADD_API_FUNCTION(p_exceptions, "String", "bool do_throw", "Throw/catch through std::function");
	ADD_API_FUNCTION(p_fenv, "String", "", "1/3 under FE_UPWARD and FE_DOWNWARD");
	ADD_API_FUNCTION(p_file, "String", "String path", "Read a file with ifstream and fopen");
	ADD_API_FUNCTION(p_threads, "String", "", "hardware_concurrency, spawn and join");
	ADD_API_FUNCTION(p_spin, "int", "int n", "n LCG steps; returns the state");
	ADD_API_FUNCTION(p_alloc, "String", "int mb", "malloc, touch and free mb MiB");
	ADD_API_FUNCTION(echo_f, "float", "float x", "Echo a float");
	ADD_API_FUNCTION(echo_i, "int", "int x", "Echo an int");
	ADD_API_FUNCTION(echo_b, "bool", "bool x", "Echo a bool");
	ADD_API_FUNCTION(echo_s, "String", "String s", "Echo a String through std::string");
	ADD_API_FUNCTION(echo_pf32, "PackedFloat32Array", "PackedFloat32Array a", "Echo floats through std::vector");
	ADD_API_FUNCTION(echo_pb, "PackedByteArray", "PackedByteArray a", "Echo bytes through std::vector");
	ADD_API_FUNCTION(f_bits, "int", "float x", "The IEEE bits of x as the guest saw it");
	ADD_API_FUNCTION(echo_var, "Variant", "Variant v", "Echo a boxed Variant");
	ADD_API_FUNCTION(p_hold, "String", "int mb", "Allocate and keep mb MiB");
	ADD_API_FUNCTION(p_release, "String", "", "Free the held allocation");
	ADD_API_FUNCTION(p_rd, "Object", "", "The guest-held local RenderingDevice");
	ADD_API_FUNCTION(fib_start, "String", "", "Start the 100-round GPU job on a fiber");
	ADD_API_FUNCTION(fib_pump, "String", "", "Resume the fiber once");
	ADD_API_FUNCTION(sm_start, "String", "", "Start the same job as a state machine");
	ADD_API_FUNCTION(sm_pump, "String", "", "Advance the state machine once");
	ADD_API_FUNCTION(big_buffer, "String", "int bytes, bool direct", "Empty buffer + clear + whole-buffer saxpby + readback");
	ADD_API_FUNCTION(refs_usets, "String", "int n", "Create and free n uniform sets in one call");
	ADD_API_FUNCTION(p_list_end, "String", "", "End a compute list a failed call left open");
	ADD_API_FUNCTION(refs_setup, "String", "", "Four +1 counters for refs_run");
	ADD_API_FUNCTION(refs_run, "String", "int n, bool aliased", "n dispatches x 3 binds, barrier every 4th, one submit");
	ADD_API_FUNCTION(f16_read, "String", "PackedByteArray halves", "64 halves through the Lean half_load kernel");
	ADD_API_FUNCTION(ggml_probe, "String", "int n", "ggml-cpu n^3 f16xf32 mul_mat + soft_max checksum");
	ADD_API_FUNCTION(zfh_probe, "float", "", "Half arithmetic compiled with Zfh");
	halt();
}
