// probes.elf -- Gate 0F: what the godot-sandbox runtime actually does.
//
// Each entry point answers one question that could break a later cut
// (exceptions, rounding modes, files, threads, limits, argument marshalling,
// the heap, a fiber across vmcalls, large RD buffers, references_max, ggml).
// The host side is project/gate_runtime.gd; it runs every probe with a
// control and writes gates/0f-runtime/results.txt. Nothing in here decides a
// verdict: the guest reports what it saw and the host compares.

#include <api.hpp>

#include <algorithm>
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
// 17. Aligned allocation. godot-sandbox's memalign fallback (vendored
// docker/api/native.cpp) used to return an already-freed block when 16
// malloc tries missed the alignment. n blocks at each of 64, 128 and 4096
// alignment, through posix_memalign, aligned_alloc, memalign and aligned
// operator new in turn, each filled with its own tag byte; a third of the
// steps free a random live block and every seventh reallocs one (which
// leaves it a plain block). At the end every live block must be aligned,
// intact (its tag in every byte) and disjoint from every other; then all are
// freed. upstream=true runs the same sequence through a copy of the old
// fallback as the control: it counts the calls that returned a freed block
// ("exhausted") and never frees those, so the host heap is not asked to
// free them twice.

extern "C" void *memalign(size_t alignment, size_t size);

static void *memalign_upstream(size_t alignment, size_t size, bool &exhausted) {
	exhausted = false;
	if (alignment <= 16) {
		return std::malloc(size);
	}
	void *list[16];
	size_t i = 0;
	void *result = nullptr;
	for (i = 0; i < 16; i++) {
		result = std::malloc(size);
		list[i] = result;
		const bool aligned = ((uintptr_t)result % alignment) == 0;
		if (result && aligned) {
			break;
		} else if (result) {
			std::free(result);
			list[i] = std::malloc(16);
		} else {
			result = nullptr;
			break;
		}
	}
	for (size_t j = 0; j < i; j++) {
		std::free(list[j]);
	}
	exhausted = (i == 16);
	return result;
}

struct AlignedBlock {
	uintptr_t a;
	size_t n;
	size_t align; // 1 after a realloc
	int kind; // 0 posix_memalign, 1 aligned_alloc, 2 memalign, 3 aligned new, 4 plain (realloc'd)
	uint8_t tag;
	bool poisoned; // upstream control: returned already freed
};

static bool block_intact(const AlignedBlock &b) {
	const uint8_t *p = reinterpret_cast<const uint8_t *>(b.a);
	for (size_t i = 0; i < b.n; i++) {
		if (p[i] != b.tag) {
			return false;
		}
	}
	return true;
}

static void block_free(const AlignedBlock &b) {
	if (b.poisoned) {
		return;
	}
	if (b.kind == 3) {
		::operator delete(reinterpret_cast<void *>(b.a), std::align_val_t(b.align));
	} else {
		std::free(reinterpret_cast<void *>(b.a));
	}
}

static Variant p_memalign(int64_t n, bool upstream) {
	static const size_t aligns[3] = { 64, 128, 4096 };
	uint64_t s = 0x9E3779B97F4A7C15ull;
	auto rnd = [&s]() {
		s = s * 6364136223846793005ull + 1442695040888963407ull;
		return uint32_t(s >> 33);
	};
	std::vector<AlignedBlock> live;
	live.reserve(size_t(n) * 3);
	int64_t made = 0, freed = 0, reallocs = 0, nulls = 0, misaligned = 0, exhausted = 0, corrupt = 0;
	uint8_t tag = 0;
	for (size_t al : aligns) {
		for (int64_t i = 0; i < n; i++) {
			size_t sz = 1 + rnd() % (al == 4096 ? 6000 : 700);
			const int kind = upstream ? 2 : int(i % 4);
			void *p = nullptr;
			bool ex = false;
			if (upstream) {
				p = memalign_upstream(al, sz, ex);
			} else if (kind == 0) {
				if (posix_memalign(&p, al, sz) != 0) {
					p = nullptr;
				}
			} else if (kind == 1) {
				sz = (sz + al - 1) / al * al; // C11: a multiple of the alignment
				p = aligned_alloc(al, sz);
			} else if (kind == 2) {
				p = memalign(al, sz);
			} else {
				p = ::operator new(sz, std::align_val_t(al));
			}
			if (p == nullptr) {
				nulls++;
				continue;
			}
			made++;
			exhausted += ex ? 1 : 0;
			if (uintptr_t(p) % al != 0) {
				misaligned++;
			}
			tag = uint8_t(tag == 255 ? 1 : tag + 1);
			std::memset(p, tag, sz);
			live.push_back({ uintptr_t(p), sz, al, kind, tag, ex });
			const uint32_t r = rnd();
			if (r % 3 == 0 && !live.empty()) {
				const size_t j = rnd() % live.size();
				if (!live[j].poisoned) {
					corrupt += block_intact(live[j]) ? 0 : 1;
					block_free(live[j]);
					live[j] = live.back();
					live.pop_back();
					freed++;
				}
			} else if (!upstream && r % 7 == 1 && !live.empty()) {
				const size_t j = rnd() % live.size();
				AlignedBlock &b = live[j];
				if (b.kind != 3) {
					corrupt += block_intact(b) ? 0 : 1;
					const size_t nn = b.n + 1 + rnd() % 300;
					void *q = std::realloc(reinterpret_cast<void *>(b.a), nn);
					if (q == nullptr) {
						nulls++;
					} else {
						const uint8_t *qp = static_cast<const uint8_t *>(q);
						for (size_t k = 0; k < b.n; k++) {
							if (qp[k] != b.tag) {
								corrupt++;
								break;
							}
						}
						std::memset(q, b.tag, nn);
						b.a = uintptr_t(q);
						b.n = nn;
						b.align = 1;
						b.kind = 4;
						reallocs++;
					}
				}
			}
		}
	}
	int64_t intact_bad = 0;
	for (const AlignedBlock &b : live) {
		intact_bad += block_intact(b) ? 0 : 1;
	}
	std::vector<AlignedBlock> sorted = live;
	std::sort(sorted.begin(), sorted.end(), [](const AlignedBlock &x, const AlignedBlock &y) { return x.a < y.a; });
	int64_t overlaps = 0;
	for (size_t k = 1; k < sorted.size(); k++) {
		if (sorted[k - 1].a + sorted[k - 1].n > sorted[k].a) {
			overlaps++;
		}
	}
	const size_t at_end = live.size();
	for (const AlignedBlock &b : live) {
		block_free(b);
	}
	return text(std::string(upstream ? "upstream" : "fixed") + " n=" + std::to_string(n) +
			" made=" + std::to_string(made) + " freed_midway=" + std::to_string(freed) +
			" reallocs=" + std::to_string(reallocs) + " live_at_end=" + std::to_string(at_end) +
			" misaligned=" + std::to_string(misaligned) + " overlaps=" + std::to_string(overlaps) +
			" corrupt=" + std::to_string(corrupt + intact_bad) + " nulls=" + std::to_string(nulls) +
			" exhausted=" + std::to_string(exhausted));
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
//
// Every RenderingDevice call goes through rdc::Device (AGENTS.md: the host's
// 32-slot method-name cache is keyed by the name's guest address, and
// rd_compute places its names so they do not collide). rdc::Device also makes
// every RID it returns permanent: a returned RID is a per-vmcall scoped
// Variant (probe 11b shows what happens without that).

static rdc::Device g_dev;

static bool ensure_rd(std::string &err) {
	if (g_dev.ok()) {
		return true;
	}
	if (!g_dev.open()) {
		err = g_dev.error();
		return false;
	}
	return true;
}

static Variant p_rd() {
	std::string err;
	if (!ensure_rd(err)) {
		return text(err);
	}
	return Variant(g_dev.object());
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
	g_sax.shader = (g_dev.shader_from_spirv(k->bytes, k->size));
	g_sax.pipeline = g_sax.shader.index ? (g_dev.compute_pipeline(g_sax.shader)) : ::RID();
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
	c.params = (g_dev.uniform_buffer(sizeof p, &p));
	c.ones = (g_dev.storage_buffer(4, &one));
	c.dst = (g_dev.storage_buffer(4));
	if (!c.params.index || !c.ones.index || !c.dst.index) {
		err = g_dev.error();
		return false;
	}
	c.set = (g_dev.uniform_set(g_sax.shader,
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

	::RID buf = g_dev.storage_buffer_uninit(size_t(bytes));
	if (!buf.index) {
		return text(r + " FAIL storage_buffer_create (empty): " + g_dev.error());
	}
	const bool cleared = g_dev.buffer_clear(buf, 0, size_t(bytes));
	r += std::string(" clear_err=") + (cleared ? "0" : "1");

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
		g_dev.buffer_copy(buf, stage, 4, size_t(offs[i]), size_t(4 * i));
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
	c.params = (g_dev.uniform_buffer(sizeof p, &p));
	c.ones = (g_dev.storage_buffer(4, &one));
	c.a = (g_dev.storage_buffer(4));
	c.b = (g_dev.storage_buffer(4));
	c.set_ab = (g_dev.uniform_set(g_sax.shader,
			{ { 0, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, c.params }, { 1, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.ones },
					{ 2, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.a }, { 3, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.b } }));
	c.set_ba = (g_dev.uniform_set(g_sax.shader,
			{ { 0, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, c.params }, { 1, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.ones },
					{ 2, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.b }, { 3, rdc::UNIFORM_TYPE_STORAGE_BUFFER, c.a } }));
	if (!c.set_ab.index || !c.set_ba.index) {
		err = g_dev.error();
		return false;
	}
	return true;
}

static bool g_rc_made[4] = {};

// Counter k (aliased and ping-pong), made on first use, then zeroed. One per
// call keeps each vmcall's scoped references small (probe 13 at 100).
static std::string refs_one(int k) {
	std::string err;
	if (!ensure_saxpby(err)) {
		return err;
	}
	if (!g_rc_made[k]) {
		if (!make_counter(g_rc[k], err) || !make_pingpong(g_pp[k], err)) {
			return err;
		}
		g_rc_made[k] = true;
	}
	g_rc_ok = g_rc_made[0] && g_rc_made[1] && g_rc_made[2] && g_rc_made[3];
	const float z = 0.0f;
	if (!g_dev.buffer_update(g_rc[k].dst, 0, 4, &z) || !g_dev.buffer_update(g_pp[k].a, 0, 4, &z) ||
			!g_dev.buffer_update(g_pp[k].b, 0, 4, &z)) {
		return "FAIL buffer_update refused: " + g_dev.error();
	}
	return "ok";
}

static Variant refs_setup_one(int64_t k) {
	if (k < 0 || k > 3) {
		return text("FAIL k in [0, 3]");
	}
	return text(refs_one(int(k)));
}

static Variant refs_setup() {
	for (int k = 0; k < 4; ++k) {
		const std::string r = refs_one(k);
		if (r != "ok") {
			return text(r);
		}
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
	if (!g_dev.ok()) {
		return text("no device");
	}
	g_dev.list_end();
	return text("list ended");
}

// ---------------------------------------------------------------------------
// 11b. A RID kept in a guest static across vmcalls, with and without
// permanence. rid_hold() makes a 16-byte buffer and keeps its RID; rid_use(),
// a later vmcall, reads it back. Without permanence the RID is the host's
// per-call scoped index, and the later call resolves it to something else or
// nothing (the first fiber run's failure). The non-permanent buffer cannot be
// freed afterwards (its RID no longer names it); it leaks 16 bytes by design.

static ::RID g_held;
static bool g_held_perm = true;
static const float kHeld[4] = { 1.25f, 2.5f, 3.75f, 5.0f };

static Variant rid_hold(bool permanent) {
	std::string err;
	if (!ensure_rd(err)) {
		return text(err);
	}
	if (g_held.index && g_held_perm) {
		g_dev.free_rid(g_held);
	}
	g_dev.set_permanent_rids(permanent);
	g_held = g_dev.storage_buffer(sizeof kHeld, kHeld);
	g_dev.set_permanent_rids(true);
	g_held_perm = permanent;
	// Same call: the RID works either way.
	std::vector<uint8_t> now = g_dev.buffer_get(g_held, 0, 16);
	const bool same_call_ok = now.size() == 16 && std::memcmp(now.data(), kHeld, 16) == 0;
	return text("held index=" + std::to_string(g_held.index) + " permanent=" + (permanent ? "yes" : "no") +
			" same_call_read=" + (same_call_ok ? "exact" : "WRONG"));
}

static Variant rid_use() {
	if (!g_held.index) {
		return text("FAIL nothing held");
	}
	const int64_t idx = g_held.index;
	std::vector<uint8_t> got = g_dev.buffer_get(g_held, 0, 16);
	const bool exact = got.size() == 16 && std::memcmp(got.data(), kHeld, 16) == 0;
	if (g_held_perm) {
		g_dev.free_rid(g_held);
	}
	g_held = ::RID();
	return text("index=" + std::to_string(idx) + " later_call_read bytes=" + std::to_string(got.size()) + " " +
			(exact ? "exact" : "WRONG"));
}

// ---------------------------------------------------------------------------
// 16. One set-0 uniform set under several pipelines (-preserve-params), and
// in-place ops across barriers. The kernels are Lean's Probes.Set0 (layout:
// b0 params, b1-b3 sources, b4 destination); gen.sh embeds each compiled with
// -preserve-params and, as the control, without (<name>_stripped).

struct ProbeParams {
	uint32_t n;
	float alpha;
	uint32_t pad[2];
};

struct Kern {
	::RID shader, pipeline;
};

static bool make_kern(const std::string &name, Kern &k, std::string &err) {
	const avbd_kernels::Entry *e = avbd_kernels::find(name.c_str());
	if (!e) {
		err = "FAIL " + name + " not embedded";
		return false;
	}
	k.shader = g_dev.shader_from_spirv(e->bytes, e->size);
	k.pipeline = k.shader.index ? g_dev.compute_pipeline(k.shader) : ::RID();
	if (!k.pipeline.index) {
		err = "FAIL " + name + ": " + g_dev.error();
		return false;
	}
	return true;
}

static void free_kern(Kern &k) {
	g_dev.free_rid(k.pipeline);
	g_dev.free_rid(k.shader);
	k = Kern{};
}

static std::vector<float> read_floats(::RID buf, size_t n) {
	std::vector<uint8_t> b = g_dev.buffer_get(buf, 0, n * 4);
	std::vector<float> f(n, -1.0f);
	if (b.size() >= n * 4) {
		std::memcpy(f.data(), b.data(), n * 4);
	}
	return f;
}

static std::vector<rdc::Binding> set0(::RID params, ::RID s0, ::RID s1, ::RID s2, ::RID dst) {
	return { { 0, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, params }, { 1, rdc::UNIFORM_TYPE_STORAGE_BUFFER, s0 },
		{ 2, rdc::UNIFORM_TYPE_STORAGE_BUFFER, s1 }, { 3, rdc::UNIFORM_TYPE_STORAGE_BUFFER, s2 },
		{ 4, rdc::UNIFORM_TYPE_STORAGE_BUFFER, dst } };
}

static void run_one(const Kern &k, ::RID set, uint32_t n) {
	g_dev.list_begin();
	g_dev.bind_pipeline(k.pipeline);
	g_dev.bind_uniform_set(set);
	g_dev.dispatch(rdc::Device::groups_for(n, 64));
	g_dev.list_end();
	g_dev.submit();
	g_dev.sync(); // a probe, not a frame loop: timed as one call by the host
}

static std::string count_exact(const std::vector<float> &got, const std::vector<float> &want) {
	int ok = 0;
	for (size_t i = 0; i < got.size(); ++i) {
		ok += bitsf(got[i]) == bitsf(want[i]);
	}
	return std::to_string(ok) + "/" + std::to_string(got.size());
}

static Variant set0_share(String variant_s) {
	std::string err;
	if (!ensure_rd(err)) {
		return text(err);
	}
	// "" = -O0 -preserve-params, "_stripped" = -O0, "_o1pp" = -O1 -preserve-params
	const std::string suffix = variant_s.utf8();
	if (suffix != "" && suffix != "_stripped" && suffix != "_o1pp") {
		return text("FAIL variant is \"\", \"_stripped\" or \"_o1pp\"");
	}
	Kern add, scale;
	if (!make_kern("probe_add" + suffix, add, err) || !make_kern("probe_scale" + suffix, scale, err)) {
		free_kern(add);
		return text(err);
	}
	const uint32_t n = 256;
	std::vector<float> x(n), y(n), zero(n, 0.0f), sum(n), x3(n);
	for (uint32_t i = 0; i < n; ++i) {
		x[i] = float(i) + 1.0f;
		y[i] = 1000.0f + float(i);
		sum[i] = x[i] + y[i];
		x3[i] = x[i] * 3.0f;
	}
	const ProbeParams pp{ n, 3.0f, { 0, 0 } };
	::RID params = g_dev.uniform_buffer(sizeof pp, &pp);
	::RID X = g_dev.storage_buffer(n * 4, x.data());
	::RID Y = g_dev.storage_buffer(n * 4, y.data());
	::RID Z = g_dev.storage_buffer(n * 4);
	::RID D = g_dev.storage_buffer(n * 4);
	std::string r = "kernels=probe_add" + suffix + ",probe_scale" + suffix;

	// A set built against probe_add, used by probe_add and then by probe_scale.
	::RID set = g_dev.uniform_set(add.shader, set0(params, X, Y, Z, D));
	if (!set.index) {
		r += " set_for_add=REFUSED(" + g_dev.error() + ")";
	} else {
		run_one(add, set, n);
		r += " add_own_set=" + count_exact(read_floats(D, n), sum);
		run_one(scale, set, n);
		const std::vector<float> d = read_floats(D, n);
		r += " scale_under_add_set=" + count_exact(d, x3);
		r += std::string(d == sum ? " (D unchanged: the dispatch was refused)" : "");
	}
	// The same kernel with a set built against its own shader: the arithmetic
	// works whatever the layout question says.
	g_dev.buffer_clear(D, 0, n * 4);
	::RID own = g_dev.uniform_set(scale.shader, set0(params, X, Y, Z, D));
	if (own.index) {
		run_one(scale, own, n);
		r += " scale_own_set=" + count_exact(read_floats(D, n), x3);
	} else {
		r += " scale_own_set=REFUSED(" + g_dev.error() + ")";
	}
	// In place, one dispatch: X bound read-only at b1 and read-write at b4.
	::RID ip = g_dev.uniform_set(scale.shader, set0(params, X, Y, Z, X));
	if (ip.index) {
		run_one(scale, ip, n);
		r += " inplace_b1_b4_one_dispatch=" + count_exact(read_floats(X, n), x3);
	} else {
		r += " inplace=REFUSED(" + g_dev.error() + ")";
	}
	g_dev.free_rid(ip);
	g_dev.free_rid(own);
	g_dev.free_rid(set);
	for (::RID b : { params, X, Y, Z, D }) {
		g_dev.free_rid(b);
	}
	free_kern(scale);
	free_kern(add);
	return text(r);
}

// rounds of "+1 on every element", one compute list, a barrier after every
// round, one submit. Each thread touches only its own element, so no
// dispatch races with itself; any loss is between rounds.
//   0 aliased     probe_add, X at b1 (read-only) and b4 (read-write): X = X + 1
//   1 rw_only     probe_acc, X at b4 only: X = X + 1 (same arithmetic)
//   2 pingpong    probe_add, A -> B then B -> A
//   3 ro_then_rw  per round: probe_scale reads X (b1) into a scratch S, then
//                 probe_acc adds 1 to X (b4); X is first seen read-only in
//                 the list (between two barriers)
//   4 rw_then_ro  the same two dispatches, probe_acc first
static Variant inplace_run(int64_t mode, int64_t rounds) {
	std::string err;
	if (!ensure_rd(err)) {
		return text(err);
	}
	static const char *kNames[5] = { "aliased", "rw_only", "pingpong", "ro_then_rw", "rw_then_ro" };
	if (mode < 0 || mode > 4 || rounds < 1) {
		return text("FAIL mode in [0, 4], rounds >= 1");
	}
	Kern add, acc, scale;
	if (!make_kern("probe_add", add, err) || !make_kern("probe_acc", acc, err) || !make_kern("probe_scale", scale, err)) {
		free_kern(acc);
		free_kern(add);
		return text(err);
	}
	const uint32_t n = 4096;
	const std::vector<float> ones(n, 1.0f);
	const ProbeParams pp{ n, 1.0f, { 0, 0 } };
	::RID params = g_dev.uniform_buffer(sizeof pp, &pp);
	::RID O = g_dev.storage_buffer(n * 4, ones.data());
	::RID A = g_dev.storage_buffer(n * 4);
	::RID B = g_dev.storage_buffer(n * 4);
	::RID Z = g_dev.storage_buffer(n * 4);
	// Modes 3 and 4 read X into a scratch S. One S would chain every round
	// through S's own write-after-write; 16 in rotation leave 16 consecutive
	// rounds ordered by X alone.
	const int kS = 16;
	::RID S[kS] = {};
	::RID sS[kS] = {};
	::RID s1 = ::RID(), s2 = ::RID();
	const Kern *k1 = nullptr, *k2 = nullptr;
	switch (mode) {
		case 0:
			s1 = g_dev.uniform_set(add.shader, set0(params, A, O, Z, A));
			k1 = &add;
			break;
		case 1:
			s1 = g_dev.uniform_set(acc.shader, set0(params, O, Z, Z, A));
			k1 = &acc;
			break;
		case 2:
			s1 = g_dev.uniform_set(add.shader, set0(params, A, O, Z, B));
			s2 = g_dev.uniform_set(add.shader, set0(params, B, O, Z, A));
			k1 = &add;
			k2 = &add;
			break;
		case 3:
		case 4:
			for (int j = 0; j < kS; ++j) {
				S[j] = g_dev.storage_buffer(n * 4);
				sS[j] = g_dev.uniform_set(scale.shader, set0(params, A, Z, Z, S[j]));
			}
			s2 = g_dev.uniform_set(acc.shader, set0(params, O, Z, Z, A));
			s1 = sS[0];
			k1 = &scale;
			k2 = &acc;
			break;
	}
	std::string r = std::string("mode=") + kNames[mode] + " rounds=" + std::to_string(rounds) + " elems=" + std::to_string(n);
	if (!s1.index || (k2 && !s2.index)) {
		r += " FAIL uniform_set: " + g_dev.error();
	} else {
		const uint32_t groups = rdc::Device::groups_for(n, 64);
		g_dev.list_begin();
		for (int64_t i = 0; i < rounds; ++i) {
			if (mode == 2) {
				// round i: A -> B on even rounds, B -> A on odd ones
				g_dev.bind_pipeline(add.pipeline);
				g_dev.bind_uniform_set(i % 2 == 0 ? s1 : s2);
				g_dev.dispatch(groups);
			} else if (mode >= 3) {
				// 3: read X (scale into S[i % 16]) then write X (acc)
				// 4: write X (acc) then read X (scale into S[i % 16])
				for (int step = 0; step < 2; ++step) {
					const bool read_step = (step == 0) == (mode == 3);
					g_dev.bind_pipeline(read_step ? scale.pipeline : acc.pipeline);
					g_dev.bind_uniform_set(read_step ? sS[i % kS] : s2);
					g_dev.dispatch(groups);
				}
			} else {
				g_dev.bind_pipeline(k1->pipeline);
				g_dev.bind_uniform_set(s1);
				g_dev.dispatch(groups);
			}
			if (i + 1 < rounds) {
				g_dev.barrier();
			}
		}
		g_dev.list_end();
		g_dev.submit();
		g_dev.sync();
		const ::RID out = (mode == 2 && rounds % 2 == 1) ? B : A;
		const std::vector<float> got = read_floats(out, n);
		int exact = 0;
		float lo = 1e30f, hi = -1e30f;
		for (float v : got) {
			exact += v == float(rounds);
			lo = v < lo ? v : lo;
			hi = v > hi ? v : hi;
		}
		char b[160];
		std::snprintf(b, sizeof b, " exact=%d/%u min=%g max=%g expect=%lld", exact, n, lo, hi, (long long)rounds);
		r += b;
	}
	g_dev.free_rid(s2);
	if (mode < 3) {
		g_dev.free_rid(s1);
	}
	for (int j = 0; j < kS; ++j) {
		g_dev.free_rid(sS[j]);
		g_dev.free_rid(S[j]);
	}
	for (::RID b : { params, O, A, B, Z }) {
		g_dev.free_rid(b);
	}
	free_kern(scale);
	free_kern(acc);
	free_kern(add);
	return text(r);
}

// Free the guest's device (and with it every buffer on it) before the host
// frees this Sandbox; the gate's fresh-sandbox arms would leak one each.
static Variant p_rd_close() {
	if (!g_dev.ok()) {
		return text("no device");
	}
	g_dev.close();
	g_sax = Saxpby{};
	g_rc_ok = false;
	for (bool &m : g_rc_made) {
		m = false;
	}
	return text("closed");
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
	ADD_API_FUNCTION(p_memalign, "String", "int n, bool upstream", "n aligned blocks at 64/128/4096, interleaved frees; overlap check");
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
	ADD_API_FUNCTION(p_rd_close, "String", "", "Free the guest's RenderingDevice");
	ADD_API_FUNCTION(refs_setup, "String", "", "Four +1 counters for refs_run");
	ADD_API_FUNCTION(refs_setup_one, "String", "int k", "Counter k for refs_run, one per call");
	ADD_API_FUNCTION(rid_hold, "String", "bool permanent", "Create a buffer and keep its RID in a static");
	ADD_API_FUNCTION(rid_use, "String", "", "Read the kept RID's buffer in a later vmcall");
	ADD_API_FUNCTION(set0_share, "String", "String variant", "One uniform set under two pipelines, and in place");
	ADD_API_FUNCTION(inplace_run, "String", "int mode, int rounds", "rounds of +1 in place across barriers");
	ADD_API_FUNCTION(refs_run, "String", "int n, bool aliased", "n dispatches x 3 binds, barrier every 4th, one submit");
	ADD_API_FUNCTION(f16_read, "String", "PackedByteArray halves", "64 halves through the Lean half_load kernel");
	ADD_API_FUNCTION(ggml_probe, "String", "int n", "ggml-cpu n^3 f16xf32 mul_mat + soft_max checksum");
	ADD_API_FUNCTION(zfh_probe, "float", "", "Half arithmetic compiled with Zfh");
	halt();
}
