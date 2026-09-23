// dress_on.elf -- the guest's public surface.
//
// Every ADD_API_FUNCTION here is reachable from GDScript as
// Sandbox.vmcall(name, ...), and project/main.gd wraps each one so an MCP
// client can reach it with no argument marshalling. It exposes the GPU
// layer's own Stage 1 probes. Each later stage is its own ELF in its own
// Sandbox node (AGENTS.md rule 6): the drape is guest/drape/main.cpp.

#include <api.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "rd_compute.h"

// The one device, held across vmcalls. rd_probe reports whether it found it
// already open, which is the Stage 1 proof that a guest static can keep the
// RenderingDevice between calls.
static rdc::Device g_dev;
static std::vector<uint8_t> g_probe_spirv;

static Variant text(const std::string &s) {
	return Variant(String(s));
}

static int64_t now_us() {
	using namespace std::chrono;
	return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

static Variant rd_open() {
	const bool was_open = g_dev.ok();
	if (!g_dev.open()) {
		return text(g_dev.error());
	}
	return text(was_open ? "ok (already open)" : "ok");
}

static Variant rd_close() {
	g_dev.close();
	return text("closed");
}

// Gate 0A's probe on top of rd_compute: dispatch a kernel that writes
// 0x00C0FFEE and read it back. The constant is deliberate -- zero or a small
// integer could be uninitialised memory.
static Variant rd_probe(PackedByteArray spirv) {
	const bool reused = g_dev.ok();
	if (!g_dev.open()) {
		return text(g_dev.error());
	}
	rdc::Device &d = g_dev;

	::RID shader = d.shader_from_spirv(spirv);
	if (shader.index == 0) {
		return text(d.error());
	}
	::RID buf = d.storage_buffer(4);
	if (buf.index == 0) {
		return text(d.error());
	}
	::RID uset = d.uniform_set(shader, { { 0, rdc::UNIFORM_TYPE_STORAGE_BUFFER, buf } });
	if (uset.index == 0) {
		return text(d.error());
	}
	::RID pipe = d.compute_pipeline(shader);
	if (pipe.index == 0) {
		return text(d.error());
	}

	d.list_begin();
	d.bind_pipeline(pipe);
	d.bind_uniform_set(uset);
	d.dispatch(1);
	d.list_end();
	d.submit();
	d.sync();

	std::vector<uint8_t> out = d.buffer_get(buf);
	d.free_rid(uset);
	d.free_rid(pipe);
	d.free_rid(shader);
	d.free_rid(buf);
	if (out.size() < 4) {
		return text("FAIL at buffer_get_data: short read");
	}
	const uint32_t got = uint32_t(out[0]) | (uint32_t(out[1]) << 8) | (uint32_t(out[2]) << 16) |
			(uint32_t(out[3]) << 24);
	if (got != 0x00C0FFEEu) {
		char b[96];
		std::snprintf(b, sizeof b, "FAIL: dispatched but read back 0x%08X", got);
		return text(b);
	}
	return text(std::string("PASS: GPU compute reached from the guest") +
			(reused ? " (device held from an earlier vmcall)" : " (device opened in this vmcall)"));
}

// The boundary-cost measurement. `spirv` is the accumulate kernel (outBuf[0]
// += 1). Records n_submit compute lists of n_dispatch dispatches each --
// binding pipeline and uniform set per dispatch, as a real kernel sequence
// would -- and reads the count back. The count must equal
// n_dispatch * n_submit; the per-dispatch and per-submit costs fall out of
// running it at (1,1), (n,1) and (1,n). The host times the whole vmcall around
// this; the guest clock is reported beside it.
static Variant rd_bench(PackedByteArray spirv, int n_dispatch, int n_submit, bool barrier) {
	if (n_dispatch < 1 || n_submit < 1) {
		return text("FAIL: n_dispatch and n_submit must be >= 1");
	}
	if (!g_dev.open()) {
		return text(g_dev.error());
	}
	rdc::Device &d = g_dev;

	// The first clock read in a vmcall comes back stale (setup_us went
	// negative by ~988 s on every call after the first); one throwaway read
	// warms it. The host timing around the vmcall is the authoritative one.
	(void)now_us();
	const int64_t t0 = now_us();
	::RID shader = d.shader_from_spirv(spirv);
	::RID buf = d.storage_buffer(4);
	::RID uset = shader.index && buf.index
			? d.uniform_set(shader, { { 0, rdc::UNIFORM_TYPE_STORAGE_BUFFER, buf } })
			: ::RID();
	::RID pipe = shader.index ? d.compute_pipeline(shader) : ::RID();
	if (!shader.index || !buf.index || !uset.index || !pipe.index) {
		return text(d.error());
	}
	const int64_t t1 = now_us();

	// Per-call-type time, so a slow call is named rather than inferred.
	int64_t us_bind_pipe = 0, us_bind_set = 0, us_dispatch = 0, us_barrier = 0, us_submit = 0;
	for (int s = 0; s < n_submit; ++s) {
		d.list_begin();
		for (int i = 0; i < n_dispatch; ++i) {
			int64_t a = now_us();
			d.bind_pipeline(pipe);
			int64_t b2 = now_us();
			d.bind_uniform_set(uset);
			int64_t c = now_us();
			d.dispatch(1);
			int64_t e = now_us();
			if (barrier && i + 1 < n_dispatch) {
				d.barrier();
			}
			int64_t f = now_us();
			us_bind_pipe += b2 - a;
			us_bind_set += c - b2;
			us_dispatch += e - c;
			us_barrier += f - e;
		}
		int64_t g = now_us();
		d.list_end();
		d.submit();
		d.sync();
		us_submit += now_us() - g;
	}
	const int64_t t2 = now_us();

	std::vector<uint8_t> out = d.buffer_get(buf);
	const int64_t t3 = now_us();
	d.free_rid(uset);
	d.free_rid(pipe);
	d.free_rid(shader);
	d.free_rid(buf);

	if (out.size() < 4) {
		return text("FAIL at buffer_get_data: short read");
	}
	const uint32_t got = uint32_t(out[0]) | (uint32_t(out[1]) << 8) | (uint32_t(out[2]) << 16) |
			(uint32_t(out[3]) << 24);
	const uint32_t expected = uint32_t(n_dispatch) * uint32_t(n_submit);
	const int64_t loop_us = t2 - t1;
	char b[384];
	std::snprintf(b, sizeof b,
			"%s value=%u expected=%u setup_us=%lld loop_us=%lld read_us=%lld us_per_dispatch=%.1f us_per_submit=%.1f"
			" | bind_pipe=%lld bind_set=%lld dispatch=%lld barrier=%lld submit=%lld",
			got == expected ? "OK" : "WRONG", got, expected, (long long)(t1 - t0), (long long)loop_us,
			(long long)(t3 - t2), double(loop_us) / double(n_dispatch * n_submit),
			double(loop_us) / double(n_submit), (long long)us_bind_pipe, (long long)us_bind_set,
			(long long)us_dispatch, (long long)us_barrier, (long long)us_submit);
	return text(b);
}

// rd_bench without any guest clock read: only the count and the host's
// timing around the vmcall. Separates "the guest clock costs this" from
// "the calls cost this".
static Variant rd_bench_quiet(PackedByteArray spirv, int n_dispatch, int n_submit, bool barrier) {
	if (n_dispatch < 1 || n_submit < 1) {
		return text("FAIL: n_dispatch and n_submit must be >= 1");
	}
	if (!g_dev.open()) {
		return text(g_dev.error());
	}
	rdc::Device &d = g_dev;
	::RID shader = d.shader_from_spirv(spirv);
	::RID buf = d.storage_buffer(4);
	::RID uset = shader.index && buf.index
			? d.uniform_set(shader, { { 0, rdc::UNIFORM_TYPE_STORAGE_BUFFER, buf } })
			: ::RID();
	::RID pipe = shader.index ? d.compute_pipeline(shader) : ::RID();
	if (!shader.index || !buf.index || !uset.index || !pipe.index) {
		return text(d.error());
	}
	for (int s = 0; s < n_submit; ++s) {
		d.list_begin();
		for (int i = 0; i < n_dispatch; ++i) {
			d.bind_pipeline(pipe);
			d.bind_uniform_set(uset);
			d.dispatch(1);
			if (barrier && i + 1 < n_dispatch) {
				d.barrier();
			}
		}
		d.list_end();
		d.submit();
		d.sync();
	}
	std::vector<uint8_t> out = d.buffer_get(buf);
	d.free_rid(uset);
	d.free_rid(pipe);
	d.free_rid(shader);
	d.free_rid(buf);
	if (out.size() < 4) {
		return text("FAIL at buffer_get_data: short read");
	}
	const uint32_t got = uint32_t(out[0]) | (uint32_t(out[1]) << 8) | (uint32_t(out[2]) << 16) |
			(uint32_t(out[3]) << 24);
	const uint32_t expected = uint32_t(n_dispatch) * uint32_t(n_submit);
	return text(std::string(got == expected ? "OK" : "WRONG") + " value=" + std::to_string(got) +
			" expected=" + std::to_string(expected));
}

// Boundary-cost instrument, timed by the HOST around the vmcall (the guest
// clock is not trusted: its readings jump between two time bases). `kind`
// picks one call type and `n` repeats it; the host divides. Kinds:
//   ticks    Time.get_ticks_usec()               a trivial singleton call
//   limit    rd.limit_get(0)                      a trivial RenderingDevice call
//   clock    std::chrono::steady_clock::now()     the guest clock syscall
//   bind     compute_list_bind_compute_pipeline   inside one list
//   barrier  compute_list_add_barrier             inside one list
//   dispatch compute_list_dispatch(1,1,1)         inside one list, no barrier
//   submit   list_begin/end + submit + sync       n empty lists
static Variant rd_calls(String kind_s, int n) {
	const std::string kind = kind_s.utf8();
	if (n < 1) {
		return text("FAIL: n must be >= 1");
	}
	if (kind == "ticks") {
		Object t("Time");
		int64_t last = 0;
		for (int i = 0; i < n; ++i) {
			last = int64_t(t.call("get_ticks_usec"));
		}
		return text(kind + " n=" + std::to_string(n) + " last=" + std::to_string(last));
	}
	if (kind == "clock") {
		int64_t last = 0;
		for (int i = 0; i < n; ++i) {
			last = now_us();
		}
		return text(kind + " n=" + std::to_string(n) + " last=" + std::to_string(last));
	}
	if (!g_dev.open()) {
		return text(g_dev.error());
	}
	rdc::Device &d = g_dev;
	if (kind == "limit") {
		int64_t last = 0;
		for (int i = 0; i < n; ++i) {
			last = d.limit_get(0);
		}
		return text(kind + " n=" + std::to_string(n) + " last=" + std::to_string(last));
	}
	if (kind == "submit") {
		for (int i = 0; i < n; ++i) {
			d.list_begin();
			d.list_end();
			d.submit();
			d.sync();
		}
		return text(kind + " n=" + std::to_string(n));
	}
	// Setup/teardown kinds: create + free, n times.
	if (kind == "buffer") {
		for (int i = 0; i < n; ++i) {
			d.free_rid(d.storage_buffer(4));
		}
		return text(kind + " n=" + std::to_string(n));
	}
	if (kind == "shader") {
		for (int i = 0; i < n; ++i) {
			d.free_rid(d.shader_from_spirv(g_probe_spirv.data(), g_probe_spirv.size()));
		}
		return text(kind + " n=" + std::to_string(n));
	}
	if (kind == "shader-pba") {
		// Same, but the PackedByteArray is built once outside the loop.
		PackedByteArray pba(g_probe_spirv.data(), g_probe_spirv.size());
		for (int i = 0; i < n; ++i) {
			d.free_rid(d.shader_from_spirv(pba));
		}
		return text(kind + " n=" + std::to_string(n));
	}
	if (kind == "pipeline" || kind == "uset" || kind == "readback" || kind == "instantiate") {
		::RID shader = d.shader_from_spirv(g_probe_spirv.data(), g_probe_spirv.size());
		::RID buf = d.storage_buffer(4);
		if (kind == "pipeline") {
			for (int i = 0; i < n; ++i) {
				d.free_rid(d.compute_pipeline(shader));
			}
		} else if (kind == "uset") {
			for (int i = 0; i < n; ++i) {
				d.free_rid(d.uniform_set(shader, { { 0, rdc::UNIFORM_TYPE_STORAGE_BUFFER, buf } }));
			}
		} else if (kind == "readback") {
			size_t total = 0;
			for (int i = 0; i < n; ++i) {
				total += d.buffer_get(buf).size();
			}
			d.free_rid(shader);
			d.free_rid(buf);
			return text(kind + " n=" + std::to_string(n) + " bytes=" + std::to_string(total));
		} else {
			for (int i = 0; i < n; ++i) {
				Object un = ClassDB::instantiate("RDUniform");
				un.voidcall("set_binding", int64_t(i));
			}
		}
		d.free_rid(shader);
		d.free_rid(buf);
		return text(kind + " n=" + std::to_string(n));
	}
	// The in-list kinds need a pipeline to bind.
	::RID shader = d.shader_from_spirv(g_probe_spirv);
	::RID buf = d.storage_buffer(4);
	::RID uset = d.uniform_set(shader, { { 0, rdc::UNIFORM_TYPE_STORAGE_BUFFER, buf } });
	::RID pipe = d.compute_pipeline(shader);
	if (!shader.index || !buf.index || !uset.index || !pipe.index) {
		return text(d.error());
	}
	d.list_begin();
	d.bind_pipeline(pipe);
	d.bind_uniform_set(uset);
	if (kind == "bind") {
		for (int i = 0; i < n; ++i) {
			d.bind_pipeline(pipe);
		}
	} else if (kind == "barrier") {
		for (int i = 0; i < n; ++i) {
			d.barrier();
		}
	} else if (kind == "dispatch") {
		for (int i = 0; i < n; ++i) {
			d.dispatch(1);
		}
	} else {
		d.list_end();
		return text("FAIL: unknown kind " + kind);
	}
	d.list_end();
	d.submit();
	d.sync();
	d.free_rid(uset);
	d.free_rid(pipe);
	d.free_rid(shader);
	d.free_rid(buf);
	return text(kind + " n=" + std::to_string(n));
}

// The probe kernel's bytes, kept by rd_set_probe so rd_calls needs no
// PackedByteArray argument per call.
static Variant rd_set_probe(PackedByteArray spirv) {
	g_probe_spirv = spirv.fetch();
	return text("ok " + std::to_string(g_probe_spirv.size()) + " bytes");
}

// Where did it get to? Named steps, not booleans (gate_rd_compute.gd).
static Variant rd_last_step() {
	return text(g_dev.step());
}

int main() {
	ADD_API_FUNCTION(rd_open, "String", "", "Create and hold a local RenderingDevice");
	ADD_API_FUNCTION(rd_close, "String", "", "Free the held RenderingDevice");
	ADD_API_FUNCTION(rd_probe, "String", "PackedByteArray spirv",
			"Dispatch the 0x00C0FFEE probe kernel through rd_compute and read it back");
	ADD_API_FUNCTION(rd_bench, "String", "PackedByteArray spirv, int n_dispatch, int n_submit, bool barrier",
			"Time n_submit lists of n_dispatch accumulate dispatches; the count must match");
	ADD_API_FUNCTION(rd_bench_quiet, "String", "PackedByteArray spirv, int n_dispatch, int n_submit, bool barrier",
			"rd_bench with no guest clock reads; host-timed only");
	ADD_API_FUNCTION(rd_set_probe, "String", "PackedByteArray spirv", "Keep the probe kernel bytes for rd_calls");
	ADD_API_FUNCTION(rd_calls, "String", "String kind, int n",
			"Repeat one call kind n times (ticks|limit|clock|bind|barrier|dispatch|submit|buffer|shader|shader-pba|pipeline|uset|readback|instantiate); time it on the host");
	ADD_API_FUNCTION(rd_last_step, "String", "", "The last RenderingDevice step attempted");
	halt();
}
