// ggml_test.elf -- Gate 3 (G3.ops): ggml's test-backend-ops in the guest,
// ggml-rd against the in-guest ggml-cpu reference, driven by the pump.
//
// The host (project/gate_ggml_rd.gd through project/infer_host.gd) owns a
// local RenderingDevice and attaches it (ggml_attach); a job is started
// (ggml_ops_start runs vendor/ggml/tests/test-backend-ops.cpp's main with the
// given arguments; ggml_probe_start runs one of probes.cpp) and advanced by
// one ggml_pump per call: WAIT_GPU after every submit (the sync lands on the
// next frame, AGENTS.md rule 4), COOP while uniform sets are being made,
// READ/UPLOAD for host files, DONE/ERROR at the end. Everything the job
// prints (stdout and stderr) is captured for ggml_output() and still goes to
// the Godot console.
//
// Adaptations around test-backend-ops (no source change beyond the two in
// vendor/ggml/CITATION.cff):
//  - it is compiled with -Dmain=ggml_test_backend_ops_main,
//    -DGGML_GUEST_REGISTER_BACKENDS (ggml_guest_register_backends() below
//    replaces ggml_backend_load_all) and -DN_THREADS=1;
//  - the link wraps ggml_backend_init_by_type so every CPU backend it makes
//    runs one thread: the reference backend is created at
//    GGML_DEFAULT_N_THREADS (4), and guest threads are serialized (Gate 0C),
//    so ggml's spinning barriers would never be released.
#include <api.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "probes.h"
#include "pump/pump.h"
#include "rd_compute.h"

// test-backend-ops.cpp's main, renamed by -Dmain= (C++ linkage, like any renamed function).
int ggml_test_backend_ops_main(int argc, char **argv);

static rdc::Device g_dev;
static bool g_attached = false;

// --- the reference backend runs one thread ------------------------------------

extern "C" ggml_backend_t __real_ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char *params);
extern "C" ggml_backend_t __wrap_ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char *params) {
	ggml_backend_t b = __real_ggml_backend_init_by_type(type, params);
	if (b != nullptr && type == GGML_BACKEND_DEVICE_TYPE_CPU) {
		ggml_backend_cpu_set_n_threads(b, 1);
	}
	return b;
}

// --- backends: CPU is registered by ggml itself; RD here ---------------------

extern "C" void ggml_guest_register_backends(void) {
	static bool registered = false;
	if (!registered) {
		registered = true;
		ggml_backend_register(ggml_backend_rd_reg());
	}
	if (ggml_backend_reg_dev_count(ggml_backend_rd_reg()) == 0) {
		std::printf("ggml-rd: no RD device (no RenderingDevice was attached; headless?)\n");
	}
}

// --- output capture -------------------------------------------------------------

static std::string g_out;

static ssize_t capture_write(void *cookie, const char *buf, size_t n) {
	g_out.append(buf, n);
	const int fd = int(reinterpret_cast<intptr_t>(cookie));
	(void)!::write(fd, buf, n); // still to the Godot console
	return ssize_t(n);
}

static void capture_install() {
	static bool done = false;
	if (done) {
		return;
	}
	done = true;
	cookie_io_functions_t io{};
	io.write = capture_write;
	FILE *o = fopencookie(reinterpret_cast<void *>(intptr_t(1)), "w", io);
	FILE *e = fopencookie(reinterpret_cast<void *>(intptr_t(2)), "w", io);
	if (o != nullptr && e != nullptr) {
		std::setvbuf(o, nullptr, _IOLBF, 0);
		std::setvbuf(e, nullptr, _IONBF, 0);
		stdout = o;
		stderr = e;
	}
}

// --- hooks: the backend waits and yields through the pump ----------------------

static void hook_wait_gpu(void *) {
	pump::wait_gpu();
}

static void hook_coop(void *) {
	pump::coop();
}

static bool hook_upload(void *, const std::string &path, uint64_t file_offset, uint64_t bytes, ::RID rid,
		uint64_t dst_offset) {
	return pump::upload(path, file_offset, bytes, rid, dst_offset);
}

static void on_ggml_abort(const char *message) {
	std::fflush(stdout);
	pump::fail(std::string("ggml abort: ") + message);
}

// --- jobs -----------------------------------------------------------------------

struct OpsJob {
	std::vector<std::string> args;
	int rc = -1;
};
static OpsJob g_ops;

static void ops_job(void *) {
	std::vector<char *> argv;
	static char prog[] = "test-backend-ops";
	argv.push_back(prog);
	for (std::string &a : g_ops.args) {
		argv.push_back(a.data());
	}
	argv.push_back(nullptr);
	g_ops.rc = ggml_test_backend_ops_main(int(argv.size() - 1), argv.data());
	std::printf("ggml_test: test-backend-ops returned %d\n", g_ops.rc);
	std::printf("ggml_test: rd stats %s\n", ggml_backend_rd_stats().c_str());
	std::fflush(stdout);
}

static std::vector<std::string> split_ws(const std::string &s) {
	std::vector<std::string> r;
	size_t i = 0;
	while (i < s.size()) {
		while (i < s.size() && s[i] == ' ') {
			++i;
		}
		size_t j = i;
		while (j < s.size() && s[j] != ' ') {
			++j;
		}
		if (j > i) {
			r.push_back(s.substr(i, j - i));
		}
		i = j;
	}
	return r;
}

// "K=V K2=V2": set for this job; the GGML_RD_* switches not named are unset.
static void apply_env(const std::string &env) {
	for (const char *k : { "GGML_RD_FAULT", "GGML_RD_BARRIER_ALL", "GGML_RD_MAX_BUFFER_MB", "GGML_RD_SERIAL" }) {
		unsetenv(k);
	}
	for (const std::string &kv : split_ws(env)) {
		const size_t eq = kv.find('=');
		if (eq != std::string::npos) {
			setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
		}
	}
}

// --- API ------------------------------------------------------------------------

// Attach the host's RenderingDevice (or null: the no-device control). The
// parameter is an Object, not a Variant: with unboxed arguments (the
// default) the host passes an Object as a bare handle in a register, and a
// Variant parameter would read that handle as a pointer (Gate 0F probe 9).
static Variant ggml_attach(Object rd, int64_t total_mb) {
	capture_install();
	std::string r;
	if (rd.is_valid()) {
		g_dev.adopt(rd);
	}
	g_attached = g_dev.ok();
	ggml_backend_rd_attach(g_attached ? &g_dev : nullptr, size_t(total_mb) << 20);
	probes::set_device(g_attached ? &g_dev : nullptr);
	ggml_rd_hooks h;
	h.wait_gpu = hook_wait_gpu;
	h.coop = hook_coop;
	h.upload = hook_upload;
	ggml_backend_rd_set_hooks(h);
	ggml_set_abort_callback(on_ggml_abort);
	if (g_attached) {
		r = "attached device=" + g_dev.device_name() + " total_mb=" + std::to_string(total_mb);
	} else {
		r = "no RD device";
	}
	return Variant(String(r));
}

static Variant ggml_ops_start(String args, String env) {
	capture_install();
	g_out.clear();
	g_ops = OpsJob{};
	g_ops.args = split_ws(args.utf8());
	apply_env(env.utf8());
	if (!pump::start(&ops_job, nullptr)) {
		return Variant(String("FAIL a job is running"));
	}
	return Variant(String("STARTED test-backend-ops " + args.utf8() + " env[" + env.utf8() + "]"));
}

static Variant ggml_probe_start(String name, String arg, String env) {
	capture_install();
	g_out.clear();
	apply_env(env.utf8());
	std::string err;
	if (!probes::start(name.utf8(), arg.utf8(), err)) {
		return Variant(String("FAIL " + err));
	}
	return Variant(String("STARTED probe " + name.utf8()));
}

static Variant ggml_pump(PackedArray<uint8_t> in) {
	return pump::step(in);
}

// Everything printed since the job started.
static Variant ggml_output() {
	std::fflush(stdout);
	return Variant(String(g_out));
}

static Variant ggml_rd_stats() {
	std::string r = ggml_backend_rd_stats();
	r += " rule4_same_frame_syncs=" + std::to_string(g_dev.same_frame_syncs());
	r += " syncs=" + std::to_string(g_dev.syncs());
	r += " submits=" + std::to_string(g_dev.submits());
	r += " recoveries=" + std::to_string(g_dev.recoveries());
	r += " permanent_slots=" + std::to_string(g_dev.permanent_slots());
	r += " | " + rdc::name_slots();
	const std::string e = ggml_backend_rd_last_error();
	if (!e.empty()) {
		r += " | last_error: " + e;
	}
	return Variant(String(r));
}

// Free every RD object ggml-rd holds; the device stays the host's.
static Variant ggml_rd_close() {
	if (pump::running()) {
		return Variant(String("BUSY a job is running"));
	}
	ggml_backend_rd_release();
	g_dev.close();
	ggml_backend_rd_attach(nullptr, 0);
	return Variant(String("CLOSED permanent_slots=" + std::to_string(g_dev.permanent_slots())));
}

int main() {
	ADD_API_FUNCTION(ggml_attach, "String", "Object rd, int total_mb", "Attach the host's RenderingDevice (null: none)");
	ADD_API_FUNCTION(ggml_ops_start, "String", "String args, String env", "Start test-backend-ops on the pump");
	ADD_API_FUNCTION(ggml_probe_start, "String", "String name, String arg, String env", "Start a ggml-rd probe on the pump");
	ADD_API_FUNCTION(ggml_pump, "Array", "PackedByteArray data", "Resume the job once: [header, text, rid]");
	ADD_API_FUNCTION(ggml_output, "String", "", "What the job printed");
	ADD_API_FUNCTION(ggml_rd_stats, "String", "", "ggml-rd and rd_compute counters");
	ADD_API_FUNCTION(ggml_rd_close, "String", "", "Release ggml-rd's RD objects");
	halt();
}
