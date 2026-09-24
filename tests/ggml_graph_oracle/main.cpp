// ggml_graph_oracle: the host-native reference for Gate 3 G3.graph.
//
// The guest runs each G3.graph net on ggml-rd only and dumps both arms'
// outputs (project/gate_ggml_graph.gd writes them under --dump). This
// program rebuilds the same net with the same builders and the same seeds
// (guest/ggml_test/graph_nets.cpp), computes it on the host and compares:
//
//   rd native vs ref native   rel-L2 <= 1e-3   (the verdict)
//   rd f32    vs ref f32      rel-L2 <= 1e-4   (the verdict)
//   the block's input as dumped vs as built here: bit-identical (the two
//   sides filled the same bytes), and info lines (rd native vs ref f32, ref
//   native vs ref f32: how much of the gap is the reference's own rounding).
//
//   --graph=<qwen|dit|sconv|kimodo_denoiser|kimodo_text>[:res]   the net (graph_nets.h)
//   --ref=<vulkan|cpu>               the reference: ggml-vulkan on the GPU for
//                                    large graphs, host ggml-cpu for small ones
//   --check=<cpu|vulkan|none>        a second host backend, compared with the
//                                    reference (a check of the oracle itself)
//                                    and with ggml-rd (info)
//   --vk=<precise|default>           precise (the default): ggml-vulkan with f16
//                                    arithmetic, cooperative matrices, bf16 and
//                                    integer dot products off, so every matmul
//                                    accumulates in f32 from f32 shared memory
//                                    as ggml-cpu and ggml-rd do; default: as
//                                    ggml-vulkan chooses (f16 accumulation and
//                                    coopmat2's f16 conversion on an RTX 4090)
//   --dump=<dir>                     the guest's dump (<dir>/<arm>/<output>.f32);
//                                    none: only the host backends run
//   --out=<file>                     write everything here instead of stdout
//   --threads=<n>                    ggml-cpu threads (default: all)
//
// ggml-vulkan is an ORACLE here: it compiles its own GLSL with glslc, never
// ships and never runs in the guest (AGENTS.md rule 2 governs the shipped
// kernels, which come from Lean). A build with -DORACLE_VULKAN=OFF (no Vulkan
// SDK) has ggml-cpu only: "vulkan" is then no backend. The last line is
// RESULT: PASS or FAIL.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef ORACLE_VULKAN
#include "ggml-vulkan.h"
#endif
#include "ggml.h"
#include "graph_nets.h"

using namespace graph_nets;

namespace {

void set_env(const char *k, const char *v) {
#ifdef _WIN32
	_putenv_s(k, v);
#else
	setenv(k, v, 1);
#endif
}

double now_ms() {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Opts {
	std::string graph = "dit";
	int res = 0;
	std::string ref = "vulkan";
	std::string check = "none";
	std::string vk = "precise";
	std::string dump;
	std::string out;
	int threads = 0;
};

int g_vk_device = -1;

ggml_backend_t make_backend(const std::string &kind, const Opts &o) {
	if (kind == "cpu") {
		ggml_backend_t b = ggml_backend_cpu_init();
		const int t = o.threads > 0 ? o.threads : int(std::max(1u, std::thread::hardware_concurrency()));
		ggml_backend_cpu_set_n_threads(b, t);
		return b;
	}
#ifdef ORACLE_VULKAN
	if (kind == "vulkan") {
		if (g_vk_device < 0) {
			const int n = ggml_backend_vk_get_device_count();
			g_vk_device = 0;
			for (int i = 0; i < n; ++i) {
				char desc[256] = {};
				ggml_backend_vk_get_device_description(i, desc, sizeof desc);
				std::printf("ORACLE vulkan device %d: %s\n", i, desc);
				if (std::strstr(desc, "4090") != nullptr) {
					g_vk_device = i;
				}
			}
			std::printf("ORACLE vulkan uses device %d\n", g_vk_device);
		}
		return ggml_backend_vk_init(size_t(g_vk_device));
	}
#endif
	return nullptr;
}

struct ArmResult {
	Outs outs;
	bool ok = false;
	double ms = 0.0;
};

// Build, allocate, fill and compute one arm on a fresh backend.
ArmResult run_arm(const std::string &kind, const Opts &o, const BuildFn &build, bool f32_arm) {
	ArmResult r;
	Net n;
	n.be = make_backend(kind, o);
	if (n.be == nullptr) {
		std::printf("ORACLE %s: no backend\n", kind.c_str());
		return r;
	}
	try {
		build(n, f32_arm);
		int unsupported = 0;
		for (int i = 0; i < ggml_graph_n_nodes(n.gf); ++i) {
			ggml_tensor *t = ggml_graph_node(n.gf, i);
			if (!ggml_backend_supports_op(n.be, t)) {
				if (unsupported++ < 8) {
					std::printf("ORACLE %s: node %d %s '%s' (%s) not supported\n", kind.c_str(), i, ggml_op_desc(t),
							t->name, ggml_type_name(t->type));
				}
			}
		}
		if (unsupported > 0) {
			std::printf("ORACLE %s: %d unsupported nodes\n", kind.c_str(), unsupported);
		} else if (allocate(n)) {
			const double t0 = now_ms();
			const ggml_status st = ggml_backend_graph_compute(n.be, n.gf);
			r.outs = read_outs(n);
			r.ms = now_ms() - t0;
			r.ok = st == GGML_STATUS_SUCCESS;
			std::printf("ORACLE %s_%s status=%d nodes=%d ms=%.1f\n", kind.c_str(), f32_arm ? "f32" : "native", int(st),
					ggml_graph_n_nodes(n.gf), r.ms);
			// A second run, timed on its own (the first includes pipeline creation).
			const double t1 = now_ms();
			const ggml_status st2 = ggml_backend_graph_compute(n.be, n.gf);
			Outs again = read_outs(n);
			std::printf("ORACLE %s_%s rerun status=%d ms=%.1f repeat=%s\n", kind.c_str(), f32_arm ? "f32" : "native",
					int(st2), now_ms() - t1, identical(again, r.outs, nullptr) ? "bit-identical" : "different");
		}
	} catch (const std::exception &e) {
		std::printf("ORACLE %s: %s\n", kind.c_str(), e.what());
	}
	ggml_backend_free(n.be);
	n.be = nullptr;
	return r;
}

bool load_f32(const std::string &path, std::vector<float> &v) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) {
		return false;
	}
	const std::streamsize n = f.tellg();
	f.seekg(0);
	v.resize(size_t(n) / 4);
	return bool(f.read(reinterpret_cast<char *>(v.data()), std::streamsize(v.size() * 4)));
}

// The guest's arm, shaped like `like` (the same output names).
bool load_dump(const std::string &dir, const std::string &arm, const Outs &like, Outs &out) {
	bool ok = true;
	for (const auto &x : like) {
		std::vector<float> v;
		const std::string p = dir + "/" + arm + "/" + x.first + ".f32";
		if (!load_f32(p, v)) {
			if (x.first != "__residual_in" || arm == "rd_native") {
				std::printf("ORACLE dump: missing %s\n", p.c_str());
				ok = false;
			}
			continue;
		}
		out.emplace_back(x.first, std::move(v));
	}
	// rd_f32 carries no input of its own: the native arm's is the same leaf.
	return ok;
}

} // namespace

int main(int argc, char **argv) {
	Opts o;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto val = [&](const char *k) -> const char * {
			const size_t n = std::strlen(k);
			return a.compare(0, n, k) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char *v = val("--graph=")) {
			const std::string g = v;
			const size_t c = g.find(':');
			o.graph = g.substr(0, c);
			o.res = c == std::string::npos ? 0 : std::atoi(g.c_str() + c + 1);
		} else if (const char *v = val("--ref=")) {
			o.ref = v;
		} else if (const char *v = val("--check=")) {
			o.check = v;
		} else if (const char *v = val("--vk=")) {
			o.vk = v;
		} else if (const char *v = val("--dump=")) {
			o.dump = v;
		} else if (const char *v = val("--out=")) {
			o.out = v;
		} else if (const char *v = val("--threads=")) {
			o.threads = std::atoi(v);
		} else {
			std::fprintf(stderr, "unknown argument %s\n", a.c_str());
			return 2;
		}
	}
	if (!o.out.empty()) {
		if (std::freopen(o.out.c_str(), "w", stdout) == nullptr) {
			return 2;
		}
	}
	std::setvbuf(stdout, nullptr, _IOLBF, 1 << 16);
	if (o.vk == "precise") {
		for (const char *k : { "GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2",
					 "GGML_VK_DISABLE_BFLOAT16", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT" }) {
			set_env(k, "1");
		}
	}
	const double t_start = now_ms();
	BuildFn build;
	std::string native, resid;
	if (!graph_builder(o.graph, o.res, build, native, resid)) {
		std::printf("ORACLE unknown graph '%s'\nRESULT: FAIL\n", o.graph.c_str());
		return 1;
	}
	const std::string g = o.graph + (o.res > 0 ? ":" + std::to_string(o.res) : "");
	const char *gc = g.c_str();
	std::printf("ORACLE graph=%s ref=%s check=%s vk=%s dump=%s\n", gc, o.ref.c_str(), o.check.c_str(), o.vk.c_str(),
			o.dump.empty() ? "(none)" : o.dump.c_str());

	std::map<std::string, ArmResult> ref, chk;
	ref["native"] = run_arm(o.ref, o, build, false);
	ref["f32"] = run_arm(o.ref, o, build, true);
	bool ok = ref["native"].ok && ref["f32"].ok;
	const std::string R = o.ref, N = native;
	auto name = [](const std::string &a, const std::string &b) { return a + "_vs_" + b; };
	if (o.check != "none") {
		chk["native"] = run_arm(o.check, o, build, false);
		chk["f32"] = run_arm(o.check, o, build, true);
		const std::string C = o.check;
		const double c_nat = compare(gc, name(C + "_" + N, R + "_" + N + "_oracle_check").c_str(), chk["native"].outs,
				ref["native"].outs, resid);
		const double c_f32 = compare(gc, name(C + "_f32", R + "_f32_oracle_check").c_str(), chk["f32"].outs,
				ref["f32"].outs, resid);
		std::printf("ORACLE check %s vs %s: native %.3e f32 %.3e\n", C.c_str(), R.c_str(), c_nat, c_f32);
	}
	const double e_rr = compare(gc, name(R + "_" + N, R + "_f32_info").c_str(), ref["native"].outs, ref["f32"].outs, resid);

	bool verdict = false;
	if (!o.dump.empty() && ok) {
		Outs rd_nat, rd_f32;
		const bool have = load_dump(o.dump, "rd_native", ref["native"].outs, rd_nat) &
				load_dump(o.dump, "rd_f32", ref["f32"].outs, rd_f32);
		// The same leaves on both sides: the block's input bit for bit.
		const std::vector<float> *in_rd = find(rd_nat, "__residual_in");
		const std::vector<float> *in_ref = find(ref["native"].outs, "__residual_in");
		bool same_in = true;
		if (in_rd && in_ref) {
			same_in = in_rd->size() == in_ref->size() &&
					std::memcmp(in_rd->data(), in_ref->data(), in_rd->size() * 4) == 0;
			std::printf("ORACLE inputs: the dumped block input vs the oracle's: %s (n=%zu)\n",
					same_in ? "bit-identical" : "DIFFERENT", in_rd->size());
		}
		// The f32 arm's rows need the residual input for the branch norm.
		if (in_ref && !find(rd_f32, "__residual_in")) {
			rd_f32.emplace_back("__residual_in", *in_ref);
		}
		const double e_nat = compare(gc, name("rd_" + N, R + "_" + N).c_str(), rd_nat, ref["native"].outs, resid);
		const double e_f32 = compare(gc, name("rd_f32", R + "_f32").c_str(), rd_f32, ref["f32"].outs, resid);
		const double e_x = compare(gc, name("rd_" + N, R + "_f32_info").c_str(), rd_nat, ref["f32"].outs, resid);
		double e_cn = -1, e_cf = -1;
		if (o.check != "none") {
			e_cn = compare(gc, name("rd_" + N, o.check + "_" + N + "_info").c_str(), rd_nat, chk["native"].outs, resid);
			e_cf = compare(gc, name("rd_f32", o.check + "_f32_info").c_str(), rd_f32, chk["f32"].outs, resid);
		}
		const bool pass_nat = e_nat <= 1e-3, pass_f32 = e_f32 <= 1e-4;
		verdict = have && same_in && pass_nat && pass_f32;
		std::printf("ORACLE %s SUMMARY ref=%s(%s) weights=%s rel_l2(rd_%s,%s_%s)=%.3e (<=1e-3: %s) "
					"rel_l2(rd_f32,%s_f32)=%.3e (<=1e-4: %s) info rel_l2(rd_%s,%s_f32)=%.3e rel_l2(%s_%s,%s_f32)=%.3e",
				gc, R.c_str(), R == "vulkan" ? o.vk.c_str() : "host", N.c_str(), N.c_str(), R.c_str(), N.c_str(), e_nat,
				pass_nat ? "yes" : "NO", R.c_str(), e_f32, pass_f32 ? "yes" : "NO", N.c_str(), R.c_str(), e_x, R.c_str(),
				N.c_str(), R.c_str(), e_rr);
		if (o.check != "none") {
			std::printf(" rel_l2(rd_%s,%s_%s)=%.3e rel_l2(rd_f32,%s_f32)=%.3e", N.c_str(), o.check.c_str(), N.c_str(),
					e_cn, o.check.c_str(), e_cf);
		}
		std::printf(" ref_ms=%.0f/%.0f inputs=%s wall_s=%.1f\n", ref["native"].ms, ref["f32"].ms,
				same_in ? "bit-identical" : "DIFFERENT", (now_ms() - t_start) / 1000.0);
	} else {
		verdict = ok && (o.check == "none" || (chk["native"].ok && chk["f32"].ok));
		std::printf("ORACLE %s SUMMARY (no dump) ref=%s ref_ms=%.0f/%.0f wall_s=%.1f\n", gc, R.c_str(),
				ref["native"].ms, ref["f32"].ms, (now_ms() - t_start) / 1000.0);
	}
	std::printf("RESULT: %s (oracle %s)\n", verdict ? "PASS" : "FAIL", gc);
	std::fflush(stdout);
	return verdict ? 0 : 1;
}
