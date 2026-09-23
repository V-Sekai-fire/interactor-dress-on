#include "census.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "pump/pump.h"

namespace census {

namespace {

struct Row {
	const char *op;
	int64_t ne[4];
	float lo, hi; // input range of the census check
	float p; // SCALE: s; DIAG_MASK_INF: n_past
	int reps; // perf: applications per graph
	int64_t check_ne1; // census check: ne[1] cut to this (0: full shape)
	const char *where;
};

// census_skintokens.csv / census_pixal3d.csv (count per model run in `where`).
const Row kRows[] = {
	// Not a census row: one tiny dispatch, the floor of the host's timing
	// (frame gap, submit, sync) that the perf rows are read against.
	{ "NEG", { 256, 1, 1, 1 }, -1, 1, 0, 1, 0, "timing floor, not census" },
	{ "SILU", { 3072, 1, 2, 1 }, -8, 8, 0, 256, 0, "skin-tokens x10724" },
	{ "SILU", { 3072, 514, 1, 1 }, -8, 8, 0, 64, 0, "skin-tokens x56" },
	{ "SILU", { 4096, 32, 1, 1 }, -8, 8, 0, 256, 0, "Pixal3D x40" },
	{ "SILU", { 16, 16, 16, 512 }, -8, 8, 0, 64, 0, "Pixal3D x17" },
	{ "GELU", { 8192, 4096, 1, 1 }, -6, 6, 0, 16, 0, "Pixal3D x60" },
	{ "GELU_ERF", { 2048, 512, 1, 1 }, -6, 6, 0, 64, 0, "skin-tokens x766" },
	{ "GELU_ERF", { 4096, 1029, 1, 1 }, -6, 6, 0, 32, 0, "Pixal3D x24" },
	{ "SIGMOID", { 1, 16384, 1, 1 }, -12, 12, 0, 256, 0, "skin-tokens x212" },
	{ "NEG", { 1, 64, 12, 4096 }, -1, 1, 0, 64, 0, "Pixal3D x120" },
	{ "NEG", { 32, 16, 1024, 1 }, -1, 1, 0, 128, 0, "Pixal3D x48" },
	{ "SCALE", { 515, 1, 16, 2 }, -1, 1, 0.088388346f, 256, 0, "skin-tokens x10724" },
	{ "SCALE", { 54000, 512, 8, 1 }, -1, 1, 0.125f, 8, 64, "skin-tokens x822" },
	{ "DIAG_MASK_INF", { 514, 514, 16, 1 }, -1, 1, 0, 64, 0, "skin-tokens x56" },
	{ "ROPE", { 128, 8, 514, 1 }, -1, 1, 0, 512, 0, "skin-tokens x21560" },
};

constexpr int kPerfIters = 5;

bool is_unary(const char *op) {
	return std::strcmp(op, "SILU") == 0 || std::strcmp(op, "GELU") == 0 || std::strcmp(op, "GELU_ERF") == 0 ||
			std::strcmp(op, "SIGMOID") == 0 || std::strcmp(op, "NEG") == 0;
}

ggml_unary_op unary_of(const char *op) {
	if (std::strcmp(op, "SILU") == 0) {
		return GGML_UNARY_OP_SILU;
	}
	if (std::strcmp(op, "GELU") == 0) {
		return GGML_UNARY_OP_GELU;
	}
	if (std::strcmp(op, "GELU_ERF") == 0) {
		return GGML_UNARY_OP_GELU_ERF;
	}
	if (std::strcmp(op, "SIGMOID") == 0) {
		return GGML_UNARY_OP_SIGMOID;
	}
	return GGML_UNARY_OP_NEG;
}

// The op as the census records it; ROPE is skin-tokens' (NEOX, n_dims 128,
// n_ctx_orig 3192, freq_base 1e6, no YaRN).
ggml_tensor *apply(ggml_context *ctx, const Row &r, ggml_tensor *x, ggml_tensor *pos, bool inplace) {
	if (is_unary(r.op)) {
		return inplace ? ggml_unary_inplace(ctx, x, unary_of(r.op)) : ggml_unary(ctx, x, unary_of(r.op));
	}
	if (std::strcmp(r.op, "SCALE") == 0) {
		return inplace ? ggml_scale_inplace(ctx, x, r.p) : ggml_scale(ctx, x, r.p);
	}
	if (std::strcmp(r.op, "DIAG_MASK_INF") == 0) {
		return inplace ? ggml_diag_mask_inf_inplace(ctx, x, int(r.p)) : ggml_diag_mask_inf(ctx, x, int(r.p));
	}
	auto f = inplace ? ggml_rope_ext_inplace : ggml_rope_ext;
	return f(ctx, x, pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 3192, 1.0e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

bool is_rope(const Row &r) {
	return std::strcmp(r.op, "ROPE") == 0;
}

double ref64(const char *op, double x) {
	if (std::strcmp(op, "SILU") == 0) {
		return x / (1.0 + std::exp(-x));
	}
	if (std::strcmp(op, "SIGMOID") == 0) {
		return 1.0 / (1.0 + std::exp(-x));
	}
	if (std::strcmp(op, "GELU") == 0) {
		return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
	}
	if (std::strcmp(op, "GELU_ERF") == 0) {
		return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476));
	}
	return -x;
}

bool isinf_or_max(float f) {
	return std::isinf(f) || f == 3.40282347e38f || f == -3.40282347e38f;
}

struct Cmp {
	double nmse = 0.0;
	int64_t exact = 0;
	bool inf_ok = true;
	bool nan = false;
};

// test-backend-ops' comparison: no NaN, infinities of one sign on both sides
// (skipped in the sums, where inf - inf would be NaN), NMSE of a against b.
Cmp compare(const std::vector<float> &a, const std::vector<float> &b) {
	Cmp c;
	double ab = 0.0, a0 = 0.0;
	for (size_t i = 0; i < a.size(); ++i) {
		c.exact += std::memcmp(&a[i], &b[i], 4) == 0;
		if (std::isnan(a[i]) || std::isnan(b[i])) {
			c.nan = true;
			continue;
		}
		if (isinf_or_max(a[i]) || isinf_or_max(b[i])) {
			if (!(isinf_or_max(a[i]) && isinf_or_max(b[i]) && std::signbit(a[i]) == std::signbit(b[i]))) {
				c.inf_ok = false;
			}
			continue;
		}
		ab += (double(a[i]) - double(b[i])) * (double(a[i]) - double(b[i]));
		a0 += double(a[i]) * double(a[i]);
	}
	c.nmse = a0 > 0.0 ? ab / a0 : ab;
	return c;
}

double nmse64(const std::vector<float> &a, const std::vector<double> &ref) {
	double ab = 0.0, a0 = 0.0;
	for (size_t i = 0; i < a.size(); ++i) {
		ab += (double(a[i]) - ref[i]) * (double(a[i]) - ref[i]);
		a0 += double(a[i]) * double(a[i]);
	}
	return a0 > 0.0 ? ab / a0 : ab;
}

ggml_context *make_ctx(int tensors) {
	ggml_init_params ip = {
		ggml_tensor_overhead() * size_t(tensors + 8) + ggml_graph_overhead_custom(size_t(tensors + 8), false),
		nullptr,
		true,
	};
	return ggml_init(ip);
}

struct Graph {
	ggml_context *ctx = nullptr;
	ggml_tensor *x = nullptr, *pos = nullptr, *out = nullptr;
	ggml_cgraph *gf = nullptr;
	ggml_backend_buffer_t buf = nullptr;
};

// x -> op applied `reps` times (in place when reps > 1), allocated on `be`.
Graph build(const Row &r, const int64_t ne[4], int reps, ggml_backend_t be) {
	Graph g;
	g.ctx = make_ctx(reps + 4);
	g.x = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
	if (is_rope(r)) {
		g.pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, ne[2]);
	}
	ggml_tensor *cur = g.x;
	for (int i = 0; i < reps; ++i) {
		cur = apply(g.ctx, r, cur, g.pos, reps > 1);
	}
	g.out = cur;
	g.gf = ggml_new_graph_custom(g.ctx, size_t(reps + 8), false);
	ggml_build_forward_expand(g.gf, g.out);
	g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
	if (g.pos != nullptr) {
		std::vector<int32_t> p(static_cast<size_t>(ne[2]));
		for (size_t i = 0; i < p.size(); ++i) {
			p[i] = int32_t(i); // skin-tokens: positions 0..n_tokens-1
		}
		ggml_backend_tensor_set(g.pos, p.data(), 0, p.size() * 4);
	}
	return g;
}

void release(Graph &g) {
	if (g.buf) {
		ggml_backend_buffer_free(g.buf);
	}
	if (g.ctx) {
		ggml_free(g.ctx);
	}
	g = Graph{};
}

std::string shape(const int64_t ne[4]) {
	char b[96];
	std::snprintf(b, sizeof b, "[%lld,%lld,%lld,%lld]", (long long)ne[0], (long long)ne[1], (long long)ne[2],
			(long long)ne[3]);
	return b;
}

ggml_backend_t rd_backend() {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		std::printf("ggml-rd: no RD device\n");
		return nullptr;
	}
	return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
}

bool check_row(const Row &r, ggml_backend_t rd, ggml_backend_t cpu) {
	int64_t ne[4] = { r.ne[0], r.ne[1], r.ne[2], r.ne[3] };
	if (r.check_ne1 > 0) {
		ne[1] = r.check_ne1;
	}
	const size_t n = size_t(ne[0] * ne[1] * ne[2] * ne[3]);
	std::vector<float> in(n);
	std::mt19937 rng(20260922u);
	std::uniform_real_distribution<float> u(r.lo, r.hi);
	for (float &v : in) {
		v = u(rng);
	}
	Graph gr = build(r, ne, 1, rd);
	Graph gc = build(r, ne, 1, cpu);
	ggml_backend_tensor_set(gr.x, in.data(), 0, n * 4);
	ggml_backend_tensor_set(gc.x, in.data(), 0, n * 4);
	const ggml_status sr = ggml_backend_graph_compute(rd, gr.gf);
	pump::coop();
	const ggml_status sc = ggml_backend_graph_compute(cpu, gc.gf);
	std::vector<float> a(n), b(n);
	ggml_backend_tensor_get(gr.out, a.data(), 0, n * 4);
	ggml_backend_tensor_get(gc.out, b.data(), 0, n * 4);
	const Cmp c = compare(a, b);
	char extra[160] = "";
	if (is_unary(r.op)) {
		std::vector<double> ref(n);
		for (size_t i = 0; i < n; ++i) {
			ref[i] = ref64(r.op, double(in[i]));
		}
		std::snprintf(extra, sizeof extra, " nmse_vs_f64: rd=%.3e cpu=%.3e", nmse64(a, ref), nmse64(b, ref));
	}
	const bool ok = sr == GGML_STATUS_SUCCESS && sc == GGML_STATUS_SUCCESS && !c.nan && c.inf_ok && c.nmse <= 1e-7;
	std::printf("CENSUS %s %s%s (%s) range [%g,%g): nmse=%.3e (max 1e-07) exact=%lld/%zu inf_ok=%d nan=%d%s %s\n", r.op,
			shape(ne).c_str(), r.check_ne1 > 0 ? " (census ne1 cut)" : "", r.where, double(r.lo), double(r.hi), c.nmse,
			(long long)c.exact, n, int(c.inf_ok), int(c.nan), extra, ok ? "OK" : "FAIL");
	std::fflush(stdout);
	release(gr);
	release(gc);
	return ok;
}

bool perf_row(const Row &r, ggml_backend_t rd) {
	Graph g = build(r, r.ne, r.reps, rd);
	ggml_backend_buffer_clear(g.buf, 0);
	if (g.pos != nullptr) { // the clear zeroed them too
		std::vector<int32_t> p(size_t(r.ne[2]));
		for (size_t i = 0; i < p.size(); ++i) {
			p[i] = int32_t(i);
		}
		ggml_backend_tensor_set(g.pos, p.data(), 0, p.size() * 4);
	}
	bool ok = true;
	// One warm-up (pipelines, sets), then the timed graphs. Each sync ends
	// its vmcall with a COOP yield, so the host's interval covers the GPU
	// work of exactly one graph.
	for (int it = 0; it <= kPerfIters && ok; ++it) {
		ok = ggml_backend_graph_compute(rd, g.gf) == GGML_STATUS_SUCCESS;
		ggml_backend_synchronize(rd);
		pump::coop();
	}
	int64_t dispatches = 0, barriers = 0;
	ggml_backend_rd_last_graph(&dispatches, &barriers);
	const int64_t n = r.ne[0] * r.ne[1] * r.ne[2] * r.ne[3];
	std::printf("PERF %s %s (%s) reps=%d iters=%d warmup=1 dispatches=%lld barriers=%lld elements=%lld bytes_per_dispatch=%lld %s\n",
			r.op, shape(r.ne).c_str(), r.where, r.reps, kPerfIters, (long long)dispatches, (long long)barriers,
			(long long)n, (long long)(n * 8 + (g.pos ? r.ne[2] * 4 : 0)), ok ? "OK" : "FAIL");
	std::fflush(stdout);
	release(g);
	return ok && dispatches == r.reps;
}

} // namespace

bool known(const std::string &probe) {
	return probe == "census" || probe == "perf";
}

void run(const std::string &probe, const std::string &op) {
	ggml_backend_t rd = rd_backend();
	if (rd == nullptr) {
		std::printf("RESULT: FAIL (%s)\n", probe.c_str());
		return;
	}
	ggml_backend_t cpu = nullptr;
	if (probe == "census") {
		cpu = ggml_backend_cpu_init();
		ggml_backend_cpu_set_n_threads(cpu, 1); // guest threads are serialized (Gate 0C)
	}
	// op: "all", an op name, or a row index (the host times one row per job).
	const bool by_index = !op.empty() && op.find_first_not_of("0123456789") == std::string::npos;
	const size_t index = by_index ? size_t(std::atoi(op.c_str())) : 0;
	int rows = 0, ok = 0;
	for (size_t k = 0; k < sizeof(kRows) / sizeof(kRows[0]); ++k) {
		const Row &r = kRows[k];
		if (by_index ? k != index : (op != "all" && op != r.op)) {
			continue;
		}
		std::printf("ROW %zu\n", k);
		++rows;
		ok += probe == "census" ? check_row(r, rd, cpu) : perf_row(r, rd);
		pump::coop();
	}
	std::printf("%s %s: %d/%d rows OK\n", probe.c_str(), op.c_str(), ok, rows);
	std::printf("RESULT: %s (%s %s)\n", rows > 0 && ok == rows ? "PASS" : "FAIL", probe.c_str(), op.c_str());
	std::fflush(stdout);
	if (cpu) {
		ggml_backend_free(cpu);
	}
	ggml_backend_free(rd);
}

} // namespace census
