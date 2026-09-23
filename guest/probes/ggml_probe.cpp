// Gate 0F probe 15: ggml-cpu, one thread. The same file is compiled into
// probes.elf (riscv64, in the sandbox) and into a host-native executable
// (gates/0f-runtime/ggml_host, llvm-mingw); the two checksum lines must agree
// to 1e-6 relative. No guest API in here on purpose.
//
// The work: A (f16, n x n) times B (f32, n x n) with ggml_mul_mat, then
// ggml_soft_max over the result's rows. Inputs come from a fixed LCG, so both
// builds see bit-identical data. A float64 reference product over the same
// f16-rounded inputs gives the in-process error.
#include "ggml_probe.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

struct Lcg {
	uint64_t s;
	// Uniform in [-1, 1), from the top 24 bits so it is exact in float.
	float next() {
		s = s * 6364136223846793005ull + 1442695040888963407ull;
		return float(int32_t(uint32_t(s >> 40)) - (1 << 23)) / float(1 << 23);
	}
};

} // namespace

std::string ggml_probe_run(int n) {
	ggml_cpu_init();
	ggml_init_params ip = {};
	ip.mem_size = size_t(64) << 20;
	ip.mem_buffer = nullptr;
	ip.no_alloc = false;
	ggml_context *ctx = ggml_init(ip);
	if (!ctx) {
		return "FAIL ggml_init";
	}

	// ggml_mul_mat(a, b): a is [K, M] (ne0 = K), b is [K, N]; result [M, N].
	ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, n);
	ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
	Lcg rng{ 0x0F0F0F0F12345678ull };
	std::vector<float> af(size_t(n) * n), bf(size_t(n) * n);
	ggml_fp16_t *ad = static_cast<ggml_fp16_t *>(a->data);
	for (size_t i = 0; i < af.size(); ++i) {
		ad[i] = ggml_fp32_to_fp16(rng.next());
		af[i] = ggml_fp16_to_fp32(ad[i]);
	}
	float *bd = static_cast<float *>(b->data);
	for (size_t i = 0; i < bf.size(); ++i) {
		bf[i] = rng.next();
		bd[i] = bf[i];
	}

	ggml_tensor *c = ggml_mul_mat(ctx, a, b);
	ggml_tensor *s = ggml_soft_max(ctx, c);
	ggml_cgraph *gf = ggml_new_graph(ctx);
	ggml_build_forward_expand(gf, s);
	const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
	if (st != GGML_STATUS_SUCCESS) {
		ggml_free(ctx);
		return "FAIL ggml_graph_compute status=" + std::to_string(int(st));
	}

	// c[m + M*j] = sum_k a[k + K*m] * b[k + K*j]
	const float *cd = static_cast<const float *>(c->data);
	const float *sd = static_cast<const float *>(s->data);
	double sum_abs = 0, sum_sq = 0, sm_w = 0, sm_rows = 0, max_err = 0, max_ref = 0;
	for (int j = 0; j < n; ++j) {
		for (int m = 0; m < n; ++m) {
			double ref = 0;
			for (int k = 0; k < n; ++k) {
				ref += double(af[size_t(k) + size_t(n) * m]) * double(bf[size_t(k) + size_t(n) * j]);
			}
			const double got = cd[size_t(m) + size_t(n) * j];
			max_err = std::fmax(max_err, std::fabs(got - ref));
			max_ref = std::fmax(max_ref, std::fabs(ref));
			sum_abs += std::fabs(got);
			sum_sq += got * got;
			const double p = sd[size_t(m) + size_t(n) * j];
			sm_w += p * double((m * 7 + j * 13) % 17 + 1);
			sm_rows += p;
		}
	}
	ggml_free(ctx);

	char buf[320];
	std::snprintf(buf, sizeof buf,
			"n=%d sum_abs=%.17g sum_sq=%.17g softmax_w=%.17g softmax_rowsum=%.17g max_abs_err_vs_f64=%.3g max_ref=%.6g",
			n, sum_abs, sum_sq, sm_w, sm_rows, max_err, max_ref);
	return buf;
}
