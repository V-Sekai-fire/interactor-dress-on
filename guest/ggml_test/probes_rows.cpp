// probe rows_perf (Gate 3, families K3/K4): the GPU time of NORM, RMS_NORM
// and SOFT_MAX on the census's hottest shapes.
//
// For each shape, one tensor in an RD buffer (created cleared: the kernels
// take the same time on any values, and the largest shape, 884 MB, never has
// to exist in the guest heap) and two graphs over it: the op applied in
// place once, and N times in a chain (every dispatch reads what the one
// before wrote, so N dispatches and N - 1 barriers, as consecutive ops of a
// model run). Each graph is computed once to make its pipelines and sets,
// then three times timed: the host's clock (Time.get_ticks_usec, a host
// call: the guest clock is not a clock, AGENTS.md) from after the submit to
// after the sync, which lands on the next frame (WAIT_GPU, rule 4). The
// chain of one pays the frame and the sync; the chain of N pays those and N
// ops' GPU time, so
//
//     per op = (min t(N) - min t(1)) / (N - 1)
//
// N grows (x4) until t(N) - t(1) is at least 20 ms, so the frame's jitter is
// small against it. GB/s counts each pass over the row (NORM: 3 reads + 1
// write per element, RMS_NORM 2 + 1, SOFT_MAX 3 + 1). A check reads the
// result's first elements (a norm of zeros is 0, a softmax of equal values
// 1/ne0). One PROBE line per shape; RESULT PASS when every shape checked and
// timed.
//
// rows_perf sweep: NORM, RMS_NORM and SOFT_MAX over rows of 32 to 65536 elements
// (2^23 elements each), once with GGML_RD_ROW_THREADS=64 and once with 256:
// where the 64-thread kernels stop winning is ops/rows.h's kShortRow.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rd.h"
#include "ggml.h"

namespace probes {

namespace {

int64_t host_usec() {
	static Object time("Time");
	return int64_t(time.call("get_ticks_usec"));
}

struct Shape {
	ggml_op op;
	int64_t ne[4];
	float p; // eps, or SOFT_MAX's scale
	const char *census;
};

const Shape kShapes[] = {
	{ GGML_OP_RMS_NORM, { 896, 512, 1, 1 }, 1.1920929e-7f, "Skin-Tokens RMS_NORM x21947" },
	{ GGML_OP_RMS_NORM, { 128, 8, 514, 1 }, 1e-6f, "Skin-Tokens q/k RMS_NORM x21560" },
	{ GGML_OP_SOFT_MAX, { 512, 512, 8, 1 }, 1.0f, "Skin-Tokens attention SOFT_MAX x11542" },
	{ GGML_OP_NORM, { 512, 54000, 1, 1 }, 1e-5f, "Skin-Tokens NORM x1964" },
	{ GGML_OP_SOFT_MAX, { 54000, 512, 8, 1 }, 1.0f, "Skin-Tokens SOFT_MAX, rows > 16384, x4" },
	{ GGML_OP_RMS_NORM, { 128, 12, 4096, 1 }, 1e-12f, "Pixal3D ss_flow q/k RMS_NORM x240" },
	{ GGML_OP_NORM, { 1536, 4096, 1, 1 }, 1e-6f, "Pixal3D ss_flow NORM x182" },
	{ GGML_OP_NORM, { 1024, 1029, 1, 1 }, 1e-5f, "Pixal3D DINO NORM x80" },
	{ GGML_OP_SOFT_MAX, { 1029, 1029, 16, 1 }, 0.125f, "Pixal3D DINO SOFT_MAX x24" },
	{ GGML_OP_NORM, { 32, 64, 64, 64 }, 1e-5f, "Pixal3D ss_dec NORM x12" },
	{ GGML_OP_NORM, { 128, 32, 32, 32 }, 1e-5f, "Pixal3D ss_dec NORM x14" },
};

ggml_tensor *apply(ggml_context *ctx, const Shape &s, ggml_tensor *x) {
	switch (s.op) {
		case GGML_OP_NORM:
			return ggml_norm_inplace(ctx, x, s.p);
		case GGML_OP_RMS_NORM:
			return ggml_rms_norm_inplace(ctx, x, s.p);
		default:
			return ggml_soft_max_ext_inplace(ctx, x, nullptr, s.p, 0.0f);
	}
}

int passes(ggml_op op) {
	return op == GGML_OP_RMS_NORM ? 3 : 4;
}

// Submit g, then time from after the submit to after the sync.
int64_t timed(ggml_backend_t be, ggml_cgraph *g) {
	if (ggml_backend_graph_compute_async(be, g) != GGML_STATUS_SUCCESS) {
		return -1;
	}
	const int64_t t0 = host_usec();
	ggml_backend_synchronize(be);
	return host_usec() - t0;
}

int64_t best_of_3(ggml_backend_t be, ggml_cgraph *g) {
	int64_t best = -1;
	for (int i = 0; i < 3; ++i) {
		const int64_t t = timed(be, g);
		if (t < 0) {
			return -1;
		}
		best = best < 0 ? t : std::min(best, t);
	}
	return best;
}

bool one_shape(ggml_backend_t be, const Shape &s, const char *tag) {
	const int64_t n = s.ne[0] * s.ne[1] * s.ne[2] * s.ne[3];
	const double bytes = double(n) * 4.0 * passes(s.op);
	// A first N from a guess of 500 GB/s and 20 ms, at least 4.
	int chain = int(std::clamp(0.020 * 500e9 / bytes, 4.0, 1024.0));
	int64_t t1 = -1, tn = -1;
	bool ok = true;
	std::string why;
	float got[4] = { -1, -1, -1, -1 };
	for (int attempt = 0; attempt < 3; ++attempt) {
		ggml_init_params ip = { ggml_tensor_overhead() * size_t(chain + 16) +
						ggml_graph_overhead_custom(size_t(chain + 16), false) * 2,
			nullptr, true };
		ggml_context *ctx = ggml_init(ip);
		ggml_tensor *x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.ne[0], s.ne[1], s.ne[2], s.ne[3]);
		ggml_cgraph *g1 = ggml_new_graph_custom(ctx, 16, false);
		ggml_build_forward_expand(g1, apply(ctx, s, x));
		ggml_cgraph *gn = ggml_new_graph_custom(ctx, size_t(chain + 16), false);
		ggml_tensor *cur = x;
		for (int i = 0; i < chain; ++i) {
			cur = apply(ctx, s, cur);
		}
		ggml_build_forward_expand(gn, cur);
		ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
		if (buf == nullptr) {
			ok = false;
			why = "no buffer: " + ggml_backend_rd_last_error();
			ggml_free(ctx);
			break;
		}
		// Warm-up: pipelines, sets and params slots.
		ok = timed(be, g1) >= 0 && timed(be, gn) >= 0;
		t1 = ok ? best_of_3(be, g1) : -1;
		tn = ok ? best_of_3(be, gn) : -1;
		ok = ok && t1 >= 0 && tn >= 0;
		if (ok) {
			ggml_backend_tensor_get(x, got, 0, sizeof got);
		} else {
			why = "compute failed: " + ggml_backend_rd_last_error();
		}
		ggml_backend_buffer_free(buf);
		ggml_free(ctx);
		if (!ok || tn - t1 >= 20000 || chain >= 1024) {
			break;
		}
		chain = std::min(chain * 4, 1024);
	}
	const float want = s.op == GGML_OP_SOFT_MAX ? 1.0f / float(s.ne[0]) : 0.0f;
	bool check = ok;
	for (float v : got) {
		check = check && std::fabs(v - want) <= 1e-6f * std::max(1.0f, std::fabs(want));
	}
	const double per_op_us = chain > 1 ? double(tn - t1) / double(chain - 1) : 0.0;
	const bool timed_ok = ok && tn > t1;
	std::printf("PROBE rows_perf%s op=%s ne=[%lld,%lld,%lld,%lld] p=%g chain=%d t1_us=%lld tN_us=%lld per_op_us=%.1f "
				"GBps=%.0f check=%s (%s)%s%s\n",
			tag, ggml_op_name(s.op), (long long)s.ne[0], (long long)s.ne[1], (long long)s.ne[2], (long long)s.ne[3],
			double(s.p), chain, (long long)t1, (long long)tn, per_op_us,
			per_op_us > 0.0 ? bytes / (per_op_us * 1e3) : 0.0, check ? "ok" : "BAD", s.census,
			why.empty() ? "" : " error: ", why.c_str());
	std::fflush(stdout);
	return check && timed_ok;
}

} // namespace

bool rows_perf(ggml_backend_t be, const std::string &arg) {
	bool all = true;
	if (arg != "sweep") {
		for (const Shape &s : kShapes) {
			all = one_shape(be, s, "") && all;
		}
		return all;
	}
	for (ggml_op op : { GGML_OP_NORM, GGML_OP_RMS_NORM, GGML_OP_SOFT_MAX }) {
		for (int64_t ne0 : { 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536 }) {
			const Shape s = { op, { ne0, (int64_t(1) << 23) / ne0, 1, 1 }, op == GGML_OP_SOFT_MAX ? 1.0f : 1e-6f, "sweep" };
			for (const char *threads : { "64", "256" }) {
				setenv("GGML_RD_ROW_THREADS", threads, 1);
				const std::string tag = std::string(" threads=") + threads;
				all = one_shape(be, s, tag.c_str()) && all;
			}
		}
	}
	unsetenv("GGML_RD_ROW_THREADS");
	return all;
}

} // namespace probes
