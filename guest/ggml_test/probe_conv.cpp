// The K7 probe (IM2COL, CONV_3D): the census' hottest shapes on the GPU,
// timed on the host clock, checked against a reference computed here.
//
//   conv_perf <shape|all>   shape: im2col_dino, conv3d_ss_a, conv3d_ss_b
//
// For each shape the probe builds R independent copies of the op on the same
// inputs (one graph, no barrier between them) and computes the graph K
// times (K = 10, after at least 300 ms of untimed ones): each compute is a submit, a WAIT_GPU
// yield and a sync on the next frame (rule 4). The time is Godot's
// Time.get_ticks_usec() (a host call; the guest clock is not a clock) around
// the K computes, for R = 1 and
// R = 17: per_graph_ms(R=1) is what one op costs a graph end to end, frame
// included, and per_op_ms = (T(17) - T(1)) / 16 is the GPU time of one more
// op of that shape. Then copy 0 of the output is read back and 4096
// elements spread over it are compared with a reference computed in the
// probe: CONV_3D by NMSE against double accumulation (conv3d_f16 rounds the
// input to f16 as ggml-cpu does), within test_conv_3d's 5e-4; f16 columns
// bit for bit against ggml_fp32_to_fp16.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <api.hpp>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rd.h"
#include "ggml.h"

namespace probes {

namespace {

int64_t host_us() {
	static const char kTicks[] = "get_ticks_usec";
	Object t("Time");
	return int64_t(t.call(kTicks));
}

struct Shape {
	const char *name;
	bool conv; // CONV_3D, else IM2COL
	// IM2COL: image [W,H,C,1], kernel [KW,KH,C,OC], stride s, no padding, f16 out.
	// CONV_3D: input [W,W,W,IC], kernel [3,3,3,IC*OC] f16, pad 1, stride 1.
	int64_t w, c, oc, k, s;
};

const Shape kShapes[] = {
	{ "im2col_dino", false, 512, 3, 1024, 16, 16 }, // Pixal3D DINO patch embedding
	{ "conv3d_ss_a", true, 16, 8, 512, 3, 1 }, // Pixal3D ss_dec, 16^3 x 8 -> 512
	{ "conv3d_ss_b", true, 64, 32, 1, 3, 1 }, // Pixal3D ss_dec, 64^3 x 32 -> 1
};

float f16r(float x) {
	return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
}

bool run_shape(ggml_backend_t be, const Shape &sh) {
	const int K = 10;
	double per_graph_ms[2] = { 0, 0 };
	const int Rs[2] = { 1, 17 };
	double max_abs = 0.0, err2 = 0.0, ref2 = 0.0;
	int64_t checked = 0, exact = 0, bad = 0;
	int64_t n_out = 0;
	ggml_status st = GGML_STATUS_SUCCESS;
	int64_t dispatches = 0, barriers = 0;
	for (int ri = 0; ri < 2; ++ri) {
		const int R = Rs[ri];
		ggml_init_params ip = { ggml_tensor_overhead() * size_t(R + 8) + ggml_graph_overhead_custom(size_t(R + 8), false),
			nullptr, true };
		ggml_context *ctx = ggml_init(ip);
		ggml_tensor *in, *ker;
		if (sh.conv) {
			in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.w, sh.w, sh.w, sh.c);
			ker = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.k, sh.k, sh.k, sh.c * sh.oc);
		} else {
			in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.w, sh.w, sh.c, 1);
			ker = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.k, sh.k, sh.c, sh.oc);
		}
		ggml_cgraph *gf = ggml_new_graph_custom(ctx, size_t(R + 8), false);
		std::vector<ggml_tensor *> outs;
		for (int r = 0; r < R; ++r) {
			ggml_tensor *o = sh.conv
					? ggml_conv_3d_direct(ctx, ker, in, 1, 1, 1, 1, 1, 1, 1, 1, 1, int(sh.c), 1, int(sh.oc))
					: ggml_im2col(ctx, ker, in, int(sh.s), int(sh.s), 0, 0, 1, 1, true, GGML_TYPE_F16);
			outs.push_back(o);
			ggml_build_forward_expand(gf, o);
		}
		ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
		std::mt19937 rng(20260922u);
		std::uniform_real_distribution<float> u(-1.0f, 1.0f);
		std::vector<float> inv(size_t(ggml_nelements(in)));
		for (float &x : inv) {
			x = u(rng);
		}
		std::vector<ggml_fp16_t> kv(size_t(ggml_nelements(ker)));
		for (ggml_fp16_t &h : kv) {
			h = ggml_fp32_to_fp16(u(rng));
		}
		ggml_backend_tensor_set(in, inv.data(), 0, inv.size() * 4);
		ggml_backend_tensor_set(ker, kv.data(), 0, kv.size() * 2);
		// Warm-up: pipelines, sets, the params table, and the GPU's clocks
		// (an idle GPU runs at a fraction of its boost clock): at least 3
		// computes and 300 ms.
		const int64_t tw = host_us();
		for (int k = 0; st == GGML_STATUS_SUCCESS && (k < 3 || host_us() - tw < 300000); ++k) {
			st = ggml_backend_graph_compute(be, gf);
		}
		const int64_t t0 = host_us();
		for (int k = 0; k < K && st == GGML_STATUS_SUCCESS; ++k) {
			st = ggml_backend_graph_compute(be, gf);
		}
		const int64_t t1 = host_us();
		per_graph_ms[ri] = double(t1 - t0) / 1000.0 / K;
		ggml_backend_rd_last_graph(&dispatches, &barriers);

		if (ri == 1 && st == GGML_STATUS_SUCCESS) { // check copy 0 against the reference
			ggml_tensor *o = outs[0];
			n_out = ggml_nelements(o);
			std::vector<uint8_t> got(ggml_nbytes(o));
			ggml_backend_tensor_get(o, got.data(), 0, got.size());
			const int64_t step = n_out / 4096 > 0 ? n_out / 4096 : 1;
			for (int64_t e = 0; e < n_out; e += step) {
				const int64_t i0 = e % o->ne[0], i1 = (e / o->ne[0]) % o->ne[1], i2 = (e / (o->ne[0] * o->ne[1])) % o->ne[2],
							  i3 = e / (o->ne[0] * o->ne[1] * o->ne[2]);
				++checked;
				if (sh.conv) {
					// dst [OW,OH,OD,OC]; pad 1, stride 1: out size == in size.
					double acc = 0.0;
					for (int64_t ic = 0; ic < sh.c; ++ic) {
						for (int64_t kz = 0; kz < 3; ++kz) {
							for (int64_t ky = 0; ky < 3; ++ky) {
								for (int64_t kx = 0; kx < 3; ++kx) {
									const int64_t x = i0 + kx - 1, y = i1 + ky - 1, z = i2 + kz - 1;
									if (x < 0 || y < 0 || z < 0 || x >= sh.w || y >= sh.w || z >= sh.w) {
										continue;
									}
									const float xv = f16r(inv[size_t(((ic * sh.w + z) * sh.w + y) * sh.w + x)]);
									const float kw = ggml_fp16_to_fp32(kv[size_t((((i3 * sh.c + ic) * 3 + kz) * 3 + ky) * 3 + kx)]);
									acc += double(xv) * double(kw);
								}
							}
						}
					}
					float g;
					std::memcpy(&g, got.data() + 4 * e, 4);
					const double d = double(g) - acc;
					max_abs = std::fabs(d) > max_abs ? std::fabs(d) : max_abs;
					err2 += d * d;
					ref2 += acc * acc;
					bad += !std::isfinite(g);
				} else {
					// dst [C*K*K, OW, OH, 1] f16; stride s, no padding: always inside.
					const int64_t iic = i0 / (sh.k * sh.k), ikh = (i0 / sh.k) % sh.k, ikw = i0 % sh.k;
					const int64_t x = i1 * sh.s + ikw, y = i2 * sh.s + ikh;
					const ggml_fp16_t want = ggml_fp32_to_fp16(inv[size_t((iic * sh.w + y) * sh.w + x)]);
					ggml_fp16_t g;
					std::memcpy(&g, got.data() + 2 * e, 2);
					exact += g == want;
					bad += g != want;
				}
			}
		}
		ggml_backend_buffer_free(buf);
		ggml_free(ctx);
		if (st != GGML_STATUS_SUCCESS) {
			break;
		}
	}
	const double per_op_ms = (per_graph_ms[1] - per_graph_ms[0]) / double(Rs[1] - Rs[0]);
	std::printf("PROBE conv_perf %s status=%d out=%lld K=%d per_graph_ms(R=1)=%.3f per_graph_ms(R=%d)=%.3f "
				"per_op_ms=%.3f dispatches(R=%d)=%lld barriers=%lld checked=%lld bad=%lld%s\n",
			sh.name, int(st), (long long)n_out, K, per_graph_ms[0], Rs[1], per_graph_ms[1], per_op_ms, Rs[1],
			(long long)dispatches, (long long)barriers, (long long)checked, (long long)bad, "");
	const double nmse = ref2 > 0.0 ? err2 / ref2 : err2;
	if (sh.conv) {
		std::printf("PROBE conv_perf %s nmse=%.3e (max 5e-4) max_abs_err=%.3e (vs double accumulation of the f16-rounded "
					"input)\n",
				sh.name, nmse, max_abs);
		bad += nmse > 5e-4;
	} else {
		std::printf("PROBE conv_perf %s f16_exact=%lld/%lld\n", sh.name, (long long)exact, (long long)checked);
	}
	std::fflush(stdout);
	return st == GGML_STATUS_SUCCESS && checked > 0 && bad == 0 && dispatches == Rs[1] && barriers == 0;
}

} // namespace

bool conv_perf(ggml_backend_t be, const std::string &which) {
	bool ok = true;
	int ran = 0;
	for (const Shape &sh : kShapes) {
		if (which == "all" || which.empty() || which == sh.name) {
			ok = run_shape(be, sh) && ok;
			++ran;
		}
	}
	return ok && ran > 0;
}

} // namespace probes
