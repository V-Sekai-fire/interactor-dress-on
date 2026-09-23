// L2 cases for IM2COL and CONV_3D (family K7).
//
// IM2COL: test-backend-ops' test_im2col shapes (2-D default, the stride /
// padding / dilation sweep, 1-D, Whisper's 1-D conv), f32 and f16 columns,
// f16 and f32 kernels (shape only), odd-length f16 output (the last word
// half outside dst), a permuted image (channels and batch swapped), and the
// census' Pixal3D DINO patch embedding (512x512x3, 16x16 stride 16, f16
// out). Columns are copies (and one f16 rounding): expected bit-exact,
// threshold 1e-7 as test_im2col. The swap-nb control swaps src1's (the
// image's) strides: src0 is read for its shape only.
//
// CONV_3D: test_conv_3d shapes (the N / IC / OC / stride / padding /
// dilation sweep, the asymmetric kernel, a 1x1x1 kernel), f32 and f16
// kernels, a permuted input, and the census' two Pixal3D shapes (16^3 x 8
// -> 512 channels, 64^3 x 32 -> 1). Threshold 5e-4 as test_conv_3d.
#include <array>
#include <cstdio>
#include <string>

#include "../l2.h"

namespace {

using Ne = std::array<int64_t, 4>;

const char *tn(ggml_type t) {
	return t == GGML_TYPE_F16 ? "f16" : "f32";
}

struct Im {
	ggml_type k = GGML_TYPE_F16, dst = GGML_TYPE_F32;
	Ne in{ 10, 10, 3, 1 }, ker{ 3, 3, 3, 1 };
	int s0 = 1, s1 = 1, p0 = 1, p1 = 1, d0 = 1, d1 = 1;
	bool is2d = true;
	bool permuted = false; // image dims 2 and 3 swapped (a view)
};

void im2col(std::vector<L2Case> &out, Im m) {
	L2Case c;
	char b[256];
	std::snprintf(b, sizeof b, "IM2COL %s k=%s in=[%lld,%lld,%lld,%lld] k=[%lld,%lld,%lld,%lld] s=%d,%d p=%d,%d d=%d,%d %s%s",
			tn(m.dst), tn(m.k), (long long)m.in[0], (long long)m.in[1], (long long)m.in[2], (long long)m.in[3],
			(long long)m.ker[0], (long long)m.ker[1], (long long)m.ker[2], (long long)m.ker[3], m.s0, m.s1, m.p0, m.p1,
			m.d0, m.d1, m.is2d ? "2d" : "1d", m.permuted ? " perm" : "");
	c.name = b;
	c.build = [m](ggml_context *ctx) {
		ggml_tensor *img;
		if (m.permuted) {
			img = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, m.in[0], m.in[1], m.in[3], m.in[2]);
			img = ggml_permute(ctx, img, 0, 1, 3, 2);
		} else {
			img = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, m.in.data());
		}
		ggml_tensor *k = ggml_new_tensor(ctx, m.k, 4, m.ker.data());
		return ggml_im2col(ctx, k, img, m.s0, m.s1, m.p0, m.p1, m.d0, m.d1, m.is2d, m.dst);
	};
	c.control_src = 1;
	out.push_back(c);
}

struct Cv {
	ggml_type k = GGML_TYPE_F32;
	int64_t N = 1, IC = 1, ID = 18, IH = 22, IW = 20, OC = 1, KD = 3, KH = 3, KW = 3;
	int s0 = 1, s1 = 1, s2 = 1, p0 = 0, p1 = 0, p2 = 0, d0 = 1, d1 = 1, d2 = 1;
	bool permuted = false; // input dims 2 and 3 swapped (a view)
};

void conv3d(std::vector<L2Case> &out, Cv m) {
	L2Case c;
	char b[256];
	std::snprintf(b, sizeof b,
			"CONV_3D k=%s N=%lld IC=%lld in=%lldx%lldx%lld OC=%lld k=%lldx%lldx%lld s=%d,%d,%d p=%d,%d,%d d=%d,%d,%d%s",
			tn(m.k), (long long)m.N, (long long)m.IC, (long long)m.ID, (long long)m.IH, (long long)m.IW, (long long)m.OC,
			(long long)m.KD, (long long)m.KH, (long long)m.KW, m.s0, m.s1, m.s2, m.p0, m.p1, m.p2, m.d0, m.d1, m.d2,
			m.permuted ? " perm" : "");
	c.name = b;
	c.build = [m](ggml_context *ctx) {
		ggml_tensor *in;
		if (m.permuted) {
			in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, m.IW, m.IH, m.IC * m.N, m.ID);
			in = ggml_permute(ctx, in, 0, 1, 3, 2);
		} else {
			in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, m.IW, m.IH, m.ID, m.IC * m.N);
		}
		ggml_tensor *k = ggml_new_tensor_4d(ctx, m.k, m.KW, m.KH, m.KD, m.IC * m.OC);
		return ggml_conv_3d_direct(ctx, k, in, m.s0, m.s1, m.s2, m.p0, m.p1, m.p2, m.d0, m.d1, m.d2, int(m.IC), int(m.N),
				int(m.OC));
	};
	c.max_nmse = 5e-4;
	out.push_back(c);
}

} // namespace

L2_CASES(conv) {
	const ggml_type F32 = GGML_TYPE_F32, F16 = GGML_TYPE_F16;
	// IM2COL 2-D: test_im2col's defaults, every kernel/dst type pair.
	for (ggml_type k : { F32, F16 }) {
		for (ggml_type d : { F32, F16 }) {
			Im m;
			m.k = k;
			m.dst = d;
			im2col(out, m);
		}
	}
	// The stride / padding / dilation sweep on [20,20,2,2] (a sample of it).
	const int sw[][6] = { { 1, 1, 0, 0, 1, 1 }, { 3, 1, 3, 0, 1, 3 }, { 1, 3, 0, 3, 3, 1 }, { 3, 3, 3, 3, 3, 3 } };
	for (const auto &q : sw) {
		for (ggml_type d : { F32, F16 }) {
			Im m;
			m.k = F32;
			m.dst = d;
			m.in = { 20, 20, 2, 2 };
			m.ker = { 3, 3, 2, 2 };
			m.s0 = q[0], m.s1 = q[1], m.p0 = q[2], m.p1 = q[3], m.d0 = q[4], m.d1 = q[5];
			im2col(out, m);
		}
	}
	{ // odd-length f16 columns: 9 x 7 x 5 = 315 halves, the last word half dst
		Im m;
		m.dst = F16;
		m.in = { 7, 5, 1, 1 };
		m.ker = { 3, 3, 1, 1 };
		m.p0 = m.p1 = 1;
		im2col(out, m);
	}
	{ // a non-square kernel, no padding, batch 32 (test_im2col)
		Im m;
		m.dst = F16;
		m.in = { 5, 5, 1, 32 };
		m.ker = { 3, 4, 1, 32 };
		m.p0 = m.p1 = 0;
		im2col(out, m);
	}
	for (ggml_type d : { F32, F16 }) { // channels and batch permuted
		Im m;
		m.dst = d;
		m.in = { 9, 7, 3, 2 };
		m.ker = { 3, 3, 3, 1 };
		m.permuted = true;
		im2col(out, m);
	}
	// IM2COL 1-D: the sweep sample and Whisper's conv (test_im2col).
	{
		Im m;
		m.k = F32;
		m.in = { 20, 2, 2, 1 };
		m.ker = { 3, 2, 2, 1 };
		m.s0 = 3, m.p0 = 3, m.d0 = 3;
		m.s1 = 0, m.p1 = 0, m.d1 = 0;
		m.is2d = false;
		im2col(out, m);
		m.dst = F16;
		im2col(out, m);
	}
	{
		Im m;
		m.dst = F16;
		m.in = { 3000, 128, 1, 1 };
		m.ker = { 3, 128, 1280, 1 };
		m.s0 = 1, m.p0 = 1, m.d0 = 1;
		m.s1 = 0, m.p1 = 0, m.d1 = 0;
		m.is2d = false;
		im2col(out, m);
	}
	{ // census: Pixal3D DINO patch embedding, f16 kernel, f16 columns
		Im m;
		m.k = F16;
		m.dst = F16;
		m.in = { 512, 512, 3, 1 };
		m.ker = { 16, 16, 3, 1024 };
		m.s0 = m.s1 = 16;
		m.p0 = m.p1 = 0;
		im2col(out, m);
	}

	// CONV_3D: a sample of test_conv_3d's sweep on 20x22x18, both kernel types.
	for (ggml_type k : { F32, F16 }) {
		const int sw3[][5] = { // N IC OC s p d
			{ 1, 1, 1, 1, 0 }, { 2, 3, 4, 1, 1 }, { 1, 3, 4, 2, 1 }, { 2, 1, 4, 2, 0 }
		};
		for (const auto &q : sw3) {
			Cv m;
			m.k = k;
			m.N = q[0], m.IC = q[1], m.OC = q[2];
			m.s0 = m.s1 = m.s2 = q[3];
			m.p0 = m.p1 = m.p2 = q[4];
			conv3d(out, m);
			m.d0 = m.d1 = m.d2 = 2; // dilated
			conv3d(out, m);
		}
		{ // the asymmetric kernel and parameters
			Cv m;
			m.k = k;
			m.N = 2, m.IC = 3, m.OC = 4;
			m.KW = 5, m.KH = 1, m.KD = 3;
			m.s0 = 2, m.s1 = 1, m.s2 = 1, m.p0 = 2, m.p1 = 0, m.p2 = 1, m.d0 = 1, m.d1 = 1, m.d2 = 2;
			conv3d(out, m);
		}
		{ // a 1x1x1 kernel
			Cv m;
			m.k = k;
			m.IC = 4, m.ID = m.IH = m.IW = 8, m.OC = 8, m.KD = m.KH = m.KW = 1;
			conv3d(out, m);
		}
		{ // a permuted input (depth and channel*batch swapped)
			Cv m;
			m.k = k;
			m.N = 2, m.IC = 3, m.OC = 2, m.ID = 7, m.IH = 6, m.IW = 5, m.p0 = m.p1 = m.p2 = 1;
			m.permuted = true;
			conv3d(out, m);
		}
	}
	{ // census: Pixal3D ss_dec, 16^3 x 8 -> 512 channels, f16 kernel
		Cv m;
		m.k = F16;
		m.IC = 8, m.ID = m.IH = m.IW = 16, m.OC = 512;
		m.p0 = m.p1 = m.p2 = 1;
		conv3d(out, m);
	}
	{ // census: Pixal3D ss_dec, 64^3 x 32 -> 1 channel, f16 kernel
		Cv m;
		m.k = F16;
		m.IC = 32, m.ID = m.IH = m.IW = 64, m.OC = 1;
		m.p0 = m.p1 = m.p2 = 1;
		conv3d(out, m);
	}
}
