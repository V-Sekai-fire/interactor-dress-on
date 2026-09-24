// L2 cases for the ops the see-through engine (weftspun/see-through-cpp, an
// SDXL-class layer-decomposition pipeline) needs beyond the earlier
// families: SQRT, GELU_QUICK, GLU (GEGLU_ERF: plain, swapped and split),
// GROUP_NORM, PAD (zero fill and circular), ARANGE, TIMESTEP_EMBEDDING,
// CONV_2D (f32 and f16 kernels) and CONV_TRANSPOSE_2D (f32 and f16
// kernels). DIV is in cases/binary.cpp with its siblings.
//
// Shapes are test-backend-ops' where it has them (test_unary's two shapes
// over [-150, 150], test_glu's and test_glu_split's with the v = 1 view,
// test_group_norm's [64,64,320] and [9,9,1280] in 32 groups, test_pad_ext's
// three source forms for both fills, test_arange's 10 and 2^20 elements,
// test_timestep_embedding's dim 320 / period 10000, a sample of
// test_conv_2d's stride / padding / dilation sweep and its ConvNet layers,
// test_conv_transpose_2d's three shapes at strides 1 and 2), plus the
// engine's own: SDXL's GroupNorm(32) over 320..1280 channels at 64x64 and
// 32x32, its GEGLU rows of 2 x 1280 and 2 x 2560, the 3x3 convolutions of
// its resnet blocks, the 2x2 stride-2 transposed convolution of a decoder
// upsample, the 320-wide timestep embedding of t in [0, 1000).
//
// Thresholds are test-backend-ops': 1e-7 for everything but the two
// convolutions (5e-4, as test_conv_2d and test_conv_transpose_2d set; the
// measured values are far below). Expected bit-exact: SQRT, PAD, ARANGE
// (copies or one IEEE operation), and, on the host libm, TIMESTEP_EMBEDDING.
// GELU_QUICK is the f32 formula against ggml-cpu's f16 table; GEGLU_ERF the
// A&S erf; GROUP_NORM f32 tree sums against double; the convolutions f32
// sequential sums against ggml-cpu's dot products.
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include "../l2.h"

namespace {

using Ne = std::array<int64_t, 4>;

std::string shape(Ne ne) {
	char b[96];
	std::snprintf(b, sizeof b, "[%lld,%lld,%lld,%lld]", (long long)ne[0], (long long)ne[1], (long long)ne[2],
			(long long)ne[3]);
	return b;
}

const char *tn(ggml_type t) {
	return t == GGML_TYPE_F16 ? "f16" : "f32";
}

// test_unary's source: plain, or a view of a tensor 3x2x5x4 larger.
ggml_tensor *unary_src(ggml_context *ctx, Ne ne, bool view) {
	if (view) {
		ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0] * 3, ne[1] * 2, ne[2] * 5, ne[3] * 4);
		return ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a->nb[1], a->nb[2], a->nb[3], 0);
	}
	return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
}

/* ---- SQRT, GELU_QUICK ---- */

void add_sqrt(std::vector<L2Case> &out, Ne ne, bool view, float lo, float hi) {
	L2Case c;
	char b[160];
	std::snprintf(b, sizeof b, "SQRT %s%s range [%g,%g)", shape(ne).c_str(), view ? " view" : "", lo, hi);
	c.name = b;
	c.lo = lo;
	c.hi = hi;
	c.build = [=](ggml_context *ctx) { return ggml_sqrt(ctx, unary_src(ctx, ne, view)); };
	out.push_back(c);
}

void add_gelu_quick(std::vector<L2Case> &out, Ne ne, bool view, bool inplace, float lo, float hi) {
	L2Case c;
	char b[160];
	std::snprintf(b, sizeof b, "GELU_QUICK %s%s%s range [%g,%g)", shape(ne).c_str(), view ? " view" : "",
			inplace ? " inplace" : "", lo, hi);
	c.name = b;
	c.lo = lo;
	c.hi = hi;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *a = unary_src(ctx, ne, view);
		return inplace ? ggml_gelu_quick_inplace(ctx, a) : ggml_gelu_quick(ctx, a);
	};
	out.push_back(c);
}

/* ---- GLU: GEGLU_ERF ---- */

// test_glu (v = 1: a view of a tensor whose rows are three times as long)
// and test_glu_split. ne is dst's shape for the non-split form (a has 2 ne0).
void add_geglu_erf(std::vector<L2Case> &out, Ne ne, bool view, bool swapped, bool split, float lo, float hi) {
	L2Case c;
	char b[200];
	std::snprintf(b, sizeof b, "GEGLU_ERF %s%s%s%s range [%g,%g)", shape(ne).c_str(), view ? " view" : "",
			swapped ? " swapped" : "", split ? " split" : "", lo, hi);
	c.name = b;
	c.lo = lo;
	c.hi = hi;
	c.build = [=](ggml_context *ctx) {
		const int64_t n0 = split ? ne[0] : 2 * ne[0];
		auto src = [&](void) {
			if (view) {
				ggml_tensor *t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0 * 3, ne[1], ne[2], ne[3]);
				return ggml_view_4d(ctx, t, n0, ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], 0);
			}
			return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, ne[1], ne[2], ne[3]);
		};
		ggml_tensor *a = src();
		if (split) {
			ggml_tensor *g = src();
			return ggml_glu_split(ctx, a, g, GGML_GLU_OP_GEGLU_ERF);
		}
		return ggml_glu(ctx, a, GGML_GLU_OP_GEGLU_ERF, swapped);
	};
	out.push_back(c);
}

/* ---- GROUP_NORM ---- */

enum class Form { PLAIN, VIEW, PERMUTED, INPLACE };

const char *form_name(Form f) {
	switch (f) {
		case Form::VIEW:
			return " view";
		case Form::PERMUTED:
			return " permuted";
		case Form::INPLACE:
			return " inplace";
		default:
			return "";
	}
}

void add_group_norm(std::vector<L2Case> &out, Ne ne, int ng, float eps, Form f) {
	L2Case c;
	char b[200];
	std::snprintf(b, sizeof b, "GROUP_NORM %s groups=%d eps=%g%s", shape(ne).c_str(), ng, eps, form_name(f));
	c.name = b;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *a;
		if (f == Form::VIEW) { // rows padded to twice their length, the higher dims nested
			a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0] * 2, ne[1], ne[2], ne[3]);
			a = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a->nb[1], a->nb[2], a->nb[3], 0);
		} else if (f == Form::PERMUTED) { // dims 1 and 2 swapped
			a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[2], ne[1], ne[3]);
			a = ggml_permute(ctx, a, 0, 2, 1, 3);
		} else {
			a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
		}
		return f == Form::INPLACE ? ggml_group_norm_inplace(ctx, a, ng, eps) : ggml_group_norm(ctx, a, ng, eps);
	};
	out.push_back(c);
}

/* ---- PAD ---- */

// test_pad_ext: tfrm 0 plain, 1 a view of the larger half of every
// dimension, 2 a permuted source (dims 0 and 2 swapped); zero fill or
// circular.
void add_pad(std::vector<L2Case> &out, Ne ne, std::array<int, 8> p, int tfrm, bool circular) {
	L2Case c;
	char b[240];
	std::snprintf(b, sizeof b, "PAD %s lp/rp=%d,%d %d,%d %d,%d %d,%d%s%s", shape(ne).c_str(), p[0], p[1], p[2], p[3], p[4],
			p[5], p[6], p[7], tfrm == 1 ? " view" : (tfrm == 2 ? " perm" : ""), circular ? " circular" : "");
	c.name = b;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
		if (tfrm == 1) {
			a = ggml_view_4d(ctx, a, (a->ne[0] + 1) / 2, (a->ne[1] + 1) / 2, (a->ne[2] + 1) / 2, (a->ne[3] + 1) / 2,
					a->nb[1], a->nb[2], a->nb[3], 0);
		} else if (tfrm == 2) {
			a = ggml_permute(ctx, a, 2, 1, 0, 3);
		}
		return circular ? ggml_pad_ext_circular(ctx, a, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
						: ggml_pad_ext(ctx, a, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
	};
	out.push_back(c);
}

/* ---- ARANGE ---- */

void add_arange(std::vector<L2Case> &out, float start, float stop, float step) {
	L2Case c;
	char b[160];
	std::snprintf(b, sizeof b, "ARANGE %g..%g step %g", start, stop, step);
	c.name = b;
	c.build = [=](ggml_context *ctx) { return ggml_arange(ctx, start, stop, step); };
	c.value_blind = true; // no source: the swapped words are zero
	out.push_back(c);
}

/* ---- TIMESTEP_EMBEDDING ---- */

void add_timestep(std::vector<L2Case> &out, int n, int dim, int max_period, float lo, float hi) {
	L2Case c;
	char b[160];
	std::snprintf(b, sizeof b, "TIMESTEP_EMBEDDING n=%d dim=%d max_period=%d t in [%g,%g)", n, dim, max_period, lo, hi);
	c.name = b;
	c.lo = lo;
	c.hi = hi;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
		return ggml_timestep_embedding(ctx, t, dim, max_period);
	};
	// The timesteps are 1-D: dims 1 and 2 have one element, so the swapped
	// strides never reach an address (the control counts it a no-op).
	out.push_back(c);
}

/* ---- CONV_2D ---- */

struct C2 {
	ggml_type k = GGML_TYPE_F32;
	int64_t W = 20, H = 20, C = 3, N = 1, KW = 3, KH = 3, OC = 4;
	int s0 = 1, s1 = 1, p0 = 1, p1 = 1, d0 = 1, d1 = 1;
	bool permuted = false; // input dims 2 and 3 swapped (a view)
};

void conv2d(std::vector<L2Case> &out, C2 m) {
	L2Case c;
	char b[256];
	std::snprintf(b, sizeof b, "CONV_2D k=%s N=%lld C=%lld in=%lldx%lld OC=%lld k=%lldx%lld s=%d,%d p=%d,%d d=%d,%d%s",
			tn(m.k), (long long)m.N, (long long)m.C, (long long)m.W, (long long)m.H, (long long)m.OC, (long long)m.KW,
			(long long)m.KH, m.s0, m.s1, m.p0, m.p1, m.d0, m.d1, m.permuted ? " perm" : "");
	c.name = b;
	c.build = [m](ggml_context *ctx) {
		ggml_tensor *in;
		if (m.permuted) {
			in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, m.W, m.H, m.N, m.C);
			in = ggml_permute(ctx, in, 0, 1, 3, 2);
		} else {
			in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, m.W, m.H, m.C, m.N);
		}
		ggml_tensor *k = ggml_new_tensor_4d(ctx, m.k, m.KW, m.KH, m.C, m.OC);
		return ggml_conv_2d_direct(ctx, k, in, m.s0, m.s1, m.p0, m.p1, m.d0, m.d1);
	};
	c.max_nmse = 5e-4;
	out.push_back(c);
}

/* ---- CONV_TRANSPOSE_2D ---- */

void conv_transpose_2d(std::vector<L2Case> &out, ggml_type k, Ne in, Ne ker, int stride) {
	L2Case c;
	char b[256];
	std::snprintf(b, sizeof b, "CONV_TRANSPOSE_2D k=%s in=%s k=%s stride=%d", tn(k), shape(in).c_str(),
			shape(ker).c_str(), stride);
	c.name = b;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, in[0], in[1], in[2], in[3]);
		ggml_tensor *w = ggml_new_tensor_4d(ctx, k, ker[0], ker[1], ker[2], ker[3]);
		return ggml_conv_transpose_2d_p0(ctx, w, x, stride);
	};
	c.max_nmse = 5e-4;
	out.push_back(c);
}

} // namespace

L2_CASES(seethrough) {
	const ggml_type F32 = GGML_TYPE_F32, F16 = GGML_TYPE_F16;

	// SQRT: test_unary's shapes over a non-negative range, a census-like row.
	add_sqrt(out, { 128, 2, 2, 2 }, false, 0.0f, 150.0f);
	add_sqrt(out, { 5, 7, 11, 13 }, true, 0.0f, 150.0f);
	add_sqrt(out, { 1280, 3, 1, 1 }, false, 0.0f, 4.0f);

	// GELU_QUICK: test_unary's shapes over [-150, 150], SDXL's 1280-wide
	// MLP rows, in place.
	add_gelu_quick(out, { 128, 2, 2, 2 }, false, false, -150.0f, 150.0f);
	add_gelu_quick(out, { 5, 7, 11, 13 }, true, false, -150.0f, 150.0f);
	add_gelu_quick(out, { 1280, 64, 1, 1 }, false, false, -6.0f, 6.0f);
	add_gelu_quick(out, { 640, 5, 3, 1 }, false, true, -6.0f, 6.0f);

	// GEGLU_ERF: test_glu's two shapes, plain and viewed, swapped or not, and
	// test_glu_split's; SDXL's 2 x 1280 and 2 x 2560 rows.
	for (bool view : { false, true }) {
		for (bool swapped : { false, true }) {
			add_geglu_erf(out, { 128, 2, 2, 2 }, view, swapped, false, -150.0f, 150.0f);
			add_geglu_erf(out, { 5, 7, 11, 13 }, view, swapped, false, -150.0f, 150.0f);
		}
		add_geglu_erf(out, { 128, 2, 2, 2 }, view, false, true, -150.0f, 150.0f);
		add_geglu_erf(out, { 5, 7, 11, 13 }, view, false, true, -150.0f, 150.0f);
	}
	add_geglu_erf(out, { 1280, 77, 1, 1 }, false, false, false, -6.0f, 6.0f);
	add_geglu_erf(out, { 2560, 16, 1, 1 }, false, true, false, -6.0f, 6.0f);

	// GROUP_NORM: test_group_norm's two shapes; SDXL's 32 groups over 320,
	// 640 and 1280 channels (spatial cut down); groups that do not divide the
	// channels (the last group short), more groups than channels (empty
	// groups), a batch of 3, a view, permuted channels, in place.
	add_group_norm(out, { 64, 64, 320, 1 }, 32, 1e-6f, Form::PLAIN);
	add_group_norm(out, { 9, 9, 1280, 1 }, 32, 1e-6f, Form::PLAIN);
	add_group_norm(out, { 16, 16, 640, 2 }, 32, 1e-5f, Form::PLAIN);
	add_group_norm(out, { 4, 3, 7, 2 }, 3, 1e-6f, Form::PLAIN);
	add_group_norm(out, { 5, 3, 2, 2 }, 4, 1e-6f, Form::PLAIN);
	add_group_norm(out, { 6, 5, 10, 3 }, 5, 1e-6f, Form::PLAIN);
	add_group_norm(out, { 8, 8, 12, 2 }, 4, 1e-6f, Form::VIEW);
	add_group_norm(out, { 7, 5, 12, 2 }, 6, 1e-6f, Form::PERMUTED);
	add_group_norm(out, { 8, 8, 32, 1 }, 32, 1e-6f, Form::INPLACE);
	add_group_norm(out, { 33, 1, 1, 1 }, 1, 0.0f, Form::PLAIN);

	// PAD: test_pad_ext's default, its two shapes in all three forms with
	// both fills; ggml_pad (right pads only); a pad of nothing.
	add_pad(out, { 512, 512, 3, 1 }, { 1, 1, 1, 1, 0, 0, 0, 0 }, 0, false);
	for (int tfrm : { 0, 1, 2 }) {
		for (bool circular : { false, true }) {
			add_pad(out, { 512, 512, 1, 1 }, { 0, 1, 0, 1, 0, 0, 0, 0 }, tfrm, circular);
			add_pad(out, { 11, 22, 33, 44 }, { 1, 2, 3, 4, 5, 6, 7, 8 }, tfrm, circular);
		}
	}
	add_pad(out, { 3, 5, 7, 2 }, { 0, 2, 0, 3, 0, 1, 0, 1 }, 0, false);
	add_pad(out, { 64, 64, 4, 1 }, { 0, 0, 0, 0, 0, 0, 0, 0 }, 0, false);
	add_pad(out, { 6, 4, 3, 2 }, { 6, 0, 4, 0, 3, 0, 2, 0 }, 0, true); // lp = ne: a whole period

	// ARANGE: test_arange's two, a fractional step, a negative start.
	add_arange(out, 0.0f, 10.0f, 1.0f);
	add_arange(out, 0.0f, 1048576.0f, 1.0f);
	add_arange(out, 0.5f, 7.25f, 0.75f);
	add_arange(out, -3.0f, 3.0f, 0.5f);
	add_arange(out, 0.0f, 1000.0f, 1.0f); // the diffusion timesteps

	// TIMESTEP_EMBEDDING: test_timestep_embedding's default (n = 2, dim 320,
	// period 10000) with t in [-1, 1) as it draws; SDXL's t in [0, 1000);
	// an odd dim (the zero column); a short dim; one timestep.
	add_timestep(out, 2, 320, 10000, -1.0f, 1.0f);
	add_timestep(out, 5, 320, 10000, 0.0f, 1000.0f);
	add_timestep(out, 3, 321, 10000, 0.0f, 1000.0f);
	add_timestep(out, 4, 8, 100, 0.0f, 10.0f);
	add_timestep(out, 1, 256, 10000, 0.0f, 1000.0f);

	// CONV_2D: test_conv_2d's default (the sweep's stride/pad/dilation
	// sample), ConvNet layers (cut down), 1x1 and 11x11 kernels, both kernel
	// types, a permuted input; SDXL's 3x3 resnet conv (channels cut down).
	for (ggml_type k : { F32, F16 }) {
		C2 m;
		m.k = k;
		conv2d(out, m);
		m = C2{};
		m.k = k, m.W = 141, m.H = 133, m.C = 25, m.N = 2, m.KW = 3, m.KH = 2, m.OC = 12, m.s0 = 3, m.s1 = 1, m.p0 = 0,
		m.p1 = 5, m.d0 = 1, m.d1 = 2;
		conv2d(out, m);
		m = C2{};
		m.k = k, m.W = 16, m.H = 16, m.C = 8, m.N = 2, m.KW = 1, m.KH = 1, m.OC = 6, m.p0 = 0, m.p1 = 0;
		conv2d(out, m);
		m = C2{};
		m.k = k, m.W = 24, m.H = 20, m.C = 2, m.N = 1, m.KW = 11, m.KH = 11, m.OC = 3, m.p0 = 5, m.p1 = 5;
		conv2d(out, m);
	}
	{
		C2 m;
		m.W = 58, m.H = 58, m.C = 8, m.N = 2, m.KW = 3, m.KH = 3, m.OC = 8, m.p0 = 0, m.p1 = 0;
		conv2d(out, m);
		m = C2{};
		m.W = 32, m.H = 32, m.C = 20, m.N = 1, m.KW = 3, m.KH = 3, m.OC = 16;
		conv2d(out, m); // SDXL resnet conv shape (channels cut down)
		m.k = F16;
		conv2d(out, m);
		m = C2{};
		m.W = 13, m.H = 11, m.C = 3, m.N = 2, m.KW = 3, m.KH = 3, m.OC = 5, m.permuted = true;
		conv2d(out, m);
		m = C2{};
		m.W = 7, m.H = 9, m.C = 4, m.N = 1, m.KW = 3, m.KH = 3, m.OC = 5, m.s0 = 2, m.s1 = 2, m.p0 = 1, m.p1 = 1;
		conv2d(out, m); // a stride-2 downsample
	}

	// CONV_TRANSPOSE_2D: test_conv_transpose_2d's three shapes, both kernel
	// types; a 2x2 stride-2 decoder upsample; a 4x4 stride-2 with padding
	// overlap.
	for (ggml_type k : { F32, F16 }) {
		conv_transpose_2d(out, k, { 3, 2, 3, 1 }, { 2, 2, 1, 3 }, 1);
		conv_transpose_2d(out, k, { 10, 10, 9, 1 }, { 3, 3, 1, 9 }, 2);
		conv_transpose_2d(out, k, { 129, 63, 35, 1 }, { 3, 3, 48, 35 }, 1);
		conv_transpose_2d(out, k, { 16, 16, 16, 1 }, { 3, 3, 8, 16 }, 1);
		conv_transpose_2d(out, k, { 12, 10, 6, 1 }, { 2, 2, 4, 6 }, 2);
		conv_transpose_2d(out, k, { 9, 7, 5, 1 }, { 4, 4, 3, 5 }, 2);
	}
}
