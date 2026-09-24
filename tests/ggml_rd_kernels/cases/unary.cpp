// L2 cases for family K1: SILU, GELU, GELU_ERF, SIGMOID, NEG, RELU, SCALE,
// DIAG_MASK_INF, LEAKY_RELU and CLAMP (f32).
//
// Each unary op runs test-backend-ops' two shapes ([128,2,2,2] contiguous,
// [5,7,11,13] as a view of a larger tensor) over its input range [-150, 150],
// then a census shape over a realistic range (where GELU_ERF's A&S 7.1.26
// erf and GELU's f32 tanh are measured against ggml-cpu, whose GELU goes
// through an f16 table). Then a permuted source (rows contiguous, dims 1
// and 2 swapped, as ggml_unary allows), an in-place SILU, test-backend-ops'
// SCALE and DIAG_MASK_INF cases, the census SCALE (1/sqrt(128), b = 0) and a
// census-like causal mask. LEAKY_RELU and CLAMP run test-backend-ops' cases
// (slope 0.1 on [128,2,2,2]; [-0.5, 0.5] on [10,5,4,3]), MotionBricks' (slope
// 0.01 over its 512-wide MLP rows; a [0, 1] clamp of a summed mask), a
// row-padded view (the one strided form ggml-cpu's row walk gets right), in place (every CLAMP is: ggml_clamp views its source) and
// a chain. Thresholds are test-backend-ops'
// (NMSE 1e-7); every K1 op but GELU and GELU_ERF is expected bit-exact.
#include <array>
#include <cstdio>
#include <string>

#include "../l2.h"

namespace {

using Ne = std::array<int64_t, 4>;

struct UOp {
	const char *name;
	ggml_unary_op op;
};

const UOp kOps[] = {
	{ "SILU", GGML_UNARY_OP_SILU },
	{ "GELU", GGML_UNARY_OP_GELU },
	{ "GELU_ERF", GGML_UNARY_OP_GELU_ERF },
	{ "SIGMOID", GGML_UNARY_OP_SIGMOID },
	{ "NEG", GGML_UNARY_OP_NEG },
	{ "RELU", GGML_UNARY_OP_RELU },
};

std::string shape(Ne ne) {
	char b[96];
	std::snprintf(b, sizeof b, "[%lld,%lld,%lld,%lld]", (long long)ne[0], (long long)ne[1], (long long)ne[2],
			(long long)ne[3]);
	return b;
}

// test_unary::build_graph: v = 1 views a tensor 3x2x5x4 larger.
ggml_tensor *unary(ggml_context *ctx, ggml_unary_op op, Ne ne, bool view) {
	ggml_tensor *a;
	if (view) {
		a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0] * 3, ne[1] * 2, ne[2] * 5, ne[3] * 4);
		a = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a->nb[1], a->nb[2], a->nb[3], 0);
	} else {
		a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
	}
	return ggml_unary(ctx, a, op);
}

// A view of ne whose rows are padded to three times ne0 (nb1 = 3 ne0, the
// higher strides nested on it): strided, and still ggml_is_contiguous_1.
// ggml-cpu's LEAKY_RELU and CLAMP walk rows by nb[1] alone (they assert
// contiguous_1, compiled out in Release), so the reference is right only for
// such a view; the kernels take any strides.
ggml_tensor *padded_rows(ggml_context *ctx, Ne ne) {
	ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0] * 3, ne[1], ne[2], ne[3]);
	return ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a->nb[1], a->nb[2], a->nb[3], 0);
}

void add_unary(std::vector<L2Case> &out, const UOp &u, Ne ne, bool view, float lo, float hi) {
	L2Case c;
	char b[160];
	std::snprintf(b, sizeof b, "%s %s%s range [%g,%g)", u.name, shape(ne).c_str(), view ? " view" : "", lo, hi);
	c.name = b;
	c.lo = lo;
	c.hi = hi;
	const ggml_unary_op op = u.op;
	c.build = [=](ggml_context *ctx) { return unary(ctx, op, ne, view); };
	out.push_back(c);
}

} // namespace

L2_CASES(unary) {
	// test-backend-ops: both shapes, contiguous and viewed, over [-150, 150].
	for (const UOp &u : kOps) {
		add_unary(out, u, { 128, 2, 2, 2 }, false, -150.0f, 150.0f);
		add_unary(out, u, { 5, 7, 11, 13 }, true, -150.0f, 150.0f);
	}
	// Census shapes (scaled down where large), realistic ranges.
	add_unary(out, kOps[0], { 3072, 2, 1, 1 }, false, -8.0f, 8.0f); // SILU, skin-tokens
	add_unary(out, kOps[1], { 1024, 64, 1, 1 }, false, -6.0f, 6.0f); // GELU, Pixal3D
	add_unary(out, kOps[2], { 2048, 16, 1, 1 }, false, -6.0f, 6.0f); // GELU_ERF, skin-tokens
	add_unary(out, kOps[3], { 1, 16384, 1, 1 }, false, -12.0f, 12.0f); // SIGMOID, skin-tokens
	add_unary(out, kOps[4], { 1, 64, 12, 32 }, false, -1.0f, 1.0f); // NEG, Pixal3D
	add_unary(out, kOps[5], { 512, 9, 1, 1 }, false, -4.0f, 4.0f); // RELU, MotionBricks' MLP rows
	add_unary(out, kOps[5], { 512, 3, 3, 4 }, true, -4.0f, 4.0f); // RELU on a strided view

	// A permuted source: dims 1 and 2 swapped, rows still contiguous.
	for (int k : { 0, 2 }) {
		L2Case c;
		c.name = std::string(kOps[k].name) + " on permute(a,0,2,1,3) [33,4,5,3]";
		const ggml_unary_op op = kOps[k].op;
		c.lo = -6.0f;
		c.hi = 6.0f;
		c.build = [=](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 33, 4, 5, 3);
			return ggml_unary(ctx, ggml_permute(ctx, a, 0, 2, 1, 3), op);
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SILU inplace [67,5,3,2]";
		c.lo = -8.0f;
		c.hi = 8.0f;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 67, 5, 3, 2);
			return ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
		};
		out.push_back(c);
	}

	// SCALE: test-backend-ops' four cases and the census one.
	struct Sc {
		Ne ne;
		float s, b;
		bool inplace;
	};
	const Sc scs[] = {
		{ { 10, 10, 10, 10 }, 2.0f, 0.0f, false },
		{ { 10, 10, 10, 10 }, 2.0f, 1.0f, false },
		{ { 10, 10, 10, 10 }, 2.0f, 1.0f, true },
		{ { 100, 10, 10, 10 }, 2.0f, 1.0f, false },
		{ { 515, 1, 16, 2 }, 0.088388346f, 0.0f, false }, // skin-tokens, 1/sqrt(128)
	};
	for (const Sc &s : scs) {
		L2Case c;
		char b[160];
		std::snprintf(b, sizeof b, "SCALE %s s=%g b=%g%s", shape(s.ne).c_str(), s.s, s.b, s.inplace ? " inplace" : "");
		c.name = b;
		c.build = [=](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.ne[0], s.ne[1], s.ne[2], s.ne[3]);
			return s.inplace ? ggml_scale_bias_inplace(ctx, a, s.s, s.b) : ggml_scale_bias(ctx, a, s.s, s.b);
		};
		out.push_back(c);
	}

	// LEAKY_RELU: test-backend-ops' (slope 0.1, in place), MotionBricks'
	// (slope 0.01), odd sizes, a strided view, a negative slope of 1 (an
	// identity that still runs the two-term sum) and a chain.
	struct Lr {
		Ne ne;
		float slope;
		bool inplace, view;
	};
	const Lr lrs[] = {
		{ { 128, 2, 2, 2 }, 0.1f, true, false },
		{ { 128, 2, 2, 2 }, 0.1f, false, false },
		{ { 512, 9, 1, 1 }, 0.01f, false, false },
		{ { 7, 3, 5, 2 }, 0.2f, false, false },
		{ { 5, 7, 11, 13 }, 0.1f, false, true },
		{ { 33, 4, 5, 3 }, 1.0f, true, false },
	};
	for (const Lr &l : lrs) {
		L2Case c;
		char b[160];
		std::snprintf(b, sizeof b, "LEAKY_RELU %s slope=%g%s%s", shape(l.ne).c_str(), l.slope, l.inplace ? " inplace" : "",
				l.view ? " view" : "");
		c.name = b;
		c.lo = -8.0f;
		c.hi = 8.0f;
		c.build = [=](ggml_context *ctx) {
			ggml_tensor *a = l.view ? padded_rows(ctx, l.ne) : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, l.ne[0], l.ne[1], l.ne[2], l.ne[3]);
			return ggml_leaky_relu(ctx, a, l.slope, l.inplace);
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "LEAKY_RELU then RELU chain [512,9,1,1]";
		c.lo = -4.0f;
		c.hi = 4.0f;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 512, 9, 1, 1);
			return ggml_relu(ctx, ggml_leaky_relu(ctx, a, 0.01f, false));
		};
		out.push_back(c);
	}

	// CLAMP: test-backend-ops' case, a wide input range clamped to [-1, 1]
	// (most elements land on a bound), MotionBricks' [0, 1] on a summed
	// mask, a strided view, odd sizes and a degenerate min = max. ggml_clamp
	// is always in place (its result is a view of the source), so every case
	// here exercises the in-place path.
	struct Cl {
		Ne ne;
		float lo, hi, min, max;
		bool view;
	};
	const Cl cls[] = {
		{ { 10, 5, 4, 3 }, -1.0f, 1.0f, -0.5f, 0.5f, false },
		{ { 128, 2, 2, 2 }, -150.0f, 150.0f, -1.0f, 1.0f, false },
		{ { 1, 13, 1, 1 }, -1.0f, 3.0f, 0.0f, 1.0f, false },
		{ { 5, 7, 11, 13 }, -2.0f, 2.0f, -0.5f, 0.5f, true },
		{ { 67, 5, 3, 2 }, -2.0f, 2.0f, -0.25f, 0.75f, false },
		{ { 9, 4, 1, 1 }, -2.0f, 2.0f, 0.5f, 0.5f, false },
	};
	for (const Cl &l : cls) {
		L2Case c;
		char b[160];
		std::snprintf(b, sizeof b, "CLAMP %s [%g,%g] range [%g,%g)%s", shape(l.ne).c_str(), l.min, l.max, l.lo, l.hi,
				l.view ? " view" : "");
		c.name = b;
		c.lo = l.lo;
		c.hi = l.hi;
		c.value_blind = l.min == l.max; // every output is min: no stride can change it
		c.build = [=](ggml_context *ctx) {
			ggml_tensor *a = l.view ? padded_rows(ctx, l.ne) : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, l.ne[0], l.ne[1], l.ne[2], l.ne[3]);
			return ggml_clamp(ctx, a, l.min, l.max);
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "ADD then CLAMP [0,1] mask sum [1,13,1,1]";
		c.lo = 0.0f;
		c.hi = 1.0f;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 13, 1, 1);
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 13, 1, 1);
			return ggml_clamp(ctx, ggml_add(ctx, a, b), 0.0f, 1.0f);
		};
		out.push_back(c);
	}

	// DIAG_MASK_INF: test-backend-ops' three, a causal mask (n_past 0) and
	// an in-place one.
	struct Dm {
		Ne ne;
		int n_past;
		bool inplace;
	};
	const Dm dms[] = {
		{ { 10, 10, 1, 1 }, 5, false },
		{ { 10, 10, 3, 1 }, 5, false },
		{ { 10, 10, 3, 2 }, 5, false },
		{ { 66, 66, 4, 1 }, 0, false }, // skin-tokens: [514,514,16] with n_past 0, scaled down
		{ { 13, 9, 3, 2 }, 2, true },
	};
	for (const Dm &d : dms) {
		L2Case c;
		char b[160];
		std::snprintf(b, sizeof b, "DIAG_MASK_INF %s n_past=%d%s", shape(d.ne).c_str(), d.n_past,
				d.inplace ? " inplace" : "");
		c.name = b;
		c.build = [=](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d.ne[0], d.ne[1], d.ne[2], d.ne[3]);
			return d.inplace ? ggml_diag_mask_inf_inplace(ctx, a, d.n_past) : ggml_diag_mask_inf(ctx, a, d.n_past);
		};
		out.push_back(c);
	}
}
