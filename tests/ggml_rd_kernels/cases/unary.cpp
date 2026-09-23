// L2 cases for family K1: SILU, GELU, GELU_ERF, SIGMOID, NEG, SCALE and
// DIAG_MASK_INF (f32).
//
// Each unary op runs test-backend-ops' two shapes ([128,2,2,2] contiguous,
// [5,7,11,13] as a view of a larger tensor) over its input range [-150, 150],
// then a census shape over a realistic range (where GELU_ERF's A&S 7.1.26
// erf and GELU's f32 tanh are measured against ggml-cpu, whose GELU goes
// through an f16 table). Then a permuted source (rows contiguous, dims 1
// and 2 swapped, as ggml_unary allows), an in-place SILU, test-backend-ops'
// SCALE and DIAG_MASK_INF cases, the census SCALE (1/sqrt(128), b = 0) and a
// census-like causal mask. Thresholds are test-backend-ops' (NMSE 1e-7).
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
