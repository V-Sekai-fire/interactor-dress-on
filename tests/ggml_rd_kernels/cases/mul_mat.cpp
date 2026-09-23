// L2 cases for MUL_MAT (family K6): every (src0, src1) pair with a kernel,
// both kernels (vec for ne11 <= 4, tiled above), broadcast in dims 2 and 3,
// the three permutations test-backend-ops uses, a k view with strided rows
// (k_v), several outputs summed (o > 1: MUL_MAT then ADD), edges on every
// side of the 64 x 64 tile and the census's shapes scaled down (skin-tokens'
// decode, its attention over a permuted KV cache, Pixal3D's bf16 blocks).
//
// build() is test_mul_mat::build_graph (vendor/ggml/tests/test-backend-ops.cpp);
// the threshold is its max_nmse_err, 5e-4. The kernels that share group
// memory run here as their cpp siblings (kernels/ggml/cpp_siblings.txt).
#include <array>
#include <cstdio>
#include <string>

#include "../l2.h"

namespace {

using A2 = std::array<int64_t, 2>;
using A4 = std::array<int64_t, 4>;

struct Mm {
	ggml_type ta = GGML_TYPE_F32, tb = GGML_TYPE_F32;
	int64_t m = 32, n = 32, k = 32;
	A2 bs = { 1, 1 }, nr = { 1, 1 };
	A4 per = { 0, 1, 2, 3 };
	int64_t k_v = 0;
	int o = 1;
};

ggml_tensor *build(ggml_context *ctx, const Mm &c) {
	ggml_tensor *a;
	ggml_tensor *b;
	const int npermuted = (c.per[0] != 0) + (c.per[1] != 1) + (c.per[2] != 2) + (c.per[3] != 3);
	if (npermuted > 0) {
		const int64_t ne_a[4] = { c.k, c.m, c.bs[0], c.bs[1] };
		const int64_t ne_b[4] = { c.k, c.n, c.bs[0] * c.nr[0], c.bs[1] * c.nr[1] };
		a = ggml_new_tensor_4d(ctx, c.ta, ne_a[c.per[0]], ne_a[c.per[1]], ne_a[c.per[2]], ne_a[c.per[3]]);
		b = ggml_new_tensor_4d(ctx, c.tb, ne_b[c.per[0]], ne_b[c.per[1]], ne_b[c.per[2]], ne_b[c.per[3]]);
		a = ggml_permute(ctx, a, int(c.per[0]), int(c.per[1]), int(c.per[2]), int(c.per[3]));
		b = ggml_permute(ctx, b, int(c.per[0]), int(c.per[1]), int(c.per[2]), int(c.per[3]));
	} else {
		const int64_t kp = c.k_v == 0 ? c.k : c.k_v;
		a = ggml_new_tensor_4d(ctx, c.ta, kp, c.m, c.bs[0], c.bs[1]);
		b = ggml_new_tensor_4d(ctx, c.tb, kp, c.n, c.bs[0] * c.nr[0], c.bs[1] * c.nr[1]);
		if (c.k_v != 0) {
			a = ggml_view_4d(ctx, a, c.k, c.m, c.bs[0], c.bs[1], a->nb[1], a->nb[2], a->nb[3], 0);
			b = ggml_view_4d(ctx, b, c.k, c.n, c.bs[0] * c.nr[0], c.bs[1] * c.nr[1], b->nb[1], b->nb[2], b->nb[3], 0);
		}
	}
	ggml_tensor *out = ggml_mul_mat(ctx, a, b);
	for (int i = 1; i < c.o; ++i) {
		out = ggml_add(ctx, out, ggml_mul_mat(ctx, a, b));
	}
	return out;
}

std::string fmt(const Mm &c, const char *note) {
	char s[256];
	std::snprintf(s, sizeof s, "MUL_MAT %s x %s m=%lld n=%lld k=%lld bs=[%lld,%lld] nr=[%lld,%lld] per=[%lld,%lld,%lld,%lld]%s%s%s",
			ggml_type_name(c.ta), ggml_type_name(c.tb), (long long)c.m, (long long)c.n, (long long)c.k,
			(long long)c.bs[0], (long long)c.bs[1], (long long)c.nr[0], (long long)c.nr[1], (long long)c.per[0],
			(long long)c.per[1], (long long)c.per[2], (long long)c.per[3],
			c.k_v ? (" k_v=" + std::to_string(c.k_v)).c_str() : "", c.o > 1 ? (" o=" + std::to_string(c.o)).c_str() : "",
			note);
	return s;
}

void add(std::vector<L2Case> &out, const Mm &c, const char *note = "") {
	L2Case l;
	l.name = fmt(c, note);
	l.build = [c](ggml_context *ctx) { return build(ctx, c); };
	l.max_nmse = 5e-4; // test_mul_mat::max_nmse_err
	out.push_back(l);
}

Mm mm(ggml_type ta, ggml_type tb, int64_t m, int64_t n, int64_t k, A2 bs = { 1, 1 }, A2 nr = { 1, 1 },
		A4 per = { 0, 1, 2, 3 }, int64_t k_v = 0, int o = 1) {
	Mm c;
	c.ta = ta;
	c.tb = tb;
	c.m = m;
	c.n = n;
	c.k = k;
	c.bs = bs;
	c.nr = nr;
	c.per = per;
	c.k_v = k_v;
	c.o = o;
	return c;
}

// skin-tokens' attention scores in decode: q [128, 1, 16, 2] against a K
// cache stored [128, 16, T, 2] and viewed [128, T, 16, 2] (the census's RC
// class: src0 rows 16 * 128 elements apart), T = 515 scaled to 67.
ggml_tensor *attn_scores(ggml_context *ctx) {
	ggml_tensor *kc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, 67, 2);
	ggml_tensor *k = ggml_permute(ctx, kc, 0, 2, 1, 3); // [128, 67, 16, 2]
	ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 1, 16, 2);
	return ggml_mul_mat(ctx, k, q);
}

// The census's RR class: both operands permuted views, [64, T, 8] x [64, T, 8].
ggml_tensor *rr_scores(ggml_context *ctx) {
	ggml_tensor *kc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 8, 100, 1);
	ggml_tensor *qc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 8, 72, 1);
	return ggml_mul_mat(ctx, ggml_permute(ctx, kc, 0, 2, 1, 3), ggml_permute(ctx, qc, 0, 2, 1, 3));
}

} // namespace

L2_CASES(mul_mat) {
	const ggml_type F32 = GGML_TYPE_F32, F16 = GGML_TYPE_F16, BF16 = GGML_TYPE_BF16;
	// f32 x f32
	add(out, mm(F32, F32, 16, 16, 256), " tiled");
	add(out, mm(F32, F32, 16, 1, 256, { 3, 2 }, { 2, 2 }), " vec, broadcast in dims 2 and 3");
	add(out, mm(F32, F32, 64, 77, 77, { 12, 1 }), " tiled, odd n and k");
	add(out, mm(F32, F32, 16, 16, 256, { 2, 3 }, { 1, 1 }, { 0, 2, 1, 3 }), " permuted");
	add(out, mm(F32, F32, 16, 8, 256, { 2, 3 }, { 1, 1 }, { 0, 1, 3, 2 }), " permuted");
	add(out, mm(F32, F32, 16, 4, 4, { 2, 3 }, { 1, 1 }, { 0, 3, 2, 1 }), " vec at n = 4, permuted");
	add(out, mm(F32, F32, 16, 32, 32, { 1, 1 }, { 1, 1 }, { 0, 1, 2, 3 }, 64, 3), " k view, three outputs summed");
	add(out, mm(F32, F32, 1, 64, 256), " m = 1");
	// f16 x f32
	add(out, mm(F16, F32, 64, 45, 128, { 8, 1 }, { 4, 1 }), " tiled, broadcast");
	add(out, mm(F16, F32, 83, 2, 64, { 8, 1 }, { 4, 1 }), " vec, broadcast");
	add(out, mm(F16, F32, 1056, 1, 67, { 1, 1 }, { 4, 1 }, { 0, 2, 1, 3 }), " vec, permuted, odd k");
	add(out, mm(F16, F32, 128, 1, 1057, { 1, 1 }, { 1, 1 }, { 0, 1, 2, 3 }, 2 * 1056 + 1), " vec, k view");
	add(out, mm(F16, F32, 33, 5, 8), " tiled at n = 5");
	add(out, mm(F16, F32, 2048, 1, 896, { 1, 1 }, { 2, 1 }), " skin-tokens decode (896 x 2048, 2 beams)");
	// bf16 x f32
	add(out, mm(BF16, F32, 16, 16, 256, { 2, 3 }, { 1, 1 }, { 0, 2, 1, 3 }), " permuted");
	add(out, mm(BF16, F32, 16, 1, 256, { 2, 3 }, { 1, 1 }, { 0, 2, 1, 3 }), " vec, permuted");
	add(out, mm(BF16, F32, 65, 67, 17), " one past the tile on every side");
	add(out, mm(BF16, F32, 129, 3, 33, { 2, 1 }, { 2, 1 }), " vec, broadcast");
	add(out, mm(BF16, F32, 192, 130, 96), " Pixal3D block, scaled (1536 x 4608 x 4096)");
	// f16 x f16
	add(out, mm(F16, F16, 64, 77, 77, { 2, 1 }), " tiled");
	add(out, mm(F16, F16, 16, 1, 256, { 3, 2 }, { 2, 2 }), " vec, broadcast");
	{
		L2Case l;
		l.name = "MUL_MAT f32 x f32 skin-tokens attention scores over a permuted K cache (RC) [128,67,16,2] x [128,1,16,2]";
		l.build = attn_scores;
		l.max_nmse = 5e-4;
		out.push_back(l);
	}
	{
		L2Case l;
		l.name = "MUL_MAT f32 x f32 both permuted (RR) [64,100,8] x [64,72,8]";
		l.build = rr_scores;
		l.max_nmse = 5e-4;
		out.push_back(l);
	}
}
