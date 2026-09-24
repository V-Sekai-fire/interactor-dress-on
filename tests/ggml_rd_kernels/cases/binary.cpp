// L2 cases for ADD, SUB, MUL and DIV (f32): the template for every family's cases.
//
// The shapes are test-backend-ops' test_bin_bcast ones (broadcast in every
// dimension, a permuted src1, src0/src1 overlapping views of one tensor),
// plus odd sizes, a strided destination (in place on a permuted view) and a
// two-node chain. MUL and DIV draw their operands from [0.9, 1.1) as
// test-backend-ops does. Every case must match ggml-cpu within NMSE 1e-7;
// ADD, SUB, MUL and DIV are one IEEE operation per element, so they are
// expected bit-exact. SUB and DIV run every ADD shape (the same broadcast
// rule); SUB also MotionBricks' masked blend, mul(sub(a, b), mask), and DIV
// test-backend-ops' in-place [16,5,4,3] and the see-through engine's
// normalisation by a [1, n] row sum.
#include <array>
#include <cstdio>
#include <string>

#include "../l2.h"

namespace {

using Ne = std::array<int64_t, 4>;
using Nr = std::array<int, 4>;

enum class Src1 { PLAIN, PERMUTED, OVERLAP };

enum class Op { ADD, SUB, MUL, DIV };
const Op kOps[] = { Op::ADD, Op::SUB, Op::MUL, Op::DIV };

const char *op_name(Op op) {
	switch (op) {
		case Op::ADD:
			return "ADD";
		case Op::SUB:
			return "SUB";
		case Op::MUL:
			return "MUL";
		default:
			return "DIV";
	}
}

ggml_tensor *apply(ggml_context *ctx, Op o, ggml_tensor *a, ggml_tensor *b, bool inplace) {
	switch (o) {
		case Op::ADD:
			return inplace ? ggml_add_inplace(ctx, a, b) : ggml_add(ctx, a, b);
		case Op::SUB:
			return inplace ? ggml_sub_inplace(ctx, a, b) : ggml_sub(ctx, a, b);
		case Op::MUL:
			return inplace ? ggml_mul_inplace(ctx, a, b) : ggml_mul(ctx, a, b);
		default:
			return inplace ? ggml_div_inplace(ctx, a, b) : ggml_div(ctx, a, b);
	}
}

// MUL and DIV draw from [0.9, 1.1), as test_bin_bcast does.
void op_range(L2Case &c, Op o) {
	if (o == Op::MUL || o == Op::DIV) {
		c.lo = 0.9f;
		c.hi = 1.1f;
	}
}

std::string fmt(const char *op, Ne ne, Nr nr, Src1 s1, const char *extra = "") {
	char b[160];
	std::snprintf(b, sizeof b, "%s ne=[%lld,%lld,%lld,%lld] nr=[%d,%d,%d,%d]%s%s", op, (long long)ne[0], (long long)ne[1],
			(long long)ne[2], (long long)ne[3], nr[0], nr[1], nr[2], nr[3],
			s1 == Src1::PERMUTED ? " perm1" : (s1 == Src1::OVERLAP ? " overlap" : ""), extra);
	return b;
}

// test_bin_bcast::build_graph for one op (nf = 1).
ggml_tensor *bcast(ggml_context *ctx, Op o, Ne ne, Nr nr, Src1 s1) {
	ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3]);
	ggml_tensor *b;
	ggml_tensor *x = a;
	if (s1 == Src1::PERMUTED) {
		const int p[4] = { 1, 2, 0, 3 };
		b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[p[0]], ne[p[1]], ne[p[2]], ne[p[3]]);
		b = ggml_permute(ctx, b, p[0], p[1], p[2], p[3]);
	} else if (s1 == Src1::OVERLAP) {
		b = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], 2 * (ne[3] / 3), a->nb[1], a->nb[2], a->nb[3], (ne[3] / 3) * a->nb[3]);
		x = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], 2 * (ne[3] / 3), a->nb[1], a->nb[2], a->nb[3], 0);
	} else {
		b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
	}
	return apply(ctx, o, x, b, false);
}

void add_bcast(std::vector<L2Case> &out, Ne ne, Nr nr, Src1 s1 = Src1::PLAIN) {
	for (Op o : kOps) {
		L2Case c;
		c.name = fmt(op_name(o), ne, nr, s1);
		c.build = [=](ggml_context *ctx) { return bcast(ctx, o, ne, nr, s1); };
		op_range(c, o);
		out.push_back(c);
	}
}

} // namespace

L2_CASES(binary) {
	add_bcast(out, { 1, 1, 8, 1 }, { 1, 1, 1, 1 });
	add_bcast(out, { 1, 1, 1, 1 }, { 32, 1, 1, 1 });
	add_bcast(out, { 10, 5, 1, 1 }, { 1, 1, 1, 1 });
	add_bcast(out, { 10, 5, 4, 1 }, { 1, 1, 1, 1 });
	add_bcast(out, { 10, 5, 4, 3 }, { 1, 1, 1, 1 });
	add_bcast(out, { 10, 5, 4, 3 }, { 2, 1, 1, 1 });
	add_bcast(out, { 10, 5, 4, 3 }, { 1, 2, 1, 1 });
	add_bcast(out, { 10, 5, 4, 3 }, { 1, 1, 2, 1 });
	add_bcast(out, { 10, 5, 4, 3 }, { 1, 1, 1, 2 });
	add_bcast(out, { 10, 5, 4, 3 }, { 1, 1, 2, 2 });
	add_bcast(out, { 10, 5, 4, 3 }, { 1, 2, 2, 2 });
	add_bcast(out, { 10, 5, 4, 3 }, { 2, 2, 2, 2 });
	add_bcast(out, { 10, 5, 4, 3 }, { 1, 1, 1, 1 }, Src1::PERMUTED);
	add_bcast(out, { 10, 5, 4, 3 }, { 2, 2, 2, 2 }, Src1::PERMUTED);
	add_bcast(out, { 10, 5, 4, 6 }, { 1, 1, 1, 1 }, Src1::OVERLAP);
	add_bcast(out, { 7, 3, 5, 2 }, { 3, 1, 1, 1 }); // odd sizes
	add_bcast(out, { 13, 1, 1, 1 }, { 1, 11, 1, 1 });
	add_bcast(out, { 1280, 1, 1, 1 }, { 1, 16, 16, 1 }); // stable diffusion
	add_bcast(out, { 1, 1, 640, 1 }, { 32, 32, 1, 1 });

	// A strided destination: in place on a permuted view of a. (Dims 1 and 2
	// swap; ggml-cpu, the reference, wants dst and src0 rows contiguous.)
	for (Op o : kOps) {
		L2Case c;
		c.name = std::string(op_name(o)) + " inplace on permute(a,0,2,1,3) [6,4,3,2]";
		c.build = [=](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 6, 4, 3, 2);
			ggml_tensor *at = ggml_permute(ctx, a, 0, 2, 1, 3); // [6,3,4,2], rows strided
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 6, 3, 4, 2);
			return apply(ctx, o, at, b, true);
		};
		op_range(c, o);
		out.push_back(c);
	}

	// Two nodes, the second reading the first: (a + b) * c.
	{
		L2Case c;
		c.name = "ADD then MUL chain [9,7,5,3]";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 9, 7, 5, 3);
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 9, 7, 5, 3);
			ggml_tensor *k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 9, 1, 5, 1);
			return ggml_mul(ctx, ggml_add(ctx, a, b), k);
		};
		out.push_back(c);
	}

	// MotionBricks' masked blend: mul(sub(target_emb, hidden), mask) with a
	// [1, n] mask broadcast over the 512 channels (root.cpp:161, pose.cpp:173).
	{
		L2Case c;
		c.name = "SUB then MUL by [1,n] mask [512,9,1,1]";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 512, 9, 1, 1);
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 512, 9, 1, 1);
			ggml_tensor *m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 9, 1, 1);
			return ggml_mul(ctx, ggml_sub(ctx, a, b), m);
		};
		out.push_back(c);
	}

	// DIV: test-backend-ops' in-place case (ggml_div_inplace on [16,5,4,3])
	// and the see-through engine's normalisation of a [n, m] block by its
	// [1, m] row sums.
	{
		L2Case c;
		c.name = "DIV inplace [16,5,4,3]";
		c.lo = 0.9f;
		c.hi = 1.1f;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 5, 4, 3);
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 5, 4, 3);
			return ggml_div_inplace(ctx, a, b);
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "DIV by [1,m] row sums [320,77,1,1]";
		c.lo = 0.9f;
		c.hi = 1.1f;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 320, 77, 1, 1);
			ggml_tensor *s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 77, 1, 1);
			return ggml_div(ctx, a, s);
		};
		out.push_back(c);
	}
}
