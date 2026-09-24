// L2 cases for the ops interactor-nx-ggml and interactor-rf-detr-ggml need
// beyond the MotionBricks set: the f32 math unaries ABS, SGN, STEP, TANH,
// EXP, FLOOR, SQR, LOG, SIN, COS (ops/unary_math.cpp), SUM and SUM_ROWS
// (ops/sum.cpp), POOL_1D max and avg (ops/pool.cpp), ARGSORT and
// ggml_argsort_top_k (ops/argsort.cpp), SET (ops/set.cpp), UPSCALE bilinear
// with and without align-corners (ops/upscale.cpp) and CONV_2D_DW
// (ops/conv2d_dw.cpp).
//
// Shapes are test-backend-ops' (test_unary's two shapes over [-150, 150];
// test_sqr, test_log over [0.9, 1.1), test_sin/test_cos over [-6.5, 6.5);
// test_sum's permute, test_sum_rows' permute and slice; test_pool1d's three
// shapes over its k0/s0/p0 grid; test_argsort's rows incl. 1023..1025 with
// unique values, both orders; test_set's dim 1..4, in place and not;
// test_upscale's x2 with a transposed source, test_interpolate's ratios and
// its three align-corners cases; test_conv_2d_dw's WHCN and CWHN pairs) plus
// the apps' own: nx-ggml's sum_all, sum_last_axis and reduce_max (a max pool
// over the row), rf-detr's exp(clamp) box widths, sin/cos of the position
// embedding, the top-k of the max-pooled class scores, the box-column and
// per-head SETs (decoder.cpp, deform_attn.cpp), the segmentation head's
// bilinear interpolate and 3x3 depthwise conv (segmentation.cpp), the loss'
// sum of abs. Thresholds are test-backend-ops' (NMSE 1e-7). Bit-exact on the
// host is expected for the unaries, POOL_1D, ARGSORT and SET (each cpp emit
// runs the same libm calls and the same f32 operations in the same order as
// ggml-cpu); SUM and SUM_ROWS differ by rounding (ggml-cpu sums in double),
// and UPSCALE bilinear and CONV_2D_DW by one ULP (the reference is built
// with -mfma and fuses the blend's and the taps' multiply-adds, the emit is
// not), NMSE ~1e-15 either way.
#include <array>
#include <cstdio>
#include <cstring>
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

ggml_tensor *leaf(ggml_context *ctx, Ne ne) {
	return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
}

// test_unary::build_graph: v = 1 views a tensor 3x2x5x4 larger.
ggml_tensor *maybe_view(ggml_context *ctx, Ne ne, bool view) {
	if (!view) {
		return leaf(ctx, ne);
	}
	ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0] * 3, ne[1] * 2, ne[2] * 5, ne[3] * 4);
	return ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a->nb[1], a->nb[2], a->nb[3], 0);
}

/* ---- the math unaries ---- */

using UnaryFn = ggml_tensor *(*)(ggml_context *, ggml_tensor *);

struct MOp {
	const char *name;
	UnaryFn fn;
	float lo, hi; // test-backend-ops' range
};

const MOp kMath[] = {
	{ "ABS", ggml_abs, -150.0f, 150.0f },
	{ "SGN", ggml_sgn, -150.0f, 150.0f },
	{ "STEP", ggml_step, -150.0f, 150.0f },
	{ "TANH", ggml_tanh, -150.0f, 150.0f },
	{ "EXP", ggml_exp, -150.0f, 150.0f },
	{ "FLOOR", ggml_floor, -150.0f, 150.0f },
	{ "SQR", ggml_sqr, -1.0f, 1.0f },
	{ "LOG", ggml_log, 0.9f, 1.1f },
	{ "SIN", ggml_sin, -6.5f, 6.5f },
	{ "COS", ggml_cos, -6.5f, 6.5f },
};

void add_math(std::vector<L2Case> &out, const MOp &m, Ne ne, bool view, float lo, float hi) {
	L2Case c;
	char b[160];
	std::snprintf(b, sizeof b, "%s %s%s range [%g,%g)", m.name, shape(ne).c_str(), view ? " view" : "", lo, hi);
	c.name = b;
	c.lo = lo;
	c.hi = hi;
	const UnaryFn fn = m.fn;
	c.build = [=](ggml_context *ctx) { return fn(ctx, maybe_view(ctx, ne, view)); };
	out.push_back(c);
}

/* ---- ARGSORT: unique values per row (test-backend-ops' fill), so the
   reference's unstable std::sort and the kernel's tie-by-index agree ---- */

bool unique_rows(ggml_tensor *t, std::mt19937 &rng) {
	if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
		return false;
	}
	const int64_t n = t->ne[0];
	const int64_t rows = ggml_nrows(t);
	float *d = static_cast<float *>(t->data);
	for (int64_t r = 0; r < rows; ++r) {
		float *row = d + r * n;
		for (int64_t i = 0; i < n; ++i) {
			row[i] = float(i);
		}
		for (int64_t i = n - 1; i > 0; --i) { // Fisher-Yates
			std::uniform_int_distribution<int64_t> u(0, i);
			std::swap(row[i], row[u(rng)]);
		}
	}
	return true;
}

// Every element of a leaf distinct: a shuffled 0 .. n-1 (a chain whose
// argsort keys are maxima over rows then has no ties either).
bool unique_all(ggml_tensor *t, std::mt19937 &rng) {
	if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
		return false;
	}
	const int64_t n = ggml_nelements(t);
	float *d = static_cast<float *>(t->data);
	for (int64_t i = 0; i < n; ++i) {
		d[i] = float(i);
	}
	for (int64_t i = n - 1; i > 0; --i) {
		std::uniform_int_distribution<int64_t> u(0, i);
		std::swap(d[i], d[u(rng)]);
	}
	return true;
}

void add_argsort(std::vector<L2Case> &out, Ne ne, ggml_sort_order order) {
	L2Case c;
	c.name = std::string("ARGSORT ") + shape(ne) + (order == GGML_SORT_ORDER_ASC ? " asc" : " desc");
	c.init = unique_rows;
	c.build = [=](ggml_context *ctx) { return ggml_argsort(ctx, leaf(ctx, ne), order); };
	out.push_back(c);
}

/* ---- POOL_1D ---- */

void add_pool(std::vector<L2Case> &out, ggml_op_pool op, Ne ne, int k0, int s0, int p0, const char *tag = "") {
	L2Case c;
	char b[160];
	std::snprintf(b, sizeof b, "POOL_1D %s %s k=%d s=%d p=%d%s", op == GGML_OP_POOL_MAX ? "max" : "avg",
			shape(ne).c_str(), k0, s0, p0, tag);
	c.name = b;
	c.build = [=](ggml_context *ctx) { return ggml_pool_1d(ctx, leaf(ctx, ne), op, k0, s0, p0); };
	out.push_back(c);
}

/* ---- SET: test_set::build_graph ---- */

void add_set_tbo(std::vector<L2Case> &out, Ne ne, int dim, bool inplace) {
	L2Case c;
	char b[160];
	std::snprintf(b, sizeof b, "SET %s dim=%d%s", shape(ne).c_str(), dim, inplace ? " inplace" : "");
	c.name = b;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *src = leaf(ctx, ne);
		Ne ne_dst = ne;
		for (int i = 0; i < dim; ++i) {
			ne_dst[i] *= 2;
		}
		ggml_tensor *dst = leaf(ctx, ne_dst);
		size_t offset = 0;
		for (int i = 0; i < dim; ++i) {
			offset += ((ne_dst[i] - ne[i]) / 2) * dst->nb[i];
		}
		return inplace ? ggml_set_inplace(ctx, dst, src, src->nb[1], src->nb[2], src->nb[3], offset)
					   : ggml_set(ctx, dst, src, src->nb[1], src->nb[2], src->nb[3], offset);
	};
	out.push_back(c);
}

/* ---- UPSCALE bilinear ---- */

void add_bilinear(std::vector<L2Case> &out, Ne ne, Ne ne_tgt, bool align, bool transpose = false) {
	L2Case c;
	c.name = std::string("UPSCALE bilinear") + (align ? " align-corners " : " ") + shape(ne) + (transpose ? " T" : "") +
			" -> " + shape(ne_tgt);
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *a = transpose ? ggml_transpose(ctx, leaf(ctx, { ne[1], ne[0], ne[2], ne[3] })) : leaf(ctx, ne);
		const uint32_t mode = GGML_SCALE_MODE_BILINEAR | (align ? GGML_SCALE_FLAG_ALIGN_CORNERS : 0);
		return ggml_interpolate(ctx, a, ne_tgt[0], ne_tgt[1], ne_tgt[2], ne_tgt[3], mode);
	};
	out.push_back(c);
}

/* ---- CONV_2D_DW: test_conv_2d_dw::build_graph ---- */

void add_conv_dw(std::vector<L2Case> &out, Ne ne_input, Ne ne_kernel, int stride, int pad, int dil, bool cwhn,
		const char *tag = "") {
	L2Case c;
	char b[200];
	std::snprintf(b, sizeof b, "CONV_2D_DW in=%s k=%s s=%d p=%d d=%d%s%s", shape(ne_input).c_str(),
			shape(ne_kernel).c_str(), stride, pad, dil, cwhn ? " cwhn" : "", tag);
	c.name = b;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *input = leaf(ctx, ne_input);
		ggml_tensor *kernel = leaf(ctx, ne_kernel);
		if (cwhn) {
			input = ggml_cont(ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
			input = ggml_permute(ctx, input, 2, 0, 1, 3);
			kernel = ggml_cont(ctx, ggml_permute(ctx, kernel, 2, 3, 1, 0));
			kernel = ggml_permute(ctx, kernel, 3, 2, 0, 1);
		}
		return ggml_conv_2d_dw_direct(ctx, kernel, input, stride, stride, pad, pad, dil, dil);
	};
	out.push_back(c);
}

} // namespace

L2_CASES(nx_rfdetr) {
	// The math unaries: test-backend-ops' two shapes over each op's range.
	for (const MOp &m : kMath) {
		add_math(out, m, { 128, 2, 2, 2 }, false, m.lo, m.hi);
		add_math(out, m, { 5, 7, 11, 13 }, true, m.lo, m.hi);
	}
	add_math(out, kMath[6], { 7, 1, 5, 3 }, false, -1.0f, 1.0f); // SQR, test_sqr
	add_math(out, kMath[7], { 10, 5, 4, 3 }, false, 0.9f, 1.1f); // LOG, test_log
	add_math(out, kMath[8], { 10, 2, 2, 2 }, false, -6.5f, 6.5f); // SIN, test_sin
	add_math(out, kMath[9], { 7, 1, 5, 3 }, false, -6.5f, 6.5f); // COS
	add_math(out, kMath[0], { 4, 300, 1, 1 }, false, -2.0f, 2.0f); // ABS, rf-detr's L1 box loss
	add_math(out, kMath[4], { 4, 1225, 1, 1 }, false, -4.0f, 4.0f); // EXP, rf-detr's box widths
	add_math(out, kMath[5], { 2, 300, 8, 4 }, false, -1.0f, 60.0f); // FLOOR, deform_attn's sample corners
	add_math(out, kMath[2], { 1, 300, 8, 4 }, false, -2.0f, 62.0f); // STEP, deform_attn's in-range masks
	add_math(out, kMath[3], { 256, 32, 1, 1 }, false, -4.0f, 4.0f); // TANH, nx-ggml
	add_math(out, kMath[8], { 128, 300, 1, 1 }, false, -700.0f, 700.0f); // SIN, rf-detr's position embedding
	add_math(out, kMath[9], { 128, 300, 1, 1 }, false, -700.0f, 700.0f); // COS
	// A permuted source (dims 1 and 2 swapped, rows contiguous) and in place.
	{
		L2Case c;
		c.name = "ABS on permute(a,0,2,1,3) [33,4,5,3]";
		c.build = [](ggml_context *ctx) {
			return ggml_abs(ctx, ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 33, 4, 5, 3), 0, 2, 1, 3));
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "EXP inplace [67,5,3,2]";
		c.lo = -8.0f;
		c.hi = 8.0f;
		c.build = [](ggml_context *ctx) { return ggml_exp_inplace(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 67, 5, 3, 2)); };
		out.push_back(c);
	}
	// rf-detr's box width: mul(exp(clamp(delta, -4, 4)), base).
	{
		L2Case c;
		c.name = "MUL(EXP(CLAMP(a,-4,4)), b) [2,1225,1,1]";
		c.lo = -6.0f;
		c.hi = 6.0f;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 1225, 1, 1);
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 1225, 1, 1);
			return ggml_mul(ctx, ggml_exp(ctx, ggml_clamp(ctx, a, -4.0f, 4.0f)), b);
		};
		out.push_back(c);
	}

	// SUM: test_sum's shape, plain and permuted, over [-0.9, 1.1); a view;
	// nx-ggml's sum_all of a wide row; rf-detr's L1 loss, sum(abs(diff)).
	{
		L2Case c;
		c.name = "SUM [10,5,4,3]";
		c.lo = -0.9f;
		c.hi = 1.1f;
		c.build = [](ggml_context *ctx) { return ggml_sum(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 10, 5, 4, 3)); };
		out.push_back(c);
	}
	// A permuted source with its rows contiguous: ggml-cpu's SUM reads each
	// row as ne00 consecutive floats (it asserts nb[0] == 4, compiled out in
	// Release), so a permutation of dim 0 would make the reference wrong,
	// not the kernel, which reads every element through its strides.
	{
		L2Case c;
		c.name = "SUM permute(a,0,2,1,3) [10,5,4,3]";
		c.lo = -0.9f;
		c.hi = 1.1f;
		c.build = [](ggml_context *ctx) {
			return ggml_sum(ctx, ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 10, 5, 4, 3), 0, 2, 1, 3));
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SUM view half [12,6,4,2] of [24,6,8,2]";
		c.lo = -0.9f;
		c.hi = 1.1f;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 24, 6, 8, 2);
			return ggml_sum(ctx, ggml_view_4d(ctx, a, 12, 6, 4, 2, a->nb[1], a->nb[2], a->nb[3], a->nb[0] * 4));
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SUM [4096,4,1,1]";
		c.lo = -0.9f;
		c.hi = 1.1f;
		c.build = [](ggml_context *ctx) { return ggml_sum(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4096, 4, 1, 1)); };
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SUM(ABS(a - b)) [4,300,1,1]";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 300, 1, 1);
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 300, 1, 1);
			return ggml_sum(ctx, ggml_abs(ctx, ggml_sub(ctx, a, b)));
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SUM [1,1,1,1]";
		c.build = [](ggml_context *ctx) { return ggml_sum(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1)); };
		out.push_back(c);
	}

	// SUM_ROWS: test_sum_rows plain, permuted (0,2,3,1) and sliced; rf-detr's
	// layer-norm variance, sum_rows(sqr(x - mean)); the keypoint boost; nx's
	// sum_last_axis of a wide row; rows of one element.
	for (int form = 0; form < 3; ++form) {
		L2Case c;
		c.name = std::string("SUM_ROWS [10,5,4,3]") + (form == 1 ? " permute(0,2,3,1)" : (form == 2 ? " slice" : ""));
		c.build = [=](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 10, 5, 4, 3);
			if (form == 2) {
				a = ggml_view_4d(ctx, a, 10, 5, 2, 2, a->nb[1], a->nb[2] * 2, a->nb[3], a->nb[3]);
			}
			if (form == 1) {
				a = ggml_permute(ctx, a, 0, 2, 3, 1);
			}
			return ggml_sum_rows(ctx, a);
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SUM_ROWS(SQR(a - b)) [256,300,1,1]";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 300, 1, 1);
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 300, 1, 1);
			return ggml_sum_rows(ctx, ggml_sqr(ctx, ggml_sub(ctx, a, b)));
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SUM_ROWS [17,300,1,1]";
		c.build = [](ggml_context *ctx) { return ggml_sum_rows(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 17, 300, 1, 1)); };
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SUM_ROWS [4096,3,2,1]";
		c.build = [](ggml_context *ctx) { return ggml_sum_rows(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4096, 3, 2, 1)); };
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SUM_ROWS [1,8,1,1]";
		c.build = [](ggml_context *ctx) { return ggml_sum_rows(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 8, 1, 1)); };
		out.push_back(c);
	}

	// POOL_1D: test_pool1d's three shapes over a slice of its k0/s0/p0 grid,
	// both ops; then the apps' max over the whole row (k0 = s0 = ne0).
	const Ne pool_shapes[] = { { 10, 3, 2, 1 }, { 11, 1, 3, 2 }, { 128, 2, 1, 3 } };
	const int ksp[][3] = { { 1, 1, 0 }, { 3, 1, 0 }, { 3, 2, 1 }, { 3, 3, 0 }, { 1, 2, 1 } };
	for (ggml_op_pool op : { GGML_OP_POOL_MAX, GGML_OP_POOL_AVG }) {
		for (const Ne &ne : pool_shapes) {
			for (const auto &k : ksp) {
				add_pool(out, op, ne, k[0], k[1], k[2]);
			}
		}
	}
	add_pool(out, GGML_OP_POOL_MAX, { 91, 1225, 1, 1 }, 91, 91, 0, " rf-detr scores");
	add_pool(out, GGML_OP_POOL_MAX, { 64, 8, 3, 1 }, 64, 64, 0, " nx reduce_max");
	add_pool(out, GGML_OP_POOL_AVG, { 33, 8, 1, 1 }, 33, 33, 0, " row mean");

	// ARGSORT: test-backend-ops' rows (the 1023..1025 band that catches an
	// off-by-one in a tiled sort), both orders; rf-detr's top-k view.
	add_argsort(out, { 8, 1, 1, 1 }, GGML_SORT_ORDER_ASC);
	add_argsort(out, { 7, 1, 1, 1 }, GGML_SORT_ORDER_DESC);
	for (ggml_sort_order order : { GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC }) {
		add_argsort(out, { 16, 10, 10, 10 }, order);
		add_argsort(out, { 60, 10, 10, 10 }, order);
		add_argsort(out, { 1023, 2, 1, 3 }, order);
		add_argsort(out, { 1025, 2, 1, 3 }, order);
	}
	{
		L2Case c;
		c.name = "ARGSORT_TOP_K [1225,1,1,1] k=300";
		c.init = unique_rows;
		c.build = [](ggml_context *ctx) {
			return ggml_argsort_top_k(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1225, 1, 1, 1), 300);
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "ARGSORT rows of a view [64,6,1,1] of [128,6,1,1]";
		c.init = unique_rows;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 6, 1, 1);
			return ggml_argsort(ctx, ggml_view_4d(ctx, a, 64, 6, 1, 1, a->nb[1], a->nb[2], a->nb[3], 0), GGML_SORT_ORDER_DESC);
		};
		out.push_back(c);
	}
	// The whole rf-detr selection: max-pool the class scores, argsort_top_k,
	// gather the boxes.
	{
		L2Case c;
		c.name = "GET_ROWS(boxes, ARGSORT_TOP_K(POOL_1D max(scores)))";
		c.init = unique_all; // distinct scores, so the row maxima have no ties
		c.build = [](ggml_context *ctx) {
			ggml_tensor *cls = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 91, 1225, 1, 1);
			ggml_tensor *boxes = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 1225, 1, 1);
			ggml_tensor *score = ggml_pool_1d(ctx, cls, GGML_OP_POOL_MAX, 91, 91, 0);
			ggml_tensor *score1d = ggml_reshape_1d(ctx, ggml_cont(ctx, score), 1225);
			ggml_tensor *idx = ggml_reshape_1d(ctx, ggml_argsort_top_k(ctx, score1d, 300), 300);
			return ggml_get_rows(ctx, boxes, idx);
		};
		out.push_back(c);
	}

	// SET: test_set's four dims, in place and not; rf-detr's two box-column
	// sets into [4, n] (decoder.cpp), the eight per-head sets into the
	// attention output (deform_attn.cpp), the class-logit column (set_2d)
	// and a set_1d.
	for (int dim = 1; dim <= 4; ++dim) {
		add_set_tbo(out, { 6, 5, 4, 3 }, dim, false);
		add_set_tbo(out, { 6, 5, 4, 3 }, dim, true);
	}
	{
		L2Case c;
		c.name = "SET box columns: cxcy at 0, wh at 8 bytes into [4,1225,1]";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 1225, 1);
			ggml_tensor *cxcy = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1225, 1);
			ggml_tensor *wh = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1225, 1);
			ggml_tensor *r = ggml_set(ctx, base, cxcy, base->nb[1], base->nb[2], base->nb[3], 0);
			return ggml_set(ctx, r, wh, base->nb[1], base->nb[2], base->nb[3], 2 * sizeof(float));
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SET_2D 8 heads of [32,300] into [256,300]";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *outp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 300);
			for (int h = 0; h < 8; ++h) {
				ggml_tensor *head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 300);
				outp = ggml_set_2d(ctx, outp, head, outp->nb[1], size_t(h) * 32 * sizeof(float));
			}
			return outp;
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SET_2D column [1,300] at 12 bytes into [91,300]";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 91, 300);
			ggml_tensor *col = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 300);
			return ggml_set_2d(ctx, logits, col, logits->nb[1], 3 * sizeof(float));
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "SET_1D [7] at 20 bytes into [33]";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 33);
			ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
			return ggml_set_1d(ctx, a, b, 5 * sizeof(float));
		};
		out.push_back(c);
	}

	// UPSCALE bilinear: test_upscale's x2 (scaled down from [512,512,3,2])
	// with and without a transposed source, test_interpolate's ratios up and
	// down, its three align-corners cases, and rf-detr's segmentation
	// interpolate (channels cut down).
	add_bilinear(out, { 64, 64, 3, 2 }, { 128, 128, 3, 2 }, false);
	add_bilinear(out, { 64, 64, 3, 2 }, { 128, 128, 3, 2 }, false, true);
	add_bilinear(out, { 2, 5, 7, 11 }, { 5, 7, 11, 13 }, false);
	add_bilinear(out, { 5, 7, 11, 13 }, { 2, 5, 7, 11 }, false);
	add_bilinear(out, { 2, 5, 7, 11 }, { 5, 7, 11, 13 }, true);
	add_bilinear(out, { 1, 4, 3, 2 }, { 2, 8, 3, 2 }, true);
	add_bilinear(out, { 4, 1, 3, 2 }, { 1, 1, 3, 2 }, true);
	add_bilinear(out, { 20, 20, 16, 1 }, { 56, 56, 16, 1 }, false);
	add_bilinear(out, { 35, 35, 8, 1 }, { 56, 56, 8, 1 }, false);
	add_bilinear(out, { 9, 7, 2, 3 }, { 9, 7, 2, 3 }, false);

	// CONV_2D_DW: test_conv_2d_dw's WHCN/CWHN pairs (stride 1 no pad; stride
	// 2 pad 1), rf-detr's 3x3 pad 1 block (channels cut down), a dilated and
	// a non-square kernel.
	add_conv_dw(out, { 17, 34, 9, 1 }, { 3, 3, 1, 9 }, 1, 0, 1, false);
	add_conv_dw(out, { 17, 34, 9, 1 }, { 3, 3, 1, 9 }, 1, 0, 1, true);
	add_conv_dw(out, { 32, 8, 64, 1 }, { 3, 3, 1, 64 }, 2, 1, 1, false);
	add_conv_dw(out, { 32, 8, 64, 1 }, { 3, 3, 1, 64 }, 2, 1, 1, true);
	add_conv_dw(out, { 56, 56, 16, 1 }, { 3, 3, 1, 16 }, 1, 1, 1, false, " rf-detr");
	add_conv_dw(out, { 16, 12, 4, 2 }, { 3, 3, 1, 4 }, 1, 2, 2, false);
	add_conv_dw(out, { 13, 11, 3, 2 }, { 5, 3, 1, 3 }, 1, 1, 1, false);
	add_conv_dw(out, { 13, 11, 3, 2 }, { 5, 3, 1, 3 }, 1, 1, 1, true);
}
