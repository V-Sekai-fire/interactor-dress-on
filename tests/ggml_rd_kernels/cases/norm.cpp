// L2 cases for NORM, RMS_NORM and MEAN (f32, family K3).
//
// test-backend-ops' own shapes (n = 64 and 1025 over [n,5,4,3], a strided
// view of half of each dimension, permuted rows, in place, eps from 0 to 10),
// the census's shapes (Pixal3D's DINO and flow NORMs, Skin-Tokens' RMS_NORMs;
// the largest cut down in rows only, so a row is still a census row), odd
// sizes, and the chains the models build (NORM * w + b, RMS_NORM * w). The
// kernels run as their Serial siblings (the same partial sums in the same
// order as the kernels, 256 or 64 threads as the packer picks by row length). Thresholds are test-backend-ops': NMSE
// 1e-7 for all three ops; RMS_NORM draws from [-10, 10) and MEAN from
// [-0.9, 1.1), as test-backend-ops does.
#include <array>
#include <cstdio>
#include <string>

#include "../l2.h"

namespace {

using Ne = std::array<int64_t, 4>;

enum class Form { PLAIN, VIEW, PERMUTED_ROWS, INPLACE };

const char *form_name(Form f) {
	switch (f) {
		case Form::VIEW:
			return " view";
		case Form::PERMUTED_ROWS:
			return " permuted-rows";
		case Form::INPLACE:
			return " inplace";
		default:
			return "";
	}
}

std::string fmt(const char *op, Ne ne, Form f, float eps, const char *extra = "") {
	char b[200];
	std::snprintf(b, sizeof b, "%s ne=[%lld,%lld,%lld,%lld]%s eps=%g%s", op, (long long)ne[0], (long long)ne[1],
			(long long)ne[2], (long long)ne[3], form_name(f), eps, extra);
	return b;
}

// test_norm / test_rms_norm's source: plain, a view of half of every
// dimension (rows strided), or permuted rows (dims 0 and 1 swapped: elements
// strided).
ggml_tensor *source(ggml_context *ctx, Ne ne, Form f) {
	if (f == Form::PERMUTED_ROWS) {
		ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[1], ne[0], ne[2], ne[3]);
		return ggml_permute(ctx, a, 1, 0, 2, 3);
	}
	ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
	if (f == Form::VIEW) {
		a = ggml_view_4d(ctx, a, a->ne[0] / 2, a->ne[1] / 2, a->ne[2] / 2, a->ne[3] / 2, a->nb[1], a->nb[2], a->nb[3], 0);
	}
	return a;
}

void add_norm(std::vector<L2Case> &out, Ne ne, Form f, float eps) {
	L2Case c;
	c.name = fmt("NORM", ne, f, eps);
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *a = source(ctx, ne, f);
		return f == Form::INPLACE ? ggml_norm_inplace(ctx, a, eps) : ggml_norm(ctx, a, eps);
	};
	out.push_back(c);
}

void add_rms(std::vector<L2Case> &out, Ne ne, Form f, float eps) {
	L2Case c;
	c.name = fmt("RMS_NORM", ne, f, eps);
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *a = source(ctx, ne, f);
		return f == Form::INPLACE ? ggml_rms_norm_inplace(ctx, a, eps) : ggml_rms_norm(ctx, a, eps);
	};
	c.lo = -10.0f;
	c.hi = 10.0f;
	out.push_back(c);
}

void add_mean(std::vector<L2Case> &out, Ne ne, Form f = Form::PLAIN) {
	L2Case c;
	c.name = fmt("MEAN", ne, f, 0.0f);
	c.build = [=](ggml_context *ctx) { return ggml_mean(ctx, source(ctx, ne, f)); };
	c.lo = -0.9f;
	c.hi = 1.1f;
	out.push_back(c);
}

} // namespace

L2_CASES(norm) {
	// test-backend-ops' NORM / RMS_NORM shapes.
	for (float eps : { 0.0f, 1e-6f, 10.0f }) {
		for (int64_t n : { 64, 1025 }) {
			add_norm(out, { n, 5, 4, 3 }, Form::PLAIN, eps);
			add_rms(out, { n, 5, 4, 3 }, Form::PLAIN, eps);
		}
	}
	add_norm(out, { 1025, 5, 4, 3 }, Form::VIEW, 1e-4f);
	add_rms(out, { 1025, 5, 4, 3 }, Form::VIEW, 1e-4f);
	add_norm(out, { 64, 5, 4, 3 }, Form::PERMUTED_ROWS, 1e-1f);
	add_norm(out, { 1025, 5, 4, 3 }, Form::PERMUTED_ROWS, 1e-6f);
	add_norm(out, { 64, 5, 4, 3 }, Form::INPLACE, 1e-6f);
	add_rms(out, { 64, 5, 4, 3 }, Form::INPLACE, 1e-6f);
	add_norm(out, { 7, 3, 5, 2 }, Form::PLAIN, 1e-5f); // odd sizes
	add_rms(out, { 1, 3, 5, 2 }, Form::PLAIN, 1e-5f); // a row of one
	add_norm(out, { 257, 1, 1, 1 }, Form::PLAIN, 1e-5f); // one element past a stride

	// The census's shapes (eps as the models pass it).
	add_norm(out, { 1024, 1029, 1, 1 }, Form::PLAIN, 1e-5f); // DINO
	add_norm(out, { 512, 16, 16, 16 }, Form::PLAIN, 1e-5f); // sparse-structure decoder
	add_norm(out, { 1536, 512, 1, 1 }, Form::PLAIN, 1e-6f); // flow blocks (4096 rows there)
	add_norm(out, { 512, 5400, 1, 1 }, Form::PLAIN, 1e-5f); // Skin-Tokens (54000 rows there)
	add_rms(out, { 896, 512, 1, 1 }, Form::PLAIN, 1.1920929e-7f); // Skin-Tokens
	add_rms(out, { 128, 8, 514, 1 }, Form::PLAIN, 1e-6f); // Skin-Tokens q/k norm
	add_rms(out, { 128, 12, 912, 1 }, Form::PLAIN, 1e-12f); // Pixal3D q/k norm

	// MEAN: test-backend-ops' shapes, and rows strided.
	add_mean(out, { 10, 5, 4, 3 });
	add_mean(out, { 33, 1, 1, 1 });
	add_mean(out, { 33, 256, 1, 1 });
	add_mean(out, { 32769, 1, 1, 1 });
	add_mean(out, { 32, 256, 1, 1 });
	add_mean(out, { 32768, 1, 1, 1 });
	add_mean(out, { 256, 256, 3, 1 });
	add_mean(out, { 66, 10, 8, 6 }, Form::VIEW);

	// Chains as the models build them: (a + w + b) normed, * w, + b (test_norm_mul_add), and RMS_NORM * w.
	{
		L2Case c;
		c.name = "NORM_MUL_ADD chain [1025,5,4,3] eps=1e-5";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1025, 5, 4, 3);
			ggml_tensor *w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1025, 5, 4, 3);
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1025, 5, 4, 3);
			a = ggml_add(ctx, ggml_add(ctx, a, w), b);
			return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, a, 1e-5f), w), b);
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "RMS_NORM_MUL chain [896,64] eps=1e-6";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 896, 64, 1, 1);
			ggml_tensor *w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 896, 1, 1, 1);
			return ggml_mul(ctx, ggml_rms_norm(ctx, a, 1e-6f), w);
		};
		c.lo = -10.0f;
		c.hi = 10.0f;
		out.push_back(c);
	}
}
