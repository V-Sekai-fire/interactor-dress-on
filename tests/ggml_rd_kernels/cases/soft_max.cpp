// L2 cases for SOFT_MAX (f32, scale, no mask, no sinks, max_bias 0; family K4).
//
// test-backend-ops' unmasked shapes ([10,5,4,3]; 16 and 1024 and one less
// in each of ne0 and ne1 at scale 1 and 0.1; in place on [16,2,32,1]; the
// 200000- and 643251-long rows), the census's (Skin-Tokens' 512x512x8
// attention and its 54000-long rows, DINO's 1029 rows at scale 0.125, cut down
// in rows only), a spread wide enough that exp without the max would
// overflow, and a chain. SOFT_MAX needs a contiguous source (ggml asserts
// it), so there is no view or permuted case. The kernels run as their Serial
// siblings (the same partial maxima and sums in the same order; 256 or 64
// threads as the packer picks by row length). Threshold:
// test-backend-ops' NMSE 1e-6 for SOFT_MAX.
#include <array>
#include <cstdio>
#include <string>

#include "../l2.h"

namespace {

using Ne = std::array<int64_t, 4>;

void add_sm(std::vector<L2Case> &out, Ne ne, float scale, bool inplace = false, float lo = -1.0f, float hi = 1.0f,
		const char *extra = "") {
	char b[200];
	std::snprintf(b, sizeof b, "SOFT_MAX ne=[%lld,%lld,%lld,%lld] scale=%g%s%s", (long long)ne[0], (long long)ne[1],
			(long long)ne[2], (long long)ne[3], scale, inplace ? " inplace" : "", extra);
	L2Case c;
	c.name = b;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
		return inplace ? ggml_soft_max_ext_inplace(ctx, a, nullptr, scale, 0.0f)
					   : ggml_soft_max_ext(ctx, a, nullptr, scale, 0.0f);
	};
	c.lo = lo;
	c.hi = hi;
	c.max_nmse = 1e-6;
	out.push_back(c);
}

} // namespace

L2_CASES(soft_max) {
	add_sm(out, { 10, 5, 4, 3 }, 1.0f);
	for (float scale : { 1.0f, 0.1f }) {
		for (int64_t ne0 : { 16, 1024 }) {
			for (int64_t ne1 : { 16, 1024 }) {
				if (ne0 == 1024 && ne1 == 1024 && scale == 1.0f) {
					continue; // the 1024x1024 pair once, at 0.1
				}
				add_sm(out, { ne0, ne1, 1, 1 }, scale);
				add_sm(out, { ne0 - 1, ne1 - 1, 1, 1 }, scale);
			}
		}
	}
	add_sm(out, { 16, 2, 32, 1 }, 0.1f, true);
	add_sm(out, { 200000, 4, 1, 1 }, 1.0f);
	add_sm(out, { 643251, 3, 1, 1 }, 1.0f);
	add_sm(out, { 7, 3, 5, 2 }, 1.0f); // odd sizes
	add_sm(out, { 1, 3, 5, 2 }, 1.0f); // a row of one: exactly 1, whatever it reads
	out.back().value_blind = true;
	add_sm(out, { 300, 64, 1, 1 }, 1.0f, false, -200.0f, 200.0f, " x in [-200,200)"); // exp(x) alone overflows

	// The census's shapes.
	add_sm(out, { 512, 512, 8, 1 }, 1.0f); // Skin-Tokens attention
	add_sm(out, { 54000, 8, 2, 1 }, 1.0f); // Skin-Tokens, rows > 16384 (512 x 8 of them there)
	add_sm(out, { 1029, 1029, 2, 1 }, 0.125f); // DINO (16 heads there)

	// A chain: softmax, then an ADD reading it.
	{
		L2Case c;
		c.name = "SOFT_MAX then ADD chain [77,64,4] scale=0.125";
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 77, 64, 4, 1);
			ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 77, 1, 4, 1);
			return ggml_add(ctx, ggml_soft_max_ext(ctx, a, nullptr, 0.125f, 0.0f), b);
		};
		c.max_nmse = 1e-6;
		out.push_back(c);
	}
}
