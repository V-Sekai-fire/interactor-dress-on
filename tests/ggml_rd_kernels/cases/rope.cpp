// L2 cases for family K5: ROPE, NEOX mode, f32, i32 positions.
//
// test-backend-ops' NEOX shapes (falcon, stablelm, phi-2, the 16x16xN one
// with N cut to 64) under its YaRN settings (freq_scale 1 or 1.4245,
// ext_factor 0 or 0.7465, attn_factor 1 or 1.4245, beta_fast = beta_slow =
// 1, freq_base 1e4, n_ctx_orig 512, positions uniform in [0, 512)); its
// views (v = 1: a strided view; v = 2: a view starting ne0 elements into
// dim 0) and its in-place case; n_dims < ne0 (the pass-through channels);
// and the skin-tokens row (n_dims 128, freq_base 1e6, n_ctx_orig 3192,
// sequential positions) with fewer tokens. Leaves in [-1, 1) as in
// test-backend-ops; NMSE 1e-7.
#include <array>
#include <cstdio>
#include <cstring>
#include <string>

#include "../l2.h"

namespace {

using Ne = std::array<int64_t, 4>;

struct Rope {
	Ne ne;
	int n_dims;
	float fs = 1.0f, ef = 0.0f, af = 1.0f;
	int v = 0; // 0 plain, 1 strided view, 2 dim-0 offset view
	bool inplace = false;
	float base = 10000.0f;
	int n_ctx_orig = 512;
	float beta_fast = 1.0f, beta_slow = 1.0f;
	bool sequential = false; // positions 0, 1, 2, ... (else uniform in [0, 512))
};

ggml_tensor *build(ggml_context *ctx, const Rope &r) {
	ggml_tensor *a;
	if (r.v == 1) {
		a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, r.ne[0] * 2, r.ne[1] * 4, r.ne[2] * 3, r.ne[3]);
		a = ggml_view_4d(ctx, a, r.ne[0], r.ne[1], r.ne[2], r.ne[3], a->nb[1], a->nb[2], a->nb[3], 0);
	} else if (r.v == 2) {
		a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, r.ne[0] * 2, r.ne[1], r.ne[2], r.ne[3]);
		a = ggml_view_4d(ctx, a, r.ne[0], r.ne[1], r.ne[2], r.ne[3], a->nb[1], a->nb[2], a->nb[3],
				size_t(r.ne[0]) * sizeof(float));
	} else {
		a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, r.ne[0], r.ne[1], r.ne[2], r.ne[3]);
	}
	ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, r.ne[2]);
	auto f = r.inplace ? ggml_rope_ext_inplace : ggml_rope_ext;
	return f(ctx, a, pos, nullptr, r.n_dims, GGML_ROPE_TYPE_NEOX, r.n_ctx_orig, r.base, r.fs, r.ef, r.af, r.beta_fast,
			r.beta_slow);
}

void add(std::vector<L2Case> &out, const Rope &r, const char *tag = "") {
	L2Case c;
	char b[200];
	std::snprintf(b, sizeof b, "ROPE neox [%lld,%lld,%lld,%lld] n_dims=%d fs=%g ef=%g af=%g v=%d%s%s", (long long)r.ne[0],
			(long long)r.ne[1], (long long)r.ne[2], (long long)r.ne[3], r.n_dims, r.fs, r.ef, r.af, r.v,
			r.inplace ? " inplace" : "", tag);
	c.name = b;
	c.build = [=](ggml_context *ctx) { return build(ctx, r); };
	const bool seq = r.sequential;
	c.init = [=](ggml_tensor *t, std::mt19937 &rng) {
		if (t->type != GGML_TYPE_I32) {
			return false;
		}
		for (int64_t i = 0; i < ggml_nelements(t); ++i) {
			const int32_t p = seq ? int32_t(i) : int32_t(rng() % 512u);
			std::memcpy(static_cast<char *>(t->data) + 4 * i, &p, 4);
		}
		return true;
	};
	out.push_back(c);
}

} // namespace

L2_CASES(rope) {
	struct Yarn {
		float fs, ef, af;
	};
	const Yarn yarns[] = { { 1.0f, 0.0f, 1.0f }, { 1.4245f, 0.7465f, 1.4245f }, { 1.0f, 0.7465f, 1.0f },
		{ 1.4245f, 0.0f, 1.0f } };
	for (const Yarn &y : yarns) {
		Rope r{ { 64, 128, 2, 1 }, 64 }; // falcon 40B, test-backend-ops' every-combination shape
		r.fs = y.fs;
		r.ef = y.ef;
		r.af = y.af;
		add(out, r);
		r.v = 1;
		add(out, r);
	}
	const Rope shapes[] = {
		{ { 64, 1, 2, 1 }, 64 },
		{ { 64, 71, 2, 1 }, 64 },
		{ { 64, 8, 2, 1 }, 64 },
		{ { 80, 32, 2, 1 }, 20 }, // stablelm: 60 pass-through channels
		{ { 80, 32, 2, 1 }, 32 },
		{ { 80, 32, 4, 1 }, 32 },
		{ { 16, 16, 64, 1 }, 16 },
	};
	for (Rope r : shapes) {
		r.fs = 1.4245f;
		r.ef = 0.7465f;
		r.af = 1.4245f;
		add(out, r);
	}
	{
		Rope r{ { 36, 16, 57, 1 }, 36 }; // build_rope_2d: a view ne0 elements into dim 0
		r.v = 2;
		add(out, r);
		r.fs = 1.4245f;
		r.ef = 0.7465f;
		r.af = 1.4245f;
		add(out, r);
	}
	{
		Rope r{ { 128, 32, 2, 3 }, 128 }; // test-backend-ops' in-place case
		r.fs = 1.4245f;
		r.ef = 0.7465f;
		r.af = 1.4245f;
		r.inplace = true;
		r.v = 1;
		add(out, r);
		r.v = 0;
		add(out, r);
	}
	{
		// skin-tokens: [128, 8, 514] with n_dims 128, freq_base 1e6,
		// n_ctx_orig 3192, positions 0..513; here 96 tokens.
		Rope r{ { 128, 8, 96, 1 }, 128 };
		r.base = 1.0e6f;
		r.n_ctx_orig = 3192;
		r.beta_fast = 32.0f;
		r.sequential = true;
		add(out, r, " base=1e6 (skin-tokens)");
	}
}
