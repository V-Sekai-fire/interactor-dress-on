// L2 cases for FLASH_ATTN_EXT without a mask (family K8).
//
// The graph is test-backend-ops' test_flash_attn_ext::build_graph with no
// mask and no sinks: Q f32 [D, N, H*nr2, nr3], K and V [D, Lk, H, nr3] as
// views of a cache twice as long (as test-backend-ops builds them), optionally
// all three permuted (0, 2, 1, 3), scale 1/sqrt(D), an optional softcap. The
// shapes cover both head sizes and both K/V types, grouped-query broadcast in
// dims 2 and 3, the census shapes (D = 128, H = 12, f32, Lk = 5 and odd
// Lk = 1029, cut down in N), Lk on and off the 32-key tile (2, 31, 32, 33),
// N on and off the 16-query block (1, 15, 16, 17, 75), and a node reading the
// result (FA then ADD). The host runs the _serial siblings (their tiled
// kernels have no cpp emit); test-backend-ops' threshold is NMSE 5e-4.
#include <cmath>
#include <cstdio>
#include <string>

#include "../l2.h"

namespace {

struct Fa {
	int64_t d = 128, nh = 4, nr2 = 1, nr3 = 1, kv = 96, nb = 8;
	float softcap = 0.0f;
	ggml_type kvt = GGML_TYPE_F32;
	bool permute = false;
	ggml_prec prec = GGML_PREC_F32;
	bool then_add = false;
};

std::string fmt(const Fa &f) {
	char b[200];
	std::snprintf(b, sizeof b, "FA d=%lld nh=%lld nr23=[%lld,%lld] kv=%lld nb=%lld kv_type=%s%s%s%s%s", (long long)f.d,
			(long long)f.nh, (long long)f.nr2, (long long)f.nr3, (long long)f.kv, (long long)f.nb, ggml_type_name(f.kvt),
			f.softcap != 0.0f ? " softcap=10" : "", f.permute ? " perm0213" : "",
			f.prec == GGML_PREC_DEFAULT ? " prec=def" : "", f.then_add ? " +ADD" : "");
	return b;
}

// test_flash_attn_ext::build_graph's create_permuted.
ggml_tensor *make(ggml_context *ctx, ggml_type t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, bool view,
		bool perm) {
	int64_t ne[4] = { ne0, ne1, ne2, ne3 };
	const int p[4] = { 0, 2, 1, 3 };
	int64_t np[4];
	for (int i = 0; i < 4; ++i) {
		np[perm ? p[i] : i] = ne[i];
	}
	ggml_tensor *x;
	if (view) {
		ggml_tensor *t0 = ggml_new_tensor_4d(ctx, t, np[0], 2 * np[1], np[2], np[3]);
		x = ggml_view_4d(ctx, t0, np[0], np[1], np[2], np[3], t0->nb[1], t0->nb[2], t0->nb[3], 0);
	} else {
		x = ggml_new_tensor_4d(ctx, t, np[0], np[1], np[2], np[3]);
	}
	if (perm) {
		x = ggml_permute(ctx, x, 0, 2, 1, 3);
	}
	return x;
}

ggml_tensor *build(ggml_context *ctx, const Fa &f) {
	ggml_tensor *q = make(ctx, GGML_TYPE_F32, f.d, f.nb, f.nh * f.nr2, f.nr3, false, f.permute);
	ggml_tensor *k = make(ctx, f.kvt, f.d, f.kv, f.nh, f.nr3, true, f.permute);
	ggml_tensor *v = make(ctx, f.kvt, f.d, f.kv, f.nh, f.nr3, true, f.permute);
	ggml_tensor *out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt(float(f.d)), 0.0f, f.softcap);
	ggml_flash_attn_ext_set_prec(out, f.prec);
	if (f.then_add) {
		ggml_tensor *r = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], out->ne[2], out->ne[3]);
		out = ggml_add(ctx, out, r);
	}
	return out;
}

void add(std::vector<L2Case> &out, const Fa &f) {
	L2Case c;
	c.name = fmt(f);
	c.build = [=](ggml_context *ctx) { return build(ctx, f); };
	c.max_nmse = 5e-4; // test_flash_attn_ext::max_nmse_err
	out.push_back(c);
}

Fa fa(int64_t d, ggml_type t, int64_t nh, int64_t kv, int64_t nb) {
	Fa f;
	f.d = d;
	f.kvt = t;
	f.nh = nh;
	f.kv = kv;
	f.nb = nb;
	return f;
}

} // namespace

L2_CASES(flash_attn_ext) {
	const ggml_type F32 = GGML_TYPE_F32, F16 = GGML_TYPE_F16;
	add(out, fa(64, F32, 4, 113, 1));
	add(out, fa(64, F32, 4, 512, 3));
	add(out, fa(64, F32, 4, 1024, 32));
	add(out, fa(64, F32, 4, 2, 17)); // two keys (with one, the output is V whatever Q is, and the swap-nb control on Q cannot show)
	add(out, fa(64, F32, 4, 31, 15));
	add(out, fa(64, F32, 4, 33, 16));
	{
		Fa f = fa(64, F32, 4, 512, 75); // grouped-query in dim 2
		f.nr2 = 4;
		add(out, f);
	}
	{
		Fa f = fa(64, F32, 4, 113, 3); // broadcast in dim 3
		f.nr3 = 3;
		add(out, f);
	}
	{
		Fa f = fa(64, F32, 4, 512, 32);
		f.permute = true;
		add(out, f);
	}
	add(out, fa(64, F16, 4, 113, 32));
	{
		Fa f = fa(64, F16, 4, 512, 3);
		f.nr2 = 4;
		f.nr3 = 3;
		add(out, f);
	}
	{
		Fa f = fa(64, F16, 4, 512, 75);
		f.permute = true;
		add(out, f);
	}
	add(out, fa(128, F32, 12, 5, 33)); // census: cross-attention, Lk = 5
	add(out, fa(128, F32, 12, 1029, 17)); // census: odd Lk
	add(out, fa(128, F32, 4, 32, 16));
	{
		Fa f = fa(128, F32, 4, 113, 3);
		f.softcap = 10.0f;
		add(out, f);
	}
	{
		Fa f = fa(128, F32, 4, 512, 32);
		f.permute = true;
		add(out, f);
	}
	{
		Fa f = fa(128, F16, 4, 512, 75);
		f.softcap = 10.0f;
		add(out, f);
	}
	{
		Fa f = fa(128, F16, 4, 512, 1);
		f.nr2 = 12;
		f.prec = GGML_PREC_DEFAULT;
		add(out, f);
	}
	{
		Fa f = fa(128, F16, 4, 1024, 8);
		f.permute = true;
		add(out, f);
	}
	{
		Fa f = fa(64, F32, 2, 77, 9); // a node reading the result
		f.then_add = true;
		add(out, f);
	}
}
