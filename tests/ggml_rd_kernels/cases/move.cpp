// L2 cases for the data-movement family: CPY/DUP/CONT, GET_ROWS, CONCAT,
// REPEAT and UPSCALE nearest (ops/cpy.cpp, get_rows.cpp, concat.cpp,
// repeat.cpp, upscale.cpp).
//
// Shapes are test-backend-ops' (test_cpy, test_dup, test_cont, test_get_rows,
// test_concat, test_repeat) plus the census's hottest ones and odd sizes:
// every type pair among f32/f16/bf16, permuted sources and destinations
// (16-bit destinations included: the word-ownership path), strided views,
// reshaping copies, every concat dim with non-contiguous operands, and
// repeats in each dimension, and nearest upscales (test_upscale's x2 with
// and without a transposed source, test_interpolate's non-integer ratios up
// and down, MotionBricks' x2 on ne0). Same-type moves must be bit-exact
// (NMSE 0 in test-backend-ops); conversions within test-backend-ops' 1e-6.
// The integer conversions are ggml-cpu's rounding, so those are expected
// exact too.
#include <array>
#include <cstdio>
#include <string>

#include "../l2.h"

namespace {

using Ne = std::array<int64_t, 4>;

std::string ne_str(Ne ne) {
	char b[64];
	std::snprintf(b, sizeof b, "[%lld,%lld,%lld,%lld]", (long long)ne[0], (long long)ne[1], (long long)ne[2],
			(long long)ne[3]);
	return b;
}

const char *tn(ggml_type t) {
	return ggml_type_name(t);
}

// test_cpy::build_graph.
void add_cpy(std::vector<L2Case> &out, ggml_type ts, ggml_type td, Ne ne, Ne ne_dst = { -1, -1, -1, -1 },
		Ne perm_src = { 0, 0, 0, 0 }, Ne perm_dst = { 0, 0, 0, 0 }, bool transpose = false, Ne alloc = { 0, 0, 0, 0 }) {
	L2Case c;
	const bool ps = perm_src[0] + perm_src[1] + perm_src[2] + perm_src[3] > 0;
	const bool pd = perm_dst[0] + perm_dst[1] + perm_dst[2] + perm_dst[3] > 0;
	const bool shaped = ne_dst[0] >= 0;
	c.name = std::string("CPY ") + tn(ts) + "->" + tn(td) + " " + ne_str(ne) + (shaped ? " to " + ne_str(ne_dst) : "") +
			(ps ? " psrc" + ne_str(perm_src) : "") + (pd ? " pdst" + ne_str(perm_dst) : "") + (transpose ? " T" : "") +
			(alloc[0] > 0 ? " alloc" + ne_str(alloc) : "");
	c.lo = -150.0f;
	c.hi = 150.0f;
	c.max_nmse = ts == td ? 0.0 : 1e-6;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *src = ggml_new_tensor(ctx, ts, 4, ne.data());
		if (ps) {
			src = ggml_permute(ctx, src, int(perm_src[0]), int(perm_src[1]), int(perm_src[2]), int(perm_src[3]));
		}
		if (transpose) {
			src = ggml_transpose(ctx, src);
		}
		Ne dn = shaped ? ne_dst : Ne{ src->ne[0], src->ne[1], src->ne[2], src->ne[3] };
		ggml_tensor *dst;
		if (alloc[0] > 0) {
			ggml_tensor *buf = ggml_new_tensor(ctx, td, 4, alloc.data());
			dst = ggml_view_4d(ctx, buf, dn[0], dn[1], dn[2], dn[3], buf->nb[1], buf->nb[2], buf->nb[3], 0);
		} else {
			dst = ggml_new_tensor(ctx, td, 4, dn.data());
			if (pd) {
				dst = ggml_permute(ctx, dst, int(perm_dst[0]), int(perm_dst[1]), int(perm_dst[2]), int(perm_dst[3]));
			}
		}
		return ggml_cpy(ctx, src, dst);
	};
	out.push_back(c);
}

void add_get_rows(std::vector<L2Case> &out, ggml_type t, int n, int m, int r, int be1, int be2, bool v) {
	L2Case c;
	char b[128];
	std::snprintf(b, sizeof b, "GET_ROWS %s n=%d m=%d r=%d b=%d,%d%s", tn(t), n, m, r, be1, be2, v ? " view" : "");
	c.name = b;
	c.lo = 0.0f; // the i32 leaf is floor(uniform[lo, hi)): a row in [0, m)
	c.hi = float(m) - 0.001f;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *in = ggml_new_tensor_4d(ctx, t, n, m, be1, be2);
		ggml_tensor *rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, r, be1, be2);
		if (v) {
			rows = ggml_view_3d(ctx, rows, r / 2, be1, be2, rows->nb[1], rows->nb[2], 0);
		}
		return ggml_get_rows(ctx, in, rows);
	};
	out.push_back(c);
}

// test_concat::build_graph.
void add_concat(std::vector<L2Case> &out, ggml_type t, Ne ne_a, int64_t ne_b_d, int dim, int v) {
	L2Case c;
	char b[128];
	std::snprintf(b, sizeof b, "CONCAT %s a=%s b_d=%lld dim=%d v=%d", tn(t), ne_str(ne_a).c_str(), (long long)ne_b_d, dim,
			v);
	c.name = b;
	c.max_nmse = 0.0;
	c.build = [=](ggml_context *ctx) {
		Ne ne_b = ne_a;
		ne_b[dim] = ne_b_d;
		ggml_tensor *a;
		if (v & 1) {
			Ne ne = ne_a;
			ne[0] *= 2;
			ne[1] *= 4;
			ne[2] *= 3;
			a = ggml_new_tensor(ctx, t, 4, ne.data());
			a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
		} else {
			a = ggml_new_tensor(ctx, t, 4, ne_a.data());
		}
		ggml_tensor *bt;
		if (v & 2) {
			Ne ne = ne_b;
			ne[0] *= 3;
			ne[1] *= 2;
			ne[2] *= 4;
			bt = ggml_new_tensor(ctx, t, 4, ne.data());
			bt = ggml_view_4d(ctx, bt, ne_b[0], ne_b[1], ne_b[2], ne_b[3], bt->nb[1], bt->nb[2], bt->nb[3], 0);
		} else {
			bt = ggml_new_tensor(ctx, t, 4, ne_b.data());
		}
		return ggml_concat(ctx, a, bt, dim);
	};
	out.push_back(c);
}

void add_repeat(std::vector<L2Case> &out, ggml_type t, Ne ne, std::array<int, 4> nr) {
	L2Case c;
	char b[128];
	std::snprintf(b, sizeof b, "REPEAT %s ne=%s nr=[%d,%d,%d,%d]", tn(t), ne_str(ne).c_str(), nr[0], nr[1], nr[2], nr[3]);
	c.name = b;
	c.max_nmse = 0.0;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *target = ggml_new_tensor_4d(ctx, t, ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3]);
		ggml_tensor *src = ggml_new_tensor(ctx, t, 4, ne.data());
		return ggml_repeat(ctx, src, target);
	};
	out.push_back(c);
}

// test_upscale / test_interpolate::build_graph, mode NEAREST: src ne, dst
// ne_tgt; a transposed source (dims 0 and 1 swapped, then interpolated).
void add_upscale(std::vector<L2Case> &out, Ne ne, Ne ne_tgt, bool transpose = false) {
	L2Case c;
	c.name = "UPSCALE nearest " + ne_str(ne) + (transpose ? " T" : "") + " -> " + ne_str(ne_tgt);
	c.lo = -150.0f;
	c.hi = 150.0f;
	c.max_nmse = 0.0;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *a = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data());
		if (transpose) {
			a = ggml_transpose(ctx, a);
		}
		return ggml_interpolate(ctx, a, ne_tgt[0], ne_tgt[1], ne_tgt[2], ne_tgt[3], GGML_SCALE_MODE_NEAREST);
	};
	out.push_back(c);
}

} // namespace

L2_CASES(move) {
	const ggml_type F32 = GGML_TYPE_F32, F16 = GGML_TYPE_F16, BF16 = GGML_TYPE_BF16;
	const ggml_type kinds[3] = { F32, F16, BF16 };

	// Every type pair, contiguous and "by rows" (test_cpy).
	for (ggml_type s : kinds) {
		for (ggml_type d : kinds) {
			add_cpy(out, s, d, { 256, 4, 4, 4 });
			add_cpy(out, s, d, { 256, 2, 3, 4 }, { -1, -1, -1, -1 }, { 0, 2, 1, 3 });
		}
	}
	// Not-contiguous and permuted destinations, 16-bit ones included.
	for (ggml_type s : { F32, F16 }) {
		for (ggml_type d : { F32, F16 }) {
			add_cpy(out, s, d, { 256, 2, 3, 4 }, { -1, -1, -1, -1 }, { 1, 0, 2, 3 });
		}
	}
	for (ggml_type t : kinds) {
		for (int k = 1; k < 4; ++k) {
			add_cpy(out, t, t, { k, 2, 3, 4 }, { -1, -1, -1, -1 }, { 0, 3, 1, 2 }, { 0, 2, 1, 3 }); // odd words
		}
	}
	add_cpy(out, F16, F16, { 256, 4, 3, 1 }, { -1, -1, -1, -1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, true);
	add_cpy(out, BF16, BF16, { 256, 4, 3, 1 }, { -1, -1, -1, -1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, true);
	add_cpy(out, F32, F32, { 256, 4, 3, 3 }, { -1, -1, -1, -1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, true);
	add_cpy(out, F16, F16, { 128, 2, 3, 1 }, { 128, 2, 3, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, false, { 128, 4, 3, 1 });
	add_cpy(out, F32, F16, { 7, 5, 3, 1 }, { 7, 5, 3, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, false, { 9, 5, 3, 1 });
	add_cpy(out, F32, BF16, { 3, 5, 7, 1 }, { -1, -1, -1, -1 }, { 0, 0, 0, 0 }, { 2, 0, 1, 3 }); // 16-bit dst, odd strides
	// Reshaping copies (test_cpy's {3, 5, 7, 32} family).
	add_cpy(out, F32, F32, { 5, 7, 32, 3 }, { 3, 5, 7, 32 });
	add_cpy(out, F16, F16, { 7, 3, 32, 5 }, { 32, 7, 5, 3 });
	add_cpy(out, F32, F16, { 32, 3, 7, 5 }, { 3, 5, 7, 32 }, { 1, 0, 2, 3 });
	// The census: CONT of permuted views, f32 and f16 (dst [1024, 1, 1024]).
	add_cpy(out, F32, F32, { 128, 8, 2, 514 }, { -1, -1, -1, -1 }, { 0, 2, 1, 3 });
	add_cpy(out, F16, F16, { 1024, 1024, 1, 1 }, { -1, -1, -1, -1 }, { 0, 2, 1, 3 });
	add_cpy(out, BF16, F32, { 1536, 7, 1, 1 });
	// Range edges of the conversions: large, tiny (f16 subnormal) values.
	{
		L2Case c;
		c.name = "CPY f32->f16->f32 wide range [4099,3]";
		c.lo = -65000.0f;
		c.hi = 65000.0f;
		c.max_nmse = 1e-6;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_2d(ctx, F32, 4099, 3);
			ggml_tensor *h = ggml_cpy(ctx, a, ggml_new_tensor_2d(ctx, F16, 4099, 3));
			return ggml_cpy(ctx, h, ggml_new_tensor_2d(ctx, F32, 4099, 3));
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "CPY f32->f16->f32 subnormal range [4099,3]";
		c.lo = -1e-4f;
		c.hi = 1e-4f;
		c.max_nmse = 1e-4;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_2d(ctx, F32, 4099, 3);
			ggml_tensor *h = ggml_cpy(ctx, a, ggml_new_tensor_2d(ctx, F16, 4099, 3));
			return ggml_cpy(ctx, h, ggml_new_tensor_2d(ctx, F32, 4099, 3));
		};
		out.push_back(c);
	}
	// DUP and CONT nodes (test_dup, test_cont).
	for (ggml_type t : { F32, F16 }) {
		L2Case c;
		c.name = std::string("DUP ") + tn(t) + " [10,10,5,1] perm [1,0,2,3]";
		c.max_nmse = 0.0;
		c.build = [=](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, t, 10, 10, 5, 1);
			return ggml_dup(ctx, ggml_permute(ctx, a, 1, 0, 2, 3));
		};
		out.push_back(c);
	}
	for (ggml_type t : kinds) {
		L2Case c;
		c.name = std::string("CONT ") + tn(t) + " transpose [1,8,17,1]";
		c.max_nmse = 0.0;
		c.build = [=](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, t, 1, 8, 17, 1);
			return ggml_cont(ctx, ggml_transpose(ctx, a));
		};
		out.push_back(c);
	}
	{
		L2Case c;
		c.name = "CONT f32 view slice [2,3,5,7]";
		c.max_nmse = 0.0;
		c.build = [](ggml_context *ctx) {
			ggml_tensor *a = ggml_new_tensor_4d(ctx, F32, 2, 3, 5, 7);
			ggml_tensor *s = ggml_view_4d(ctx, a, 2, 1, 5, 7, a->nb[1], a->nb[2], a->nb[3], a->nb[0] * 2);
			return ggml_cont(ctx, s);
		};
		out.push_back(c);
	}

	// GET_ROWS (test_get_rows; the census: f32 [896, 33036] rows).
	for (ggml_type t : kinds) {
		for (int b : { 1, 7 }) {
			for (bool v : { false, true }) {
				add_get_rows(out, t, 256, 5, 4, b, 1, v);
			}
		}
	}
	add_get_rows(out, F32, 896, 3304, 18, 1, 1, false);
	add_get_rows(out, BF16, 33, 17, 9, 2, 3, false);
	add_get_rows(out, GGML_TYPE_I32, 13, 5, 4, 2, 1, true);

	// CONCAT: every dim, non-contiguous operands (test_concat), the KV cache.
	for (int v : { 0, 1, 2, 3 }) {
		for (int dim = 0; dim < 4; ++dim) {
			add_concat(out, F32, { 11, 12, 13, 14 }, 7, dim, v);
			add_concat(out, F16, { 11, 12, 13, 14 }, 7, dim, v);
		}
	}
	add_concat(out, BF16, { 11, 12, 13, 14 }, 7, 0, 3);
	add_concat(out, F32, { 128, 8, 514, 1 }, 1, 2, 0); // skin-tokens KV cache
	add_concat(out, F32, { 128, 8, 515, 1 }, 1, 3, 0);
	add_concat(out, F32, { 1, 64, 12, 40 }, 1, 0, 0);

	// REPEAT (test_repeat) and the census.
	for (int ne3 : { 1, 3 }) {
		add_repeat(out, F32, { 10, 5, 4, ne3 }, { 2, 1, 1, 1 });
		add_repeat(out, F32, { 10, 5, 4, ne3 }, { 1, 2, 1, 1 });
		add_repeat(out, F32, { 10, 5, 4, ne3 }, { 1, 1, 2, 1 });
		add_repeat(out, F32, { 10, 5, 4, ne3 }, { 1, 1, 1, 2 });
		add_repeat(out, F16, { 10, 5, 4, ne3 }, { 2, 1, 1, 1 });
		add_repeat(out, BF16, { 9, 5, 3, ne3 }, { 1, 3, 1, 1 });
	}
	add_repeat(out, F32, { 128, 8, 1, 514 }, { 1, 1, 2, 1 });
	add_repeat(out, F32, { 128, 1, 1, 1 }, { 1, 8, 514, 1 });

	// UPSCALE nearest: test_upscale (x2, scaled down from [512,512,3,2]; and
	// its transposed source), test_interpolate's ratios up and down,
	// MotionBricks' x2 on ne0 (decoder.cpp:173, root.cpp:170), x3, a
	// mixed up/down, and one dimension left alone.
	add_upscale(out, { 64, 64, 3, 2 }, { 128, 128, 3, 2 });
	add_upscale(out, { 64, 64, 3, 2 }, { 128, 128, 3, 2 }, true);
	add_upscale(out, { 2, 5, 7, 11 }, { 5, 7, 11, 13 });
	add_upscale(out, { 5, 7, 11, 13 }, { 2, 5, 7, 11 });
	add_upscale(out, { 512, 9, 1, 1 }, { 1024, 9, 1, 1 });
	add_upscale(out, { 9, 7, 5, 3 }, { 27, 21, 15, 9 });
	add_upscale(out, { 10, 6, 4, 3 }, { 5, 12, 2, 6 });
	add_upscale(out, { 16, 3, 2, 1 }, { 16, 3, 2, 1 });
}
