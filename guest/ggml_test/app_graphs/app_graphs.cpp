// The apps' graph builders (app_graphs.h says which, from where). The bodies
// are the apps' code with the model lookups made parameters; comments marked
// "adapted:" say where a line differs from its source.
#include "app_graphs.h"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace app_graphs {

// --- skin-tokens-ggml src/qwen.cpp @097a0cc ------------------------------------

namespace qwen {

namespace {

ggml_tensor *required(const WeightFn &weights, const std::string &name) {
	ggml_tensor *value = weights(name);
	if (value == nullptr) {
		throw std::runtime_error("missing TokenRig tensor: " + name);
	}
	return value;
}

} // namespace

ggml_tensor *linear(ggml_context *context, ggml_tensor *input, ggml_tensor *weight) {
	auto *output = ggml_mul_mat(context, weight, input);
	ggml_mul_mat_set_prec(output, GGML_PREC_F32);
	return output;
}

ggml_tensor *rms(ggml_context *context, ggml_tensor *input, ggml_tensor *weight, float epsilon) {
	auto *normalized = ggml_rms_norm(context, input, epsilon);
	return ggml_mul(context, normalized, ggml_repeat(context, weight, normalized));
}

ggml_tensor *repeat_kv(ggml_context *context, ggml_tensor *value, int64_t sequence) {
	auto *grouped = ggml_reshape_4d(context, value, head_dim, kv_heads, 1, sequence);
	auto *shape = ggml_new_tensor_4d(context, value->type, head_dim, kv_heads, heads / kv_heads, sequence);
	auto *repeated = ggml_repeat(context, grouped, shape);
	repeated = ggml_cont(context, ggml_permute(context, repeated, 0, 2, 1, 3));
	return ggml_reshape_3d(context, repeated, head_dim, heads, sequence);
}

ggml_tensor *repeat_kv_batched(ggml_context *context, ggml_tensor *value, int64_t sequence, int64_t batch) {
	value = ggml_reshape_3d(context, value, head_dim, kv_heads, sequence * batch);
	value = repeat_kv(context, value, sequence * batch);
	return ggml_reshape_4d(context, value, head_dim, heads, sequence, batch);
}

ggml_tensor *cache_view(ggml_context *context, ggml_tensor *cache, size_t slot, size_t length, size_t position) {
	return ggml_view_4d(context, cache, head_dim, kv_heads, static_cast<int64_t>(length), 1, cache->nb[1],
			cache->nb[2], cache->nb[3], slot * cache->nb[3] + position * cache->nb[2]);
}

// adapted: a member of qwen_graph_evaluator there; keys_[layer]/values_[layer]
// and weights_ are parameters here.
ggml_tensor *decode_layer(ggml_context *context, ggml_tensor *input, ggml_tensor *position,
		const std::vector<size_t> &slots, size_t layer, size_t past, const WeightFn &weights, ggml_tensor *keys,
		ggml_tensor *values) {
	const std::string prefix = "llm.l." + std::to_string(layer) + ".";
	const auto get = [&](const char *suffix) { return required(weights, prefix + suffix); };
	const auto batch = static_cast<int64_t>(slots.size());
	auto *normalized = rms(context, input, get("an.w"), 1e-6F);
	auto *q = ggml_reshape_4d(context, linear(context, normalized, get("attn.q.w")), head_dim, heads, 1, batch);
	auto *k_new = ggml_reshape_4d(context, linear(context, normalized, get("attn.k.w")), head_dim, kv_heads, 1, batch);
	auto *v_new = ggml_reshape_4d(context, linear(context, normalized, get("attn.v.w")), head_dim, kv_heads, 1, batch);
	q = rms(context, q, get("attn.q_norm.w"), 1e-6F);
	k_new = rms(context, k_new, get("attn.k_norm.w"), 1e-6F);
	q = ggml_rope_ext(context, q, position, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 3192, 1'000'000.0F, 1.0F, 0.0F,
			1.0F, 0.0F, 0.0F);
	k_new = ggml_rope_ext(context, k_new, position, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 3192, 1'000'000.0F, 1.0F,
			0.0F, 1.0F, 0.0F, 0.0F);
	ggml_tensor *all_k = nullptr;
	ggml_tensor *all_v = nullptr;
	for (size_t beam = 0; beam < slots.size(); ++beam) {
		auto *beam_k = ggml_view_4d(context, k_new, head_dim, kv_heads, 1, 1, k_new->nb[1], k_new->nb[2], k_new->nb[3],
				beam * k_new->nb[3]);
		auto *beam_v = ggml_view_4d(context, v_new, head_dim, kv_heads, 1, 1, v_new->nb[1], v_new->nb[2], v_new->nb[3],
				beam * v_new->nb[3]);
		beam_k = ggml_cpy(context, beam_k, cache_view(context, keys, slots[beam], 1U, past));
		beam_v = ggml_cpy(context, beam_v, cache_view(context, values, slots[beam], 1U, past));
		auto *slot_k = ggml_concat(context, cache_view(context, keys, slots[beam], past), beam_k, 2);
		auto *slot_v = ggml_concat(context, cache_view(context, values, slots[beam], past), beam_v, 2);
		all_k = all_k == nullptr ? slot_k : ggml_concat(context, all_k, slot_k, 3);
		all_v = all_v == nullptr ? slot_v : ggml_concat(context, all_v, slot_v, 3);
	}
	const auto sequence = static_cast<int64_t>(past + 1U);
	all_k = repeat_kv_batched(context, all_k, sequence, batch);
	all_v = repeat_kv_batched(context, all_v, sequence, batch);
	q = ggml_permute(context, q, 0, 2, 1, 3);
	all_k = ggml_permute(context, all_k, 0, 2, 1, 3);
	all_v = ggml_permute(context, all_v, 0, 2, 1, 3);
	auto *scores = ggml_mul_mat(context, all_k, q);
	ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
	scores = ggml_scale(context, scores, 1.0F / std::sqrt(static_cast<float>(head_dim)));
	auto *attended = ggml_mul_mat(context, ggml_cont(context, ggml_transpose(context, all_v)), ggml_soft_max(context, scores));
	ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
	attended = ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3));
	auto *state = ggml_add(context, input,
			linear(context, ggml_reshape_3d(context, attended, heads * head_dim, 1, batch), get("attn.o.w")));
	normalized = rms(context, state, get("fn.w"), 1e-6F);
	auto *gate = ggml_silu(context, linear(context, normalized, get("mlp.g.w")));
	auto *up = linear(context, normalized, get("mlp.u.w"));
	return ggml_add(context, state, linear(context, ggml_mul(context, gate, up), get("mlp.d.w")));
}

// adapted: the graph half of qwen_graph_evaluator::decode (allocation, the
// inputs and the readback are the caller's); n_layers is 28 there.
ggml_tensor *decode_step(ggml_context *context, ggml_tensor *ids, ggml_tensor *position,
		const std::vector<size_t> &slots, size_t past, const WeightFn &weights,
		const std::vector<ggml_tensor *> &keys, const std::vector<ggml_tensor *> &values, int n_layers) {
	const size_t batch = slots.size();
	auto *state = ggml_reshape_3d(context, ggml_get_rows(context, required(weights, "llm.tok.w"), ids), hidden, 1,
			static_cast<int64_t>(batch));
	for (int layer = 0; layer < n_layers; ++layer) {
		state = decode_layer(context, state, position, slots, size_t(layer), past, weights, keys[size_t(layer)],
				values[size_t(layer)]);
	}
	auto *normalized = rms(context, state, required(weights, "llm.norm.w"), 1e-6F);
	auto *last = ggml_reshape_2d(context, normalized, hidden, static_cast<int64_t>(batch));
	return linear(context, last, required(weights, "llm.out.w"));
}

} // namespace qwen

// --- pixal3d-ggml trellis2.cpp @1c22f5e: the sparse-structure flow DiT ----------

namespace dit {

// Scaled dot-product attention via ggml_flash_attn_ext; q3/k3/v3 are
// [head_dim, n_head, L]; returns [n_head*head_dim, L_q]. adapted: the
// TRELLIS2_SDPA_EXACT switch (the materialized path) is not taken here.
ggml_tensor *sdpa_auto(ggml_context *ctx, ggml_tensor *q3, ggml_tensor *k3, ggml_tensor *v3, int C, float scale) {
	ggml_tensor *qp = ggml_cont(ctx, ggml_permute(ctx, q3, 0, 2, 1, 3)); // [hd, Lq, H]
	ggml_tensor *kp = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3)); // [hd, Lk, H]
	ggml_tensor *vp = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3)); // [hd, Lk, H]
	ggml_tensor *o = ggml_flash_attn_ext(ctx, qp, kp, vp, nullptr, scale, 0.0f, 0.0f);
	ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
	return ggml_reshape_2d(ctx, o, C, o->ne[2]); // [C, Lq]
}

std::vector<float> timestep_embedding(float t, int dim) {
	std::vector<float> e((size_t)dim, 0.0f);
	const int half = dim / 2;
	for (int i = 0; i < half; ++i) {
		const float freq = std::exp(-std::log(10000.0f) * (float)i / (float)half);
		const float arg = t * freq;
		e[i] = std::cos(arg);
		e[half + i] = std::sin(arg);
	}
	return e;
}

void rope_tables(int res, int head_dim, float freq_min, float freq_base, std::vector<float> &cos_t,
		std::vector<float> &sin_t) {
	const int dim = 3;
	const int freq_dim = head_dim / 2 / dim;
	const int N = res * res * res;
	std::vector<float> freqs((size_t)freq_dim);
	for (int mi = 0; mi < freq_dim; ++mi) {
		freqs[mi] = freq_min / std::pow(freq_base, (float)mi / (float)freq_dim);
	}
	cos_t.assign((size_t)head_dim * N, 1.0f);
	sin_t.assign((size_t)head_dim * N, 0.0f);
	const int pairs = head_dim / 2;
	for (int n = 0; n < N; ++n) {
		const int coord[3] = { n / (res * res), (n / res) % res, n % res };
		for (int p = 0; p < pairs; ++p) {
			float theta = 0.0f;
			if (p < dim * freq_dim) {
				theta = (float)coord[p / freq_dim] * freqs[p % freq_dim];
			}
			const size_t base = (size_t)n * head_dim + (size_t)2 * p;
			cos_t[base] = cos_t[base + 1] = std::cos(theta);
			sin_t[base] = sin_t[base + 1] = std::sin(theta);
		}
	}
}

namespace {

// adapted: trellis2_ss_flow_forward's lambdas (lin, modulate, rope, qk_norm,
// sdpa) as members, so block() and forward() share them.
struct B {
	ggml_context *ctx;
	const Hparams &hp;
	const WeightFn &W;
	const Leaves &in;
	int C, N, H, hd;
	float attn_scale;
	std::string missing;

	B(ggml_context *c, const Hparams &h, const WeightFn &w, const Leaves &l) :
			ctx(c), hp(h), W(w), in(l), C(h.model_channels), N(h.tokens()), H(h.num_heads), hd(h.head_dim()),
			attn_scale(1.0f / std::sqrt((float)h.head_dim())) {}

	ggml_tensor *w(const std::string &n) {
		ggml_tensor *t = W(n);
		if (t == nullptr && missing.empty()) {
			missing = n;
		}
		return t;
	}
	ggml_tensor *lin(ggml_tensor *x, const std::string &pfx) {
		ggml_tensor *y = ggml_mul_mat(ctx, w(pfx + ".weight"), x);
		ggml_tensor *b = W(pfx + ".bias");
		if (b) {
			y = ggml_add(ctx, y, b);
		}
		return y;
	}
	// h * (1 + scale) + shift, broadcasting the [C] vectors over tokens.
	ggml_tensor *modulate(ggml_tensor *h, ggml_tensor *scale, ggml_tensor *shift) {
		return ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, h, scale), h), shift);
	}
	// interleaved RoPE on a [hd, H, N] tensor using the cos/sin tables.
	ggml_tensor *rope(ggml_tensor *q3) {
		ggml_tensor *q4 = ggml_reshape_4d(ctx, q3, 2, hd / 2, H, N);
		ggml_tensor *q0 = ggml_cont(ctx, ggml_view_4d(ctx, q4, 1, hd / 2, H, N, q4->nb[1], q4->nb[2], q4->nb[3], 0));
		ggml_tensor *q1 = ggml_cont(ctx, ggml_view_4d(ctx, q4, 1, hd / 2, H, N, q4->nb[1], q4->nb[2], q4->nb[3], q4->nb[0]));
		ggml_tensor *swap = ggml_concat(ctx, ggml_neg(ctx, q1), q0, 0); // [2,hd/2,H,N]
		swap = ggml_reshape_3d(ctx, swap, hd, H, N);
		return ggml_add(ctx, ggml_mul(ctx, q3, in.cos_t), ggml_mul(ctx, swap, in.sin_t));
	}
	// QK-RMSNorm: F.normalize(x)*gamma*sqrt(hd) == rms_norm(x)*gamma.
	ggml_tensor *qk_norm(ggml_tensor *v3, const std::string &gname) {
		return ggml_mul(ctx, ggml_rms_norm(ctx, v3, 1e-12f), w(gname));
	}
	ggml_tensor *sdpa(ggml_tensor *q3, ggml_tensor *k3, ggml_tensor *v3) {
		return sdpa_auto(ctx, q3, k3, v3, C, attn_scale);
	}

	ggml_tensor *block(ggml_tensor *h, ggml_tensor *tmod, int b) {
		const size_t es = sizeof(float);
		const int Lkv = int(in.cnd->ne[1]);
		const std::string blk = "blocks." + std::to_string(b);
		ggml_tensor *mods = ggml_add(ctx, w(blk + ".modulation"), tmod); // [6C]
		auto chunk = [&](int idx) { return ggml_view_1d(ctx, mods, C, (size_t)idx * C * es); };
		ggml_tensor *shift_msa = chunk(0), *scale_msa = chunk(1), *gate_msa = chunk(2);
		ggml_tensor *shift_mlp = chunk(3), *scale_mlp = chunk(4), *gate_mlp = chunk(5);

		// self-attention (norm1 affine-free, modulated; RoPE + QK-RMSNorm)
		ggml_tensor *hn = modulate(ggml_norm(ctx, h, 1e-6f), scale_msa, shift_msa);
		ggml_tensor *qkv = lin(hn, blk + ".self_attn.to_qkv"); // [3C, N]
		ggml_tensor *q = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], 0)), hd, H, N);
		ggml_tensor *k = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], (size_t)C * es)), hd, H, N);
		ggml_tensor *v = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], (size_t)2 * C * es)), hd, H, N);
		q = rope(qk_norm(q, blk + ".self_attn.q_rms_norm.gamma"));
		k = rope(qk_norm(k, blk + ".self_attn.k_rms_norm.gamma"));
		ggml_tensor *sa = lin(sdpa(q, k, v), blk + ".self_attn.to_out");
		h = ggml_add(ctx, h, ggml_mul(ctx, sa, gate_msa));

		// cross-attention (norm2 affine; QK-RMSNorm, no RoPE, no gate)
		ggml_tensor *h2 = ggml_norm(ctx, h, 1e-6f);
		h2 = ggml_add(ctx, ggml_mul(ctx, h2, w(blk + ".norm2.weight")), w(blk + ".norm2.bias"));
		ggml_tensor *cq = ggml_reshape_3d(ctx, lin(h2, blk + ".cross_attn.to_q"), hd, H, N);
		cq = qk_norm(cq, blk + ".cross_attn.q_rms_norm.gamma");
		ggml_tensor *kv = lin(in.cnd, blk + ".cross_attn.to_kv"); // [2C, Lkv]
		ggml_tensor *ck = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, kv, C, Lkv, kv->nb[1], 0)), hd, H, Lkv);
		ggml_tensor *cv = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, kv, C, Lkv, kv->nb[1], (size_t)C * es)), hd, H, Lkv);
		ck = qk_norm(ck, blk + ".cross_attn.k_rms_norm.gamma");
		ggml_tensor *ca = lin(sdpa(cq, ck, cv), blk + ".cross_attn.to_out");
		if (in.proj != nullptr) {
			// adapted: Pixal3D's ProjectAttention (proj_attention.py) returns
			// proj_linear(proj) + CrossAttn(h, global); TRELLIS.2 has no proj.
			ca = ggml_add(ctx, lin(in.proj, blk + ".cross_attn.proj_linear"), ca);
		}
		h = ggml_add(ctx, h, ca);

		// feed-forward (norm3 affine-free, modulated; GELU-tanh)
		ggml_tensor *hm = modulate(ggml_norm(ctx, h, 1e-6f), scale_mlp, shift_mlp);
		hm = lin(hm, blk + ".mlp.mlp.0");
		hm = ggml_gelu(ctx, hm);
		hm = lin(hm, blk + ".mlp.mlp.2");
		h = ggml_add(ctx, h, ggml_mul(ctx, hm, gate_mlp));
		return h;
	}
};

void check(const B &b) {
	if (!b.missing.empty()) {
		throw std::runtime_error("missing tensor: " + b.missing);
	}
}

} // namespace

ggml_tensor *block(ggml_context *ctx, ggml_tensor *h, ggml_tensor *tmod, int b, const Hparams &hp, const WeightFn &W,
		const Leaves &in) {
	B bb(ctx, hp, W, in);
	ggml_tensor *r = bb.block(h, tmod, b);
	check(bb);
	return r;
}

ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x_t, ggml_tensor *temb, const Hparams &hp, const WeightFn &W,
		const Leaves &in) {
	B bb(ctx, hp, W, in);
	// stem: input projection
	ggml_tensor *h = ggml_cont(ctx, ggml_transpose(ctx, x_t)); // [Cin, N]
	h = bb.lin(h, "input_layer"); // [C, N]
	// shared modulation from the timestep
	ggml_tensor *te = bb.lin(temb, "t_embedder.mlp.0");
	te = ggml_silu(ctx, te);
	te = bb.lin(te, "t_embedder.mlp.2"); // [C]
	ggml_tensor *tmod = bb.lin(ggml_silu(ctx, te), "adaLN_modulation.1"); // [6C]
	for (int b = 0; b < hp.num_blocks; ++b) {
		h = bb.block(h, tmod, b);
	}
	// head: affine-free LayerNorm (eps 1e-5) + output projection
	h = ggml_norm(ctx, h, 1e-5f);
	h = bb.lin(h, "out_layer"); // [out_channels, N]
	ggml_tensor *y = ggml_cont(ctx, ggml_transpose(ctx, h)); // [N, out_channels]
	check(bb);
	return y;
}

} // namespace dit

// --- pixal3d-ggml trellis2.cpp @1c22f5e: the shape-SLAT decoder's sparse convs --

namespace sparse {

namespace {

inline uint64_t voxel_key(int32_t c1, int32_t c2, int32_t c3) {
	return ((uint64_t)(uint32_t)c1 << 40) | ((uint64_t)(uint32_t)c2 << 20) | (uint64_t)(uint32_t)c3;
}

} // namespace

void build_neighbor_indices(const std::vector<int32_t> &coords, int L, std::vector<std::vector<int32_t>> &idx) {
	std::unordered_map<uint64_t, int32_t> map;
	map.reserve((size_t)L * 2);
	for (int v = 0; v < L; ++v) {
		map[voxel_key(coords[(size_t)v * 3], coords[(size_t)v * 3 + 1], coords[(size_t)v * 3 + 2])] = v;
	}
	idx.assign(27, std::vector<int32_t>((size_t)L));
	for (int k = 0; k < 27; ++k) {
		const int d1 = k / 9 - 1, d2 = (k / 3) % 3 - 1, d3 = k % 3 - 1;
		std::vector<int32_t> &ik = idx[k];
		for (int v = 0; v < L; ++v) {
			const int32_t c1 = coords[(size_t)v * 3] + d1;
			const int32_t c2 = coords[(size_t)v * 3 + 1] + d2;
			const int32_t c3 = coords[(size_t)v * 3 + 2] + d3;
			if (c1 < 0 || c2 < 0 || c3 < 0) {
				ik[v] = L;
				continue;
			}
			auto it = map.find(voxel_key(c1, c2, c3));
			ik[v] = (it == map.end()) ? L : it->second;
		}
	}
}

std::vector<std::pair<std::string, ggml_tensor *>> level_graph(ggml_context *ctx, const Level &lv, const WeightFn &W,
		const std::vector<ggml_tensor *> &idx_t, const std::vector<ggml_tensor *> &mask_t, ggml_tensor *in_a,
		ggml_tensor *in_hch, ggml_tensor *in_xch) {
	const int lvl = lv.lvl, C = lv.C, prev_C = lv.prev_C, L = lv.L;
	const bool has_up = lv.C_next > 0;
	const float eps = lv.eps;
	std::string missing;
	auto w = [&](const std::string &n) -> ggml_tensor * {
		ggml_tensor *t = W(n);
		if (t == nullptr && missing.empty()) {
			missing = n;
		}
		return t;
	};
	auto lin = [&](ggml_tensor *in, const std::string &pfx) -> ggml_tensor * {
		ggml_tensor *y = ggml_mul_mat(ctx, w(pfx + ".weight"), in);
		ggml_tensor *b = W(pfx + ".bias");
		if (b) {
			y = ggml_add(ctx, y, b);
		}
		return y;
	};
	auto ln_affine = [&](ggml_tensor *h, const std::string &pfx) -> ggml_tensor * {
		ggml_tensor *y = ggml_norm(ctx, h, eps);
		y = ggml_mul(ctx, y, w(pfx + ".weight"));
		y = ggml_add(ctx, y, w(pfx + ".bias"));
		return y;
	};
	// submanifold conv: x [Cin, L] -> [Cout, L]
	auto conv = [&](ggml_tensor *x, const std::string &pfx) -> ggml_tensor * {
		ggml_tensor *wt = w(pfx + ".weight"); // ne [Ci, 27, Co]
		ggml_tensor *b = w(pfx + ".bias"); // [Co]
		if (!wt || !b) {
			return x;
		}
		const int64_t Ci = wt->ne[0], Co = wt->ne[2];
		ggml_tensor *acc = nullptr;
		for (int k = 0; k < 27; ++k) {
			ggml_tensor *wk = ggml_cont(ctx, ggml_view_3d(ctx, wt, Ci, 1, Co, wt->nb[1], wt->nb[2], (size_t)k * wt->nb[1]));
			wk = ggml_reshape_2d(ctx, wk, Ci, Co);
			ggml_tensor *g = ggml_get_rows(ctx, x, idx_t[k]); // [Ci, L]
			g = ggml_mul(ctx, g, mask_t[k]); // zero missing (broadcast [1,L])
			ggml_tensor *y = ggml_mul_mat(ctx, wk, g); // [Co, L]
			acc = acc ? ggml_add(ctx, acc, y) : y;
		}
		return ggml_add(ctx, acc, b);
	};
	// ConvNeXt block: x + mlp(LN(conv(x)))
	auto convnext = [&](ggml_tensor *x, const std::string &pfx) -> ggml_tensor * {
		ggml_tensor *h = conv(x, pfx + ".conv");
		h = ln_affine(h, pfx + ".norm");
		h = lin(h, pfx + ".mlp.0");
		h = ggml_silu(ctx, h);
		h = lin(h, pfx + ".mlp.2");
		return ggml_add(ctx, h, x);
	};

	ggml_tensor *h = nullptr;
	if (lvl == 0) {
		h = lin(in_a, "from_latent");
	} else {
		// adapted: the level count of the previous level's blocks is taken as
		// this level's (the up-block's name is blocks.<lvl-1>.<num_blocks>).
		const std::string up = "blocks." + std::to_string(lvl - 1) + "." + std::to_string(lv.num_blocks);
		const int r = C / (prev_C / 8); // repeat_interleave factor
		ggml_tensor *skip = ggml_reshape_3d(ctx, in_xch, 1, prev_C / 8, L);
		skip = ggml_repeat(ctx, skip, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, r, prev_C / 8, L));
		skip = ggml_reshape_2d(ctx, skip, C, L);
		ggml_tensor *hn = ggml_norm(ctx, in_hch, eps); // norm2, affine-free
		hn = ggml_silu(ctx, hn);
		hn = conv(hn, up + ".conv2");
		h = ggml_add(ctx, hn, skip);
	}
	for (int b = 0; b < lv.num_blocks; ++b) {
		const std::string pfx = "blocks." + std::to_string(lvl) + "." + std::to_string(b);
		h = convnext(h, pfx);
	}
	std::vector<std::pair<std::string, ggml_tensor *>> outs;
	if (has_up) {
		const std::string up = "blocks." + std::to_string(lvl) + "." + std::to_string(lv.num_blocks);
		ggml_tensor *subdiv = lin(h, up + ".to_subdiv"); // [8, L]
		outs.emplace_back("subdiv", ggml_cont(ctx, subdiv));
		ggml_tensor *hn = ln_affine(h, up + ".norm1");
		hn = ggml_silu(ctx, hn);
		ggml_tensor *h1 = conv(hn, up + ".conv1"); // [C_next*8, L]
		outs.emplace_back("h1", ggml_cont(ctx, h1));
		outs.emplace_back("x", ggml_cont(ctx, h));
	} else {
		ggml_tensor *hn = ggml_norm(ctx, h, 1e-5f);
		ggml_tensor *o = lin(hn, "output_layer"); // [7, L]
		outs.emplace_back("out", ggml_cont(ctx, o));
	}
	if (!missing.empty()) {
		throw std::runtime_error("missing tensor: " + missing + " (level " + std::to_string(lvl) + ")");
	}
	return outs;
}

} // namespace sparse

// --- kimodo-ggml src/denoiser.cpp @568b025: the motion denoiser's encoder layer --

namespace kimodo {

namespace denoiser {

namespace {

// adapted: `weight(const ggml_motion_weights &, string_view)` there; the
// lookup is the WeightFn here.
ggml_tensor *weight(const WeightFn &w, const std::string &n) {
	auto *t = w(n);
	if (!t) {
		throw std::runtime_error("missing GGML tensor: " + n);
	}
	return t;
}

} // namespace

ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *bias) {
	auto *y = ggml_mul_mat(ctx, w, x);
	// F32 parity takes precedence over Tensor Core throughput.  In
	// particular, do not let a Vulkan backend lower the accumulation
	// precision for the reference model.
	ggml_mul_mat_set_prec(y, GGML_PREC_F32);
	return ggml_add(ctx, y, ggml_repeat(ctx, bias, y));
}

ggml_tensor *norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale, ggml_tensor *bias) {
	auto *n = ggml_norm(ctx, x, 1.e-5f);
	return ggml_add(ctx, ggml_mul(ctx, n, ggml_repeat(ctx, scale, n)), ggml_repeat(ctx, bias, n));
}

// adapted: `layer(ctx, x, const ggml_motion_weights &w, string_view p, seq, batch)`
// there; w is the WeightFn.
ggml_tensor *layer(ggml_context *ctx, ggml_tensor *x, const WeightFn &w, const std::string &p, int seq, int batch) {
	const std::string s(p);
	auto *qkv = linear(ctx, x, weight(w, s + "self_attn.in_proj_weight"), weight(w, s + "self_attn.in_proj_bias"));
	// Use explicit [head, batch] branches for the F32 reference graph.  The
	// packed 4-D variant is faster, but differs slightly across Vulkan
	// backends; this layout exactly matches the PyTorch tensor boundaries.
	auto head = [&](int block, int h, int b) {
		return ggml_view_2d(ctx, qkv, head_width, seq, qkv->nb[1],
				static_cast<size_t>(block * width) * sizeof(float) + static_cast<size_t>(b) * qkv->nb[2] +
						static_cast<size_t>(h * head_width) * sizeof(float));
	};
	std::vector<ggml_tensor *> batches;
	batches.reserve(static_cast<size_t>(batch));
	for (int b = 0; b < batch; ++b) {
		std::vector<ggml_tensor *> joined_heads;
		joined_heads.reserve(heads);
		for (int h = 0; h < heads; ++h) {
			auto *q = ggml_cont(ctx, head(0, h, b)), *k = ggml_cont(ctx, head(1, h, b)), *v = ggml_cont(ctx, head(2, h, b));
			auto *scores = ggml_mul_mat(ctx, k, q);
			ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
			auto *prob = ggml_soft_max(ctx, ggml_scale(ctx, scores, 1.f / std::sqrt(float(head_width))));
			auto *value_product = ggml_mul_mat(ctx, prob, ggml_cont(ctx, ggml_transpose(ctx, v)));
			ggml_mul_mat_set_prec(value_product, GGML_PREC_F32);
			joined_heads.push_back(ggml_transpose(ctx, value_product));
		}
		auto *joined = joined_heads.front();
		for (int h = 1; h < heads; ++h) {
			joined = ggml_concat(ctx, joined, joined_heads[static_cast<size_t>(h)], 0);
		}
		batches.push_back(ggml_reshape_3d(ctx, joined, width, seq, 1));
	}
	auto *a = batches.front();
	for (int b = 1; b < batch; ++b) {
		a = ggml_concat(ctx, a, batches[static_cast<size_t>(b)], 2);
	}
	a = linear(ctx, a, weight(w, s + "self_attn.out_proj.weight"), weight(w, s + "self_attn.out_proj.bias"));
	x = norm(ctx, ggml_add(ctx, x, a), weight(w, s + "norm1.weight"), weight(w, s + "norm1.bias"));
	auto *ff = linear(ctx, x, weight(w, s + "linear1.weight"), weight(w, s + "linear1.bias"));
	ff = ggml_gelu_erf(ctx, ff);
	ff = linear(ctx, ff, weight(w, s + "linear2.weight"), weight(w, s + "linear2.bias"));
	return norm(ctx, ggml_add(ctx, x, ff), weight(w, s + "norm2.weight"), weight(w, s + "norm2.bias"));
}

} // namespace denoiser

// --- kimodo-ggml src/llm_text_encoder.cpp @568b025: one LLM2Vec Llama layer ----

namespace text {

ggml_tensor *norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight) {
	auto *normalized = ggml_rms_norm(ctx, x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32), 1e-5F);
	if (x->type == GGML_TYPE_BF16) {
		normalized = ggml_cast(ctx, normalized, GGML_TYPE_BF16);
		auto *repeated = ggml_repeat(ctx, weight, normalized);
		return ggml_cast(ctx, ggml_mul(ctx, ggml_cast(ctx, normalized, GGML_TYPE_F32),
				ggml_cast(ctx, repeated, GGML_TYPE_F32)), GGML_TYPE_BF16);
	}
	return ggml_mul(ctx, normalized, ggml_repeat(ctx, ggml_cast(ctx, weight, GGML_TYPE_F32), normalized));
}

ggml_tensor *repeat_kv(ggml_context *ctx, ggml_tensor *x, int64_t seq) {
	auto *value = ggml_reshape_4d(ctx, x, head_dim, kv_heads, 1, seq);
	auto *shape = ggml_new_tensor_4d(ctx, x->type, head_dim, kv_heads, heads / kv_heads, seq);
	value = ggml_repeat(ctx, value, shape);
	value = ggml_cont(ctx, ggml_permute(ctx, value, 0, 2, 1, 3));
	return ggml_reshape_3d(ctx, value, head_dim, heads, seq);
}

// adapted: `layer_graph(ctx, x, positions, const component &model, seq)`
// there; model.tensor(name) is the WeightFn.
ggml_tensor *layer_graph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *positions, const WeightFn &model, int64_t seq) {
	auto base = [&](const char *name, ggml_tensor *value) {
		const std::string prefix(name);
		auto *weight = model(prefix + "_base.weight");
		// adapted: the app requires BF16 (the layer GGUFs' type); the f32 arm
		// hands the same values widened to F32.
		if (!weight || (weight->type != GGML_TYPE_BF16 && weight->type != GGML_TYPE_F32)) {
			throw std::runtime_error("missing base projection");
		}
		// Vulkan's BF16 matrix-vector kernel rejects BF16 right operands. A
		// F32 cast preserves the BF16 values while taking its supported path.
		return ggml_mul_mat(ctx, weight, value->type == GGML_TYPE_F32 ? value : ggml_cast(ctx, value, GGML_TYPE_F32));
	};
	auto linear = [&](const char *name, ggml_tensor *value) {
		const std::string prefix(name);
		auto *a = model(prefix + "_lora_a.weight");
		auto *b = model(prefix + "_lora_b.weight");
		if (!a || !b) {
			throw std::runtime_error("missing LoRA projection");
		}
		auto *lora = ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, a,
				value->type == GGML_TYPE_F32 ? value : ggml_cast(ctx, value, GGML_TYPE_F32)));
		return ggml_add(ctx, ggml_cast(ctx, base(name, value), GGML_TYPE_F32), ggml_scale(ctx, lora, 2.F));
	};
	auto *attn_norm = model("attn_norm.weight");
	auto *ffn_norm = model("ffn_norm.weight");
	if (!attn_norm || !ffn_norm) {
		throw std::runtime_error("missing layer norm");
	}
	auto *residual = ggml_cast(ctx, x, GGML_TYPE_BF16);
	auto *q = linear("attn_q_proj", norm(ctx, residual, attn_norm));
	auto *k = linear("attn_k_proj", norm(ctx, residual, attn_norm));
	auto *v = linear("attn_v_proj", norm(ctx, residual, attn_norm));
	q = ggml_reshape_3d(ctx, q, head_dim, heads, seq);
	k = ggml_reshape_3d(ctx, k, head_dim, kv_heads, seq);
	v = ggml_reshape_3d(ctx, v, head_dim, kv_heads, seq);
	q = ggml_rope_ext(ctx, q, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 500000.F, 1, 0, 1, 0, 0);
	k = ggml_rope_ext(ctx, k, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 500000.F, 1, 0, 1, 0, 0);
	k = repeat_kv(ctx, k, seq);
	v = repeat_kv(ctx, v, seq);
	q = ggml_permute(ctx, q, 0, 2, 1, 3);
	k = ggml_permute(ctx, k, 0, 2, 1, 3);
	v = ggml_permute(ctx, v, 0, 2, 1, 3);
	auto *probability = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.F / std::sqrt(float(head_dim))));
	v = ggml_cont(ctx, ggml_transpose(ctx, v));
	auto *attention = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, v, probability), 0, 2, 1, 3));
	auto *output = ggml_add(ctx, ggml_cast(ctx, residual, GGML_TYPE_F32),
			linear("attn_o_proj", ggml_reshape_2d(ctx, attention, hidden, seq)));
	auto *hidden_norm = norm(ctx, output, ffn_norm);
	auto *gate = ggml_silu(ctx, linear("ffn_gate_proj", hidden_norm));
	output = ggml_add(ctx, output, linear("ffn_down_proj", ggml_mul(ctx, gate, linear("ffn_up_proj", hidden_norm))));
	return output;
}

} // namespace text

} // namespace kimodo

} // namespace app_graphs
