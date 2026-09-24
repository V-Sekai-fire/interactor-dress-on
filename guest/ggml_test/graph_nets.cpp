// The G3.graph nets, shared by the guest probe (probe_graph.cpp: ggml-rd) and
// the host-native oracle (tests/ggml_graph_oracle: ggml-vulkan, ggml-cpu):
// each graph built by the apps' own builders (app_graphs/) with every leaf
// filled from a seed of its name, so both sides compute on the same bytes.
// graph_nets.h says what each piece is.
#include "graph_nets.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>

#include "ggml-alloc.h"

namespace graph_nets {

// --- leaves: named, filled from a seed of their name ---------------------------

uint64_t fnv1a(const std::string &s) {
	uint64_t h = 1469598103934665603ull;
	for (unsigned char c : s) {
		h = (h ^ c) * 1099511628211ull;
	}
	return h;
}

ggml_context *new_ctx(size_t tensors) {
	ggml_init_params ip = { ggml_tensor_overhead() * tensors + ggml_graph_overhead_custom(tensors, false), nullptr, true };
	return ggml_init(ip);
}

ggml_tensor *leaf(Net &n, const std::string &name, ggml_type type, std::vector<int64_t> ne, float lo, float hi,
		ggml_type round) {
	while (ne.size() < 4) {
		ne.push_back(1);
	}
	ggml_tensor *t = ggml_new_tensor_4d(n.lctx, type, ne[0], ne[1], ne[2], ne[3]);
	ggml_set_name(t, name.substr(0, GGML_MAX_NAME - 1).c_str());
	Leaf l;
	l.t = t;
	l.name = name;
	l.lo = lo;
	l.hi = hi;
	l.round = round;
	n.leaves.push_back(l);
	n.named[name] = t;
	return t;
}

ggml_tensor *custom_leaf(Net &n, const std::string &name, ggml_type type, std::vector<int64_t> ne,
		std::function<void(std::vector<uint8_t> &)> fill) {
	ggml_tensor *t = leaf(n, name, type, std::move(ne), 0.0f, 0.0f);
	n.leaves.back().custom = std::move(fill);
	return t;
}

// A weight matrix: the model's type (f16/bf16) in the native arm, f32 holding
// the same (rounded) values in the f32 arm. PyTorch nn.Linear's default
// bound, 1/sqrt(fan_in).
ggml_tensor *matrix(Net &n, const std::string &name, ggml_type native, bool f32_arm, std::vector<int64_t> ne,
		int64_t fan_in) {
	const float b = 1.0f / std::sqrt(float(fan_in));
	return f32_arm ? leaf(n, name, GGML_TYPE_F32, std::move(ne), -b, b, native)
				   : leaf(n, name, native, std::move(ne), -b, b);
}

// Fill every leaf in chunks (tensor_set: buffer_update on RD, memcpy on CPU).
void fill_leaves(Net &n, const std::vector<Leaf> &only) {
	const std::vector<Leaf> &ls = only.empty() ? n.leaves : only;
	std::vector<float> f;
	std::vector<uint8_t> bytes;
	for (const Leaf &l : ls) {
		const size_t nb = ggml_nbytes(l.t);
		if (l.custom) {
			bytes.assign(nb, 0);
			l.custom(bytes);
			ggml_backend_tensor_set(l.t, bytes.data(), 0, nb);
			continue;
		}
		uint64_t s = fnv1a(l.name) | 1;
		const int64_t total = ggml_nelements(l.t);
		const int64_t chunk = int64_t(1) << 20;
		const size_t es = ggml_type_size(l.t->type);
		for (int64_t at = 0; at < total; at += chunk) {
			const int64_t m = std::min(chunk, total - at);
			f.resize(size_t(m));
			for (int64_t i = 0; i < m; ++i) {
				s = s * 6364136223846793005ull + 1442695040888963407ull;
				f[size_t(i)] = l.lo + (l.hi - l.lo) * float(uint32_t(s >> 40)) * (1.0f / 16777216.0f);
			}
			if (l.round == GGML_TYPE_F16) {
				std::vector<ggml_fp16_t> h(static_cast<size_t>(m));
				ggml_fp32_to_fp16_row(f.data(), h.data(), m);
				ggml_fp16_to_fp32_row(h.data(), f.data(), m);
			} else if (l.round == GGML_TYPE_BF16) {
				std::vector<ggml_bf16_t> h(static_cast<size_t>(m));
				ggml_fp32_to_bf16_row(f.data(), h.data(), m);
				ggml_bf16_to_fp32_row(h.data(), f.data(), m);
			}
			bytes.resize(size_t(m) * es);
			if (l.t->type == GGML_TYPE_F32) {
				std::memcpy(bytes.data(), f.data(), bytes.size());
			} else if (l.t->type == GGML_TYPE_F16) {
				ggml_fp32_to_fp16_row(f.data(), reinterpret_cast<ggml_fp16_t *>(bytes.data()), m);
			} else if (l.t->type == GGML_TYPE_BF16) {
				ggml_fp32_to_bf16_row(f.data(), reinterpret_cast<ggml_bf16_t *>(bytes.data()), m);
			} else {
				std::printf("PROBE graph: leaf %s has type %s and no custom fill\n", l.name.c_str(), ggml_type_name(l.t->type));
			}
			ggml_backend_tensor_set(l.t, bytes.data(), size_t(at) * es, bytes.size());
		}
	}
}

// --- the graphs -------------------------------------------------------------------

// skin-tokens: decode_layer, 2 beams, past 514 (a 515-token KV cache): graph_nets.h.

void qwen_layer_weights(Net &n, int layer, bool f32_arm) {
	using namespace app_graphs::qwen;
	const std::string p = "llm.l." + std::to_string(layer) + ".";
	const ggml_type t = GGML_TYPE_F16;
	leaf(n, p + "an.w", GGML_TYPE_F32, { hidden }, 0.8f, 1.2f);
	matrix(n, p + "attn.q.w", t, f32_arm, { hidden, heads * head_dim }, hidden);
	matrix(n, p + "attn.k.w", t, f32_arm, { hidden, kv_heads * head_dim }, hidden);
	matrix(n, p + "attn.v.w", t, f32_arm, { hidden, kv_heads * head_dim }, hidden);
	leaf(n, p + "attn.q_norm.w", GGML_TYPE_F32, { head_dim }, 0.8f, 1.2f);
	leaf(n, p + "attn.k_norm.w", GGML_TYPE_F32, { head_dim }, 0.8f, 1.2f);
	matrix(n, p + "attn.o.w", t, f32_arm, { heads * head_dim, hidden }, heads * head_dim);
	leaf(n, p + "fn.w", GGML_TYPE_F32, { hidden }, 0.8f, 1.2f);
	matrix(n, p + "mlp.g.w", t, f32_arm, { hidden, intermediate }, hidden);
	matrix(n, p + "mlp.u.w", t, f32_arm, { hidden, intermediate }, hidden);
	matrix(n, p + "mlp.d.w", t, f32_arm, { intermediate, hidden }, intermediate);
}

std::function<void(std::vector<uint8_t> &)> i32_fill(std::vector<int32_t> v) {
	return [v](std::vector<uint8_t> &b) { std::memcpy(b.data(), v.data(), std::min(b.size(), v.size() * 4)); };
}

void build_qwen(Net &n, bool f32_arm) {
	using namespace app_graphs::qwen;
	n.lctx = new_ctx(64);
	const int64_t cap = int64_t(kQwenPast + 2);
	ggml_tensor *x = leaf(n, "x", GGML_TYPE_F32, { hidden, 1, int64_t(kQwenBeams) }, -1.0f, 1.0f);
	ggml_tensor *pos = custom_leaf(n, "position", GGML_TYPE_I32, { 1 }, i32_fill({ int32_t(kQwenPast) }));
	ggml_tensor *keys = leaf(n, "cache.k.0", GGML_TYPE_F32, { head_dim, kv_heads, cap, int64_t(kQwenBeams) }, -1.0f, 1.0f);
	ggml_tensor *values = leaf(n, "cache.v.0", GGML_TYPE_F32, { head_dim, kv_heads, cap, int64_t(kQwenBeams) }, -1.0f, 1.0f);
	qwen_layer_weights(n, 0, f32_arm);
	n.gctx = new_ctx(2048);
	n.gf = ggml_new_graph_custom(n.gctx, 2048, false);
	std::vector<size_t> slots = { 0, 1 };
	ggml_tensor *out = decode_layer(n.gctx, x, pos, slots, 0, kQwenPast, n.lookup(), keys, values);
	ggml_set_output(out);
	ggml_build_forward_expand(n.gf, out);
	n.outs.emplace_back("layer_out", out);
	n.residual_in = x;
	n.residual_out = "layer_out";
	for (size_t b = 0; b < kQwenBeams; ++b) {
		const size_t off = b * keys->nb[3] + kQwenPast * keys->nb[2];
		n.slices.push_back({ "k_cache_row_beam" + std::to_string(b), keys, off, size_t(head_dim * kv_heads * 4) });
		n.slices.push_back({ "v_cache_row_beam" + std::to_string(b), values, off, size_t(head_dim * kv_heads * 4) });
	}
}

void dit_block_weights(Net &n, int b, const app_graphs::dit::Hparams &hp, bool f32_arm) {
	const std::string p = "blocks." + std::to_string(b) + ".";
	const ggml_type t = GGML_TYPE_BF16;
	const int64_t C = hp.model_channels, Cc = hp.cond_channels, M = hp.mlp_hidden, hd = hp.head_dim(), H = hp.num_heads;
	auto lin = [&](const std::string &name, int64_t in, int64_t out) {
		matrix(n, p + name + ".weight", t, f32_arm, { in, out }, in);
		const float bb = 1.0f / std::sqrt(float(in));
		leaf(n, p + name + ".bias", GGML_TYPE_F32, { out }, -bb, bb);
	};
	leaf(n, p + "modulation", GGML_TYPE_F32, { 6 * C }, -0.5f, 0.5f);
	lin("self_attn.to_qkv", C, 3 * C);
	leaf(n, p + "self_attn.q_rms_norm.gamma", GGML_TYPE_F32, { hd, H }, 0.8f, 1.2f);
	leaf(n, p + "self_attn.k_rms_norm.gamma", GGML_TYPE_F32, { hd, H }, 0.8f, 1.2f);
	lin("self_attn.to_out", C, C);
	leaf(n, p + "norm2.weight", GGML_TYPE_F32, { C }, 0.8f, 1.2f);
	leaf(n, p + "norm2.bias", GGML_TYPE_F32, { C }, -0.1f, 0.1f);
	lin("cross_attn.to_q", C, C);
	leaf(n, p + "cross_attn.q_rms_norm.gamma", GGML_TYPE_F32, { hd, H }, 0.8f, 1.2f);
	lin("cross_attn.to_kv", Cc, 2 * C);
	leaf(n, p + "cross_attn.k_rms_norm.gamma", GGML_TYPE_F32, { hd, H }, 0.8f, 1.2f);
	lin("cross_attn.to_out", C, C);
	lin("cross_attn.proj_linear", Cc, C);
	lin("mlp.mlp.0", C, M);
	lin("mlp.mlp.2", M, C);
}

// Pixal3D: cross-attention over 5 global tokens (CLS + 4 registers).
constexpr int kPixal3dGlobalTokens = 5;

app_graphs::dit::Leaves dit_leaves(Net &n, const app_graphs::dit::Hparams &hp) {
	app_graphs::dit::Leaves in;
	const int N = hp.tokens(), hd = hp.head_dim();
	std::vector<float> cosv, sinv;
	app_graphs::dit::rope_tables(hp.resolution, hd, hp.rope_freq_min, hp.rope_freq_base, cosv, sinv);
	auto tab = [](std::vector<float> v) {
		return [v](std::vector<uint8_t> &b) { std::memcpy(b.data(), v.data(), std::min(b.size(), v.size() * 4)); };
	};
	in.cos_t = custom_leaf(n, "rope.cos", GGML_TYPE_F32, { hd, 1, N }, tab(cosv));
	in.sin_t = custom_leaf(n, "rope.sin", GGML_TYPE_F32, { hd, 1, N }, tab(sinv));
	in.cnd = leaf(n, "cond.global", GGML_TYPE_F32, { hp.cond_channels, kPixal3dGlobalTokens }, -1.0f, 1.0f);
	in.proj = leaf(n, "cond.proj", GGML_TYPE_F32, { hp.cond_channels, N }, -1.0f, 1.0f);
	return in;
}

app_graphs::dit::Hparams dit_hp(int res) {
	app_graphs::dit::Hparams hp;
	if (res > 0) {
		hp.resolution = res;
	}
	return hp;
}

void build_dit_block(Net &n, bool f32_arm, int res) {
	const app_graphs::dit::Hparams hp = dit_hp(res);
	n.lctx = new_ctx(64);
	const int64_t C = hp.model_channels, N = hp.tokens();
	ggml_tensor *h = leaf(n, "h", GGML_TYPE_F32, { C, N }, -1.0f, 1.0f);
	ggml_tensor *tmod = leaf(n, "tmod", GGML_TYPE_F32, { 6 * C }, -0.5f, 0.5f);
	const app_graphs::dit::Leaves in = dit_leaves(n, hp);
	dit_block_weights(n, 0, hp, f32_arm);
	n.gctx = new_ctx(1024);
	n.gf = ggml_new_graph_custom(n.gctx, 1024, false);
	ggml_tensor *out = app_graphs::dit::block(n.gctx, h, tmod, 0, hp, n.lookup(), in);
	ggml_set_output(out);
	ggml_build_forward_expand(n.gf, out);
	n.outs.emplace_back("block_out", out);
	n.residual_in = h;
	n.residual_out = "block_out";
}

// A shell of voxels (a surface, as the decoder's levels are): the cells of a
// 24^3 grid within half a cell of radius 7.4 around its centre.
std::vector<int32_t> shell_coords() {
	std::vector<int32_t> c;
	for (int i = 0; i < 24; ++i) {
		for (int j = 0; j < 24; ++j) {
			for (int k = 0; k < 24; ++k) {
				const float d = std::sqrt(float((i - 11.5) * (i - 11.5) + (j - 11.5) * (j - 11.5) + (k - 11.5) * (k - 11.5)));
				if (std::fabs(d - 7.4f) < 0.5f) {
					c.push_back(i);
					c.push_back(j);
					c.push_back(k);
				}
			}
		}
	}
	return c;
}

int g_sconv_neighbours = 0; // real neighbours over the 27 offsets (reported)

void build_sconv(Net &n, bool f32_arm) {
	using app_graphs::sparse::Level;
	const std::vector<int32_t> coords = shell_coords();
	Level lv;
	lv.L = int(coords.size() / 3);
	std::vector<std::vector<int32_t>> nidx;
	app_graphs::sparse::build_neighbor_indices(coords, lv.L, nidx);
	n.lctx = new_ctx(256);
	std::vector<ggml_tensor *> idx_t(27), mask_t(27);
	g_sconv_neighbours = 0;
	for (int k = 0; k < 27; ++k) {
		// shape_dec_run's upload: the missing-neighbour sentinel L clamped to 0, mask 0 there.
		std::vector<int32_t> clamped(size_t(lv.L));
		std::vector<float> mask(size_t(lv.L));
		for (int v = 0; v < lv.L; ++v) {
			const bool miss = nidx[size_t(k)][size_t(v)] >= lv.L;
			clamped[size_t(v)] = miss ? 0 : nidx[size_t(k)][size_t(v)];
			mask[size_t(v)] = miss ? 0.0f : 1.0f;
			g_sconv_neighbours += miss ? 0 : 1;
		}
		idx_t[size_t(k)] = custom_leaf(n, "nbr.idx." + std::to_string(k), GGML_TYPE_I32, { lv.L }, i32_fill(clamped));
		mask_t[size_t(k)] = custom_leaf(n, "nbr.mask." + std::to_string(k), GGML_TYPE_F32, { 1, lv.L },
				[mask](std::vector<uint8_t> &b) { std::memcpy(b.data(), mask.data(), b.size()); });
	}
	ggml_tensor *hch = leaf(n, "in.hch", GGML_TYPE_F32, { lv.C, lv.L }, -1.0f, 1.0f);
	ggml_tensor *xch = leaf(n, "in.xch", GGML_TYPE_F32, { lv.prev_C / 8, lv.L }, -1.0f, 1.0f);
	const ggml_type t = GGML_TYPE_F16;
	auto conv_w = [&](const std::string &p, int64_t ci, int64_t co) {
		matrix(n, p + ".weight", t, f32_arm, { ci, 27, co }, ci * 27);
		const float bb = 1.0f / std::sqrt(float(ci * 27));
		leaf(n, p + ".bias", GGML_TYPE_F32, { co }, -bb, bb);
	};
	auto lin_w = [&](const std::string &p, int64_t in, int64_t out) {
		matrix(n, p + ".weight", t, f32_arm, { in, out }, in);
		const float bb = 1.0f / std::sqrt(float(in));
		leaf(n, p + ".bias", GGML_TYPE_F32, { out }, -bb, bb);
	};
	auto ln_w = [&](const std::string &p, int64_t c) {
		leaf(n, p + ".weight", GGML_TYPE_F32, { c }, 0.8f, 1.2f);
		leaf(n, p + ".bias", GGML_TYPE_F32, { c }, -0.1f, 0.1f);
	};
	const std::string prev_up = "blocks." + std::to_string(lv.lvl - 1) + "." + std::to_string(lv.num_blocks);
	conv_w(prev_up + ".conv2", lv.C, lv.C);
	for (int b = 0; b < lv.num_blocks; ++b) {
		const std::string p = "blocks." + std::to_string(lv.lvl) + "." + std::to_string(b);
		conv_w(p + ".conv", lv.C, lv.C);
		ln_w(p + ".norm", lv.C);
		lin_w(p + ".mlp.0", lv.C, 4 * lv.C);
		lin_w(p + ".mlp.2", 4 * lv.C, lv.C);
	}
	const std::string up = "blocks." + std::to_string(lv.lvl) + "." + std::to_string(lv.num_blocks);
	lin_w(up + ".to_subdiv", lv.C, 8);
	ln_w(up + ".norm1", lv.C);
	conv_w(up + ".conv1", lv.C, int64_t(lv.C_next) * 8);
	n.gctx = new_ctx(4096);
	n.gf = ggml_new_graph_custom(n.gctx, 4096, false);
	const auto outs = app_graphs::sparse::level_graph(n.gctx, lv, n.lookup(), idx_t, mask_t, nullptr, hch, xch);
	for (const auto &o : outs) {
		ggml_set_output(o.second);
		ggml_build_forward_expand(n.gf, o.second);
		n.outs.push_back(o);
	}
}

// Kimodo: the motion denoiser's TransformerEncoderLayer and the LLM2Vec text
// encoder's Llama layer (app_graphs.h, namespace kimodo): graph_nets.h.

// The layer's tensors as convert_motion_to_gguf.py stores them (PyTorch's
// row-major [out, in] as ggml ne {in, out}), all F32: the model's type
// (docs/IMPLEMENTATION.md:86, "F32 is required for initial parity").
void kimodo_denoiser_layer_weights(Net &n, const std::string &prefix, bool f32_arm) {
	using namespace app_graphs::kimodo::denoiser;
	const ggml_type t = GGML_TYPE_F32;
	auto lin = [&](const std::string &name, int64_t in, int64_t out) {
		matrix(n, prefix + name + ".weight", t, f32_arm, { in, out }, in);
		const float bb = 1.0f / std::sqrt(float(in));
		leaf(n, prefix + name + ".bias", GGML_TYPE_F32, { out }, -bb, bb);
	};
	auto ln = [&](const std::string &name) {
		leaf(n, prefix + name + ".weight", GGML_TYPE_F32, { width }, 0.8f, 1.2f);
		leaf(n, prefix + name + ".bias", GGML_TYPE_F32, { width }, -0.1f, 0.1f);
	};
	// nn.MultiheadAttention's packed in_proj: [3 * width, width] and [3 * width].
	matrix(n, prefix + "self_attn.in_proj_weight", t, f32_arm, { width, 3 * width }, width);
	{
		const float bb = 1.0f / std::sqrt(float(width));
		leaf(n, prefix + "self_attn.in_proj_bias", GGML_TYPE_F32, { 3 * width }, -bb, bb);
	}
	lin("self_attn.out_proj", width, width);
	ln("norm1");
	lin("linear1", width, feed_forward);
	lin("linear2", feed_forward, width);
	ln("norm2");
}

void build_kimodo_denoiser(Net &n, bool f32_arm) {
	using namespace app_graphs::kimodo::denoiser;
	const int seq = prefix_tokens + kKimodoFrames; // denoiser.cpp:80
	const std::string prefix = "root_model.seqTransEncoder.layers.0."; // denoiser.cpp:85, 97
	n.lctx = new_ctx(64);
	ggml_tensor *x = leaf(n, "x", GGML_TYPE_F32, { width, seq, kKimodoBatch }, -1.0f, 1.0f);
	kimodo_denoiser_layer_weights(n, prefix, f32_arm);
	n.gctx = new_ctx(2048);
	n.gf = ggml_new_graph_custom(n.gctx, 2048, false);
	ggml_tensor *out = layer(n.gctx, x, n.lookup(), prefix, seq, kKimodoBatch);
	ggml_set_output(out);
	ggml_build_forward_expand(n.gf, out);
	n.outs.emplace_back("layer_out", out);
	// Post-LN: the output is norm2(x + ...), not x + branches, so no branch
	// norm; the input is still read back for the oracle's byte check.
	n.residual_in = x;
	n.residual_out = "";
}

// The layer GGUF's tensors (convert_llm2vec_layer_to_gguf.py:176-183): the
// two RMSNorm gains and the seven projections' merged MNTP base in BF16,
// their supervised LoRA A [in, r] and B [r, out] in F32. The LoRA B is bounded
// like its base (1/sqrt(in)) so the branch is of the base's size.
void kimodo_text_layer_weights(Net &n, bool f32_arm) {
	using namespace app_graphs::kimodo::text;
	const ggml_type t = GGML_TYPE_BF16;
	auto proj = [&](const std::string &name, int64_t in, int64_t out) {
		matrix(n, name + "_base.weight", t, f32_arm, { in, out }, in);
		matrix(n, name + "_lora_a.weight", GGML_TYPE_F32, true, { in, lora_rank }, in);
		matrix(n, name + "_lora_b.weight", GGML_TYPE_F32, true, { lora_rank, out }, in);
	};
	// The gains: bf16 in the GGUF; the f32 arm holds the same values widened.
	leaf(n, "attn_norm.weight", f32_arm ? GGML_TYPE_F32 : t, { hidden }, 0.8f, 1.2f, t);
	leaf(n, "ffn_norm.weight", f32_arm ? GGML_TYPE_F32 : t, { hidden }, 0.8f, 1.2f, t);
	proj("attn_q_proj", hidden, heads * head_dim);
	proj("attn_k_proj", hidden, kv_heads * head_dim);
	proj("attn_v_proj", hidden, kv_heads * head_dim);
	proj("attn_o_proj", heads * head_dim, hidden);
	proj("ffn_gate_proj", hidden, feed_forward);
	proj("ffn_up_proj", hidden, feed_forward);
	proj("ffn_down_proj", feed_forward, hidden);
}

void build_kimodo_text(Net &n, bool f32_arm) {
	using namespace app_graphs::kimodo::text;
	const int64_t seq = kKimodoTextSeq;
	n.lctx = new_ctx(64);
	ggml_tensor *x = leaf(n, "x", GGML_TYPE_F32, { hidden, seq }, -1.0f, 1.0f);
	std::vector<int32_t> pos(static_cast<size_t>(seq));
	for (int32_t i = 0; i < seq; ++i) {
		pos[size_t(i)] = i; // run_layer_chunk: positions 0..seq-1
	}
	ggml_tensor *positions = custom_leaf(n, "positions", GGML_TYPE_I32, { seq }, i32_fill(pos));
	kimodo_text_layer_weights(n, f32_arm);
	n.gctx = new_ctx(1024);
	n.gf = ggml_new_graph_custom(n.gctx, 1024, false);
	ggml_tensor *out = layer_graph(n.gctx, x, positions, n.lookup(), seq);
	ggml_set_output(out);
	ggml_build_forward_expand(n.gf, out);
	n.outs.emplace_back("layer_out", out);
	n.residual_in = x;
	n.residual_out = "layer_out";
}

bool graph_builder(const std::string &which, int res, BuildFn &build, std::string &native, std::string &residual_out) {
	if (which == "qwen") {
		build = [](Net &n, bool f32) { build_qwen(n, f32); };
		native = "f16";
		residual_out = "layer_out";
	} else if (which == "dit") {
		build = [res](Net &n, bool f32) { build_dit_block(n, f32, res); };
		native = "bf16";
		residual_out = "block_out";
	} else if (which == "sconv") {
		build = [](Net &n, bool f32) { build_sconv(n, f32); };
		native = "f16";
		residual_out = "";
	} else if (which == "kimodo_denoiser") {
		build = [](Net &n, bool f32) { build_kimodo_denoiser(n, f32); };
		native = "f32";
		residual_out = "";
	} else if (which == "kimodo_text") {
		build = [](Net &n, bool f32) { build_kimodo_text(n, f32); };
		native = "bf16";
		residual_out = "layer_out";
	} else {
		return false;
	}
	return true;
}

// --- running a net ------------------------------------------------------------------

// Allocate the leaves and the graph on n.be and fill the leaves.
bool allocate(Net &n) {
	n.lbuf = ggml_backend_alloc_ctx_tensors(n.lctx, n.be);
	if (n.lbuf == nullptr) {
		std::printf("PROBE graph: leaf allocation failed\n");
		return false;
	}
	n.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(n.be));
	if (n.galloc == nullptr || !ggml_gallocr_alloc_graph(n.galloc, n.gf)) {
		std::printf("PROBE graph: graph allocation failed\n");
		return false;
	}
	fill_leaves(n);
	return true;
}


Outs read_outs(const Net &n) {
	Outs o;
	for (const auto &x : n.outs) {
		std::vector<float> v(size_t(ggml_nelements(x.second)));
		ggml_backend_tensor_get(x.second, v.data(), 0, v.size() * 4);
		o.emplace_back(x.first, std::move(v));
	}
	for (const auto &s : n.slices) {
		std::vector<float> v(s.bytes / 4);
		ggml_backend_tensor_get(s.t, v.data(), s.off, s.bytes);
		o.emplace_back(s.name, std::move(v));
	}
	if (n.residual_in != nullptr) {
		std::vector<float> v(size_t(ggml_nelements(n.residual_in)));
		ggml_backend_tensor_get(n.residual_in, v.data(), 0, v.size() * 4);
		o.emplace_back("__residual_in", std::move(v));
	}
	return o;
}

// --- comparing -----------------------------------------------------------------------

const std::vector<float> *find(const Outs &o, const std::string &name) {
	for (const auto &x : o) {
		if (x.first == name) {
			return &x.second;
		}
	}
	return nullptr;
}

double rel_l2(const std::vector<float> &a, const std::vector<float> &ref) {
	double num = 0.0, den = 0.0;
	for (size_t i = 0; i < a.size() && i < ref.size(); ++i) {
		const double d = double(a[i]) - double(ref[i]);
		num += d * d;
		den += double(ref[i]) * double(ref[i]);
	}
	if (a.size() != ref.size()) {
		return INFINITY;
	}
	return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

bool finite(const std::vector<float> &a) {
	for (float v : a) {
		if (!std::isfinite(v)) {
			return false;
		}
	}
	return true;
}

// The worst rel-L2 over the outputs (not the residual input), with a line per output.
double compare(const char *graph, const char *what, const Outs &a, const Outs &ref, const std::string &residual_out,
		bool print) {
	double worst = 0.0;
	const std::vector<float> *xin = find(ref, "__residual_in");
	for (const auto &x : ref) {
		if (x.first == "__residual_in") {
			continue;
		}
		const std::vector<float> *y = find(a, x.first);
		const double r = y ? rel_l2(*y, x.second) : INFINITY;
		const bool fin = y && finite(*y);
		worst = std::max(worst, fin ? r : double(INFINITY));
		if (!print) {
			continue;
		}
		// The residual branch alone: out - in, a stricter norm (the input is exact on both sides).
		double rd = -1.0;
		if (y && xin != nullptr && x.first == residual_out && xin->size() == x.second.size()) {
			double num = 0.0, den = 0.0;
			for (size_t i = 0; i < x.second.size(); ++i) {
				const double d = double((*y)[i]) - double(x.second[i]);
				const double br = double(x.second[i]) - double((*xin)[i]);
				num += d * d;
				den += br * br;
			}
			rd = den > 0 ? std::sqrt(num / den) : -1.0;
		}
		std::printf("PROBE graph %s %s output=%s n=%zu rel_l2=%.3e%s finite=%d\n", graph, what, x.first.c_str(),
				x.second.size(), r, rd >= 0 ? (" rel_l2_of_branch=" + std::to_string(rd)).c_str() : "", fin ? 1 : 0);
	}
	return worst;
}

// Bit-for-bit equality of every output.
bool identical(const Outs &a, const Outs &b, size_t *differing) {
	size_t diff = 0;
	bool same = a.size() == b.size();
	for (size_t k = 0; k < a.size() && k < b.size(); ++k) {
		if (a[k].first != b[k].first || a[k].second.size() != b[k].second.size()) {
			same = false;
			continue;
		}
		for (size_t i = 0; i < a[k].second.size(); ++i) {
			if (std::memcmp(&a[k].second[i], &b[k].second[i], 4) != 0) {
				++diff;
			}
		}
	}
	if (differing) {
		*differing = diff;
	}
	return same && diff == 0;
}

} // namespace graph_nets
