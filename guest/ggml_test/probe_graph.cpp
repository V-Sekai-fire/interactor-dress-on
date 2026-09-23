// Gate 3 G3.graph and G3.cost: the apps' own graph builders
// (app_graphs/, copied from skin-tokens-ggml and pixal3d-ggml) on random
// weights, ggml-rd against the in-guest ggml-cpu.
//
//   graph <qwen|dit|sconv>[:res]   G3.graph for one graph:
//     qwen   one skin-tokens Qwen3 decoder layer, KV-cached decode step
//            (decode_layer: 2 beams, past 514), f16 weights
//     dit    one Pixal3D flow DiT block (trellis2's ModulatedTransformerCrossBlock
//            plus Pixal3D's proj_linear; 4096 tokens x 1536, cross-attention
//            over 5 global tokens), bf16 weights; dit:<res> runs res^3 tokens
//     sconv  one shape-decoder level (level 1: C 256, the child head, two
//            ConvNeXt blocks, the up-block's part A; every 3^3 submanifold conv
//            27 x get_rows + mask mul + mul_mat), f16 weights
//   Arms, each built by the builder from scratch on its backend with the same
//   leaves (every leaf filled from a seed of its name):
//     cpu native / rd native   the weights in the model's type (f16, bf16):
//                              rel-L2 <= 1e-3
//     cpu f32 / rd f32         the same weight values widened to f32: rel-L2
//                              <= 1e-4 (and rd native vs cpu f32, info: both
//                              multiply f32 activations by the same weights)
//   and on the rd native net: barrier elision vs GGML_RD_BARRIER_ALL=1 and a
//   second elision run, every output bit for bit; then the control, one run
//   per barrier elision placed (at most 48, spread) with
//   GGML_RD_DROP_BARRIER=k, each compared bit for bit with the barrier-all
//   outputs: a run that differs has "detected" its dropped barrier. Before
//   every RD run the graph's compute buffers are cleared to 0, so a read
//   that races its producer sees zeros, not the previous run's result.
//
//   cost <decode|dit>   G3.cost, RD only: the host's microseconds per graph
//     and per node (Time.get_ticks_usec around graph_compute; its phases from
//     GGML_RD_PROFILE=1; a per-kernel table from one GGML_RD_PROFILE=2 run),
//     dispatches and barriers per graph, frames per graph (the pump's
//     WAIT_GPU + COOP yields from building the graph to reading its output),
//     and the GPU time (timestamps around the compute list):
//     decode  a skin-tokens decode step as the app builds it every token
//             (qwen_graph_evaluator::decode: embedding rows, 28 layers, norm,
//             logits; a new context and allocator per step), 2 beams, past 514
//     dit     a Pixal3D flow forward (30 blocks, as trellis2_ss_flow_forward
//             builds it every call), bf16 weights
//   Layer/block 0's weights are random and copied into the other layers on
//   the GPU (distinct buffers, so no layer reads another's from cache).
#include <api.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "app_graphs/app_graphs.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "pump/pump.h"
#include "rd_compute.h"

namespace probes {

namespace {

using app_graphs::WeightFn;

// --- leaves: named, filled from a seed of their name ---------------------------

uint64_t fnv1a(const std::string &s) {
	uint64_t h = 1469598103934665603ull;
	for (unsigned char c : s) {
		h = (h ^ c) * 1099511628211ull;
	}
	return h;
}

struct Leaf {
	ggml_tensor *t = nullptr;
	std::string name;
	float lo = -1.0f, hi = 1.0f;
	ggml_type round = GGML_TYPE_F32; // values rounded through this type (the f32 arm's widened weights)
	std::function<void(std::vector<uint8_t> &)> custom; // exact bytes instead (indices, masks, tables)
};

struct Net {
	ggml_backend_t be = nullptr;
	bool rd = false;
	ggml_context *lctx = nullptr; // leaves
	ggml_context *gctx = nullptr; // the graph
	ggml_cgraph *gf = nullptr;
	std::unordered_map<std::string, ggml_tensor *> named;
	std::vector<Leaf> leaves;
	std::vector<std::pair<std::string, ggml_tensor *>> outs;
	// Extra reads: (name, tensor, byte offset, bytes) of a leaf the graph writes (the KV cache rows).
	struct Slice {
		std::string name;
		ggml_tensor *t;
		size_t off, bytes;
	};
	std::vector<Slice> slices;
	ggml_tensor *residual_in = nullptr; // the block's input, for the delta norm
	std::string residual_out; // the output that is input + branches
	ggml_backend_buffer_t lbuf = nullptr;
	ggml_gallocr_t galloc = nullptr;

	Net() = default;
	Net(const Net &) = delete;
	~Net() {
		if (galloc) {
			ggml_gallocr_free(galloc);
		}
		if (lbuf) {
			ggml_backend_buffer_free(lbuf);
		}
		if (gctx) {
			ggml_free(gctx);
		}
		if (lctx) {
			ggml_free(lctx);
		}
	}
	WeightFn lookup() {
		return [this](const std::string &n) -> ggml_tensor * {
			auto it = named.find(n);
			return it == named.end() ? nullptr : it->second;
		};
	}
};

ggml_context *new_ctx(size_t tensors) {
	ggml_init_params ip = { ggml_tensor_overhead() * tensors + ggml_graph_overhead_custom(tensors, false), nullptr, true };
	return ggml_init(ip);
}

ggml_tensor *leaf(Net &n, const std::string &name, ggml_type type, std::vector<int64_t> ne, float lo, float hi,
		ggml_type round = GGML_TYPE_F32) {
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
void fill_leaves(Net &n, const std::vector<Leaf> &only = {}) {
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

struct Built {
	std::string name;
	ggml_type native = GGML_TYPE_F16;
	std::string shape;
};

// skin-tokens: decode_layer, 2 beams, past 514 (a 515-token KV cache).
constexpr size_t kQwenPast = 514;
constexpr size_t kQwenBeams = 2;

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

// --- running a net ------------------------------------------------------------------

ggml_backend_t cpu_backend() {
	ggml_backend_t b = ggml_backend_cpu_init();
	if (b != nullptr) {
		ggml_backend_cpu_set_n_threads(b, 1); // guest threads are serialized (Gate 0C)
		// One node per pump call: every node's end gives the frame back (COOP),
		// so a long reference graph never runs as one vmcall.
		ggml_backend_cpu_set_abort_callback(
				b, [](void *) -> bool {
					pump::coop();
					return false;
				},
				nullptr);
	}
	return b;
}

ggml_backend_t rd_backend_or_null() {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		return nullptr;
	}
	return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
}

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

// The graph's own buffers (gallocr's), not the leaves'.
std::vector<ggml_backend_buffer_t> compute_buffers(const Net &n) {
	std::set<ggml_backend_buffer_t> s;
	for (int i = 0; i < ggml_graph_n_nodes(n.gf); ++i) {
		const ggml_tensor *t = ggml_graph_node(n.gf, i);
		const ggml_tensor *base = t->view_src ? t->view_src : t;
		if (base->buffer != nullptr && base->buffer != n.lbuf) {
			s.insert(base->buffer);
		}
	}
	return std::vector<ggml_backend_buffer_t>(s.begin(), s.end());
}

using Outs = std::vector<std::pair<std::string, std::vector<float>>>;

struct RunInfo {
	ggml_status st = GGML_STATUS_FAILED;
	int64_t dispatches = 0, barriers = 0;
	int64_t compute_us = 0; // RD: graph_compute's host time from entry to submit (GGML_RD_PROFILE=1)
	int64_t frames = 0; // WAIT_GPU + COOP yields from compute to the outputs read
	std::string dropped;
};

int64_t frame_yields() {
	return pump::yields(pump::WAIT_GPU) + pump::yields(pump::COOP);
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

RunInfo run(Net &n, Outs &out, bool clear) {
	RunInfo r;
	if (n.rd && clear) {
		for (ggml_backend_buffer_t b : compute_buffers(n)) {
			ggml_backend_buffer_clear(b, 0);
		}
		ggml_backend_rd_ensure_idle();
	}
	const int64_t f0 = frame_yields();
	if (n.rd) {
		ggml_backend_rd_set_profile(1);
	}
	r.st = ggml_backend_graph_compute(n.be, n.gf); // ggml's: compute, then synchronize
	if (n.rd) {
		ggml_backend_rd_set_profile(0);
		r.compute_us = ggml_backend_rd_last_profile().us_total; // entry to submit
		ggml_backend_rd_last_graph(&r.dispatches, &r.barriers);
		r.dropped = ggml_backend_rd_last_dropped();
	}
	out = read_outs(n);
	r.frames = frame_yields() - f0;
	return r;
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
		bool print = true) {
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
bool identical(const Outs &a, const Outs &b, size_t *differing = nullptr) {
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

void set_env(const char *k, const char *v) {
	if (v) {
		setenv(k, v, 1);
	} else {
		unsetenv(k);
	}
}

// --- G3.graph -------------------------------------------------------------------------

using BuildFn = std::function<void(Net &, bool f32_arm)>;

// Build, allocate, fill and run one arm on a backend; the outputs in `out`.
bool cpu_arm(const char *graph, const BuildFn &build, bool f32_arm, Outs &out) {
	Net n;
	n.be = cpu_backend();
	if (n.be == nullptr) {
		return false;
	}
	bool ok = false;
	try {
		build(n, f32_arm);
		if (allocate(n)) {
			const int64_t t0 = rdc::host_usec();
			const RunInfo r = run(n, out, false);
			std::printf("PROBE graph %s cpu_%s status=%d nodes=%d host_ms=%.0f frames=%lld\n", graph,
					f32_arm ? "f32" : "native", int(r.st), ggml_graph_n_nodes(n.gf), (rdc::host_usec() - t0) / 1000.0,
					(long long)r.frames);
			ok = r.st == GGML_STATUS_SUCCESS;
		}
	} catch (const std::exception &e) {
		std::printf("PROBE graph %s: %s\n", graph, e.what());
	}
	ggml_backend_free(n.be);
	n.be = nullptr;
	return ok;
}

bool graph_one(const std::string &which, int res) {
	BuildFn build;
	const char *g = which.c_str();
	const char *native = "f16";
	if (which == "qwen") {
		build = [](Net &n, bool f32) { build_qwen(n, f32); };
	} else if (which == "dit") {
		build = [res](Net &n, bool f32) { build_dit_block(n, f32, res); };
		native = "bf16";
	} else if (which == "sconv") {
		build = [](Net &n, bool f32) { build_sconv(n, f32); };
	} else {
		std::printf("PROBE graph: unknown graph '%s' (qwen, dit, sconv)\n", g);
		return false;
	}
	for (const char *k : { "GGML_RD_BARRIER_ALL", "GGML_RD_DROP_BARRIER", "GGML_RD_FAULT", "GGML_RD_PROFILE" }) {
		unsetenv(k);
	}
	const int64_t t_start = rdc::host_usec();

	// The in-guest reference, both arms.
	Outs cpu_nat, cpu_f32;
	bool ok = cpu_arm(g, build, false, cpu_nat);
	ok = cpu_arm(g, build, true, cpu_f32) && ok;

	// ggml-rd, the native arm: elision, barrier-all, elision again, the drop sweep.
	Outs rd_nat, rd_all, rd_again, rd_f32;
	RunInfo r_el, r_all, r_again;
	int drops_run = 0, drops_detected = 0;
	{
		Net n;
		n.be = rd_backend_or_null();
		n.rd = true;
		if (n.be == nullptr) {
			std::printf("PROBE graph %s: no RD device\n", g);
			return false;
		}
		try {
			build(n, false);
			if (!allocate(n)) {
				throw std::runtime_error("allocation");
			}
			// A first run makes the pipelines and sets (COOP yields); not compared.
			Outs warm;
			run(n, warm, true);
			r_el = run(n, rd_nat, true);
			set_env("GGML_RD_BARRIER_ALL", "1");
			r_all = run(n, rd_all, true);
			set_env("GGML_RD_BARRIER_ALL", nullptr);
			r_again = run(n, rd_again, true);
			std::printf("PROBE graph %s rd_native nodes=%d dispatches=%lld barriers=%lld (barrier_all: %lld) "
						"host_us(entry..submit)=%lld frames=%lld status=%d/%d/%d\n",
					g, ggml_graph_n_nodes(n.gf), (long long)r_el.dispatches, (long long)r_el.barriers,
					(long long)r_all.barriers, (long long)r_el.compute_us, (long long)r_el.frames, int(r_el.st),
					int(r_all.st), int(r_again.st));
			// The control: leave out one barrier elision placed.
			const int64_t nb = r_el.barriers;
			std::vector<int64_t> ks;
			const int64_t kmax = 48;
			for (int64_t i = 0; i < std::min(nb, kmax); ++i) {
				ks.push_back(nb <= kmax ? i + 1 : 1 + (i * (nb - 1)) / (kmax - 1));
			}
			for (int64_t k : ks) {
				set_env("GGML_RD_DROP_BARRIER", std::to_string(k).c_str());
				Outs d;
				const RunInfo rk = run(n, d, true);
				size_t differing = 0;
				const bool same = identical(d, rd_all, &differing);
				const double rl = compare(g, "drop", d, rd_all, n.residual_out, false);
				++drops_run;
				drops_detected += same ? 0 : 1;
				std::printf("PROBE graph %s drop k=%lld barriers=%lld detected=%d differing=%zu worst_rel_l2=%.3e [%s]\n",
						g, (long long)k, (long long)rk.barriers, same ? 0 : 1, differing, rl, rk.dropped.c_str());
			}
			set_env("GGML_RD_DROP_BARRIER", nullptr);
		} catch (const std::exception &e) {
			std::printf("PROBE graph %s: %s\n", g, e.what());
			ok = false;
		}
		ggml_backend_free(n.be);
		n.be = nullptr;
	}
	{
		Net n;
		n.be = rd_backend_or_null();
		n.rd = true;
		try {
			build(n, true);
			if (!allocate(n)) {
				throw std::runtime_error("allocation");
			}
			run(n, rd_f32, true);
		} catch (const std::exception &e) {
			std::printf("PROBE graph %s: %s\n", g, e.what());
			ok = false;
		}
		ggml_backend_free(n.be);
		n.be = nullptr;
	}

	const std::string resid = which == "qwen" ? "layer_out" : which == "dit" ? "block_out" : "";
	const double e_nat = compare(g, (std::string("rd_") + native + "_vs_cpu_" + native).c_str(), rd_nat, cpu_nat, resid);
	const double e_f32 = compare(g, "rd_f32_vs_cpu_f32", rd_f32, cpu_f32, resid);
	const double e_x = compare(g, (std::string("rd_") + native + "_vs_cpu_f32_info").c_str(), rd_nat, cpu_f32, resid);
	const double e_cc = compare(g, (std::string("cpu_") + native + "_vs_cpu_f32_info").c_str(), cpu_nat, cpu_f32, resid);
	size_t d_all = 0, d_again = 0;
	const bool same_all = identical(rd_nat, rd_all, &d_all);
	const bool same_again = identical(rd_nat, rd_again, &d_again);
	const bool pass_nat = e_nat <= 1e-3, pass_f32 = e_f32 <= 1e-4;
	const bool pass_ctl = drops_detected > 0;
	std::printf("PROBE graph %s SUMMARY weights=%s rel_l2(rd_%s,cpu_%s)=%.3e (<=1e-3: %s) rel_l2(rd_f32,cpu_f32)=%.3e "
				"(<=1e-4: %s) info rel_l2(rd_%s,cpu_f32)=%.3e rel_l2(cpu_%s,cpu_f32)=%.3e elision_vs_barrier_all=%s "
				"(differing %zu) elision_repeat=%s (differing %zu) drop_control=%d/%d detected dispatches=%lld "
				"barriers=%lld/%lld wall_s=%.1f%s\n",
			g, native, native, native, e_nat, pass_nat ? "yes" : "NO", e_f32, pass_f32 ? "yes" : "NO", native, e_x,
			native, e_cc, same_all ? "bit-identical" : "DIFFERENT", d_all, same_again ? "bit-identical" : "DIFFERENT",
			d_again, drops_detected, drops_run, (long long)r_el.dispatches, (long long)r_el.barriers,
			(long long)r_all.barriers, (rdc::host_usec() - t_start) / 1e6,
			which == "sconv" ? (" L=" + std::to_string(shell_coords().size() / 3) + " neighbours=" +
									   std::to_string(g_sconv_neighbours) + "/" +
									   std::to_string(27 * shell_coords().size() / 3))
									  .c_str()
							 : "");
	return ok && pass_nat && pass_f32 && same_all && same_again && pass_ctl && r_el.st == GGML_STATUS_SUCCESS &&
			r_all.st == GGML_STATUS_SUCCESS;
}

// --- G3.cost ---------------------------------------------------------------------------

struct CostRow {
	int64_t build_us = 0, alloc_us = 0, set_us = 0, compute_us = 0, read_us = 0, total_us = 0;
	int64_t frames = 0, nodes = 0, dispatches = 0, barriers = 0, gpu_ns = -1;
	ggml_rd_profile prof;
};

void print_profile_table(const char *what, const ggml_rd_profile &p, int64_t clock_us_x1000) {
	struct Agg {
		int64_t n = 0, pack = 0, record = 0, barriers = 0;
	};
	std::map<std::string, Agg> by;
	for (const auto &d : p.per_dispatch) {
		Agg &a = by[ggml_backend_rd_kernel_name(d.kernel)];
		++a.n;
		a.pack += d.pack_us;
		a.record += d.record_us;
		a.barriers += d.barrier ? 1 : 0;
	}
	std::vector<std::pair<std::string, Agg>> rows(by.begin(), by.end());
	std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.second.pack + a.second.record > b.second.pack + b.second.record; });
	std::printf("PROBE cost %s profile2 dispatches=%lld total_us=%lld pack_us=%lld prepare_us=%lld upload_us=%lld "
				"record_us=%lld submit_us=%lld (per reading %.2f us; 3 readings per dispatch in this run)\n",
			what, (long long)p.dispatches, (long long)p.us_total, (long long)p.us_pack, (long long)p.us_prepare,
			(long long)p.us_upload, (long long)p.us_record, (long long)p.us_submit, clock_us_x1000 / 1000.0);
	for (const auto &r : rows) {
		std::printf("PROBE cost %s kernel=%-26s dispatches=%5lld barriers_before=%5lld pack_us/dispatch=%7.2f "
					"record_us/dispatch=%7.2f\n",
				what, r.first.c_str(), (long long)r.second.n, (long long)r.second.barriers,
				double(r.second.pack) / double(r.second.n), double(r.second.record) / double(r.second.n));
	}
}

// Microseconds per host clock reading, x1000 (1000 back-to-back readings).
int64_t clock_cost_x1000() {
	const int64_t t0 = rdc::host_usec();
	for (int i = 0; i < 1000; ++i) {
		(void)rdc::host_usec();
	}
	return rdc::host_usec() - t0;
}

// host_us: ggml-rd's graph_compute from entry to submit (GGML_RD_PROFILE=1);
// wait_us: the rest of ggml_backend_graph_compute, which is ggml's own
// synchronize: the WAIT_GPU yield, the next frame and the sync (the GPU time
// and a frame, not host work).
void print_cost_row(const char *what, int step, const CostRow &c) {
	std::printf("PROBE cost %s step=%d nodes=%lld dispatches=%lld barriers=%lld frames=%lld build_us=%lld "
				"alloc_us=%lld set_us=%lld host_us=%lld (pack %lld, prepare %lld, upload %lld, record %lld, "
				"submit %lld) host_us_per_node=%.2f host_us_per_dispatch=%.2f wait_us=%lld read_us=%lld "
				"step_total_us=%lld gpu_us=%.1f\n",
			what, step, (long long)c.nodes, (long long)c.dispatches, (long long)c.barriers, (long long)c.frames,
			(long long)c.build_us, (long long)c.alloc_us, (long long)c.set_us, (long long)c.prof.us_total,
			(long long)c.prof.us_pack, (long long)c.prof.us_prepare, (long long)c.prof.us_upload,
			(long long)c.prof.us_record, (long long)c.prof.us_submit,
			c.nodes ? double(c.prof.us_total) / double(c.nodes) : 0.0,
			c.dispatches ? double(c.prof.us_total) / double(c.dispatches) : 0.0,
			(long long)(c.compute_us - c.prof.us_total), (long long)c.read_us, (long long)c.total_us,
			c.gpu_ns >= 0 ? c.gpu_ns / 1000.0 : -1.0);
}

void print_cost_summary(const char *what, const std::vector<CostRow> &rows) {
	// Median over the timed steps (row 0 is the cold one).
	auto med = [&](auto f) {
		std::vector<double> v;
		for (size_t i = 1; i < rows.size(); ++i) {
			v.push_back(double(f(rows[i])));
		}
		std::sort(v.begin(), v.end());
		return v.empty() ? 0.0 : v[v.size() / 2];
	};
	const CostRow &c = rows.back();
	std::printf("PROBE cost %s SUMMARY steps=%zu nodes=%lld dispatches/graph=%lld barriers/graph=%lld "
				"frames/graph=%.0f (cold %lld) host_us=%.0f (cold %lld; pack %.0f prepare %.0f upload %.0f "
				"record %.0f submit %.0f) host_us/node=%.2f host_us/dispatch=%.2f build_us=%.0f alloc_us=%.0f "
				"wait_us=%.0f read_us=%.0f step_us=%.0f gpu_ms=%.2f\n",
			what, rows.size() - 1, (long long)c.nodes, (long long)c.dispatches, (long long)c.barriers,
			med([](const CostRow &r) { return r.frames; }), (long long)rows[0].frames,
			med([](const CostRow &r) { return r.prof.us_total; }), (long long)rows[0].prof.us_total,
			med([](const CostRow &r) { return r.prof.us_pack; }), med([](const CostRow &r) { return r.prof.us_prepare; }),
			med([](const CostRow &r) { return r.prof.us_upload; }), med([](const CostRow &r) { return r.prof.us_record; }),
			med([](const CostRow &r) { return r.prof.us_submit; }),
			med([](const CostRow &r) { return r.prof.us_total; }) / double(std::max<int64_t>(1, c.nodes)),
			med([](const CostRow &r) { return r.prof.us_total; }) / double(std::max<int64_t>(1, c.dispatches)),
			med([](const CostRow &r) { return r.build_us; }), med([](const CostRow &r) { return r.alloc_us; }),
			med([](const CostRow &r) { return r.compute_us - r.prof.us_total; }),
			med([](const CostRow &r) { return r.read_us; }), med([](const CostRow &r) { return r.total_us; }),
			med([](const CostRow &r) { return r.gpu_ns; }) / 1e6);
}

// Copy leaf `from` into leaf `to` on the GPU (buffer_copy).
void copy_leaf(Net &n, const std::string &from, const std::string &to) {
	ggml_backend_tensor_copy(n.named.at(from), n.named.at(to));
}

bool cost_decode(int steps) {
	using namespace app_graphs::qwen;
	ggml_backend_t be = rd_backend_or_null();
	if (be == nullptr) {
		return false;
	}
	Net w; // the model: weights and the KV caches, as the evaluator holds them
	w.be = be;
	w.rd = true;
	w.lctx = new_ctx(64 + 16 * layers);
	const int64_t cap = int64_t(kQwenPast + 2);
	leaf(w, "llm.tok.w", GGML_TYPE_F16, { hidden, vocab }, -0.05f, 0.05f);
	leaf(w, "llm.norm.w", GGML_TYPE_F32, { hidden }, 0.8f, 1.2f);
	matrix(w, "llm.out.w", GGML_TYPE_F16, false, { hidden, vocab }, hidden);
	std::vector<ggml_tensor *> keys, values;
	for (int l = 0; l < layers; ++l) {
		qwen_layer_weights(w, l, false);
		keys.push_back(leaf(w, "cache.k." + std::to_string(l), GGML_TYPE_F32, { head_dim, kv_heads, cap, int64_t(kQwenBeams) }, -1.0f, 1.0f));
		values.push_back(leaf(w, "cache.v." + std::to_string(l), GGML_TYPE_F32, { head_dim, kv_heads, cap, int64_t(kQwenBeams) }, -1.0f, 1.0f));
	}
	w.lbuf = ggml_backend_alloc_ctx_tensors(w.lctx, be);
	if (w.lbuf == nullptr) {
		std::printf("PROBE cost decode: weight allocation failed\n");
		return false;
	}
	// Random: the embedding, the head, layer 0 and its caches; the other layers are copies.
	std::vector<Leaf> first;
	for (const Leaf &l : w.leaves) {
		const bool per_layer = l.name.rfind("llm.l.", 0) == 0 || l.name.rfind("cache.", 0) == 0;
		const bool layer0 = l.name.rfind("llm.l.0.", 0) == 0 || l.name == "cache.k.0" || l.name == "cache.v.0";
		if (!per_layer || layer0) {
			first.push_back(l);
		}
	}
	fill_leaves(w, first);
	for (int l = 1; l < layers; ++l) {
		for (const char *s : { "an.w", "attn.q.w", "attn.k.w", "attn.v.w", "attn.q_norm.w", "attn.k_norm.w", "attn.o.w",
					 "fn.w", "mlp.g.w", "mlp.u.w", "mlp.d.w" }) {
			copy_leaf(w, std::string("llm.l.0.") + s, "llm.l." + std::to_string(l) + "." + s);
		}
		copy_leaf(w, "cache.k.0", "cache.k." + std::to_string(l));
		copy_leaf(w, "cache.v.0", "cache.v." + std::to_string(l));
	}
	ggml_backend_rd_ensure_idle();
	std::printf("PROBE cost decode model: %d layers, %.1f MiB of weights and caches on RD0\n", layers,
			ggml_backend_buffer_get_size(w.lbuf) / 1048576.0);

	const int64_t clock_x1000 = clock_cost_x1000();
	ggml_backend_rd_set_timestamps(true);
	std::vector<CostRow> rows;
	bool ok = true;
	const std::vector<size_t> slots = { 0, 1 };
	for (int step = 0; step <= steps + 1 && ok; ++step) {
		const bool prof2 = step == steps + 1; // the last: per-dispatch profile
		ggml_backend_rd_set_profile(prof2 ? 2 : 1);
		CostRow c;
		const int64_t f0 = frame_yields();
		const int64_t t0 = rdc::host_usec();
		// qwen_graph_evaluator::decode: a context, the graph, an allocator, every step.
		ggml_context *gctx = new_ctx(8192);
		ggml_tensor *ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(kQwenBeams));
		ggml_tensor *position = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
		ggml_set_input(ids);
		ggml_set_input(position);
		ggml_tensor *logits = decode_step(gctx, ids, position, slots, kQwenPast, w.lookup(), keys, values, layers);
		ggml_cgraph *gf = ggml_new_graph_custom(gctx, 8192, false);
		ggml_build_forward_expand(gf, logits);
		const int64_t t1 = rdc::host_usec();
		ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
		if (!ggml_gallocr_reserve(galloc, gf) || !ggml_gallocr_alloc_graph(galloc, gf)) {
			std::printf("PROBE cost decode: allocation failed\n");
			ok = false;
		}
		const int64_t t2 = rdc::host_usec();
		const int32_t id_v[2] = { 1000 + step, 2000 + step }, pos_v = int32_t(kQwenPast);
		ggml_backend_tensor_set(ids, id_v, 0, sizeof id_v);
		ggml_backend_tensor_set(position, &pos_v, 0, sizeof pos_v);
		const int64_t t3 = rdc::host_usec();
		const ggml_status st = ok ? ggml_backend_graph_compute(be, gf) : GGML_STATUS_FAILED;
		const int64_t t4 = rdc::host_usec();
		c.prof = ggml_backend_rd_last_profile();
		ggml_backend_rd_last_graph(&c.dispatches, &c.barriers);
		std::vector<float> out(size_t(vocab) * kQwenBeams);
		ggml_backend_tensor_get(logits, out.data(), 0, out.size() * 4); // WAIT_GPU, then the sync
		const int64_t t5 = rdc::host_usec();
		c.gpu_ns = ggml_backend_rd_last_gpu_ns();
		c.frames = frame_yields() - f0;
		c.build_us = t1 - t0;
		c.alloc_us = t2 - t1;
		c.set_us = t3 - t2;
		c.compute_us = t4 - t3;
		c.read_us = t5 - t4;
		c.total_us = t5 - t0;
		c.nodes = ggml_graph_n_nodes(gf);
		ok = ok && st == GGML_STATUS_SUCCESS && finite(out);
		if (!finite(out)) {
			std::printf("PROBE cost decode: non-finite logits\n");
		}
		ggml_gallocr_free(galloc);
		ggml_free(gctx);
		if (prof2) {
			print_cost_row("decode(profile2)", step, c);
			print_profile_table("decode", c.prof, clock_x1000);
		} else {
			print_cost_row("decode", step, c);
			rows.push_back(c);
		}
	}
	ggml_backend_rd_set_profile(0);
	ggml_backend_rd_set_timestamps(false);
	if (!rows.empty()) {
		print_cost_summary("decode", rows);
	}
	std::printf("PROBE cost decode host_clock_us_per_reading=%.2f\n", clock_x1000 / 1000.0);
	ggml_backend_buffer_free(w.lbuf);
	w.lbuf = nullptr;
	ggml_backend_free(be);
	return ok;
}

bool cost_dit(int steps) {
	using namespace app_graphs::dit;
	ggml_backend_t be = rd_backend_or_null();
	if (be == nullptr) {
		return false;
	}
	const Hparams hp;
	Net w;
	w.be = be;
	w.rd = true;
	w.lctx = new_ctx(64 + 40 * hp.num_blocks);
	const int64_t C = hp.model_channels, N = hp.tokens();
	auto lin = [&](const std::string &name, ggml_type t, int64_t in, int64_t out) {
		matrix(w, name + ".weight", t, false, { in, out }, in);
		const float bb = 1.0f / std::sqrt(float(in));
		leaf(w, name + ".bias", GGML_TYPE_F32, { out }, -bb, bb);
	};
	lin("input_layer", GGML_TYPE_F16, hp.in_channels, C);
	lin("t_embedder.mlp.0", GGML_TYPE_F16, 256, C);
	lin("t_embedder.mlp.2", GGML_TYPE_F16, C, C);
	lin("adaLN_modulation.1", GGML_TYPE_F32, C, 6 * C);
	lin("out_layer", GGML_TYPE_F16, C, hp.out_channels);
	for (int b = 0; b < hp.num_blocks; ++b) {
		dit_block_weights(w, b, hp, false);
	}
	const Leaves in = dit_leaves(w, hp);
	ggml_tensor *x_t = leaf(w, "x", GGML_TYPE_F32, { N, hp.in_channels }, -1.0f, 1.0f);
	const std::vector<float> te = timestep_embedding(0.75f * 1000.0f, 256);
	ggml_tensor *temb = custom_leaf(w, "temb", GGML_TYPE_F32, { 256 },
			[te](std::vector<uint8_t> &b) { std::memcpy(b.data(), te.data(), b.size()); });
	w.lbuf = ggml_backend_alloc_ctx_tensors(w.lctx, be);
	if (w.lbuf == nullptr) {
		std::printf("PROBE cost dit: weight allocation failed\n");
		return false;
	}
	std::vector<Leaf> first;
	for (const Leaf &l : w.leaves) {
		if (l.name.rfind("blocks.", 0) != 0 || l.name.rfind("blocks.0.", 0) == 0) {
			first.push_back(l);
		}
	}
	fill_leaves(w, first);
	for (int b = 1; b < hp.num_blocks; ++b) {
		for (const Leaf &l : w.leaves) {
			if (l.name.rfind("blocks.0.", 0) == 0) {
				copy_leaf(w, l.name, "blocks." + std::to_string(b) + "." + l.name.substr(9));
			}
		}
	}
	ggml_backend_rd_ensure_idle();
	std::printf("PROBE cost dit model: %d blocks, %lld tokens, %.1f MiB of weights on RD0 (bf16 matrices)\n",
			hp.num_blocks, (long long)N, ggml_backend_buffer_get_size(w.lbuf) / 1048576.0);

	const int64_t clock_x1000 = clock_cost_x1000();
	ggml_backend_rd_set_timestamps(true);
	std::vector<CostRow> rows;
	bool ok = true;
	for (int step = 0; step <= steps + 1 && ok; ++step) {
		const bool prof2 = step == steps + 1;
		ggml_backend_rd_set_profile(prof2 ? 2 : 1);
		CostRow c;
		const int64_t f0 = frame_yields();
		const int64_t t0 = rdc::host_usec();
		// trellis2_ss_flow_forward: a context, the graph, an allocator, every call.
		ggml_context *gctx = new_ctx(32768);
		ggml_cgraph *gf = ggml_new_graph_custom(gctx, 32768, false);
		ggml_tensor *y = nullptr;
		try {
			y = forward(gctx, x_t, temb, hp, w.lookup(), in);
		} catch (const std::exception &e) {
			std::printf("PROBE cost dit: %s\n", e.what());
			ggml_free(gctx);
			ok = false;
			break;
		}
		ggml_set_output(y);
		ggml_build_forward_expand(gf, y);
		const int64_t t1 = rdc::host_usec();
		ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
		if (!ggml_gallocr_alloc_graph(galloc, gf)) {
			std::printf("PROBE cost dit: allocation failed\n");
			ok = false;
		}
		const int64_t t2 = rdc::host_usec();
		const int64_t t3 = t2; // the inputs are leaves here, set once
		const ggml_status st = ok ? ggml_backend_graph_compute(be, gf) : GGML_STATUS_FAILED;
		const int64_t t4 = rdc::host_usec();
		c.prof = ggml_backend_rd_last_profile();
		ggml_backend_rd_last_graph(&c.dispatches, &c.barriers);
		std::vector<float> out(size_t(ggml_nelements(y)));
		ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);
		const int64_t t5 = rdc::host_usec();
		c.gpu_ns = ggml_backend_rd_last_gpu_ns();
		c.frames = frame_yields() - f0;
		c.build_us = t1 - t0;
		c.alloc_us = t2 - t1;
		c.set_us = t3 - t2;
		c.compute_us = t4 - t3;
		c.read_us = t5 - t4;
		c.total_us = t5 - t0;
		c.nodes = ggml_graph_n_nodes(gf);
		ok = ok && st == GGML_STATUS_SUCCESS && finite(out);
		if (!finite(out)) {
			std::printf("PROBE cost dit: non-finite output\n");
		}
		ggml_gallocr_free(galloc);
		ggml_free(gctx);
		if (prof2) {
			print_cost_row("dit(profile2)", step, c);
			print_profile_table("dit", c.prof, clock_x1000);
		} else {
			print_cost_row("dit", step, c);
			rows.push_back(c);
		}
	}
	ggml_backend_rd_set_profile(0);
	ggml_backend_rd_set_timestamps(false);
	if (!rows.empty()) {
		print_cost_summary("dit", rows);
	}
	std::printf("PROBE cost dit host_clock_us_per_reading=%.2f\n", clock_x1000 / 1000.0);
	ggml_backend_buffer_free(w.lbuf);
	w.lbuf = nullptr;
	ggml_backend_free(be);
	return ok;
}

} // namespace

// arg: "qwen", "dit", "dit:8" (8^3 tokens), "sconv".
bool graph_probe(const std::string &arg) {
	const size_t colon = arg.find(':');
	const std::string which = arg.substr(0, colon);
	const int res = colon == std::string::npos ? 0 : std::atoi(arg.c_str() + colon + 1);
	return graph_one(which, res);
}

// arg: "decode[:steps]" or "dit[:steps]".
bool cost_probe(const std::string &arg) {
	const size_t colon = arg.find(':');
	const std::string which = arg.substr(0, colon);
	const int steps = colon == std::string::npos ? 5 : std::max(1, std::atoi(arg.c_str() + colon + 1));
	for (const char *k : { "GGML_RD_BARRIER_ALL", "GGML_RD_DROP_BARRIER", "GGML_RD_FAULT", "GGML_RD_PROFILE" }) {
		unsetenv(k);
	}
	try {
		if (which == "decode") {
			return cost_decode(steps);
		}
		if (which == "dit") {
			return cost_dit(steps);
		}
	} catch (const std::exception &e) {
		std::printf("PROBE cost %s: %s\n", which.c_str(), e.what());
		return false;
	}
	std::printf("PROBE cost: unknown '%s' (decode, dit)\n", which.c_str());
	return false;
}

} // namespace probes
