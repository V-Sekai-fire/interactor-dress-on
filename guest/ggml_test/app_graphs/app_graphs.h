// The apps' own ggml graph builders, copied (not vendored whole) for Gate 3
// G3.graph and G3.cost: the smallest set of functions that builds the graphs
// the gate runs, with the apps' op sequences unchanged. Weights and inputs
// are asked for by name through a lookup, as the apps ask their loaded
// models; the gate hands out random tensors instead (probe_graph.cpp).
//
//   qwen::   skin-tokens-ggml src/qwen.cpp @097a0cc (rms, linear, repeat_kv,
//            repeat_kv_batched, cache_view, qwen_graph_evaluator::decode_layer
//            and the graph half of ::decode)
//   dit::    pixal3d-ggml trellis2.cpp @1c22f5e (sdpa_auto, rope_tables,
//            timestep_embedding, and trellis2_ss_flow_forward's graph: the
//            stem, the shared modulation, the ModulatedTransformerCrossBlock
//            body and the head), plus Pixal3D's one change to the block
//            (proj_attention.py: x += proj_linear(proj) + CrossAttn(h, global))
//   sparse:: pixal3d-ggml trellis2.cpp @1c22f5e (build_neighbor_indices and
//            one level of shape_dec_run's graph: the child head, the ConvNeXt
//            blocks and the up-block's part A, each 3^3 submanifold conv as
//            27 x (get_rows + mask mul + mul_mat))
//
// CITATION.cff beside this file names the sources and the adaptations.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ggml.h"

namespace app_graphs {

// A model's tensor by name (the apps' `required(weights, name)` /
// `m->tensors.find(name)`); nullptr if it has none.
using WeightFn = std::function<ggml_tensor *(const std::string &)>;

namespace qwen {

constexpr int64_t hidden = 896;
constexpr int64_t heads = 16;
constexpr int64_t kv_heads = 8;
constexpr int64_t head_dim = 128;
constexpr int64_t intermediate = 2048; // mlp.g/u: [896, 2048], mlp.d: [2048, 896]
constexpr int64_t vocab = 33036;
constexpr int layers = 28;

ggml_tensor *linear(ggml_context *context, ggml_tensor *input, ggml_tensor *weight);
ggml_tensor *rms(ggml_context *context, ggml_tensor *input, ggml_tensor *weight, float epsilon);
ggml_tensor *repeat_kv(ggml_context *context, ggml_tensor *value, int64_t sequence);
ggml_tensor *repeat_kv_batched(ggml_context *context, ggml_tensor *value, int64_t sequence, int64_t batch);
// The KV cache is [head_dim, kv_heads, capacity, beams] f32 per layer.
ggml_tensor *cache_view(ggml_context *context, ggml_tensor *cache, size_t slot, size_t length, size_t position = 0);
// One decoder layer of a KV-cached decode step (qwen_graph_evaluator::decode_layer):
// input [hidden, 1, beams], position i32 [1] = past, slots[b] = beam b's cache slot.
ggml_tensor *decode_layer(ggml_context *context, ggml_tensor *input, ggml_tensor *position,
		const std::vector<size_t> &slots, size_t layer, size_t past, const WeightFn &weights,
		ggml_tensor *keys, ggml_tensor *values);
// A whole decode step's graph (qwen_graph_evaluator::decode): the embedding
// rows of ids, `n_layers` decode layers, the final norm and the logits.
ggml_tensor *decode_step(ggml_context *context, ggml_tensor *ids, ggml_tensor *position,
		const std::vector<size_t> &slots, size_t past, const WeightFn &weights,
		const std::vector<ggml_tensor *> &keys, const std::vector<ggml_tensor *> &values, int n_layers);

} // namespace qwen

namespace dit {

// trellis2_ss_flow_hparams at the ss_flow GGUF's values (trellis2.h:95).
struct Hparams {
	int in_channels = 8;
	int out_channels = 8;
	int model_channels = 1536;
	int cond_channels = 1024;
	int resolution = 16; // N = 16^3 = 4096 tokens
	int num_blocks = 30;
	int num_heads = 12;
	int mlp_hidden = 8192; // mlp_ratio 5.3334 x 1536
	float rope_freq_min = 1.0f;
	float rope_freq_base = 10000.0f;
	int head_dim() const { return model_channels / num_heads; }
	int tokens() const { return resolution * resolution * resolution; }
};

std::vector<float> timestep_embedding(float t, int dim);
void rope_tables(int res, int head_dim, float freq_min, float freq_base, std::vector<float> &cos_t,
		std::vector<float> &sin_t);
ggml_tensor *sdpa_auto(ggml_context *ctx, ggml_tensor *q3, ggml_tensor *k3, ggml_tensor *v3, int C, float scale);

// The leaves a block reads besides its weights.
struct Leaves {
	ggml_tensor *cos_t = nullptr; // [hd, 1, N]
	ggml_tensor *sin_t = nullptr; // [hd, 1, N]
	ggml_tensor *cnd = nullptr; // [cond_channels, Lkv]: TRELLIS.2's 1029 DINO tokens, Pixal3D's 5 global ones
	ggml_tensor *proj = nullptr; // [cond_channels, N]: Pixal3D's back-projected features; null = TRELLIS.2
};

// Block b of the flow (the loop body of trellis2_ss_flow_forward): h [C, N],
// tmod [6C] the shared modulation.
ggml_tensor *block(ggml_context *ctx, ggml_tensor *h, ggml_tensor *tmod, int b, const Hparams &hp,
		const WeightFn &W, const Leaves &in);
// The whole forward: x_t [N, in_channels] channel-major, temb [256] -> [N, out_channels].
ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x_t, ggml_tensor *temb, const Hparams &hp, const WeightFn &W,
		const Leaves &in);

} // namespace dit

namespace sparse {

// idx[k][v] = row of voxel v's neighbour at offset k, or L (none).
void build_neighbor_indices(const std::vector<int32_t> &coords, int L, std::vector<std::vector<int32_t>> &idx);

struct Level {
	int lvl = 1; // > 0: the level starts with the previous up-block's child head
	int C = 256;
	int prev_C = 512;
	int C_next = 128; // 0: the last level (norm + output_layer)
	int num_blocks = 2;
	int L = 0;
	float eps = 1e-5f;
};

// One level's graph (shape_dec_run's per-level graph, shape decoder: the
// to_subdiv head is predicted). idx_t/mask_t: the 27 neighbour leaves
// (i32 [L], f32 [1, L]); in_hch [C, L] and in_xch [prev_C/8, L] (lvl > 0) or
// in_a [latent, L] (lvl 0). Returns the level's outputs by name.
std::vector<std::pair<std::string, ggml_tensor *>> level_graph(ggml_context *ctx, const Level &lv, const WeightFn &W,
		const std::vector<ggml_tensor *> &idx_t, const std::vector<ggml_tensor *> &mask_t, ggml_tensor *in_a,
		ggml_tensor *in_hch, ggml_tensor *in_xch);

} // namespace sparse

} // namespace app_graphs
