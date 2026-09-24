// The G3.graph nets: the apps' own graph builders (app_graphs/) on random
// weights, as one Net per arm, shared by the guest probe (probe_graph.cpp,
// ggml-rd) and the host-native oracle (tests/ggml_graph_oracle, ggml-vulkan
// and ggml-cpu). Pure ggml: nothing here touches the pump, RenderingDevice
// or the sandbox, so the same file compiles for riscv64 and x86_64 and both
// sides build the same graph on the same bytes.
//
//   qwen   one skin-tokens Qwen3 decoder layer, KV-cached decode step
//          (decode_layer: 2 beams, past 514), f16 weights
//   dit    one Pixal3D flow DiT block (trellis2's ModulatedTransformerCrossBlock
//          plus Pixal3D's proj_linear; 4096 tokens x 1536, cross-attention over
//          5 global tokens), bf16 weights; res > 0 runs res^3 tokens
//   sconv  one shape-decoder level (level 1: C 256, the child head, two
//          ConvNeXt blocks, the up-block's part A; every 3^3 submanifold conv
//          27 x get_rows + mask mul + mul_mat), f16 weights
//   kimodo_denoiser  one Kimodo motion-denoiser TransformerEncoderLayer
//          (denoiser.cpp's layer: 1024 wide, 8 heads x 128, FFN 2048, the
//          attention as 8 x 3 explicit [head, batch] branches) on the
//          separated-CFG batch of 3 and 52 + 8 tokens, f32 weights (the
//          model's type: its f32 arm is the same bytes)
//   kimodo_text  one LLM2Vec Llama-3-8B layer (llm_text_encoder.cpp's
//          layer_graph: 4096 wide, 32 heads / 8 KV heads x 128, FFN 14336,
//          NEOX RoPE at theta 500000, bidirectional) on 16 tokens, the base
//          weights bf16 with the F32 rank-16 LoRA branch; f32 arm: the base
//          widened, the bf16 activation casts kept (they are the app's ops)
//
// Arms: native (the weights in the model's type) and f32 (the same values,
// rounded through that type, stored as f32). Every leaf is filled from a
// seed of its name (fnv1a -> an LCG), so a leaf's bytes depend only on its
// name, type and shape, never on the backend or the order of allocation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "app_graphs/app_graphs.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace graph_nets {

using app_graphs::WeightFn;

uint64_t fnv1a(const std::string &s);

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

ggml_context *new_ctx(size_t tensors);
ggml_tensor *leaf(Net &n, const std::string &name, ggml_type type, std::vector<int64_t> ne, float lo, float hi,
		ggml_type round = GGML_TYPE_F32);
ggml_tensor *custom_leaf(Net &n, const std::string &name, ggml_type type, std::vector<int64_t> ne,
		std::function<void(std::vector<uint8_t> &)> fill);
// A weight matrix: the model's type in the native arm, f32 holding the same
// (rounded) values in the f32 arm; PyTorch nn.Linear's bound 1/sqrt(fan_in).
ggml_tensor *matrix(Net &n, const std::string &name, ggml_type native, bool f32_arm, std::vector<int64_t> ne,
		int64_t fan_in);
// Fill every leaf (or only these) in chunks through ggml_backend_tensor_set.
void fill_leaves(Net &n, const std::vector<Leaf> &only = {});

// skin-tokens: decode_layer, 2 beams, past 514 (a 515-token KV cache).
constexpr size_t kQwenPast = 514;
constexpr size_t kQwenBeams = 2;

void qwen_layer_weights(Net &n, int layer, bool f32_arm);
void dit_block_weights(Net &n, int b, const app_graphs::dit::Hparams &hp, bool f32_arm);
app_graphs::dit::Leaves dit_leaves(Net &n, const app_graphs::dit::Hparams &hp);
app_graphs::dit::Hparams dit_hp(int res);

void build_qwen(Net &n, bool f32_arm);
void build_dit_block(Net &n, bool f32_arm, int res);
void build_sconv(Net &n, bool f32_arm);
std::vector<int32_t> shell_coords();
extern int g_sconv_neighbours; // real neighbours over the 27 offsets, set by build_sconv

// Kimodo: the denoiser layer on the separated-CFG batch (denoiser.cpp:138,
// cfg_batch 3) of 8 frames after the 52 prefix tokens (tests/root_parity.cpp:21,
// T = 8, B = 3); the text layer on 16 tokens (tests/llm_layer_parity.cpp:148).
constexpr int kKimodoFrames = 8;
constexpr int kKimodoBatch = 3;
constexpr int64_t kKimodoTextSeq = 16;

void kimodo_denoiser_layer_weights(Net &n, const std::string &prefix, bool f32_arm);
void kimodo_text_layer_weights(Net &n, bool f32_arm);
void build_kimodo_denoiser(Net &n, bool f32_arm);
void build_kimodo_text(Net &n, bool f32_arm);

// "qwen", "dit" (res^3 tokens, 0 = 16), "sconv", "kimodo_denoiser",
// "kimodo_text": the builder, the native weight type's name and the output
// that is input + branches. False if `which` names none.
using BuildFn = std::function<void(Net &, bool f32_arm)>;
bool graph_builder(const std::string &which, int res, BuildFn &build, std::string &native, std::string &residual_out);

// Allocate the leaves (one buffer) and the graph (gallocr) on n.be, then fill the leaves.
bool allocate(Net &n);

using Outs = std::vector<std::pair<std::string, std::vector<float>>>;
// The net's outputs, its slices and "__residual_in" (the block's input).
Outs read_outs(const Net &n);
const std::vector<float> *find(const Outs &o, const std::string &name);
double rel_l2(const std::vector<float> &a, const std::vector<float> &ref);
bool finite(const std::vector<float> &a);
// The worst rel-L2 over the outputs (not the residual input), with a line per
// output ("PROBE graph <graph> <what> output=..."); for residual_out also the
// rel-L2 of the residual branch (out - in).
double compare(const char *graph, const char *what, const Outs &a, const Outs &ref, const std::string &residual_out,
		bool print = true);
// Bit-for-bit equality of every output.
bool identical(const Outs &a, const Outs &b, size_t *differing = nullptr);

} // namespace graph_nets
