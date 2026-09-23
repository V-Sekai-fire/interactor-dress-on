// ggml-rd ops, family K5: ROPE, GGML_ROPE_TYPE_NEOX, f32, forward.
//
// The kernel is lean/Ggml/SlangCodegen/Rope.lean (rope_neox_f32): one thread
// per pair of a row (ne0 / 2 per row); thread j < n_dims/2 rotates the pair
// (j, j + n_dims/2), the others copy (2j, 2j + 1). It computes each pair's
// angle with ggml-cpu's running product, then rope_yarn line for line.
//
// supports(): src0 and dst f32 of one shape, src1 i32 positions with at
// least ne2 of them, no src2 (frequency factors are refused), mode exactly
// NEOX (NORMAL, MROPE, IMROPE and VISION are refused), n_dims even with
// 0 < n_dims <= ne0 and ne0 even. ROPE_BACK is another op and has no packer.
//
// pack(): the derived words the kernel reads, each computed with ggml-cpu's
// own expression (ggml_compute_forward_rope_flt / rope_yarn), so the host's
// powf and logf, not the GPU's, set them:
//   55  ne0 / 2 (threads per row)          56  n_dims / 2
//   57  theta_scale = powf(freq_base, -2.0f/n_dims)
//   58  corr_dims[0]   59  corr_dims[1]    (ggml_rope_yarn_corr_dims)
//   60  mscale = attn_factor, times 1 + 0.1 logf(1/freq_scale) if ext_factor != 0
// freq_scale and ext_factor stay in op_params (words 43 and 44).
#include <cmath>
#include <cstring>

#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

constexpr uint32_t W_ROPE_PAIRS = W_DERIVED + 2;
constexpr uint32_t W_ROPE_HALF = W_DERIVED + 3;
constexpr uint32_t W_ROPE_THETA_SCALE = W_DERIVED + 4;
constexpr uint32_t W_ROPE_CORR0 = W_DERIVED + 5;
constexpr uint32_t W_ROPE_CORR1 = W_DERIVED + 6;
constexpr uint32_t W_ROPE_MSCALE = W_DERIVED + 7;

float param_f32(const ggml_tensor *op, int k) {
	float f;
	std::memcpy(&f, op->op_params + k, sizeof f);
	return f;
}

uint32_t bits(float f) {
	uint32_t u;
	std::memcpy(&u, &f, sizeof u);
	return u;
}

bool supports_rope_neox_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	const ggml_tensor *pos = op->src[1];
	if (a == nullptr || pos == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32) {
		return false;
	}
	const int n_dims = op->op_params[1];
	const int mode = op->op_params[2];
	if (mode != GGML_ROPE_TYPE_NEOX) {
		return false;
	}
	if (!ggml_are_same_shape(a, op) || a->ne[0] % 2 != 0 || n_dims <= 0 || n_dims % 2 != 0 || n_dims > a->ne[0]) {
		return false;
	}
	if (pos->ne[0] < a->ne[2]) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a) && tensor_fits(pos);
}

bool pack_rope_neox_f32(Pack &p) {
	const ggml_tensor *op = p.node;
	p.kernel = kernel_index("rope_neox_f32");
	if (p.kernel < 0) {
		return false;
	}
	const int n_dims = op->op_params[1];
	const int n_ctx_orig = op->op_params[4];
	const float freq_base = param_f32(op, 5);
	const float freq_scale = param_f32(op, 6);
	const float ext_factor = param_f32(op, 7);
	const float attn_factor = param_f32(op, 8);
	const float beta_fast = param_f32(op, 9);
	const float beta_slow = param_f32(op, 10);

	const float theta_scale = powf(freq_base, -2.0f / n_dims);
	float corr_dims[2];
	ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
	float mscale = attn_factor;
	if (ext_factor != 0.0f) {
		mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
	}

	const uint64_t pairs = uint64_t(op->ne[0]) / 2;
	p.w[W_ROPE_PAIRS] = uint32_t(pairs);
	p.w[W_ROPE_HALF] = uint32_t(n_dims / 2);
	p.w[W_ROPE_THETA_SCALE] = bits(theta_scale);
	p.w[W_ROPE_CORR0] = bits(corr_dims[0]);
	p.w[W_ROPE_CORR1] = bits(corr_dims[1]);
	p.w[W_ROPE_MSCALE] = bits(mscale);
	return grid_1d(p, pairs * uint64_t(ggml_nrows(op)));
}

} // namespace

GGML_RD_OP(rope_neox_f32, GGML_OP_ROPE, -1, supports_rope_neox_f32, pack_rope_neox_f32);
