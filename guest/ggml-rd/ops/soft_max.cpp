// ggml-rd op: SOFT_MAX, f32, with scale (family K4).
//
// The kernels are lean/Ggml/SlangCodegen/SoftMax.lean (soft_max_f32 and the
// 64-thread soft_max_f32_t64, picked by row length in ops/rows.h, and their
// _serial siblings for the host tests): one work group per row, y = exp(x*scale - max) / sum, scale from op_params[0] (word
// 37). Supported exactly as the census uses it (Skin-Tokens' attention,
// scale 1; Pixal3D's DINO, scale 0.125): no mask (src1), no sinks (src2),
// max_bias (op_params[1]) 0; every other case reports "not supported".
#include <cstring>

#include "rows.h"

namespace {

using namespace ggml_rd;

bool supports_soft_max(const ggml_tensor *op) {
	if (!rows_f32_unary(op) || !ggml_are_same_shape(op->src[0], op)) {
		return false; // rows_f32_unary also refuses a mask (src1) and sinks (src2)
	}
	float max_bias;
	std::memcpy(&max_bias, reinterpret_cast<const float *>(op->op_params) + 1, sizeof(float));
	return max_bias == 0.0f;
}

bool pack_soft_max(Pack &p) {
	return pack_rows(p, pick_threads(p.node, "soft_max_f32", "soft_max_f32_t64"));
}

} // namespace

GGML_RD_OP(soft_max_f32, GGML_OP_SOFT_MAX, -1, supports_soft_max, pack_soft_max);
