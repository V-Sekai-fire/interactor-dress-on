// ggml-rd ops: NORM, RMS_NORM and MEAN, f32 (family K3).
//
// The kernels are lean/Ggml/SlangCodegen/Norm.lean (norm_f32, rms_norm_f32,
// mean_f32, and their _serial siblings for the host tests): one 256-thread
// work group per row, grid-strided, the sums tree-reduced in group-shared
// memory (ops/rows.h); NORM and RMS_NORM rows of at most kShortRow elements
// take the 64-thread norm_f32_t64 / rms_norm_f32_t64. Every operand may be strided or permuted (element
// strides, as the census's contiguous rows and test-backend-ops' views and
// permuted rows); eps is op_params[0], read by the kernel from word 37.
//
//   NORM, RMS_NORM  src0 f32, dst f32 of src0's shape;
//   MEAN            src0 f32, dst f32 [1, ne1, ne2, ne3].
#include "rows.h"

namespace {

using namespace ggml_rd;

bool supports_norm(const ggml_tensor *op) {
	return rows_f32_unary(op) && ggml_are_same_shape(op->src[0], op);
}

bool supports_mean(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	return rows_f32_unary(op) && op->ne[0] == 1 && op->ne[1] == a->ne[1] && op->ne[2] == a->ne[2] &&
			op->ne[3] == a->ne[3];
}

bool pack_norm(Pack &p) {
	return pack_rows(p, pick_threads(p.node, "norm_f32", "norm_f32_t64"));
}

bool pack_rms_norm(Pack &p) {
	return pack_rows(p, pick_threads(p.node, "rms_norm_f32", "rms_norm_f32_t64"));
}

bool pack_mean(Pack &p) {
	return pack_rows(p, "mean_f32");
}

} // namespace

GGML_RD_OP(norm_f32, GGML_OP_NORM, -1, supports_norm, pack_norm);
GGML_RD_OP(rms_norm_f32, GGML_OP_RMS_NORM, -1, supports_norm, pack_rms_norm);
GGML_RD_OP(mean_f32, GGML_OP_MEAN, -1, supports_mean, pack_mean);
