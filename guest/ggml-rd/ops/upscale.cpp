// ggml-rd op: UPSCALE (ggml_interpolate / ggml_upscale), mode NEAREST, f32.
//
// The kernel is lean/Ggml/SlangCodegen/Upscale.lean (upscale_nearest_f32):
// one thread per dst element, dst[i] = src0[(i * src0.ne) / dst.ne] per
// dimension (integer division), both operands strided. dst's shape is the
// op's own (ggml_interpolate sets it); src0 may be any f32 view
// (test-backend-ops runs a transposed one).
//
// supports(): src0 and dst f32, one source, op_params[0] exactly
// GGML_SCALE_MODE_NEAREST: no GGML_SCALE_FLAG_ALIGN_CORNERS (ggml-cpu then
// takes (ne - 1) / (ne00 - 1) for the factor), no ANTIALIAS, no BILINEAR or
// BICUBIC. Every extent and stride must fit the uint32 words, and so must
// each dimension's product dst.ne * src0.ne (the kernel's numerator).
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_upscale_nearest_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32) {
		return false;
	}
	if (op->op_params[0] != int32_t(GGML_SCALE_MODE_NEAREST)) {
		return false;
	}
	for (int k = 0; k < 4; ++k) {
		if (op->ne[k] <= 0 || a->ne[k] <= 0 || uint64_t(op->ne[k] - 1) * uint64_t(a->ne[k]) > UINT32_MAX) {
			return false;
		}
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_upscale_nearest_f32(Pack &p) {
	p.kernel = kernel_index("upscale_nearest_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(upscale_nearest_f32, GGML_OP_UPSCALE, -1, supports_upscale_nearest_f32, pack_upscale_nearest_f32);
