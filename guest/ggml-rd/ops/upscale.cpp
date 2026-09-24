// ggml-rd op: UPSCALE (ggml_interpolate / ggml_upscale), modes NEAREST and
// BILINEAR, f32.
//
// The kernels are lean/Ggml/SlangCodegen/Upscale.lean:
//
//   upscale_nearest_f32   one thread per dst element, dst[i] =
//                         src0[(i * src0.ne) / dst.ne] per dimension
//                         (integer division);
//   upscale_bilinear_f32  dims 0 and 1 interpolated as ggml-cpu's bilinear
//                         branch (f32 factors, pixel offset 0.5, or 0 and
//                         the (ne - 1) / (ne00 - 1) factors under
//                         GGML_SCALE_FLAG_ALIGN_CORNERS, which the kernel
//                         reads from op_params word 0), dims 2 and 3 nearest.
//
// Both operands strided; dst's shape is the op's own (ggml_interpolate sets
// it); src0 may be any f32 view (test-backend-ops runs a transposed one).
//
// supports(): src0 and dst f32, one source, op_params[0] exactly NEAREST,
// or BILINEAR with or without ALIGN_CORNERS: no ANTIALIAS (another
// filter), no BICUBIC. Every extent and stride must fit the uint32 words,
// and so must each dimension's product dst.ne * src0.ne (the nearest
// index's numerator).
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

const char *kernel_for(const ggml_tensor *op) {
	const int32_t mode = op->op_params[0];
	if (mode == int32_t(GGML_SCALE_MODE_NEAREST)) {
		return "upscale_nearest_f32";
	}
	if (mode == int32_t(GGML_SCALE_MODE_BILINEAR) ||
			mode == int32_t(GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS)) {
		return "upscale_bilinear_f32";
	}
	return nullptr;
}

bool supports_upscale_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32) {
		return false;
	}
	if (kernel_for(op) == nullptr) {
		return false;
	}
	for (int k = 0; k < 4; ++k) {
		if (op->ne[k] <= 0 || a->ne[k] <= 0 || uint64_t(op->ne[k] - 1) * uint64_t(a->ne[k]) > UINT32_MAX) {
			return false;
		}
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_upscale_f32(Pack &p) {
	p.kernel = kernel_index(kernel_for(p.node));
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(upscale_f32, GGML_OP_UPSCALE, -1, supports_upscale_f32, pack_upscale_f32);
