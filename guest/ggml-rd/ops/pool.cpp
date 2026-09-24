// ggml-rd op: POOL_1D, max and avg, f32.
//
// The kernels are lean/Ggml/SlangCodegen/Pool.lean (pool_1d_max_f32,
// pool_1d_avg_f32): one thread per dst element, the window
// [ow * s0 - p0, + k0) of its row of src0 folded as ggml-cpu folds it
// (std::max from -FLT_MAX; the f32 sum in window order over the count of
// in-range taps), both operands strided. op_params: op, k0, s0, p0 (words
// 37-40); the op picks the kernel, the kernel reads the rest.
//
// supports(): src0 and dst f32, one source, dst [OW, ne1, ne2, ne3] with
// src0's ne1..ne3, op MAX or AVG, k0 >= 1, s0 >= 1, p0 >= 0 (the kernel's
// padding test is unsigned), every extent and stride in the words and
// ow * s0 + k0 in a word. ggml-cpu walks rows by nb[1] alone; the kernel
// takes any strides.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_pool_1d_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32) {
		return false;
	}
	if (op->ne[1] != a->ne[1] || op->ne[2] != a->ne[2] || op->ne[3] != a->ne[3]) {
		return false;
	}
	const int32_t *pp = reinterpret_cast<const int32_t *>(op->op_params);
	const int32_t mode = pp[0], k0 = pp[1], s0 = pp[2], p0 = pp[3];
	if (mode != int32_t(GGML_OP_POOL_MAX) && mode != int32_t(GGML_OP_POOL_AVG)) {
		return false;
	}
	if (k0 < 1 || s0 < 1 || p0 < 0) {
		return false;
	}
	if (uint64_t(op->ne[0] - 1) * uint64_t(s0) + uint64_t(k0) > UINT32_MAX) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_pool_1d_f32(Pack &p) {
	const int32_t mode = reinterpret_cast<const int32_t *>(p.node->op_params)[0];
	p.kernel = kernel_index(mode == int32_t(GGML_OP_POOL_MAX) ? "pool_1d_max_f32" : "pool_1d_avg_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(pool_1d_f32, GGML_OP_POOL_1D, -1, supports_pool_1d_f32, pack_pool_1d_f32);
