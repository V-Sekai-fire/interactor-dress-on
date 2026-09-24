// ggml-rd ops, family K1: SQRT (GGML_OP_SQRT) and GELU_QUICK
// (GGML_OP_UNARY), f32: the two element-wise ops the see-through engine
// needs beyond ops/unary.cpp. A file of their own so a family added in
// parallel does not touch the same lines.
//
// The kernels are lean/Ggml/SlangCodegen/UnarySeeThrough.lean (sqrt_f32,
// gelu_quick_f32): one thread per dst element, dst[i] = f(s0[i]), both
// strided; no op_params, no derived words beyond the 1-D grid.
//
// supports(): src0 and dst f32, one source, the same shape, every stride
// and extent in the uint32 params words (ops/unary.cpp's rule).
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_elementwise_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32) {
		return false;
	}
	if (!ggml_are_same_shape(a, op)) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_named(Pack &p, const char *kernel) {
	p.kernel = kernel_index(kernel);
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

bool pack_sqrt(Pack &p) { return pack_named(p, "sqrt_f32"); }
bool pack_gelu_quick(Pack &p) { return pack_named(p, "gelu_quick_f32"); }

} // namespace

GGML_RD_OP(sqrt_f32, GGML_OP_SQRT, -1, supports_elementwise_f32, pack_sqrt);
GGML_RD_OP(gelu_quick_f32, GGML_OP_UNARY, GGML_UNARY_OP_GELU_QUICK, supports_elementwise_f32, pack_gelu_quick);
