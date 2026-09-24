// ggml-rd ops, family K1: SILU, GELU, GELU_ERF, SIGMOID, NEG, RELU
// (GGML_OP_UNARY), SCALE, DIAG_MASK_INF, LEAKY_RELU and CLAMP, f32.
//
// The kernels are lean/Ggml/SlangCodegen/Unary.lean: one thread per dst
// element, dst[i] = f(s0[i]), both strided (any view ggml builds: a unary
// needs rows contiguous, SCALE and DIAG_MASK_INF nothing more from us). The
// op_params words the kernels read (SCALE's s and b, DIAG_MASK_INF's n_past,
// LEAKY_RELU's negative_slope, CLAMP's min and max) are ggml's own, copied
// raw by fill_standard; no derived words beyond the 1-D grid.
//
// supports(): src0 and dst f32, one source, the same shape, every stride
// and extent in the uint32 params words. f16 is not supported (no census
// row needs it); DIAG_MASK_INF also needs n_past >= 0 (ggml asserts it).
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

bool supports_diag_mask_inf_f32(const ggml_tensor *op) {
	return supports_elementwise_f32(op) && op->op_params[0] >= 0;
}

bool pack_named(Pack &p, const char *kernel) {
	p.kernel = kernel_index(kernel);
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

bool pack_silu(Pack &p) { return pack_named(p, "silu_f32"); }
bool pack_gelu(Pack &p) { return pack_named(p, "gelu_f32"); }
bool pack_gelu_erf(Pack &p) { return pack_named(p, "gelu_erf_f32"); }
bool pack_sigmoid(Pack &p) { return pack_named(p, "sigmoid_f32"); }
bool pack_neg(Pack &p) { return pack_named(p, "neg_f32"); }
bool pack_scale(Pack &p) { return pack_named(p, "scale_f32"); }
bool pack_diag_mask_inf(Pack &p) { return pack_named(p, "diag_mask_inf_f32"); }
bool pack_relu(Pack &p) { return pack_named(p, "relu_f32"); }
bool pack_leaky_relu(Pack &p) { return pack_named(p, "leaky_relu_f32"); }
bool pack_clamp(Pack &p) { return pack_named(p, "clamp_f32"); }

} // namespace

GGML_RD_OP(silu_f32, GGML_OP_UNARY, GGML_UNARY_OP_SILU, supports_elementwise_f32, pack_silu);
GGML_RD_OP(gelu_f32, GGML_OP_UNARY, GGML_UNARY_OP_GELU, supports_elementwise_f32, pack_gelu);
GGML_RD_OP(gelu_erf_f32, GGML_OP_UNARY, GGML_UNARY_OP_GELU_ERF, supports_elementwise_f32, pack_gelu_erf);
GGML_RD_OP(sigmoid_f32, GGML_OP_UNARY, GGML_UNARY_OP_SIGMOID, supports_elementwise_f32, pack_sigmoid);
GGML_RD_OP(neg_f32, GGML_OP_UNARY, GGML_UNARY_OP_NEG, supports_elementwise_f32, pack_neg);
GGML_RD_OP(scale_f32, GGML_OP_SCALE, -1, supports_elementwise_f32, pack_scale);
GGML_RD_OP(diag_mask_inf_f32, GGML_OP_DIAG_MASK_INF, -1, supports_diag_mask_inf_f32, pack_diag_mask_inf);
GGML_RD_OP(relu_f32, GGML_OP_UNARY, GGML_UNARY_OP_RELU, supports_elementwise_f32, pack_relu);
GGML_RD_OP(leaky_relu_f32, GGML_OP_LEAKY_RELU, -1, supports_elementwise_f32, pack_leaky_relu);
GGML_RD_OP(clamp_f32, GGML_OP_CLAMP, -1, supports_elementwise_f32, pack_clamp);
