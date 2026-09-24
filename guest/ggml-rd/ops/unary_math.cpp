// ggml-rd ops, family K1: the f32 math unaries ABS, SGN, STEP, TANH, EXP,
// FLOOR (GGML_OP_UNARY) and SQR, LOG, SIN, COS (ops of their own).
//
// The kernels are lean/Ggml/SlangCodegen/UnaryMath.lean: one thread per
// dst element, dst[i] = f(s0[i]), both strided, ggml-cpu's f32 formulas
// (none of these goes through an f16 table on the CPU). No op_params, no
// derived words beyond the 1-D grid.
//
// supports(): src0 and dst f32, one source, the same shape, every stride
// and extent in the uint32 params words (ops/unary.cpp's rule; ggml-cpu
// wants rows contiguous for these, the kernels take any strides).
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_math_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, op)) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_named(Pack &p, const char *name) {
	p.kernel = kernel_index(name);
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

bool pack_abs(Pack &p) { return pack_named(p, "abs_f32"); }
bool pack_sgn(Pack &p) { return pack_named(p, "sgn_f32"); }
bool pack_step(Pack &p) { return pack_named(p, "step_f32"); }
bool pack_tanh(Pack &p) { return pack_named(p, "tanh_f32"); }
bool pack_exp(Pack &p) { return pack_named(p, "exp_f32"); }
bool pack_floor(Pack &p) { return pack_named(p, "floor_f32"); }
bool pack_sqr(Pack &p) { return pack_named(p, "sqr_f32"); }
bool pack_log(Pack &p) { return pack_named(p, "log_f32"); }
bool pack_sin(Pack &p) { return pack_named(p, "sin_f32"); }
bool pack_cos(Pack &p) { return pack_named(p, "cos_f32"); }

} // namespace

GGML_RD_OP(abs_f32, GGML_OP_UNARY, GGML_UNARY_OP_ABS, supports_math_f32, pack_abs);
GGML_RD_OP(sgn_f32, GGML_OP_UNARY, GGML_UNARY_OP_SGN, supports_math_f32, pack_sgn);
GGML_RD_OP(step_f32, GGML_OP_UNARY, GGML_UNARY_OP_STEP, supports_math_f32, pack_step);
GGML_RD_OP(tanh_f32, GGML_OP_UNARY, GGML_UNARY_OP_TANH, supports_math_f32, pack_tanh);
GGML_RD_OP(exp_f32, GGML_OP_UNARY, GGML_UNARY_OP_EXP, supports_math_f32, pack_exp);
GGML_RD_OP(floor_f32, GGML_OP_UNARY, GGML_UNARY_OP_FLOOR, supports_math_f32, pack_floor);
GGML_RD_OP(sqr_f32, GGML_OP_SQR, -1, supports_math_f32, pack_sqr);
GGML_RD_OP(log_f32, GGML_OP_LOG, -1, supports_math_f32, pack_log);
GGML_RD_OP(sin_f32, GGML_OP_SIN, -1, supports_math_f32, pack_sin);
GGML_RD_OP(cos_f32, GGML_OP_COS, -1, supports_math_f32, pack_cos);
