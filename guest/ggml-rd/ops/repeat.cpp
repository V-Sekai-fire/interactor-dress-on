// ggml-rd op: REPEAT, f32/i32 or f16/bf16/i16 (src0 and dst the same type),
// any strides.
//
// The kernels are lean/Ggml/SlangCodegen/Repeat.lean:
// dst[i] = src0[i mod src0.ne]; the packer permutes dst and src0 alike into
// dst's iteration order (ops/move.h).
#include "move.h"

namespace {

using namespace ggml_rd;

const char *kernel_for(ggml_type t) {
	switch (t) {
		case GGML_TYPE_F32:
		case GGML_TYPE_I32:
			return "repeat_b32";
		case GGML_TYPE_F16:
		case GGML_TYPE_BF16:
		case GGML_TYPE_I16:
			return "repeat_b16";
		default:
			return nullptr;
	}
}

bool supports_repeat(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (kernel_for(op->type) == nullptr || a->type != op->type || !ggml_can_repeat(a, op)) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a) && move::dst_ok(op);
}

bool pack_repeat(Pack &p) {
	p.kernel = kernel_index(kernel_for(p.node->type));
	move::Order o;
	return p.kernel >= 0 && move::apply_order(p, o, { W_SRC0 }) && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(repeat, GGML_OP_REPEAT, -1, supports_repeat, pack_repeat);
