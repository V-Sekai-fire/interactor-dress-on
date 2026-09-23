// ggml-rd op: CONCAT along any dimension, f32/i32 or f16/bf16/i16 (the same
// type on all three tensors), any strides.
//
// The kernels are lean/Ggml/SlangCodegen/Concat.lean: an index inside src0's
// extent reads src0, else src1 at the index minus src0's extent on the one
// dimension it is past; the dimension itself is never needed, so the packer
// permutes dst, src0 and src1 alike into dst's iteration order (ops/move.h)
// for coalesced writes and a word per thread on a 16-bit dst.
#include "move.h"

namespace {

using namespace ggml_rd;

const char *kernel_for(ggml_type t) {
	switch (t) {
		case GGML_TYPE_F32:
		case GGML_TYPE_I32:
			return "concat_b32";
		case GGML_TYPE_F16:
		case GGML_TYPE_BF16:
		case GGML_TYPE_I16:
			return "concat_b16";
		default:
			return nullptr;
	}
}

bool supports_concat(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	const ggml_tensor *b = op->src[1];
	if (a == nullptr || b == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (kernel_for(op->type) == nullptr || a->type != op->type || b->type != op->type) {
		return false;
	}
	const int dim = op->op_params[0]; // ggml_concat: op_params[0] = dim
	if (dim < 0 || dim > 3) {
		return false;
	}
	for (int k = 0; k < 4; ++k) {
		const int64_t want = k == dim ? a->ne[k] + b->ne[k] : a->ne[k];
		if (op->ne[k] != want || (k != dim && b->ne[k] != a->ne[k])) {
			return false;
		}
	}
	return tensor_fits(op) && tensor_fits(a) && tensor_fits(b) && move::dst_ok(op);
}

bool pack_concat(Pack &p) {
	p.kernel = kernel_index(kernel_for(p.node->type));
	move::Order o;
	return p.kernel >= 0 && move::apply_order(p, o, { W_SRC0, W_SRC1 }) &&
			grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(concat, GGML_OP_CONCAT, -1, supports_concat, pack_concat);
