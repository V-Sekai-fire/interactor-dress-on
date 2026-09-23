// ggml-rd ops: CPY, DUP and CONT among f32, f16 and bf16 (and i32 -> i32,
// i16 -> i16 as bits), any strides on both sides, and reshaping copies (the
// shapes may differ; ggml only asks for equal element counts).
//
// The kernels are lean/Ggml/SlangCodegen/Cpy.lean: the destination element
// at iteration index i has ggml linear index l = sum_k i_k * w[56 + k], and
// its source is element l of src0 in ggml order. The iteration order is
// dst's dimensions sorted by stride (ops/move.h), so writes are coalesced
// whatever dst's permutation, and a 16-bit dst is written a word per thread.
//
// supports: src0 and dst types one of the (src, dst) pairs below, equal
// element counts, every stride and extent in the uint32 words, a 16-bit dst
// nested (move.h). A CPY's node is a view of src1 with src1's layout; the
// kernel writes through the node and never reads src1.
#include "move.h"

namespace {

using namespace ggml_rd;

const char *kernel_for(ggml_type s, ggml_type d) {
	if (s == d) {
		switch (s) {
			case GGML_TYPE_F32:
			case GGML_TYPE_I32:
				return "cpy_b32";
			case GGML_TYPE_F16:
			case GGML_TYPE_BF16:
			case GGML_TYPE_I16:
				return "cpy_b16";
			default:
				return nullptr;
		}
	}
	if (s == GGML_TYPE_F32 && d == GGML_TYPE_F16) return "cpy_f32_f16";
	if (s == GGML_TYPE_F32 && d == GGML_TYPE_BF16) return "cpy_f32_bf16";
	if (s == GGML_TYPE_F16 && d == GGML_TYPE_F32) return "cpy_f16_f32";
	if (s == GGML_TYPE_BF16 && d == GGML_TYPE_F32) return "cpy_bf16_f32";
	if (s == GGML_TYPE_F16 && d == GGML_TYPE_BF16) return "cpy_f16_bf16";
	if (s == GGML_TYPE_BF16 && d == GGML_TYPE_F16) return "cpy_bf16_f16";
	return nullptr;
}

bool supports_cpy(const ggml_tensor *op) {
	const ggml_tensor *s = op->src[0];
	if (s == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->op == GGML_OP_CPY ? op->src[1] == nullptr : op->src[1] != nullptr) {
		return false;
	}
	if (kernel_for(s->type, op->type) == nullptr || ggml_nelements(s) != ggml_nelements(op)) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(s) && move::dst_ok(op);
}

bool pack_cpy(Pack &p) {
	p.kernel = kernel_index(kernel_for(p.node->src[0]->type, p.node->type));
	if (p.kernel < 0) {
		return false;
	}
	uint64_t lin[4]; // dst's ggml linear weight of each original dimension
	lin[0] = 1;
	for (int k = 1; k < 4; ++k) {
		lin[k] = lin[k - 1] * uint64_t(p.node->ne[k - 1]);
	}
	move::Order o;
	if (!move::apply_order(p, o, {})) {
		return false;
	}
	for (int k = 0; k < 4; ++k) {
		p.w[move::W_LINW + k] = uint32_t(lin[o.perm[k]]);
	}
	return grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(cpy, GGML_OP_CPY, -1, supports_cpy, pack_cpy);
GGML_RD_OP(dup, GGML_OP_DUP, -1, supports_cpy, pack_cpy);
GGML_RD_OP(cont, GGML_OP_CONT, -1, supports_cpy, pack_cpy);
