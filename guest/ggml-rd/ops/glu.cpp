// ggml-rd op: GLU with GGML_GLU_OP_GEGLU_ERF, f32 (ggml_geglu_erf,
// ggml_geglu_erf_swapped, ggml_geglu_erf_split).
//
// The kernel is lean/Ggml/SlangCodegen/Glu.lean (geglu_erf_f32): one thread
// per dst element, dst[i0, i1, i2, i3] = gelu_erf(x) * g with x and g the
// two halves of src0's row (or src0 and src1 in the split form), every
// operand strided. The packer writes three derived words:
//
//   55  x's column offset in src0 (0, or nc when swapped)
//   56  the gate's column offset in src0 (nc, or 0 when swapped)
//   57  1 when the gate is src1 (split), else 0
//
// supports(): dst f32 of [nc, ne1, ne2, ne3]; src0 f32 of [2 nc, ...]
// (non-split) or of dst's shape with src1 f32 of the same shape (split, as
// ggml_glu_impl asserts); the swapped flag (op_params[1]) 0 or 1.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

constexpr uint32_t W_XOFF = W_DERIVED + 2;
constexpr uint32_t W_GOFF = W_DERIVED + 3;
constexpr uint32_t W_SPLIT = W_DERIVED + 4;
static_assert(W_XOFF == 55 && W_GOFF == 56 && W_SPLIT == 57, "Glu.lean: words 55..57");

bool supports_geglu_erf_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	const ggml_tensor *b = op->src[1];
	if (a == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || (b != nullptr && b->type != GGML_TYPE_F32)) {
		return false;
	}
	if (op->op_params[1] != 0 && op->op_params[1] != 1) {
		return false;
	}
	if (b != nullptr) {
		if (!ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, op)) {
			return false;
		}
	} else if (a->ne[0] != 2 * op->ne[0] || a->ne[1] != op->ne[1] || a->ne[2] != op->ne[2] || a->ne[3] != op->ne[3]) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a) && tensor_fits(b);
}

bool pack_geglu_erf_f32(Pack &p) {
	p.kernel = kernel_index("geglu_erf_f32");
	if (p.kernel < 0) {
		return false;
	}
	const bool split = p.node->src[1] != nullptr;
	const bool swapped = p.node->op_params[1] != 0;
	const uint32_t nc = uint32_t(p.node->ne[0]);
	p.w[W_XOFF] = split ? 0 : (swapped ? nc : 0);
	p.w[W_GOFF] = split ? 0 : (swapped ? 0 : nc);
	p.w[W_SPLIT] = split ? 1 : 0;
	return grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(geglu_erf_f32, GGML_OP_GLU, GGML_GLU_OP_GEGLU_ERF, supports_geglu_erf_f32, pack_geglu_erf_f32);
